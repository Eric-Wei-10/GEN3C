import argparse
import os
import glob

import numpy as np
import torch
import torch.nn.functional as F
import cv2


def backproject_depth_to_world(ref_depth, ref_K, ref_w2c):
    """
    Backproject first-frame (input-frame) depth into world coordinates.

    ref_depth: [H, W]
    ref_K:     [3, 3]  (direction camera-to-pixel intrinsics of first frame)
    ref_w2c:   [4, 4]  (world-to-camera extrinsics of first frame)

    returns:
        X_world: [3, H * W]  (3D points in world coordinates (flattened))
    """
    device = ref_depth.device
    H, W = ref_depth.shape

    # Pixel grid (u = x, v = y).
    # use same dtype as ref_depth / ref_K.
    dtype = ref_depth.dtype

    # ys: [H, W], ys[v, u] = v (row index).
    # xs: [H, W], xs[v, u] = u (column index).
    ys, xs = torch.meshgrid(
        torch.arange(H, device=device, dtype=dtype),
        torch.arange(W, device=device, dtype=dtype),
        indexing="ij"
    )

    # 1. Build homogeneous pixel coordinates (u, v, 1)^T for each pixel and then flatten.
    ones = torch.ones_like(xs, dtype=dtype)
    pix = torch.stack([xs, ys, ones], dim=-1)           # [H, W, 3]
    pix_flat = pix.reshape(-1, 3).T                     # [3, H * W]

    # Depths flattened.
    depth_flat = ref_depth.reshape(-1)                  # [H * W]

    # 2. Pixels -> Camera rays in first-frame camera coords.
    # For each pixel (u, v, 1)^T: X_cam = depth * K_inv @ (u, v, 1)^T 
    # where the direction of the ray is K_inv @ (u, v, 1)^T.
    K_inv = torch.inverse(ref_K)                        # [3, 3]
    rays_cam = K_inv @ pix_flat                         # [3, H * W]
    # Scale rays by depth.
    X_cam = rays_cam * depth_flat                       # [3, H * W]

    # 3. Camera -> World.
    # Convert to homogeneous coordinates (x, y, z, 1)^T so we can apply 4x4 transform.
    ones_row = torch.ones(1, X_cam.shape[1], device=device)
    X_cam_h = torch.cat([X_cam, ones_row], dim=0)       # [4, H * W]
    ref_c2w = torch.inverse(ref_w2c)                    # [4, 4]
    # Transform all camera-space points to world space
    X_world_h = ref_c2w @ X_cam_h                       # [4, H * W]
    # Drop the homogeneous coordinate.
    X_world = X_world_h[:3]                             # [3, H * W]

    return X_world


def warp_frame_to_first_view(frame, K_t, w2c_t, X_world, H, W):
    """
    Warp ONE video frame at time t into first-frame pixel coordinates.

    frame:  [C, H, W]     (source frame at time t)
    K_t:    [3, 3]        (direction camera-to-pixel intrinsics at time t)
    w2c_t:  [4, 4]        (world-to-camera at time t)
    X_world:[3, H * W]    (3D points from first-frame depth)
    H, W:   ints          (shape of the first frame)

    returns:
        warped: [1, C, H, W]
        mask:   [1, 1, H, W]   (1 where valid projection, 0 otherwise)
    """
    device = frame.device
    N = X_world.shape[1]

    # 1. Convert X_world to homogeneous coordinates for matrix multiplication.
    ones_row = torch.ones(1, N, device=device)
    X_world_h = torch.cat([X_world, ones_row], dim=0)   # [4, H * W]

    # 2. World -> camera at frame t for this video.
    X_cam_h = w2c_t @ X_world_h                         # [4, H * W]
    X_cam = X_cam_h[:3]                                 # [3, H * W]
    # Extract depth (Z) from camera coordinates.
    Z = X_cam[2:3]                                      # [1, H * W]
    # Must be in front of camera: keep only points in front of the camera (z > 0);
    # otherwise invalid projection.
    valid_z = Z > 0
    # Normalize by depth to get direction in camera coordinates: (X, Y, Z) -> (X/Z, Y/Z, 1).
    X_norm = X_cam / (Z + 1e-8)                         # [3, H * W]

    # 3. Project the normalized camera coordinates to pixels at frame t.
    uv_h = K_t @ X_norm                                 # [3, H * W]
    u = uv_h[0]                                         # [H * W]
    v = uv_h[1]                                         # [H * W]
    # Reshape to [H, W].
    u_img = u.reshape(H, W)
    v_img = v.reshape(H, W)

    # 4. Check which pixels are valid (inside image bounds).
    # A pixel is valid if:
    # - the corresponding 3D point is in front of the camera (valid_z).
    # - the projected (u, v) falls within [0, W-1] and [0, H-1].
    valid_xy = (
        (u_img >= 0) & (u_img <= (W - 1)) &
        (v_img >= 0) & (v_img <= (H - 1))
    )
    valid = (valid_z.reshape(H, W) & valid_xy)          # [H, W]

    # 5. Pixel -> normalized coords in normalized grid [-1, 1] (align_corners=True).
    # F.grid_sample expects normalized coordinates in [-1, 1] where -1, 1 correspond to the image borders.
    # -1: left/top border (u = 0 or v = 0).
    # 1: right/bottom border (u = W - 1 or v = H - 1).
    x_norm = 2.0 * (u_img / (W - 1)) - 1.0
    y_norm = 2.0 * (v_img / (H - 1)) - 1.0
    grid = torch.stack([x_norm, y_norm], dim=-1)        # [H, W, 2]
    grid = grid.unsqueeze(0)                            # [1, H, W, 2]
    # Frame to [1, C, H, W]: make the frame batch-shaped (batch size 1) for grid_sample.
    frame_b = frame.unsqueeze(0)                        # [1, C, H, W]

    # 6. Sample those source frame (frame t) at the projected locations using grid_sample.
    # For each pixel in the first frame, we fetch the color from frame t at the projected pixel.
    # Use bilinear interpolation for smooth sampling, and zero padding for out-of-bounds.
    warped = F.grid_sample(
        frame_b, grid,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=True,
    )  # [1, C, H, W]

    # 7. Build a visibility mask indicating where sampling is meaningful (in front of camera and in-bounds).
    mask = valid.unsqueeze(0).unsqueeze(0).float()      # [1, 1, H, W]

    return warped, mask


def masked_variance_across_videos(warped_all, masks_all):
    """
    Variance across N videos with masks.

    warped_all: [N, C, H, W]
    masks_all:  [N, 1, H, W]  (1 = valid sample from that video)

    returns:
        var_map:      [C, H, W]
        valid_counts: [H, W]
    """
    _, C, _, _ = warped_all.shape

    # Broadcast masks to channels: expand masks 
    # from [N, 1, H, W] to [N, C, H, W] to match the color channels.
    mask_bc = masks_all.expand(-1, C, -1, -1)               # [N, C, H, W]

    # valid_counts[v, u]: number of videos that have a valid sample at pixel (u, v) (sees the 3D point).
    valid_counts = masks_all.sum(dim=0)                     # [1, 1, H, W]

    # print("min valid_counts:", valid_counts.min().item())
    # print("max valid_counts:", valid_counts.max().item())

    # Clamp to avoid division by zero when computing mean.
    valid_counts_clamped = torch.clamp(valid_counts, min=1.0)

    # Mask / Zero out invalid entries (since mask is 0 there).
    masked_vals = warped_all * mask_bc                      # [N, C, H, W]
    # Sum along the video dimension.
    sum_vals = masked_vals.sum(dim=0)                       # [C, H, W]
    # Compute the mean color per pixel across videos.
    mean = sum_vals / valid_counts_clamped.squeeze(0)       # [C, H, W]

    # Variance: E[(X - mu)^2]
    # Compute squared differences from the mean, only on valid samples.
    diff = (warped_all - mean.unsqueeze(0)) * mask_bc
    sq_sum = (diff ** 2).sum(dim=0)                         # [C, H, W]

    # Compute unbiased variance: divide by (valid_counts - 1).
    # Clamp denominator to at least 1 to avoid division-by-zero.
    denom = torch.clamp(valid_counts - 1.0, min=1.0)        # [1, 1, H, W]
    var = sq_sum / denom.squeeze(0)                         # [C, H, W]

    # If fewer than 2 videos are valid at a pixel, variance is undefined; set variance = 0.
    at_least_two = (valid_counts >= 2.0)                    # [1, 1, H, W]
    var = torch.where(
        at_least_two.squeeze(0).expand_as(var),
        var,
        torch.zeros_like(var)
    )

    return var, valid_counts.squeeze(0).squeeze(0)          # [H, W]


def compute_variance_on_first_frame_across_videos(
    frames_t,      # [N, C, H, W]   (frames at chosen time t from each video)
    K_0,           # [3, 3]         (intrinsics of frame 0)
    w2c_0,         # [4, 4]         (w2c of frame 0)
    K_t,           # [3, 3]         (intrinsics at frame t)
    w2c_t,         # [4, 4]         (w2c at frame t)
    depth_0        # [H, W]         (depth of frame 0)
):
    """
    1. Use first-frame depth to define 3D points.
    2. Use camera at t (K_t, w2c_t) to project those 3D points into each video's frame t.
    3. Compute variance across videos for each first-frame pixel,
    using only the intersection of visibility across videos.

    returns:
        var_map:      [C, H, W]
        valid_counts: [H, W]
        inter_mask:   [H, W]   (1 where all videos see the 3D point, else 0)
    """
    N, _, H, W = frames_t.shape

    # 1. Backproject first-frame depth -> worLd: use first frame's depth + (K_0, w2c_0)
    # to get the canonical 3D points in world coordinates.
    X_world = backproject_depth_to_world(depth_0, K_0, w2c_0)       # [3, H * W]

    # 2. Warp each video's frame t into first-frame coords.
    # For each video, project same 3D points X_world into that video's frame t using (K_t, w2c_t).
    # And store the warped images and visibility masks.
    warped_list = []
    mask_list = []
    for i in range(N):
        warped, mask = warp_frame_to_first_view(
            frame=frames_t[i],
            K_t=K_t,
            w2c_t=w2c_t,
            X_world=X_world,
            H=H,
            W=W,
        )
        warped_list.append(warped)                                  # [1, C, H, W]
        mask_list.append(mask)                                      # [1, 1, H, W]

    warped_all = torch.cat(warped_list, dim=0)                      # [N, C, H, W]
    masks_all = torch.cat(mask_list, dim=0)                         # [N, 1, H, W]

    var_map, valid_counts = masked_variance_across_videos(
        warped_all, masks_all
    )

    # print("valid counts:", valid_counts)
    # print("min valid_counts:", valid_counts.min().item())
    # print("max valid_counts:", valid_counts.max().item())

    min_videos = N
    inter_mask = (valid_counts >= min_videos).float()  # [H, W]

    return var_map, valid_counts, inter_mask


def load_frame_from_video(video_path, frame_index):
    """
    Load a single frame (frame_index) from a video file using OpenCV.

    Returns:
        frame: [H, W, 3] float32 in [0, 1], RGB order.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if frame_index < 0 or frame_index >= total_frames:
        cap.release()
        raise ValueError(
            f"Requested frame_index {frame_index} out of range for {video_path} "
            f"(0..{total_frames-1})."
        )

    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ret, frame = cap.read()
    cap.release()

    if not ret:
        raise RuntimeError(f"Failed to read frame {frame_index} from {video_path}")

    # frame: [H, W, 3] BGR uint8
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = frame.astype(np.float32) / 255.0  # [0,1]
    return frame


def load_frames_at_t_from_dir(videos_dir, frame_index):
    """
    Load frame_index from ALL videos in videos_dir.

    Returns:
        frames_t: [N, C, H, W] float32.
        video_paths: list of paths used (sorted).
    """
    exts = (".mp4", ".avi", ".mov", ".mkv", ".mpg", ".mpeg", ".wmv", ".m4v")
    all_files = sorted(
        f for f in glob.glob(os.path.join(videos_dir, "*"))
        if f.lower().endswith(exts)
    )

    if len(all_files) == 0:
        raise RuntimeError(f"No video files found in {videos_dir}.")

    frames = []
    H_ref, W_ref = None, None

    for path in all_files:
        frame = load_frame_from_video(path, frame_index)  # [H, W, 3]
        H, W, _ = frame.shape

        if H_ref is None:
            H_ref, W_ref = H, W
        else:
            if (H, W) != (H_ref, W_ref):
                raise ValueError(
                    f"Video {path} has resolution {(H, W)}, expected {(H_ref, W_ref)}."
                )

        # Convert to [C, H, W]
        frame_chw = np.transpose(frame, (2, 0, 1))  # [3, H, W]
        frames.append(frame_chw)

    frames_t = np.stack(frames, axis=0)  # [N, C, H, W]
    return frames_t, all_files


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Compute per-pixel variance across videos at a chosen frame index, "
            "and express it on the first frame's pixel grid, "
            "using depth0 + (K, w2c) from a camera_info .npz."
        )
    )
    parser.add_argument(
        "--videos_dir", type=str, required=True,
        help="Path to folder containing video files (e.g. mp4, avi, ...)."
    )
    parser.add_argument(
        "--camera_npz", type=str, required=True,
        help=(
            "Path to camera .npz file with keys:\n"
            "  depth0 -> (H, W)\n"
            "  w2c    -> (T, 4, 4)\n"
            "  K      -> (T, 3, 3)"
        )
    )
    parser.add_argument(
        "--frame_index", type=int, required=True,
        help="Frame index t at which to take frames from each video."
    )
    parser.add_argument(
        "--output_npz", type=str, required=True,
        help="Output .npz path to save: var_map, valid_counts, intersection_mask, frame_index."
    )

    args = parser.parse_args()

    # Load camera info (depth0, K_all, w2c_all).
    cam_data = np.load(args.camera_npz)
    D0 = cam_data["depth0"].astype(np.float32)          # (H, W)
    w2c_all = cam_data["w2c"].astype(np.float32)        # (T, 4, 4)
    K_all = cam_data["K"].astype(np.float32)            # (T, 3, 3)

    # Check 1 (Dimensions): check that we have K and w2c for all frames.
    T_cam = w2c_all.shape[0]
    if K_all.shape[0] != T_cam:
        raise ValueError(
            f"K_all and w2c_all length mismatch: {K_all.shape[0]} vs {T_cam}."
        )

    # Check 2 (frame_index in range): check that the requested frame_index is valid.
    t = args.frame_index
    if t < 0 or t >= T_cam:
        raise ValueError(f"frame_index {t} out of range [0, {T_cam-1}].")

    # Extract camera intrinsics/extrinsics for frame 0 and frame t.
    K0 = K_all[0]           # (3, 3)
    w2c0 = w2c_all[0]       # (4, 4)
    K_t = K_all[t]          # (3, 3)
    w2c_t = w2c_all[t]      # (4, 4)

    # Load frame t from all videos in videos_dir.
    frames_t_np, video_paths = load_frames_at_t_from_dir(args.videos_dir, t)
    _, _, H, W = frames_t_np.shape

    # Check 3 (Depth resolution): check that depth0 resolution matches video resolution.
    if D0.shape != (H, W):
        raise ValueError(
            f"Depth resolution {D0.shape} does not match video resolution {(H, W)}."
        )

    # Convert NumPy arrays to PyTorch tensors and move them to CPU or GPU.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    frames_t = torch.from_numpy(frames_t_np).to(device)     # [N, C, H, W]
    depth0 = torch.from_numpy(D0).to(device)                # [H, W]
    K0_t = torch.from_numpy(K0).to(device)                  # [3, 3]
    w2c0_t = torch.from_numpy(w2c0).to(device)              # [4, 4]
    Kt_t = torch.from_numpy(K_t).to(device)                 # [3, 3]
    w2ct_t = torch.from_numpy(w2c_t).to(device)             # [4, 4]

    # Compute the per-pixel variance map on the first frame across videos and compute intersection mask.
    var_map, valid_counts, inter_mask = compute_variance_on_first_frame_across_videos(
        frames_t=frames_t,
        K_0=K0_t,
        w2c_0=w2c0_t,
        K_t=Kt_t,
        w2c_t=w2ct_t,
        depth_0=depth0,
    )

    # Save outputs.
    var_map_np = var_map.cpu().numpy()                  # [C, H, W] (per-pixel per-channel variance)
    valid_counts_np = valid_counts.cpu().numpy()        # [H, W]
    inter_mask_np = inter_mask.cpu().numpy()            # [H, W]

    np.savez(
        args.output_npz,
        var_map=var_map_np,
        valid_counts=valid_counts_np,
        intersection_mask=inter_mask_np,
        frame_index=np.int64(t),
        video_paths=np.array(video_paths),
    )


if __name__ == "__main__":
    main()
