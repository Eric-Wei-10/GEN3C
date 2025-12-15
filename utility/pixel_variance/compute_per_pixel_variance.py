import argparse
import os
import glob

import numpy as np
import torch
import torch.nn.functional as F
import cv2


# ============================================================
# IO / Loading Helpers
# ============================================================

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


# ============================================================
# Geometry Helpers
# ============================================================

def make_pixel_grid(H, W, device, dtype):
    """
    Create a pixel grid of shape [3, H * W] with homogeneous coordinates.
    """
    ys, xs = torch.meshgrid(
        torch.arange(H, device=device, dtype=dtype),
        torch.arange(W, device=device, dtype=dtype),
        indexing="ij",
    )
    ones = torch.ones_like(xs, dtype=dtype)
    pix = torch.stack([xs, ys, ones], dim=-1)   # [H, W, 3]
    return pix.reshape(-1, 3).T                 # [3, H * W]


def backproject_depth0_to_world(depth0, K0, w2c0):
    """
    Backproject first-frame (input-frame) depth into world coordinates.

    depth0:    [H, W]
    K0:        [3, 3]  (direction camera-to-pixel intrinsics of the first frame)
    w2c0:      [4, 4]  (world-to-camera extrinsics of the first frame)

    Returns:
      X_world: [3, H * W]  (3D points in world coordinates (flattened))
      valid0:  [H * W]     (bool; indicates valid depths)
    """
    device = depth0.device
    dtype = depth0.dtype
    H, W = depth0.shape

    # 1. Build homogeneous pixel coordinates (u, v, 1)^T for each pixel and then flatten.
    pix_flat = make_pixel_grid(H, W, device, dtype)         # [3, H * W]
    # Depths flattened.
    depth_flat = depth0.reshape(-1)                         # [H * W]

    # Only backproject valid depths (>0).
    valid0 = depth_flat > 0

    # 2. Pixels -> Camera rays in first-frame camera coords.
    # For each pixel (u, v, 1)^T: X_cam = depth * K_inv @ (u, v, 1)^T 
    # where the direction of the ray is K_inv @ (u, v, 1)^T.
    K_inv = torch.inverse(K0)
    rays_cam0 = K_inv @ pix_flat                            # [3, H * W]
    # Scale rays by depth.
    X_cam0 = rays_cam0 * depth_flat                         # [3, H * W]

    # 3. Camera -> World.
    # Convert to homogeneous coordinates (x, y, z, 1)^T so we can apply 4x4 transform.
    ones = torch.ones((1, H * W), device=device, dtype=dtype)
    X_cam0_h = torch.cat([X_cam0, ones], dim=0)             # [4, H * W]
    c2w0 = torch.inverse(w2c0)                              # [4, 4]
    X_world_h = c2w0 @ X_cam0_h                             # [4, H * W]
    X_world = X_world_h[:3]                                 # [3, H * W]

    return X_world, valid0


# ============================================================
# Forward Mode
# ============================================================

def build_forward_grid_and_mask(X_world, valid0, K_t, w2c_t, H, W):
    """
    Build a sampling grid and mask for forward warping from world to frame t,
    which can be used with F.grid_sample to sample frame t into frame 0.

    X_world: [3, H * W]     (3D points in world coordinates from first-frame depth)
    valid0:  [H * W]        (bool; indicates which 3D points are valid)
    K_t:     [3, 3]         (direction camera-to-pixel intrinsics of frame t)
    w2c_t:   [4, 4]         (world-to-camera extrinsics of frame t)
    H, W:    int            (shape of the target frame)

    Returns:
      grid: [1, H, W, 2]    (float; normalized coordinates for grid_sample)
      mask: [1, 1, H, W]    (float; 1.0 where valid projection; 0.0 = invalid)
    """
    device = X_world.device
    dtype = X_world.dtype
    HW = X_world.shape[1]

    # 1. Convert X_world to homogeneous coordinates for matrix multiplication.
    ones = torch.ones((1, HW), device=device, dtype=dtype)
    X_world_h = torch.cat([X_world, ones], dim=0)               # [4, H * W]

    # 2. World -> camera at frame t for this video.
    Xt_cam_h = w2c_t @ X_world_h                                # [4, H * W]
    Xt_cam = Xt_cam_h[:3]                                       # [3, H * W]
    # Extract depth (Z) from camera coordinates.
    Z_t = Xt_cam[2:3]                                           # [1, H * W]
    # Must be in front of camera: keep only points in front of the camera (z > 0);
    # otherwise invalid projection.
    valid_z = Z_t > 0
    # Normalize by depth to get direction in camera coordinates: (X, Y, Z) -> (X/Z, Y/Z, 1).
    Xt_norm = Xt_cam / (Z_t + 1e-8)                             # [3, H * W]

    # 3. Project the normalized camera coordinates to pixels at frame t.
    uv_h = K_t @ Xt_norm                                        # [3, H * W]
    u = uv_h[0]                                                 # [H * W]
    v = uv_h[1]             # [H * W]
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
    valid = (valid0.reshape(H, W) & valid_z.reshape(H, W) & valid_xy.reshape(H, W))         # [H, W] bool

    # 5. Pixel -> normalized coords in normalized grid [-1, 1] (align_corners=True).
    # F.grid_sample expects normalized coordinates in [-1, 1] where -1, 1 correspond to the image borders.
    # -1: left/top border (u = 0 or v = 0).
    # 1: right/bottom border (u = W - 1 or v = H - 1).
    x_norm = 2.0 * (u_img / (W - 1)) - 1.0
    y_norm = 2.0 * (v_img / (H - 1)) - 1.0
    grid = torch.stack([x_norm, y_norm], dim=-1).unsqueeze(0)   # [1, H, W, 2]

    # 6. Build a visibility mask indicating where sampling is meaningful (in front of camera and in-bounds).
    mask = valid.view(H, W).unsqueeze(0).unsqueeze(0).float()   # [1, 1, H, W]

    return grid, mask


def warp_batch_forward(frames_t, grid, mask):
    """
    Forward warp a batch of frames using the provided grid and mask.
    Sample frames_t at the locations specified by grid using bilinear interpolation 
    and zero padding for out-of-bounds.
    For each pixel in the first frame, we fetch the color from frame t at the projected pixel.

    frames_t: [N, C, H, W]      (input frames at time t)
    grid: [1, H, W, 2]          (sampling grid for grid_sample)
    mask: [1, 1, H, W]          (visibility mask)

    Returns:
      warped: [N, C, H, W]      (warped frames)
      masks:  [N, 1, H, W]      (expanded masks; 1.0 where valid, 0.0 where invalid)
    """
    N = frames_t.shape[0]
    gridN = grid.expand(N, -1, -1, -1)
    warped = F.grid_sample(
        frames_t, gridN,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=True,
    )
    masks = mask.expand(N, -1, -1, -1)

    return warped, masks


# ============================================================
# Backward Mode
# ============================================================

def project_t_to_0_and_splat_bilinear(frame_t, depth_t, mask_t, K_t, w2c_t, K0, w2c0, pix_flat, H, W):
    """
    Backward warp frame t to frame 0: pixel in frame t -> 3D via depth_t -> camera0 -> project -> splat to frame0 grid.
    Then splat the pixel color onto frame0 grid using bilinear weights.

    frame_t:  [C, H, W]     (float32; the frame at time t for a this video)
    depth_t:  [H, W]        (float32; depth at time t for this video)
    mask_t:   [H, W]        (float32; validity mask at time t for this video)
    K_t:       [3, 3]       (direction camera-to-pixel intrinsics of frame t)
    w2c_t:     [4, 4]       (world-to-camera extrinsics of frame t)
    K0:       [3, 3]        (direction camera-to-pixel intrinsics of frame 0)
    w2c0:     [4, 4]        (world-to-camera extrinsics of frame 0)
    pix_flat: [3, H*W]      (homogeneous pixel coords)
    H, W:     int           (height, width)

    Returns:
      warped: [1, C, H, W]   (warped frame t projected to frame0 grid)
      mask:   [1, 1, H, W]   (1.0 where valid projection; 0.0 = invalid)
    """
    device = frame_t.device
    dtype = depth_t.dtype
    C = frame_t.shape[0]

    # Flatten depth and mask.
    d = depth_t.reshape(-1)                     # [H * W]
    reliable = (mask_t.reshape(-1) > 0.5)       # [H * W] bool
    # Only keep pixels with reliable depth and positive depth.
    valid = reliable & (d > 0)                  # [H * W] bool

    # If no valid pixels, return all zeros and an all-zero mask.
    if not valid.any():
        warped = torch.zeros((1, C, H, W), device=device, dtype=frame_t.dtype)
        m = torch.zeros((1, 1, H, W), device=device, dtype=torch.float32)
        return warped, m

    # 1. Backproject pixels in frame t to 3D points in world coordinates.
    Kt_inv = torch.inverse(K_t)
    rays = Kt_inv @ pix_flat                    # [3, H * W]
    Xt = rays * d                               # [3, H * W]

    # 2. Convert 3D points of frame t in camera t coordinates to 
    # homogeneous coordinates (x, y, z, 1) for matrix multiplication.
    ones = torch.ones((1, H * W), device=device, dtype=dtype)
    Xt_h = torch.cat([Xt, ones], dim=0)         # [4, H * W]

    # 3. Convert points from camera t to world coordinates.
    c2w_t = torch.inverse(w2c_t)
    X_world_h = c2w_t @ Xt_h

    # 4. Convert points from world coordinates to camera 0 coordinates.
    X0_h = w2c0 @ X_world_h
    X0 = X0_h[:3]
    Z0 = X0[2]

    # Keep only points in front of camera 0.
    valid = valid & (Z0 > 0)
    # If nothing is in front of camera 0, return empty outputs.
    if not valid.any():
        warped = torch.zeros((1, C, H, W), device=device, dtype=frame_t.dtype)
        m = torch.zeros((1, 1, H, W), device=device, dtype=torch.float32)
        return warped, m

    # 5. Project points in camera 0 to pixels.
    # Perspective divide to normalized camera coords.
    X0_norm = X0 / (Z0.unsqueeze(0) + 1e-8)
    # Multiply by intrinsics to get pixel coordinates (u0, v0) in frame 0.
    uv0 = K0 @ X0_norm
    u0 = uv0[0]
    v0 = uv0[1]

    # Keep only points that project inside frame 0 bounds.
    in_bound = (u0 >= 0) & (u0 <= (W - 1)) & (v0 >= 0) & (v0 <= (H - 1))
    valid = valid & in_bound
    # If all points are out of bounds, return empty outputs.
    if not valid.any():
        warped = torch.zeros((1, C, H, W), device=device, dtype=frame_t.dtype)
        m = torch.zeros((1, 1, H, W), device=device, dtype=torch.float32)
        return warped, m

    # Get indices of valid pixels.
    # P: number of valid pixels.
    valid_idx = torch.nonzero(valid, as_tuple=False).squeeze(1)  # [P]

    # Flatten frame_t for easy indexing.
    frame_flat = frame_t.reshape(C, H * W)
    # Get colors of valid pixels in frame_t.
    colors = frame_flat[:, valid_idx]                           # [C, P]
    # Get projected pixel coordinates in frame0 for valid pixels.
    u = u0[valid_idx]
    v = v0[valid_idx]

    # 6. Splat colors onto frame0 grid using bilinear weights.
    # (u0f, v0f): top-left integer pixel coordinates.
    # du, dv: fractional parts inside the pixel cell.
    u0f = torch.floor(u)
    v0f = torch.floor(v)
    du = (u - u0f).clamp(0, 1)
    dv = (v - v0f).clamp(0, 1)
    # Find the 4 neighboring pixel indices and clamp to image bounds.
    u00 = u0f.long().clamp(0, W - 1)
    v00 = v0f.long().clamp(0, H - 1)
    u01 = (u00 + 1).clamp(0, W - 1)
    v01 = (v00 + 1).clamp(0, H - 1)
    # Stardard bilinear weights for the 4 neighbors.
    w00 = (1 - du) * (1 - dv)
    w10 = du * (1 - dv)
    w01 = (1 - du) * dv
    w11 = du * dv
    # Compute flattened indices for the 4 neighbors.
    idx00 = v00 * W + u00
    idx10 = v00 * W + u01
    idx01 = v01 * W + u00
    idx11 = v01 * W + u01

    # sum_colors: sum of weighted colors for each destination pixel in frame 0.
    # sum_weights: sum of weights for each destination pixel in frame 0.
    sum_colors = torch.zeros((C, H * W), device=device, dtype=frame_t.dtype)
    sum_weights = torch.zeros((H * W,), device=device, dtype=frame_t.dtype)

    # For each of the 4 neighbors,
    # - Add the weighted color to sum_colors into the destination pixel bin.
    # - Add the weight to sum_weights into the destination pixel bin.
    # Note: using scatter_add_ to handle multiple contributions to the same pixel.
    for idx, w in [(idx00, w00), (idx10, w10), (idx01, w01), (idx11, w11)]:
        sum_colors.scatter_add_(1, idx.unsqueeze(0).expand(C, -1), colors * w.unsqueeze(0))
        sum_weights.scatter_add_(0, idx, w)

    # Compute final warped colors by normalizing with sum_weights: sum_colors / sum_weights.
    # Avoid division by zero by adding a small epsilon.
    warped_flat = sum_colors / (sum_weights.unsqueeze(0) + 1e-8)
    # Mask indicating which pixels received any contribution (sum_weights > 0).
    mask_flat = (sum_weights > 0).float()

    # Reshape back to image tensor [1, C, H, W] and [1, 1, H, W].
    warped = warped_flat.view(C, H, W).unsqueeze(0)
    m = mask_flat.view(1, H, W).unsqueeze(0)

    return warped, m


# ============================================================
# Variance Helpers
# ============================================================

def masked_variance_across_videos(warped_all, masks_all):
    """
    Compute per-pixel variance across videos, considering only valid pixels indicated by masks.

    warped_all: [N, C, H, W]    (warped frames from N videos)
    masks_all:  [N, 1, H, W]    (1.0 where valid pixels from that video; 0.0 = invalid)

    Returns:
      var_map: [C, H, W]        (unbiased, 0 where valid_counts<2)
      valid_counts: [H, W]      (number of valid pixels per location)
    """
    _, C, _, _ = warped_all.shape

    # Broadcast masks to channels: expand masks 
    # from [N, 1, H, W] to [N, C, H, W] to match the color channels.
    mask_bc = masks_all.expand(-1, C, -1, -1)               # [N, C, H, W]

    # valid_counts[v, u]: number of videos that have a valid sample at pixel (u, v) (sees the 3D point).
    valid_counts = masks_all.sum(dim=0)                     # [1, 1, H, W]
    # Clamp to avoid division by zero when computing mean.
    valid_counts_clamped = torch.clamp(valid_counts, min=1.0)

    # Sum along the video dimension only where mask is valid.
    sum_vals = (warped_all * mask_bc).sum(dim=0)            # [C, H, W]
    # Compute the mean color per pixel per channel across valid videos.
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
    at_least_two = (valid_counts >= 2.0).squeeze(0)         # [1, H, W]
    var = torch.where(at_least_two.expand_as(var), var, torch.zeros_like(var))

    return var, valid_counts.squeeze(0).squeeze(0)          # [H, W]


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Compute per-pixel variance across videos at a chosen frame index t.\n"
            "Modes:\n"
            "  frame_t  : variance directly on frame t grid (no projection).\n"
            "  forward  : forward warp frame t to frame0 using depth0.\n"
            "  backward : backward warp frame t to frame0 using per-video depth_t.\n"
            "  hybrid   : backward where available, else forward.\n"
        )
    )
    parser.add_argument("--videos_dir", type=str, required=True)
    parser.add_argument("--camera_npz", type=str, required=True)
    parser.add_argument("--frame_index", type=int, required=True)
    parser.add_argument("--output_npz", type=str, required=True)

    parser.add_argument("--mode", type=str, default="frame_t",
                        choices=["frame_t", "forward", "backward", "hybrid"])

    args = parser.parse_args()

    # Load frames at t.
    frames_t_np, video_paths = load_frames_at_t_from_dir(args.videos_dir, args.frame_index)
    N, _, H, W = frames_t_np.shape

    if N < 2:
        raise ValueError("Need at least 2 videos to compute variance.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    frames_t = torch.from_numpy(frames_t_np).to(device)         # [N, 3, H, W]

    # Mode: frame_t
    if args.mode == "frame_t":
        var_map = torch.var(frames_t, dim=0, unbiased=True)     # [3, H, W]
        var_scalar = var_map.mean(dim=0)                        # [H, W]

        valid_counts = torch.full((H, W), float(N), device=device, dtype=torch.float32)
        inter_mask = torch.ones((H, W), device=device, dtype=torch.float32)

        np.savez(
            args.output_npz,
            mode=np.array(args.mode),
            var_map=var_map.detach().cpu().numpy(),
            var_scalar=var_scalar.detach().cpu().numpy(),
            valid_counts=valid_counts.detach().cpu().numpy(),
            intersection_mask=inter_mask.detach().cpu().numpy(),
            frame_index=np.int64(args.frame_index),
            video_paths=np.array(video_paths),
        )
        return

    # Mode: forward / backward / hybrid.
    # Load camera info.
    cam = np.load(args.camera_npz)
    w2c = cam["w2c"].astype(np.float32)                 # (T, 4, 4)
    K = cam["K"].astype(np.float32)                     # (T, 3, 3)
    depth0_np = cam["depth0"].astype(np.float32)        # (H, W)
    depth_all = cam["depth_all"].astype(np.float32)     # (N, T, H, W)
    mask_all = cam["mask_all"].astype(np.float32)       # (N, T, H, W)
    H_npz = int(cam["height"])
    W_npz = int(cam["width"])

    T = w2c.shape[0]
    t = args.frame_index

    # Checks.
    if t < 0 or t >= T:
        raise ValueError(f"frame_index {t} out of range [0, {T-1}].")

    if (H_npz, W_npz) != (H, W):
        raise ValueError(f"NPZ (height,width)=({H_npz},{W_npz}) != video ({H},{W}).")

    if depth0_np.shape != (H, W):
        raise ValueError(f"depth0 shape {depth0_np.shape} != ({H},{W}).")

    if depth_all.shape != (N, T, H, W):
        raise ValueError(f"depth_all shape {depth_all.shape} != (N,T,H,W)=({N},{T},{H},{W}).")
    
    if mask_all.shape != depth_all.shape:
        raise ValueError("mask_all shape mismatch.")

    K0 = torch.from_numpy(K[0]).to(device)
    w2c0 = torch.from_numpy(w2c[0]).to(device)
    K_t = torch.from_numpy(K[t]).to(device)
    w2c_t = torch.from_numpy(w2c[t]).to(device)

    depth0 = torch.from_numpy(depth0_np).to(device)
    pix_flat = make_pixel_grid(H, W, device=device, dtype=torch.float32)

    # Mode: Forward.
    warped_fwd = masks_fwd = None
    if args.mode in ("forward", "hybrid"):
        Xw, valid0 = backproject_depth0_to_world(depth0, K0, w2c0)
        grid, mask = build_forward_grid_and_mask(Xw, valid0, K_t, w2c_t, H, W)
        warped_fwd, masks_fwd = warp_batch_forward(frames_t, grid, mask)

    # Mode: Backward.
    warped_bwd = masks_bwd = None
    if args.mode in ("backward", "hybrid"):
        warped_list, mask_list = [], []
        for i in range(N):
            depth_t = torch.from_numpy(depth_all[i, t]).to(device)
            mask_t = torch.from_numpy(mask_all[i, t]).to(device)
            w_i, m_i = project_t_to_0_and_splat_bilinear(
                frame_t=frames_t[i],
                depth_t=depth_t,
                mask_t=mask_t,
                K_t=K_t, w2c_t=w2c_t,
                K0=K0, w2c0=w2c0,
                pix_flat=pix_flat,
                H=H, W=W
            )
            warped_list.append(w_i)
            mask_list.append(m_i)
        warped_bwd = torch.cat(warped_list, dim=0)      # [N, 3, H, W]
        masks_bwd = torch.cat(mask_list, dim=0)         # [N, 1, H, W]

    if args.mode == "forward":
        warped_all, masks_all = warped_fwd, masks_fwd
    elif args.mode == "backward":
        warped_all, masks_all = warped_bwd, masks_bwd
    # Mode: Hybrid.
    else:
        # If mask_fwd is 1 (backward has a value at that pixel), use backward warped value.
        # If mask_bwd is 0, use forward warped value.
        warped_all = warped_bwd * masks_bwd + warped_fwd * (1.0 - masks_bwd)
        masks_all = torch.clamp(masks_bwd + masks_fwd, max=1.0)

    var_map, valid_counts = masked_variance_across_videos(warped_all, masks_all)
    # var_scalar: per-pixel scalar variance by averaging over channels with shape [H, W].
    var_scalar = var_map.mean(dim=0)
    inter_mask = (valid_counts >= float(N)).float()

    # Save results.
    np.savez(
        args.output_npz,
        mode=np.array(args.mode),
        var_map=var_map.detach().cpu().numpy(),
        var_scalar=var_scalar.detach().cpu().numpy(),
        valid_counts=valid_counts.detach().cpu().numpy(),
        intersection_mask=inter_mask.detach().cpu().numpy(),
        frame_index=np.int64(t),
        video_paths=np.array(video_paths),
    )


if __name__ == "__main__":
    main()
