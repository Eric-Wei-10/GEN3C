# variance/single_T.py
# Compute variance for a single trajectory directory.
import os
import zipfile
import numpy as np
import torch

from .core import (
    make_pixel_grid,
    backproject_depth0_to_world,
    build_forward_grid_and_mask,
    warp_batch_forward,
    project_t_to_0_and_splat_bilinear,
    masked_variance_across_videos,
)
from .io_load import find_camera_npz, get_ordered_videos_by_seed, load_frames_at_t_from_list


def _npz_signature_ok(path: str) -> bool:
    """
    Check if the file at `path` has a valid .npz (zip) signature.
    
    :param path: Path to the file to check.
    :return: True if the file has a valid .npz signature, False otherwise. 
    """
    try:
        with open(path, "rb") as f:
            sig = f.read(4)
        # .npz files start with the ZIP file signature 'PK\x03\x04'.
        return sig[:2] == b"PK"
    except Exception:
        return False


def _file_head_preview(path: str, n: int = 200) -> str:
    """
    Read the first `n` bytes of the file at `path` and return as a string.
    For a diagnostic preview when the .npz file is corrupted.
    
    :param path: Path to the file to read.
    :param n: Number of bytes to read from the start of the file.
    :return: The first `n` bytes of the file decoded as a string.
    """
    try:
        with open(path, "rb") as f:
            head = f.read(n)
        return head.decode("latin-1", errors="replace")
    except Exception as e:
        return f"<failed to read head: {e}>"


def _safe_np_load_npz(path: str):
    """
    Safely load a .npz file, with checks and informative error messages.
    
    :param path: Path to the .npz file to load.
    """
    size = None
    try:
        size = os.path.getsize(path)
    except Exception:
        pass

    if not _npz_signature_ok(path):
        preview = _file_head_preview(path, 200)
        raise RuntimeError(
            f"camera_data.npz is not a valid .npz (zip) file:\n"
            f"  path: {path}\n"
            f"  size: {size}\n"
            f"  head(200): {repr(preview)}"
        )

    try:
        return np.load(path, allow_pickle=True)
    except zipfile.BadZipFile as e:
        preview = _file_head_preview(path, 200)
        raise RuntimeError(
            f"Failed to open .npz (BadZipFile):\n"
            f"  path: {path}\n"
            f"  size: {size}\n"
            f"  head(200): {repr(preview)}\n"
            f"  error: {e}"
        ) from e


def compute_variance_for_one_trajectory_dir(
    traj_dir: str,
    frame_index: int,
    mode: str,
    device: torch.device,
):
    """
    Compute variance for a single trajectory directory.
    
    :param traj_dir: Path to the trajectory directory.
    :param frame_index: The chosen frame index.
    :param mode: One of "frame_t", "forward", "backward", or "hybrid".
    :param device: Torch device to use for computation.
    """
    camera_npz = find_camera_npz(traj_dir)
    video_paths = get_ordered_videos_by_seed(traj_dir)

    # Load the same frame index from all videos.
    frames_t_np, ordered_paths = load_frames_at_t_from_list(video_paths, frame_index)
    N, _, H, W = frames_t_np.shape
    if N < 2:
        raise ValueError(f"Need at least 2 videos in {traj_dir} to compute variance.")
    frames_t = torch.from_numpy(frames_t_np).to(device)         # [N, 3, H, W]

    # Mode: frame_t.
    if mode == "frame_t":
        var_map = torch.var(frames_t, dim=0, unbiased=True)     # [3, H, W]
        var_scalar = var_map.mean(dim=0)                        # [H, W]
        valid_counts = torch.full((H, W), float(N), device=device, dtype=torch.float32)

        fill_mask = torch.ones((H, W), device=device, dtype=torch.float32)
        intersection_mask = torch.ones((H, W), device=device, dtype=torch.float32)

        return {
            "var_map": var_map,
            "var_scalar": var_scalar,
            "valid_counts": valid_counts,
            "fill_mask": fill_mask,
            "intersection_mask": intersection_mask,
            "H": H, "W": W, "N": N,
            "video_paths": ordered_paths,
            "camera_npz": camera_npz,
        }

    # Mode: forward, backward, hybrid.
    # Load camera data.
    cam = _safe_np_load_npz(camera_npz)
    required = ["w2c", "K", "depth0", "depth_all", "mask_all", "height", "width"]
    missing = [k for k in required if k not in cam.files]
    if missing:
        raise KeyError(f"camera_data.npz missing keys {missing}\n  file: {camera_npz}\n  keys: {cam.files}")

    w2c = cam["w2c"].astype(np.float32)                 # (T, 4, 4)
    K = cam["K"].astype(np.float32)                     # (T, 3, 3)
    depth0_np = cam["depth0"].astype(np.float32)        # (H, W)
    depth_all = cam["depth_all"].astype(np.float32)     # (N, T, H, W)
    mask_all = cam["mask_all"].astype(np.float32)       # (N, T, H, W)
    H_npz = int(cam["height"])
    W_npz = int(cam["width"])

    T = w2c.shape[0]
    t = int(frame_index)

    # Sanity checks.
    if t < 0 or t >= T:
        raise ValueError(f"frame_index {t} out of range [0,{T-1}] in {camera_npz}")
    if (H_npz, W_npz) != (H, W):
        raise ValueError(f"NPZ (H,W)=({H_npz},{W_npz}) != video ({H},{W}) in {traj_dir}")
    if depth_all.shape != (N, T, H, W):
        raise ValueError(
            f"depth_all shape {depth_all.shape} != ({N},{T},{H},{W}). "
            f"Check that camera_data.npz N matches number of videos."
        )
    if mask_all.shape != depth_all.shape:
        raise ValueError(f"mask_all shape {mask_all.shape} != depth_all shape {depth_all.shape}")

    # Convert camera arrays to torch tensors.
    K0 = torch.from_numpy(K[0]).to(device)
    w2c0 = torch.from_numpy(w2c[0]).to(device)
    K_t = torch.from_numpy(K[t]).to(device)
    w2c_t = torch.from_numpy(w2c[t]).to(device)

    depth0 = torch.from_numpy(depth0_np).to(device)
    pix_flat = make_pixel_grid(H, W, device=device, dtype=torch.float32)

    # Mode: forward.
    warped_fwd = masks_fwd = None
    if mode in ("forward", "hybrid"):
        Xw, valid0 = backproject_depth0_to_world(depth0, K0, w2c0)
        grid, mask = build_forward_grid_and_mask(Xw, valid0, K_t, w2c_t, H, W)
        warped_fwd, masks_fwd = warp_batch_forward(frames_t, grid, mask)

    # Mode: backward.
    warped_bwd = masks_bwd = None
    if mode in ("backward", "hybrid"):
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
        warped_bwd = torch.cat(warped_list, dim=0)  # [N,3,H,W]
        masks_bwd = torch.cat(mask_list, dim=0)     # [N,1,H,W]

    # Choose warped and masks based on mode.
    if mode == "forward":
        warped_all, masks_all = warped_fwd, masks_fwd
    elif mode == "backward":
        warped_all, masks_all = warped_bwd, masks_bwd
    elif mode == "hybrid":
        warped_all = warped_bwd * masks_bwd + warped_fwd * (1.0 - masks_bwd)
        masks_all = torch.clamp(masks_bwd + masks_fwd, max=1.0)
    else:
        raise ValueError(f"Unknown mode: {mode}")

    # Compute masked variance across videos.
    var_map, valid_counts = masked_variance_across_videos(warped_all, masks_all)
    var_scalar = var_map.mean(dim=0)

    # Two types of coverage masks: "fill" and "intersection".
    fill_mask = (valid_counts >= 2.0).float()
    intersection_mask = (valid_counts >= float(N)).float()

    return {
        "var_map": var_map,
        "var_scalar": var_scalar,
        "valid_counts": valid_counts,
        "fill_mask": fill_mask,
        "intersection_mask": intersection_mask,
        "H": H, "W": W, "N": N,
        "video_paths": ordered_paths,
        "camera_npz": camera_npz,
    }
