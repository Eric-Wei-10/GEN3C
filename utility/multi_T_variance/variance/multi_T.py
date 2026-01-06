# variance/multi_T.py
# Run multi-trajectory variance computation from a root directory containing trajectory folders.
import os
import numpy as np
import torch

from .single_T import compute_variance_for_one_trajectory_dir
from .orientation import parse_theta_phi_from_dir, orientation_distance_rad
from .combine import combine_priority, combine_soft_blend
from .io_load import find_camera_npz
from .core import backproject_depth0_to_world, build_forward_grid_and_mask


def _get_trajectory_dirs(inputs_root: str):
    """
    Given the inputs_root directory, list all trajectory subdirectories
    whose names start with "result_".
    
    :param inputs_root: Path to the root directory containing trajectory folders.
    :return: List of full paths to trajectory directories.
    """
    traj_dirs = []
    for name in sorted(os.listdir(inputs_root)):
        d = os.path.join(inputs_root, name)
        if os.path.isdir(d) and name.startswith("result_"):
            traj_dirs.append(d)
    if not traj_dirs:
        raise RuntimeError(f"No trajectory folders found under {inputs_root}")
    return traj_dirs


def _forward_visibility_mask_only(traj_dir: str, frame_index: int) -> torch.Tensor:
    """
    Geometry-only forward validity mask for this trajectory at time t.
    Returns: bool [H, W] on CPU.
    """
    camera_npz = find_camera_npz(traj_dir)
    cam = np.load(camera_npz, allow_pickle=True)

    w2c = cam["w2c"].astype(np.float32)                                 # (T, 4, 4)
    K = cam["K"].astype(np.float32)                                     # (T, 3, 3)
    depth0_np = cam["depth0"].astype(np.float32)                        # (H, W)

    T = w2c.shape[0]
    t = int(frame_index)
    if t < 0 or t >= T:
        raise ValueError(f"frame_index {t} out of range [0,{T-1}] in {camera_npz}")

    # compute on CPU to avoid GPU memory churn.
    dev = torch.device("cpu")
    depth0 = torch.from_numpy(depth0_np).to(dev)
    K0 = torch.from_numpy(K[0]).to(dev)
    w2c0 = torch.from_numpy(w2c[0]).to(dev)
    Kt = torch.from_numpy(K[t]).to(dev)
    w2ct = torch.from_numpy(w2c[t]).to(dev)

    H, W = depth0.shape
    Xw, valid0 = backproject_depth0_to_world(depth0, K0, w2c0)
    _, mask = build_forward_grid_and_mask(Xw, valid0, Kt, w2ct, H, W)   # mask [1, 1, H, W]

    return (mask.squeeze(0).squeeze(0) > 0.5).cpu()


def run_multi_from_inputs_root(
    inputs_root: str,
    frame_index: int,
    mode: str,
    output_npz: str,
    device: torch.device,
    combine_policy: str = "priority",
    sigma_deg: float = 15.0,
    traj_mask: str = "fill",
    save_per_trajectory: bool = False,
    channel_type: str = "rgb",
    occlude: bool = False,
):
    """
    Run multi-trajectory variance computation from a root directory containing trajectory folders.
    
    :param inputs_root: Path to the root directory containing trajectory folders.
    :param frame_index: The chosen frame index.
    :param mode: One of "frame_t", "forward", "backward", "hybrid".
    :param output_npz: Path to save the output .npz file.
    :param device: Torch device to use for computation.
    :param combine_policy: Either "priority" or "soft".
    :param sigma_deg: Sigma value in degrees for soft blending.
    :param traj_mask: Either "fill" or "intersection".
    :param save_per_trajectory: Whether to save results for each trajectory separately.
    :param channel_type: "rgb" or "dino".
    :param occlude: Whether to apply occlusion masking (backward valid & forward invalid).
    """
    if occlude and mode != "backward":
        raise ValueError("occlude=True currently supports mode='backward' only.")

    traj_dirs = _get_trajectory_dirs(inputs_root)

    # per_traj_results: list of dicts from compute_variance_for_one_trajectory_dir.
    # per_traj_dist: list of orientation distances in radians.
    # traj_names: list of trajectory directory basenames.
    # traj_theta_phi: list of (theta, phi) tuples for each trajectory.
    per_traj_results = []
    per_traj_orien_d = []
    traj_names = []
    traj_theta_phi = []

    # For occlusion, we need combined forward mask (union across trajectories).
    per_traj_fwd_mask = []          # list[bool H x W], CPU

    # Loop over trajectory folders to compute per-trajectory variance and orientation distance.
    for td in traj_dirs:
        theta, phi = parse_theta_phi_from_dir(td)
        dist = orientation_distance_rad(theta, phi)

        # NOT apply occlusion inside single_T in multi mode, but apply occlusion AFTER combining across trajectories.
        result = compute_variance_for_one_trajectory_dir(
            traj_dir=td,
            frame_index=frame_index,
            mode=mode,
            device=device,
            channel_type=channel_type,
            occlude=False,
        )

        # IMPORTANT: move big tensors off GPU right away to avoid OOM.
        result_cpu = {}
        for k, v in result.items():
            if torch.is_tensor(v):
                result_cpu[k] = v.detach().cpu()
            else:
                result_cpu[k] = v
        # drop GPU tensors from this trajectory
        del result
        if device.type == "cuda":
            torch.cuda.empty_cache()

        per_traj_results.append(result_cpu)
        
        per_traj_orien_d.append(dist)
        traj_names.append(os.path.basename(td))
        traj_theta_phi.append((theta, phi))

        if occlude:
            per_traj_fwd_mask.append(_forward_visibility_mask_only(td, frame_index))

    # Combine per-trajectory results according to the specified policy: "priority" or "soft".
    # priority: take variance from the trajectory with the smallest orientation distance to 0-orientation that covers each pixel.
    if combine_policy == "priority":
        combined_var_map, combined_var_scalar, combined_mask, source_idx = combine_priority(
            per_traj_results, per_traj_orien_d, traj_mask=traj_mask
        )
        extra = {"source_trajectory_index": source_idx.detach().cpu().numpy()}
    # soft: blend variances from all trajectories using weights based on orientation distance and sigma_deg.
    elif combine_policy == "soft":
        combined_var_map, combined_var_scalar, combined_mask, source_idx, weight_sum = combine_soft_blend(
            per_traj_results, per_traj_orien_d, sigma_deg=sigma_deg, traj_mask=traj_mask
        )
        extra = {
            "source_trajectory_index": source_idx.detach().cpu().numpy(),
            "weight_sum": weight_sum.detach().cpu().numpy(),
            "sigma_deg": np.float32(sigma_deg),
        }
    else:
        raise ValueError("combine_policy must be 'priority' or 'soft'.")

    # M_occ = M_bwd_combined & ~M_fwd_combined.
    if occlude:
        bwd_mask_bool = (combined_mask > 0.5)

        # Union forward masks across trajectories.
        H, W = bwd_mask_bool.shape
        fwd_union = torch.zeros((H, W), dtype=torch.bool)
        for m in per_traj_fwd_mask:
            if m.shape != (H, W):
                raise ValueError(f"Forward mask shape mismatch {m.shape} vs expected {(H,W)}.")
            fwd_union |= m.to(torch.bool)

        occ = bwd_mask_bool & (~fwd_union)

        # Apply to outputs.
        combined_mask = occ.float()
        combined_var_map = combined_var_map * occ.unsqueeze(0).to(combined_var_map.dtype)
        combined_var_scalar = combined_var_scalar * occ.to(combined_var_scalar.dtype)
        source_idx = source_idx.clone()
        source_idx[~occ] = -1
        extra["source_trajectory_index"] = source_idx.detach().cpu().numpy()
        extra["occlude"] = np.bool_(True)
    else:
        extra["occlude"] = np.bool_(False)

    # Save per-trajectory results if requested.
    if save_per_trajectory:
        base, ext = os.path.splitext(output_npz)
        for name, dist, (theta, phi), result in zip(traj_names, per_traj_orien_d, traj_theta_phi, per_traj_results):
            out_i = f"{base}_{name}{ext}"
            np.savez(
                out_i,
                trajectory=np.array(name),
                theta=np.float32(theta),
                phi=np.float32(phi),
                orientation_distance_rad=np.float32(dist),
                mode=np.array(mode),
                channel_type=np.array(channel_type),

                var_map=result["var_map"].numpy() if torch.is_tensor(result["var_map"]) else result["var_map"],
                var_scalar=result["var_scalar"].numpy() if torch.is_tensor(result["var_scalar"]) else result["var_scalar"],
                valid_counts=result["valid_counts"].numpy() if torch.is_tensor(result["valid_counts"]) else result["valid_counts"],
                fill_mask=result["fill_mask"].numpy() if torch.is_tensor(result["fill_mask"]) else result["fill_mask"],
                intersection_mask=result["intersection_mask"].numpy() if torch.is_tensor(result["intersection_mask"]) else result["intersection_mask"],

                frame_index=np.int64(frame_index),
                video_paths=np.array(result["video_paths"], dtype=object),
                camera_npz=np.array(result["camera_npz"]),
            )

    # Save combined result (.npz).
    payload = dict(
        mode=np.array(mode),
        channel_type=np.array(channel_type),
        frame_index=np.int64(frame_index),

        combine_policy=np.array(combine_policy),
        traj_mask=np.array(traj_mask),

        combined_var_map=combined_var_map.detach().cpu().numpy(),
        combined_var_scalar=combined_var_scalar.detach().cpu().numpy(),
        combined_mask=combined_mask.detach().cpu().numpy(),

        trajectory_names=np.array(traj_names, dtype=object),
        trajectory_theta_phi=np.array(traj_theta_phi, dtype=np.float32),
        trajectory_orientation_distance_rad=np.array(per_traj_orien_d, dtype=np.float32),
    )
    payload.update(extra)

    np.savez(output_npz, **payload)
