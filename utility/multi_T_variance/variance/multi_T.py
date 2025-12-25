# variance/multi_T.py
# Run multi-trajectory variance computation from a root directory containing trajectory folders.
import os
import numpy as np
import torch

from .single_T import compute_variance_for_one_trajectory_dir
from .orientation import parse_theta_phi_from_dir, orientation_distance_rad
from .combine import combine_priority, combine_soft_blend


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
    """
    traj_dirs = _get_trajectory_dirs(inputs_root)

    # per_traj_results: list of dicts from compute_variance_for_one_trajectory_dir.
    # per_traj_dist: list of orientation distances in radians.
    # traj_names: list of trajectory directory basenames.
    # traj_theta_phi: list of (theta, phi) tuples for each trajectory.
    per_traj_results = []
    per_traj_orien_d = []
    traj_names = []
    traj_theta_phi = []

    # Loop over trajectory folders to compute per-trajectory variance and orientation distance.
    for td in traj_dirs:
        theta, phi = parse_theta_phi_from_dir(td)
        dist = orientation_distance_rad(theta, phi)

        result = compute_variance_for_one_trajectory_dir(
            traj_dir=td,
            frame_index=frame_index,
            mode=mode,
            device=device,
        )

        per_traj_results.append(result)
        per_traj_orien_d.append(dist)
        traj_names.append(os.path.basename(td))
        traj_theta_phi.append((theta, phi))

    # Combine per-trajectory results according to the specified policy: "priority" or "soft".
    # priority: take variance from the trajectory with the smallest orientation distance to 0-orientation that covers each pixel.
    if combine_policy == "priority":
        combined_var_map, combined_var_scalar, combined_mask, source_idx = combine_priority(
            per_traj_results, per_traj_orien_d, traj_mask=traj_mask
        )
        extra = {
            "source_trajectory_index": source_idx.detach().cpu().numpy(),
        }
    # soft: blend variances from all trajectories using weights based on orientation distance and sigma_deg.
    elif combine_policy == "soft":
        combined_var_map, combined_var_scalar, combined_mask, source_idx, weight_sum = combine_soft_blend(
            per_traj_results, per_traj_orien_d, sigma_deg=sigma_deg, traj_mask=traj_mask
        )
        extra = {
            "source_trajectory_index": source_idx.detach().cpu().numpy(),  # now exists for soft too
            "weight_sum": weight_sum.detach().cpu().numpy(),
            "sigma_deg": np.float32(sigma_deg),
        }
    else:
        raise ValueError("combine_policy must be 'priority' or 'soft'.")

    # Save per-trajectory results (.npz) if requested.
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

                var_map=result["var_map"].detach().cpu().numpy(),
                var_scalar=result["var_scalar"].detach().cpu().numpy(),
                valid_counts=result["valid_counts"].detach().cpu().numpy(),
                fill_mask=result["fill_mask"].detach().cpu().numpy(),
                intersection_mask=result["intersection_mask"].detach().cpu().numpy(),
                frame_index=np.int64(frame_index),
                video_paths=np.array(result["video_paths"], dtype=object),
                camera_npz=np.array(result["camera_npz"]),
            )

    # Save combined result (.npz).
    payload = dict(
        mode=np.array(mode),
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
