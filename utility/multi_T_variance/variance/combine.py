# variance/combine.py
# Combining per-trajectory variance maps into a single variance map
# using either priority-based selection or soft blending based on orientation distance.
import math
import torch


def _select_mask(result, traj_mask: str):
    """
    Indicates which pixels are valid according to the chosen per-trajectory mask.
    
    :param result: Trajectory result dictionary from `compute_variance_for_one_trajectory_dir`.
    :param traj_mask: Either 'fill' or 'intersection'.
    :return: Boolean mask tensor of shape [H, W].
    """
    if traj_mask == "fill":
        return (result["fill_mask"] > 0.5)
    if traj_mask == "intersection":
        return (result["intersection_mask"] > 0.5)
    raise ValueError("traj_mask must be 'fill' or 'intersection'.")


def combine_priority(per_traj_results, per_traj_dist_rad, traj_mask: str):
    """
    For each pixel, selects the variance from the trajectory with the smallest orientation distance
    (i.e., closest to 0-orientation) that has a valid mask at that pixel.
    
    :param per_traj_results: List of trajectory result dictionaries from `compute_variance_for_one_trajectory_dir`.
    :param per_traj_dist_rad: List of orientation distances in radians for each trajectory.
    :param traj_mask: Either 'fill' or 'intersection'.
    :return: combined_var_map: Tensor of shape [C, H, W] indicating combined variance map,
             combined_var_scalar: Tensor of shape [H, W] indicating combined scalar variance (mean across channels),
             combined_mask: Tensor of shape [H, W] indicating valid pixels in the combined result,
             source_idx: Tensor of shape [H, W] indicating the index of the source trajectory.
    """
    # M = number of trajectories.
    M = len(per_traj_results)
    if M == 0:
        raise ValueError("No trajectory results to combine.")
    if len(per_traj_dist_rad) != M:
        raise ValueError(f"per_traj_dist_rad length {len(per_traj_dist_rad)} != num trajectories {M}.")

    # Sort trajectories by orientation distance from 0-orientation (ascending).
    # From most reliable to least.
    order = sorted(range(M), key=lambda i: per_traj_dist_rad[i])

    C = per_traj_results[0]["var_map"].shape[0]
    H = per_traj_results[0]["H"]
    W = per_traj_results[0]["W"]
    device = per_traj_results[0]["var_map"].device
    dtype = per_traj_results[0]["var_map"].dtype

    # Initialize all combined outputs to zeros / -1.
    combined_var_map = torch.zeros((C, H, W), device=device, dtype=dtype)
    combined_var_scalar = torch.zeros((H, W), device=device, dtype=per_traj_results[0]["var_scalar"].dtype)
    # combined_mask (binary): 0 means "not filled yet"; 1 means "filled".
    combined_mask = torch.zeros((H, W), device=device, dtype=torch.float32)
    # Indicate which trajectory index wrote each pixel; -1 means "no trajectory".
    source_idx = torch.full((H, W), -1, device=device, dtype=torch.int32)

    # Iterate through trajectories from most reliable to least.
    for i in order:
        result = per_traj_results[i]
        valid_mask = _select_mask(result, traj_mask)

        # Only fill pixels that are not yet filled and are valid in this trajectory.
        take_mask = (combined_mask < 0.5) & valid_mask

        if take_mask.any():
            combined_var_map[:, take_mask] = result["var_map"][:, take_mask]
            combined_var_scalar[take_mask] = result["var_scalar"][take_mask]
            combined_mask[take_mask] = 1.0
            source_idx[take_mask] = int(i)

    return combined_var_map, combined_var_scalar, combined_mask, source_idx


def combine_soft_blend(per_traj_results, per_traj_dist_rad, sigma_deg: float, traj_mask: str):
    """
    Weighted average combination of per-trajectory variance maps, where weights are determined
    by a Gaussian function of the orientation distance from 0-orientation (nearer to 0-orientation
    get higher weight).
    
    :param per_traj_results: List of trajectory result dictionaries from `compute_variance_for_one_trajectory_dir`.
    :param per_traj_dist_rad: List of orientation distances in radians for each trajectory.
    :param sigma_deg: Standard deviation of the Gaussian weighting function in degrees.
    :param traj_mask: Either 'fill' or 'intersection'.
    :return: combined_var_map: Tensor of shape [C, H, W] indicating combined variance map,
             combined_var_scalar: Tensor of shape [H, W] indicating combined scalar variance (mean across channels),
             combined_mask: Tensor of shape [H, W] indicating valid pixels in the combined result,
             source_idx: Tensor of shape [H, W] indicating the index of the closest-to-0-orientation trajectory among trajectories that cover that pixel,
             den: Tensor of shape [H, W] indicating the sum of weights used for normalization.
    """
    # M = number of trajectories.
    M = len(per_traj_results)
    if M == 0:
        raise ValueError("No trajectory results to combine.")
    if len(per_traj_dist_rad) != M:
        raise ValueError(f"per_traj_dist_rad length {len(per_traj_dist_rad)} != num trajectories {M}.")

    # Convert sigma from degrees to radians.
    sigma = math.radians(float(sigma_deg))
    if sigma <= 1e-9:
        raise ValueError("sigma_deg must be > 0.")

    C = per_traj_results[0]["var_map"].shape[0]
    H = per_traj_results[0]["H"]
    W = per_traj_results[0]["W"]
    device = per_traj_results[0]["var_map"].device
    dtype = per_traj_results[0]["var_map"].dtype

    # num: numerator for per-channel weighted sum.
    # num_s: numerator for scalar variance weighted sum.
    # den: denominator for normalization (sum of weights per pixel).
    num = torch.zeros((C, H, W), device=device, dtype=dtype)
    num_s = torch.zeros((H, W), device=device, dtype=per_traj_results[0]["var_scalar"].dtype)
    den = torch.zeros((H, W), device=device, dtype=torch.float32)

    # source_idx: for each pixel, the index of the trajectory that is closest to 0-orientation 
    # among trajectories that cover that specific pixel.
    source_idx = torch.full((H, W), -1, device=device, dtype=torch.int32)
    # best_d: records the smallest orientation distance found so far for each pixel.
    best_d = torch.full((H, W), float("inf"), device=device, dtype=torch.float32)

    # V(u, v) = sum_i [ w_i(u,v) * V_i(u,v) ] / sum_i [ w_i(u,v) ]
    # w_i(u,v) = exp( - (d_i^2) / (2 * sigma^2) ) if pixel (u,v) is valid in trajectory i, else 0.
    for i in range(M):
        result = per_traj_results[i]
        valid_mask = _select_mask(result, traj_mask).float()
        
        # Orientation distance to 0-orientation for this trajectory.
        d_i = float(per_traj_dist_rad[i])

        # Gaussian weight for this trajectory (same for all pixels).
        w_i = math.exp(-(d_i * d_i) / (2.0 * sigma * sigma))
        if w_i <= 0:
            continue
        
        # ws_i: per-pixel weights (0 where invalid).
        ws_i = float(w_i) * valid_mask
        num += result["var_map"] * ws_i.unsqueeze(0)
        num_s += result["var_scalar"] * ws_i
        den += ws_i

        # Track closest trajectory index for each pixel.
        # better: pixels where this trajectory is valid and has smaller d_i than best_d so far.
        # best_d: best (smallest) orientation distance found so far for each pixel.
        # source_idx: index of trajectory that provided best_d for each pixel.
        better = (valid_mask > 0.5) & (d_i < best_d)
        best_d[better] = d_i
        source_idx[better] = int(i)

    # Pixel is valid if at least one trajectory contributed.
    combined_mask = (den > 0).float()

    combined_var_map = torch.where(
        combined_mask.unsqueeze(0) > 0,
        num / (den.unsqueeze(0) + 1e-8),
        torch.zeros_like(num),
    )
    combined_var_scalar = torch.where(
        combined_mask > 0,
        num_s / (den + 1e-8),
        torch.zeros_like(num_s),
    )

    return combined_var_map, combined_var_scalar, combined_mask, source_idx, den
