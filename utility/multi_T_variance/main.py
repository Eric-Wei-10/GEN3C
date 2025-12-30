import argparse
import numpy as np
import torch

from variance.single_T import compute_variance_for_one_trajectory_dir
from variance.multi_T import run_multi_from_inputs_root


def main():
    p = argparse.ArgumentParser(description="Variance pipeline for single or multi trajectory inputs/")

    p.add_argument("--frame_index", type=int, required=True)
    p.add_argument("--mode", type=str, default="backward", choices=["frame_t", "forward", "backward", "hybrid"])
    p.add_argument("--output_npz", type=str, required=True)

    # Choose to run single or multi mode.
    # --traj_dir: single mode.
    # --inputs_root: multi mode.
    p.add_argument("--traj_dir", type=str, default=None)
    p.add_argument("--inputs_root", type=str, default=None)

    # Multi options.
    p.add_argument("--combine_policy", type=str, default="priority", choices=["priority", "soft"])
    p.add_argument("--sigma_deg", type=float, default=15.0)
    p.add_argument("--traj_mask", type=str, default="fill", choices=["fill", "intersection"],
                   help="Which per-trajectory mask defines validity when combining across trajectories.")
    p.add_argument("--save_per_trajectory", action="store_true")

    # Channel type option (rgb or dino) for single mode.
    p.add_argument("--channel_type", type=str, default="rgb", choices=["rgb", "dino"],
                   help="Channel type to use for variance computation in single mode.")

    args = p.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Run multi mode.
    if args.inputs_root is not None:
        run_multi_from_inputs_root(
            inputs_root=args.inputs_root,
            frame_index=args.frame_index,
            mode=args.mode,
            output_npz=args.output_npz,
            device=device,
            combine_policy=args.combine_policy,
            sigma_deg=args.sigma_deg,
            traj_mask=args.traj_mask,
            save_per_trajectory=args.save_per_trajectory,
            channel_type=args.channel_type,
        )
        return

    if args.traj_dir is None:
        raise ValueError("Provide either --inputs_root (multi) or --traj_dir (single).")
    
    # Run single mode.
    r = compute_variance_for_one_trajectory_dir(
        traj_dir=args.traj_dir,
        frame_index=args.frame_index,
        mode=args.mode,
        device=device,
        channel_type=args.channel_type,
    )

    np.savez(
        args.output_npz,
        mode=np.array(args.mode),
        channel_type=np.array(args.channel_type),
        var_map=r["var_map"].detach().cpu().numpy(),
        var_scalar=r["var_scalar"].detach().cpu().numpy(),
        valid_counts=r["valid_counts"].detach().cpu().numpy(),
        fill_mask=r["fill_mask"].detach().cpu().numpy(),
        intersection_mask=r["intersection_mask"].detach().cpu().numpy(),
        frame_index=np.int64(args.frame_index),
        video_paths=np.array(r["video_paths"], dtype=object),
        camera_npz=np.array(r["camera_npz"]),
    )


if __name__ == "__main__":
    main()
