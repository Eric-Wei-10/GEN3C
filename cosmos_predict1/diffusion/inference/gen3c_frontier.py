import os
import argparse
import numpy as np
import torch
from PIL import Image
from pathlib import Path
from cosmos_predict1.diffusion.inference.inference_utils import (
    add_common_arguments
)
from cosmos_predict1.diffusion.inference.camera_utils import generate_camera_trajectory_frontier

from gen3c_persistent import Gen3cPersistentModel
from cosmos_predict1.utils import misc

import math
from tqdm import tqdm
from itertools import product


def check_path(path_str):
    path = Path(path_str)

    if not path.exists():
        raise FileNotFoundError(f"Path not found: {path_str}")

    if path.is_dir():
        return True
    
    if path.is_file():
        img_extensions = {'.jpg', '.jpeg', '.png'}
        if path.suffix.lower() in img_extensions:
            return False
        else:
            raise ValueError(f"Unsupported file type: {path.name}, expected an image file.")


def load_image_to_numpy(image_path, target_height, target_width):
    """Loads image, resizes, and returns [1, H, W, 3] float32 array in [0, 1]."""
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")
    
    pil_image = Image.open(image_path).convert("RGB")
    pil_image = pil_image.resize((target_width, target_height), Image.LANCZOS)
    img_np = np.array(pil_image).astype(np.float32) / 255.0
    return img_np[None, ...]

def prepare_static_intrinsics(focal, principal_point, width, height, num_frames):
    """Creates a static intrinsics tensor repeated over time."""
    # Changed from [None, None, ...] to [None, ...] to produce (T, 3, 3)
    # The model class will add the Batch dimension internally.
    intrinsics = np.eye(3)[None, ...].repeat(num_frames, axis=0)
    
    intrinsics[:, 0, 0] = focal[0, 0]
    intrinsics[:, 1, 1] = focal[0, 1]
    intrinsics[:, 0, 2] = principal_point[0, 0] * width
    intrinsics[:, 1, 2] = principal_point[0, 1] * height
    return intrinsics

def run_multi_seed(path_stem, model, seeds, prompt, theta_deg, phi_deg, w2c_tensor, intrinsics_tensor, input_depth, distance):
    """
    Logic: Load Image -> Seed/MoGE (ONCE) -> [Loop: Update Seed -> Infer]
    """
 
    try:
        generated_w2cs, generated_intrinsics = generate_camera_trajectory_frontier(
            initial_w2c=w2c_tensor,
            initial_intrinsics=intrinsics_tensor,
            num_frames=model.args.num_video_frames,
            movement_distance=distance,
            center_depth=1.0,
            device=model.device.type,
            theta_deg=theta_deg,
            phi_deg=phi_deg
        )
    except (ValueError, NotImplementedError) as e:
        print(f"Failed to generate trajectory: {e}")

    # save camera poses and intrinsics ONCE
    # Charlotte
    # ==============================
    original_video_save_folder = model.args.video_save_folder
    model.args.video_save_folder = model.args.video_save_folder + "/" + path_stem + f"/result_{theta_deg}_{phi_deg}"
    cam_save_path = os.path.join(model.args.video_save_folder, f"camera_data.npz")
    
    w2c_all = generated_w2cs[0].detach().cpu().numpy()          # (T, 4, 4)
    K_all   = generated_intrinsics[0].detach().cpu().numpy()    # (T, 3, 3)
    depth0 = input_depth[0, 0].detach().cpu().numpy()   # (H, W)

    os.makedirs(model.args.video_save_folder, exist_ok=True)
    np.savez(
        cam_save_path,
        w2c=w2c_all,        # (T, 4, 4)
        K=K_all,            # (T, 3, 3)
        depth0=depth0,      # (H, W)
        height=model.H,
        width=model.W,
    )

    # Accumulate per-seed depth/mask from MoGe over all frames
    depth_all_list = []  # will become (N, T, H, W)
    mask_all_list  = []  # optional: (N, T, H, W)
    
    print(f"Saved camera + depth info to {cam_save_path}")
    # ====================================
    
    # 2. Loop Seeds (Reuse Cache)
    for seed in seeds:
        print(f"  > Generating with Seed: {seed}")
        
        # Update Generator States
        model.generator.manual_seed(seed)
        misc.set_random_seed(seed)
        model.args.seed = seed
        
        model.args.prompt = prompt
        model.args.video_save_name = f"video_{seed}"
        
        
        
        if isinstance(generated_w2cs, torch.Tensor) and isinstance(generated_intrinsics, torch.Tensor):
            generated_w2cs = generated_w2cs.cpu().numpy()
            generated_intrinsics = generated_intrinsics[0].cpu().numpy()
        
        # Charlotte
        # ======================
        result = model.inference_on_cameras(
            view_cameras_w2cs=generated_w2cs,
            view_camera_intrinsics=generated_intrinsics,
            fps=model.args.fps,
            return_estimated_depths=True,
        )

        # 3. Process and Move to CPU immediately
        if result is not None:
            depth_all_seed = result.get("predicted_depth_all", None)
            mask_all_seed  = result.get("predicted_mask_all", None)

            if depth_all_seed is not None:
                # Squeeze channel dimension -> (T, H, W)
                depth_all_list.append(depth_all_seed[:, 0, ...])
            if mask_all_seed is not None:
                mask_all_list.append(mask_all_seed[:, 0, ...])
        
        del result
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache() # Frees the VRAM for the next seed
        # =======================
        # cleanup per-seed cache
        

    # Charlotte
    # =======================
    if depth_all_list:
        # depth_all: (N, T, H, W)
        depth_all = np.stack(depth_all_list, axis=0)
        print(f"Collected depth_all with shape {depth_all.shape}")

        # Load existing camera_data and update with depth_all (and mask_all if present)
        cam_data = np.load(cam_save_path)
        save_kwargs = {k: cam_data[k] for k in cam_data.files}
        cam_data.close()

        save_kwargs["depth_all"] = depth_all  # (N, T, H, W)

        if mask_all_list:
            mask_all = np.stack(mask_all_list, axis=0)  # (N, T, H, W)
            save_kwargs["mask_all"] = mask_all

        np.savez(cam_save_path, **save_kwargs)

        if mask_all_list:
            print(f"Updated {cam_save_path} with depth_all and mask_all")
        else:
            print(f"Updated {cam_save_path} with depth_all")
    else:
        print("Warning: depth_all_list is empty; no MoGe depth was saved for later frames.")
    # =========================================

    model.args.video_save_folder = original_video_save_folder    

def run_multi_traj_multi_seed(model, image_path, seeds, prompt, sample_angle=30, debug=False):
    print(f"\n=== Genetrating Videos for {image_path} with debug mode: {debug} ===")

    image_path_stem = Path(image_path).stem

    img_np = load_image_to_numpy(image_path, model.H, model.W) 
    
    # Initialize with user specified seed to generate the depth and build cache ONCE
    initial_seed = seeds[0]
    misc.set_random_seed(initial_seed)
    model.generator.manual_seed(initial_seed)

    print(f"  > Building Initial Cache with seed {initial_seed}...")
    model.clear_cache()
    
    (est_w2c, est_focal, est_pp, resolution, input_depth) = model.seed_model_from_values(
        images_np=img_np, depths_np=None,
        world_to_cameras_np=np.zeros((1, 4, 4)),
        focal_lengths_np=np.zeros((1, 2)),
        principal_point_rel_np=np.zeros((1, 2)),
        resolutions=np.array([[model.W, model.H]])
    )

    intrinsics = prepare_static_intrinsics(est_focal, est_pp / resolution, model.W, model.H, model.frames_per_batch)
    input_depth = torch.from_numpy(input_depth).to(model.device)
    
    agent_speed = model.args.agent_speed # m/s
    t = (model.args.num_video_frames - 1)/ model.args.fps # seconds
    distance = agent_speed * t
    # if numpy, convert w2c and intrinsics to tensor
    if isinstance(est_w2c, np.ndarray) and isinstance(intrinsics, np.ndarray):
        w2c_tensor = torch.from_numpy(est_w2c[0]).to(model.device)
        intrinsics_tensor = torch.from_numpy(intrinsics[0]).to(model.device)
        
    if not debug:
        # equi-angular sampling    
        HFOV = 2 * math.atan(model.W / (2 * intrinsics_tensor[0, 0])) * (180.0 / math.pi)  # in degrees
        # VFOV = 2 * math.atan(model.H / (2 * intrinsics_tensor[1, 1])) * (180.0 / math.pi)  # in degrees
        theta_deg_list = [0]
        for theta in range(sample_angle, math.ceil(HFOV/2), sample_angle):
            theta_deg_list.append(theta)
            theta_deg_list.append(-theta)
        phi_deg_list = [0, 5, -5]
        print(f"theta_deg_list: {theta_deg_list}")
        
    else:
        # debug mode: only one trajectory
        theta_deg_list = [0]
        phi_deg_list = [0]

    for theta_deg, phi_deg in tqdm(product(theta_deg_list, phi_deg_list), 
                               total=len(theta_deg_list) * len(phi_deg_list)):
        print(f"Genereating Trajectory theta: {theta_deg}, phi: {phi_deg}")
        run_multi_seed(image_path_stem, model, seeds, prompt, theta_deg, phi_deg, 
                            w2c_tensor, intrinsics_tensor, input_depth, distance)
# -----------------------------------------------------------------------------
# MAIN DRIVER
# -----------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Gen3C Persistent Mode Runner")

    # Add common arguments
    add_common_arguments(parser)

    # single_image args
    parser.add_argument(
        "--prompt_upsampler_dir",
        type=str,
        default="Pixtral-12B",
        help="Prompt upsampler weights directory relative to checkpoint_dir",
    ) 
    # TODO: do we need this?
    
    parser.add_argument(
        "--movement_distance",
        type=float,
        default=0.3,
        help="Distance of the camera from the center of the scene",
    )
    parser.add_argument(
        "--noise_aug_strength",
        type=float,
        default=0.0,
        help="Strength of noise augmentation on warped frames",
    )
    parser.add_argument(
        "--save_buffer",
        action="store_true",
        help="If set, save the warped images (buffer) side by side with the output video.",
    )
    parser.add_argument(
        "--filter_points_threshold",
        type=float,
        default=0.05,
        help="If set, filter the points continuity of the warped images.",
    )
    parser.add_argument(
        "--foreground_masking",
        action="store_true",
        help="If set, use foreground masking for the warped images.",
    )
    
    # Args for this script
    
    parser.add_argument(
        "--input_image", 
        type=str, 
        help="Path to single image (for trajectory/seed modes)"
    )
    
    # args for post processing and saving
    parser.add_argument(
        "--pose_save_path",
        type=str,
        default=None,
        help="Path to save camera poses (w2c), e.g. outputs/gen3c/panda_w2c.npy",
    )
    parser.add_argument(
        "--K_save_path",
        type=str,
        default=None,
        help="Path to save camera intrinsics, e.g. outputs/gen3c/panda_K.npy",
    )
    
    # trajectory related args
    parser.add_argument(
        "--agent_speed",
        type=float,
        default=0.5,
        help="the physical speed of the agent in m/s (used in forward trajectory only. 0.5 by default)",
    )
    
    parser.add_argument(
        "--sample_angle",
        type=int,
        default=10,
        help="the theta angle increment (in degrees) to sample the trajectories (used in equi-angular sampling only. 30 by default)",
    )
    
    
    parser.add_argument(
        "--num_seeds",
        type=int,
        default=4,
        help="number of seeds to use for multi-seed generation. The seeds are generated with a rng for reproducibility.",
    )
    
    # Standard GEN3C Args (Mocked/Defaults)
    args = parser.parse_args()

    print(args.negative_prompt)
    
    # Apply Persistent Mode Constraints
    args.batch_input_path = None
    args.input_image_path = None
    args.trajectory = 'none'
    args.video_save_name = ""
    args.prompt = " "
    args.disable_guardrail = True
    args.disable_prompt_upsampler = True

    # 1. Initialize Model (DONE ONCE)
    print("Initializing GEN3C Model...")
    model = Gen3cPersistentModel(args)
    
    
    if not args.input_image or not os.path.exists(args.input_image):
        print("Error: --input_image is required and must exist for seed mode.")
        return
    
    n = args.num_seeds
    limit = 100
    # Generate a permutation and take the first n elements
    unique_random = torch.randperm(limit)[:n]

    # Sort them
    sorted_unique, _ = torch.sort(unique_random)
    seeds_to_test = sorted_unique.tolist()
    print(f"Using seeds to generate videos: {seeds_to_test}")

    # Start running the pipeline

    # First check if the file is valid
    is_dir = check_path(args.input_image)

    if is_dir:
        # if it is a directory, loop over
        image_dir = Path(args.input_image)
        for image_path in image_dir.iterdir():
            print(f"Processing image: {image_path}")
            if not check_path(image_path):
                run_multi_traj_multi_seed(model, image_path, seeds_to_test, args.prompt, sample_angle=args.sample_angle, debug=True)
    else:
        run_multi_traj_multi_seed(model, args.input_image, seeds_to_test, args.prompt, sample_angle=args.sample_angle)
    
    # 3. Cleanup
    model.cleanup()

if __name__ == "__main__":
    main()