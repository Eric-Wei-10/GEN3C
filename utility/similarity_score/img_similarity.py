#!/usr/bin/env python3
import argparse
import os
import glob

import cv2
import numpy as np
import torch


def parse_args():
    # python img_similarity.py --help.
    parser = argparse.ArgumentParser(
        description=(
            "Compute per-frame variance across multiple videos.\n"
            "Precondition: All videos must have the same number of frames and frame size.\n"
            "Computation can run on GPU if available."
        )
    )
    # python img_similarity.py --video-dir path/to/videos.
    parser.add_argument(
        "--video-dir",
        default=None,
        help="Directory containing input videos "
             "(default: 'for_variance' folder).",
    )
    # python img_similarity.py --format "*.mp4,*.avi,*.mov,*.mkv".
    parser.add_argument(
        "--format",
        default="*.mp4,*.avi,*.mov,*.mkv",
        help="Comma-separated list of glob patterns to find videos (default: %(default)s).",
    )
    # python img_similarity.py --output-npy frame_variance.npy.
    parser.add_argument(
        "--output-npy",
        default="frame_variance.npy",
        help="Path to save per-frame variance vector as NumPy .npy (default: %(default)s).",
    )
    # python img_similarity.py --cpu.
    parser.add_argument(
        "--cpu",
        action="store_true",
        help="Force CPU even if a CUDA GPU is available.",
    )
    return parser.parse_args()


def find_videos(video_dir, formats):
    """
    Find video files in the given directory matching the specified patterns.
    Returns a sorted list of unique file paths.
    """
    formats = [p.strip() for p in formats.split(",") if p.strip()]
    paths = []
    for fmt in formats:
        paths.extend(glob.glob(os.path.join(video_dir, fmt)))
    paths = sorted(set(paths))
    return paths


def load_videos_as_array(video_paths):
    """
    Load videos as float32 arrays and ensure they all have the same
    number of frames and frame size.

    Returns:
        videos : np.ndarray or None
            Array of shape (V, T, H, W, 3) if successful; otherwise None.
        num_frames : int or None
            Number of frames per video, if successful.
        frame_size : tuple or None
            (width, height), if successful.
    """
    videos = []
    num_frames = set()
    frame_sizes = set()

    # Load each video.
    for path in video_paths:
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            print(f"Warning: failed to open video: {path}")
            continue
        
        # Read all frames.
        frames = []
        while True:
            ret, frame = cap.read()
            # Break if no more frames.
            if not ret:
                break

            # Convert BGR to RGB (OpenCV gives BGR).
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame.astype(np.float32))

        cap.release()

        # Safety Check 1: check if any frame was read.
        if not frames:
            print(f"Warning: failed to read frames from video: {path}")
            continue
        
        # Stack frames into array.
        frames_arr = np.stack(frames, axis=0)       # (T, H, W, 3)
        T, H, W, C = frames_arr.shape
        num_frames.add(T)
        frame_sizes.add((W, H))
        videos.append(frames_arr)
        print(f"Loaded video {path} with {T} frame(s), size {W}x{H}, channels={C}.")

    # Safety Check 2: check if any video was loaded.
    if not videos:
        print("No valid videos could be loaded.")
        return None, None, None

    # Safety Check 3: check if all videos have the same number of frames.
    if len(num_frames) != 1:
        print("Error: Not all videos have the same number of frames.")
        print(f"Found number of frames: {num_frames}")
        print("Please trim or pad your videos so they all match.")
        return None, None, None

    # Safety Check 4: check if all videos have the same frame size.
    if len(frame_sizes) != 1:
        print("Error: Not all videos have the same frame size.")
        print(f"Found sizes: {frame_sizes}")
        print("Please resize your videos so they all match.")
        return None, None, None

    # Stack videos into array.
    videos_arr = np.stack(videos, axis=0)           # (V, T, H, W, 3)
    num_frames = list(num_frames)[0]
    frame_size = list(frame_sizes)[0]

    return videos_arr, num_frames, frame_size


def compute_per_frame_variance_torch(videos_tensor):
    """
    Compute per-frame variance across videos.

    Parameters:
        videos_tensor : torch.Tensor of shape (V, T, H, W, 3), 
                        dtype float32 or float64, on CPU or GPU.

    Returns:
        frame_var : torch.Tensor of shape (T,), 
                    dtype float64, per-frame variance averaged over all pixels and channels.
                    Lower means = Frames are more similar across videos.
    """
    # Convert to float64 for better numerical precision.
    x = videos_tensor.to(torch.float64)                 # (V, T, H, W, 3)

    # Variance across videos (dim=0) with unbiased=False.
    var_video = torch.var(x, dim=0, unbiased=False)     # (T, H, W, 3)

    # Average over (H, W, C).
    frame_var = var_video.mean(dim=(1, 2, 3))           # (T,)

    return frame_var


def main():
    args = parse_args()

    # Default video directory: 'for_variance'.
    script_dir = os.path.dirname(os.path.abspath(__file__))
    default_video_dir = os.path.join(script_dir, "for_variance")
    video_dir = args.video_dir or default_video_dir

    print(f"Using video directory: {video_dir}")

    video_paths = find_videos(video_dir, args.format)
    if not video_paths:
        print(f"No videos found in {video_dir} with format(s) {args.format}.")
        return

    print(f"Found {len(video_paths)} video file(s). Loading and checking shapes...")

    videos_np, _, _ = load_videos_as_array(video_paths)
    if videos_np is None:
        return

    N, T, H, W, C = videos_np.shape
    print(
        f"\nLoaded {N} valid video(s) with {T} frame(s) each, "
        f"frame size {W}x{H}, channels={C}."
    )

    # Decide device: GPU if available and not forced to CPU; otherwise CPU.
    if not args.cpu and torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    # Compute per-frame variance across videos and convert them into a list.
    print("\nComputing per-frame variance across videos...")
    frame_var_torch = compute_per_frame_variance_torch(torch.from_numpy(videos_np).to(device))
    frame_var = frame_var_torch.cpu().numpy()
    frame_var_list = frame_var.tolist()

    # Save per-frame variance as a 1D array in the specified .npy file.
    np.save(args.output_npy, frame_var)
    print(f"\nSaved per-frame variance vector to {args.output_npy}.")

    # Print summary.
    print(f"\nNumber of frames: {len(frame_var_list)}")
    print("frame_variances =", frame_var_list)


if __name__ == "__main__":
    main()
