# variance/io_load.py
# Handle trajectory-folder I/O: finding camera_data.npz, listing videos in correct order, loading frames.
import os
import glob
import re
import numpy as np

from .core import load_frame_from_video

_VIDEO_EXTS = (".mp4", ".avi", ".mov", ".mkv", ".mpg", ".mpeg", ".wmv", ".m4v")
_SEED_RE = re.compile(r"^(?:video)_(\d+)")


def _extract_seed(path: str) -> int:
    """
    Extract the numeric seed from a video filename.
    
    :param path: The video file path.
    :return: The extracted numeric seed.
    """
    base = os.path.basename(path)
    stem = os.path.splitext(base)[0]
    m = _SEED_RE.match(stem)
    if not m:
        raise ValueError(
            f"Cannot parse seed from filename '{base}'. "
            f"Expected 'video_<seed>.mp4'."
        )
    return int(m.group(1))


def find_camera_npz(traj_dir: str) -> str:
    """
    Find the camera_data.npz file in the given trajectory directory.

    :param traj_dir: The trajectory directory path.
    :return: The full path to camera_data.npz.
    """
    p = os.path.join(traj_dir, "camera_data.npz")
    if not os.path.isfile(p):
        raise RuntimeError(f"Expected camera_data.npz in {traj_dir} but not found.")
    return p


def get_ordered_videos_by_seed(traj_dir: str):
    """
    Get a list of video file paths in the given trajectory directory, ordered by their numeric seed.

    :param traj_dir: The trajectory directory path.
    :return: A list of ordered video file paths.
    """
    videos = [
        p for p in glob.glob(os.path.join(traj_dir, "*"))
        if p.lower().endswith(_VIDEO_EXTS)
    ]
    if not videos:
        raise RuntimeError(f"No video files found in {traj_dir}")

    # Sort by numeric seed.
    videos_sorted = sorted(videos, key=_extract_seed)
    return videos_sorted


def load_frames_at_t_from_list(video_paths, frame_index: int):
    """
    Load the chosen frame index from each video in the given ordered list of video paths.
    
    :param video_paths: A sorted list of video file paths.
    :param frame_index: The index of the frame to load from each video.
    :return: A tuple (frames_t, ordered_paths) where frames_t is a numpy array of shape [N, 3, H, W] 
             and ordered_paths is the list of video paths.
    """
    # frames: holds one [3, H, W] frame per video in the end.
    frames = []
    # Store resolution of the first video to verify consistency.
    H_ref, W_ref = None, None

    for path in video_paths:
        # Load the frame at frame_index.
        frame = load_frame_from_video(path, frame_index)    # [H, W, 3]
        H, W, _ = frame.shape

        # Ensure all videos have the same resolution.
        if H_ref is None:
            H_ref, W_ref = H, W
        elif (H, W) != (H_ref, W_ref):
            raise ValueError(f"Video {path} has resolution {(H,W)}, expected {(H_ref,W_ref)}")

        frames.append(np.transpose(frame, (2, 0, 1)))       # [3, H, W]

    # frames_t[i] = frame from video i (video_paths[i]) at frame index t,
    # and corresponds to depth_all[i] in camera_data.npz.
    frames_t = np.stack(frames, axis=0)                     # [N, 3, H, W]
    
    return frames_t, list(video_paths)
