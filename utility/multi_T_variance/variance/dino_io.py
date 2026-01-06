# variance/dino_io.py
# Load DINO features from videos at specified frame indices.
import numpy as np

from dino_model import load_dino, DEFAULT_MODEL_NAME
from dino_feature import extract_frame_from_video_np, extract_dense_feature


def load_dino_features_at_t_from_list(video_paths, frame_index: int):
    """
    Load the t-th frame from each video, extract dense DINO features.

    Args:
        video_paths: list[str]
        frame_index: 0-based frame index (original convention in variance computation)

    Returns:
        feats_np: [N, D, H, W] float32 numpy
        ordered_paths: list[str]
    """
    # dino_feature.extract_frame_from_video_np uses 1-based frame index.
    t_1based = int(frame_index) + 1

    # Load DINO once (cached by lru_cache).
    model, processor, device = load_dino(DEFAULT_MODEL_NAME)

    feats = []
    H_ref, W_ref, D_ref = None, None, None

    for p in video_paths:
        frame_uint8 = extract_frame_from_video_np(p, t=t_1based)        # [H, W, 3] uint8 RGB
        H, W, _ = frame_uint8.shape

        feat_hwD = extract_dense_feature(
            frame_uint8,
            model=model,
            processor=processor,
            device=device,
        )  # torch [H, W, D] float32 CPU

        feat_np = feat_hwD.numpy()                                      # [H, W, D] float32
        H2, W2, D = feat_np.shape
        if (H2, W2) != (H, W):
            raise ValueError(f"Unexpected DINO feature resolution: got {(H2,W2)} vs frame {(H,W)} for {p}")

        feat_chw = np.transpose(feat_np, (2, 0, 1)).astype(np.float32)  # [D, H, W]

        if H_ref is None:
            H_ref, W_ref, D_ref = H, W, D
        else:
            if (H, W) != (H_ref, W_ref):
                raise ValueError(f"Resolution mismatch: {p} is {(H,W)} vs expected {(H_ref,W_ref)}")
            if D != D_ref:
                raise ValueError(f"D mismatch: {p} produced D={D} vs expected D={D_ref}")

        feats.append(feat_chw)

    feats_np = np.stack(feats, axis=0)                      # [N, D, H, W]
    return feats_np, list(video_paths)
