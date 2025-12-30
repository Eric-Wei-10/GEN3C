# dino_feature.py
import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from dino_model import load_dino, DEFAULT_MODEL_NAME


# --------
UPSCALE_FACTOR = 4.0       
WINDOW_PX = 518            
STRIDE_PX = 259           
BATCH_SIZE = 4            
PATCH_SIZE = 14            # DINOv2 patch size
# -------------------


def extract_frame_from_video_np(video_path: str, t: int = 10) -> np.ndarray:
    """
    from mp4 extract t-th frame (1-based), return numpy array [H, W, 3] (uint8, RGB).

    Args:
        video_path: path to mp4 file
        t         : 1-based frame index (clamped to [1, total_frames])

    Returns:
        frame_hwc : np.ndarray [H, W, 3], dtype=uint8, RGB
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        raise RuntimeError(f"Invalid frame count for video: {video_path}")

    idx = max(0, min(t - 1, total_frames - 1))
    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ret, frame_bgr = cap.read()
    cap.release()

    if not ret:
        raise RuntimeError(f"Cannot read frame {t} (index {idx}) from {video_path}")

    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)  # [H,W,3]
    print(f"[VIDEO] Extracted frame {t}/{total_frames}, size={frame_rgb.shape[1]}x{frame_rgb.shape[0]}")
    return frame_rgb  # HWC, uint8, RGB


def _ensure_hwc_uint8(frame: np.ndarray) -> np.ndarray:
    """
    acept [H,W,3] or [3,H,W], return [H,W,3]、uint8。
    """
    if frame.ndim != 3:
        raise ValueError(f"Expected 3D array, got shape {frame.shape}")

    if frame.shape[0] == 3 and frame.shape[2] != 3:
        # CHW ->  HWC
        frame = np.transpose(frame, (1, 2, 0))

    if frame.shape[2] != 3:
        raise ValueError(f"Expected 3 channels in last dim, got shape {frame.shape}")

    if frame.dtype != np.uint8:
        frame = np.clip(frame, 0, 255).astype(np.uint8)

    return frame  # [H,W,3], uint8


def extract_dense_feature(
    frame_np: np.ndarray,
    model=None,
    processor=None,
    device: torch.device | None = None,
    upscale_factor: float = UPSCALE_FACTOR,
    window_px: int = WINDOW_PX,
    stride_px: int = STRIDE_PX,
    batch_size: int = BATCH_SIZE,
) -> torch.Tensor:
    """
   sliding window + upscale_factor=4 -> dense DINO feature

    Args:
        frame_np      : np.ndarray, [H,W,3] or [3,H,W] (RGB)
        model         : DINOv2 model; if None, load_dino()
        processor     : AutoImageProcessor
        device        : torch.device, if None, use model device
        upscale_factor: default 4.0
        window_px     : default 518
        stride_px     : 0.5 * window_px, default 259
        batch_size    : default 4

    Returns:
        dense_feat: torch.Tensor [H, W, C] (float32, CPU)
    """
    # 1) HWC uint8
    frame_hwc = _ensure_hwc_uint8(frame_np)
    H, W, _ = frame_hwc.shape

    # 2) load model
    if model is None or processor is None or device is None:
        model, processor, device = load_dino(DEFAULT_MODEL_NAME)
    else:
        device = device or next(model.parameters()).device

    # 3) 
    inputs = processor(
        images=frame_hwc,              # HWC uint8
        return_tensors="pt",
        do_resize=False,
        do_center_crop=False,
    )
    full = inputs["pixel_values"].to(device)  # [1,3,H,W]

    emb_dim = model.config.hidden_size

    # 4) (GPU, float16)
    feat_sum = torch.zeros(H, W, emb_dim, dtype=torch.float16, device=device)
    feat_cnt = torch.zeros(H, W, 1,       dtype=torch.float16, device=device)

    # 5)
    def make_starts(L, win, stride):
        if L <= win:
            return [0]
        s = list(range(0, L - win + 1, stride))
        if s[-1] != L - win:
            s.append(L - win)
        return s

    h_starts = make_starts(H, window_px, stride_px)
    w_starts = make_starts(W, window_px, stride_px)

    windows = []
    for top in h_starts:
        for left in w_starts:
            bottom = min(top + window_px, H)
            right = min(left + window_px, W)
            h = bottom - top
            w = right - left
            windows.append((top, bottom, left, right, h, w))

    print(f"[DENSE] H={H}, W={W}, total windows={len(windows)}, batch_size={batch_size}")

    use_autocast = device.type == "cuda"

    # 6) 
    for i in range(0, len(windows), batch_size):
        batch = windows[i: i + batch_size]

        crops = []
        meta = []

        # (a) upscale
        for (top, bottom, left, right, h, w) in batch:
            crop = full[:, :, top:bottom, left:right]  # [1,3,h,w]

            h_up = int(round(h * upscale_factor))
            w_up = int(round(w * upscale_factor))

            if h_up > 2072 or w_up > 2072:
                raise ValueError(
                    f"Upscaled window too large: {h_up}x{w_up}, "
                    f"reduce window_px or upscale_factor."
                )

            up_crop = F.interpolate(
                crop,
                size=(h_up, w_up),
                mode="bicubic",
                align_corners=False,
            )  # [1,3,h_up,w_up]

            crops.append(up_crop)
            meta.append((top, bottom, left, right, h, w, h_up, w_up))

        x = torch.cat(crops, dim=0)  # [B,3,H',W']

        # (b) DINO
        with torch.no_grad():
            if use_autocast:
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    out = model(x).last_hidden_state  # [B,1+N,C]
            else:
                out = model(x).last_hidden_state

        # (c) 
        for b, (top, bottom, left, right, h, w, h_up, w_up) in enumerate(meta):
            tokens = out[b:b + 1, 1:, :]  # [1,N,C]

            Hp = h_up // PATCH_SIZE
            Wp = w_up // PATCH_SIZE
            tokens = tokens[:, : Hp * Wp, :]          # [1,Hp*Wp,C]
            feats_pw = tokens.reshape(1, Hp, Wp, emb_dim)  # [1,Hp,Wp,C]

            feats_chw = feats_pw.permute(0, 3, 1, 2)       # [1,C,Hp,Wp]
            feats_up = F.interpolate(
                feats_chw,
                size=(h, w),
                mode="bilinear",
                align_corners=False,
            )                                              # [1,C,h,w]
            feats_hwC = feats_up.squeeze(0).permute(1, 2, 0)  # [h,w,C]
            feats_hwC = feats_hwC.to(torch.float16)

            feat_sum[top:bottom, left:right, :] += feats_hwC
            feat_cnt[top:bottom, left:right, :] += 1.0

    # 7)  CPU + float32
    dense_feat = (feat_sum / (feat_cnt + 1e-6)).cpu().float()  # [H,W,C]
    return dense_feat


# ---------- CLI dense feature ----------
def main():
    parser = argparse.ArgumentParser(
        description="Extract DINOv2 dense feature from frame t of an mp4 video "
                    "using sliding window + upscale_factor=4."
    )
    parser.add_argument("--input_path", type=str, required=True,
                        help="Path to input mp4 video.")
    parser.add_argument("--t", type=int, default=10,
                        help="1-based frame index (default: 10).")
    parser.add_argument("--output_path", type=str, default=None,
                        help="Where to save .npy dense feature. "
                             "Default: ./<video_stem>_t<t>_dense.npy")
    args = parser.parse_args()

    video_path = Path(args.input_path)
    if not video_path.is_file():
        raise FileNotFoundError(f"Video not found: {video_path}")

    if args.output_path is not None:
        out_path = Path(args.output_path)
    else:
        out_name = f"{video_path.stem}_t{args.t}_dense.npy"
        out_path = Path.cwd() / out_name

    print(f"[CLI] Input video : {video_path}")
    print(f"[CLI] Output path : {out_path}")

    # 1) 
    frame_np = extract_frame_from_video_np(str(video_path), t=args.t)

    # 2) 
    model, processor, device = load_dino(DEFAULT_MODEL_NAME)

    # 3) 
    dense_feat = extract_dense_feature(
        frame_np,
        model=model,
        processor=processor,
        device=device,
        upscale_factor=UPSCALE_FACTOR,
        window_px=WINDOW_PX,
        stride_px=STRIDE_PX,
        batch_size=BATCH_SIZE,
    )

    print(f"[CLI] Dense feature shape: {dense_feat.shape}, dtype={dense_feat.dtype}")

    # 4) save as .npy
    np.save(out_path, dense_feat.numpy())
    print(f"[CLI] Saved dense feature to: {out_path}")


if __name__ == "__main__":
    main()

# how to import and use:
# from dino_model import load_dino
# from dino_feature import extract_dense_feature

# model, processor, device = load_dino()  

# # Get dense DINO features [H, W, C]
# dense_feat = extract_dense_feature(
#     frame_np,
#     model=model,
#     processor=processor,
#     device=device,
# )

#how to run:
# python utility/dino_feature/dino_feature.py \
#   --input_path outputs/persistent_charlotte/mode3_seed_42.mp4 \
#   --t 40 \
#   --output_path ./video_t40_dense.npy
