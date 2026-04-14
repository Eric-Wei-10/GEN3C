"""
data_filtering.py

Given a directory of RGB images, run MoGE depth estimation on each one,
unproject to 3-D, simulate a forward camera move, and check for
collision / explorability. Outputs are saved per-image under --out_dir.

Usage:
  python data_filtering.py --image_dir assets/diffusion/dataset_all/dataset_shard_1 \
                           --out_dir   $SCRATCH/data_filtering_output

Run inside gen3c_env:
  /cluster/project/cvg/students/shangwu/gen3c_env/bin/python data_filtering.py ...

Already-processed images are skipped (resume-safe).
"""

import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from moge.model.v1 import MoGeModel


# ---------------------------------------------------------------------------
# Image I/O
# ---------------------------------------------------------------------------

def load_image_rgb(path: str) -> np.ndarray:
    """Load image as [H, W, 3] uint8 RGB array."""
    img_bgr = cv2.imread(str(path))
    if img_bgr is None:
        raise FileNotFoundError(f"Image not found: {path}")
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)


# ---------------------------------------------------------------------------
# MoGE
# ---------------------------------------------------------------------------

def run_moge(image_rgb: np.ndarray, device: str, moge_model=None):
    """
    Run MoGE on a [H, W, 3] uint8 RGB image.

    MoGE works best at 720x1280.  We resize before inference and then
    scale the intrinsics back to the *original* resolution so that the
    unprojected point cloud is consistent with the displayed image.

    Returns
    -------
    depth  : (H, W)  float32  — depth in metres at original resolution
    K      : (3, 3)  float64  — intrinsics in pixel coords (original res)
    mask   : (H, W)  bool     — valid depth pixels
    """
    orig_h, orig_w = image_rgb.shape[:2]

    # MoGE preferred inference resolution
    moge_h, moge_w = 720, 1280

    img_resized = cv2.resize(image_rgb, (moge_w, moge_h))
    img_tensor = (
        torch.tensor(img_resized / 255.0, dtype=torch.float32, device=device)
        .permute(2, 0, 1)          # (C, H, W)
    )

    if moge_model is None:
        print("Loading MoGE model …")
        moge_model = MoGeModel.from_pretrained("Ruicheng/moge-vitl").to(device)
        moge_model.eval()

    print("Running MoGE inference …")
    with torch.no_grad():
        out = moge_model.infer(img_tensor)

    depth_moge  = out["depth"].cpu()        # (moge_h, moge_w)
    intr_norm   = out["intrinsics"].cpu()   # (3, 3)  normalised
    mask_moge   = out["mask"].cpu()         # (moge_h, moge_w)  bool

    # --- Denormalise intrinsics to pixel coords at MoGE resolution ----------
    K_moge = intr_norm.clone().numpy().astype(np.float64)
    K_moge[0, 0] *= moge_w   # fx
    K_moge[1, 1] *= moge_h   # fy
    K_moge[0, 2] *= moge_w   # cx
    K_moge[1, 2] *= moge_h   # cy

    # --- Resize depth and mask back to original resolution ------------------
    depth_orig = F.interpolate(
        depth_moge.unsqueeze(0).unsqueeze(0),
        size=(orig_h, orig_w), mode="bilinear", align_corners=False
    ).squeeze().numpy().astype(np.float32)

    mask_orig = F.interpolate(
        mask_moge.float().unsqueeze(0).unsqueeze(0),
        size=(orig_h, orig_w), mode="nearest"
    ).squeeze().numpy().astype(bool)

    # --- Scale intrinsics to original resolution ----------------------------
    K = K_moge.copy()
    K[0, 0] *= orig_w / moge_w   # fx
    K[1, 1] *= orig_h / moge_h   # fy
    K[0, 2] *= orig_w / moge_w   # cx
    K[1, 2] *= orig_h / moge_h   # cy

    return depth_orig, K, mask_orig


# ---------------------------------------------------------------------------
# 3-D unprojection
# ---------------------------------------------------------------------------

def unproject(image_rgb: np.ndarray, depth: np.ndarray,
              K: np.ndarray, mask: np.ndarray):
    """
    Backproject pixels to 3-D using the pinhole camera model.

    Parameters
    ----------
    image_rgb : (H, W, 3) uint8
    depth     : (H, W)    float32   depth in metres
    K         : (3, 3)    float64   intrinsics in pixel coords
    mask      : (H, W)    bool      valid pixels

    Returns
    -------
    pts   : (N, 3) float32   XYZ in camera coords
    colors: (N, 3) uint8     RGB colours
    """
    H, W = depth.shape
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    # pixel grid
    us, vs = np.meshgrid(np.arange(W, dtype=np.float32),
                         np.arange(H, dtype=np.float32))

    X = (us - cx) * depth / fx
    Y = (vs - cy) * depth / fy
    Z = depth

    pts = np.stack([X, Y, Z], axis=-1)[mask]          # (N, 3)
    colors = image_rgb[mask]                           # (N, 3)
    return pts.astype(np.float32), colors


# ---------------------------------------------------------------------------
# PLY export
# ---------------------------------------------------------------------------

def save_ply(path: str, pts: np.ndarray, colors: np.ndarray) -> None:
    """
    Save a coloured point cloud as a binary-little-endian PLY file.

    Parameters
    ----------
    path   : output file path
    pts    : (N, 3) float32   XYZ
    colors : (N, 3) uint8     RGB
    """
    N = len(pts)
    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        f"element vertex {N}\n"
        "property float x\n"
        "property float y\n"
        "property float z\n"
        "property uchar red\n"
        "property uchar green\n"
        "property uchar blue\n"
        "end_header\n"
    )
    # interleave xyz (float32) and rgb (uint8) into a structured array
    dt = np.dtype([("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
                   ("r", "u1"), ("g", "u1"), ("b", "u1")])
    data = np.empty(N, dtype=dt)
    data["x"], data["y"], data["z"] = pts[:, 0], pts[:, 1], pts[:, 2]
    data["r"], data["g"], data["b"] = colors[:, 0], colors[:, 1], colors[:, 2]

    with open(path, "wb") as f:
        f.write(header.encode("ascii"))
        f.write(data.tobytes())

    print(f"Saved {N:,} points → {path}")


# ---------------------------------------------------------------------------
# Depth visualisation helper
# ---------------------------------------------------------------------------

def visualize_depth(depth: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """
    Render a depth map as an INFERNO-colourmap image with a metric colorbar.

    near → dark (0), far → bright (255).  Holes / invalid pixels are black.

    Returns an (H, W+bar_w, 3) uint8 BGR image ready for cv2.imwrite.
    """
    H, W = depth.shape
    d_min    = float(depth[valid].min())
    d_max    = float(depth[valid].max())
    d_median = float(np.median(depth[valid]))

    depth_vis = np.zeros((H, W), dtype=np.float32)
    depth_vis[valid] = (depth[valid] - d_min) / (d_max - d_min) * 255
    depth_color = cv2.applyColorMap(depth_vis.astype(np.uint8), cv2.COLORMAP_INFERNO)
    depth_color[~valid] = 0

    bar_w    = 60
    n_ticks  = 6
    bar_grad = np.linspace(0, 255, H, dtype=np.uint8)   # top=dark=near, bottom=bright=far
    bar_img  = cv2.applyColorMap(
        np.tile(bar_grad[:, None], (1, bar_w)), cv2.COLORMAP_INFERNO)

    font, fs, th = cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1
    white = (255, 255, 255)
    for i in range(n_ticks):
        t   = i / (n_ticks - 1)
        y   = int(t * (H - 1))
        val = d_min + t * (d_max - d_min)
        cv2.line(bar_img, (0, y), (8, y), white, 1)
        cv2.putText(bar_img, f"{val:.2f}m", (10, y + 4), font, fs, white, th, cv2.LINE_AA)

    stats_h = 80
    stats   = np.zeros((stats_h, bar_w, 3), dtype=np.uint8)
    for i, txt in enumerate([f"min:{d_min:.2f}m", f"max:{d_max:.2f}m", f"med:{d_median:.2f}m"]):
        cv2.putText(stats, txt, (4, 18 + i * 22), font, 0.40, white, th, cv2.LINE_AA)

    sidebar = np.vstack([bar_img, stats])
    pad     = np.zeros((stats_h, W, 3), dtype=np.uint8)
    return np.hstack([np.vstack([depth_color, pad]), sidebar])


# ---------------------------------------------------------------------------
# Camera translation
# ---------------------------------------------------------------------------

def move_camera_forward(pts: np.ndarray, colors: np.ndarray,
                        forward_m: float = 1.0):
    """
    Simulate moving the camera forward by `forward_m` metres along +Z.

    In camera coordinates the world shifts by -forward_m along Z, so every
    point's Z decreases by `forward_m`.  Points that end up at Z <= 0 are
    behind the new camera and are discarded.

    Parameters
    ----------
    pts       : (N, 3) float32   XYZ in original camera coords
    colors    : (N, 3) uint8     RGB colours
    forward_m : metres to advance along +Z

    Returns
    -------
    pts_new    : (M, 3) float32  — only points in front of new camera
    colors_new : (M, 3) uint8
    """
    pts_new = pts.copy()
    pts_new[:, 2] -= forward_m          # world shifts back by forward_m
    visible = pts_new[:, 2] > 0         # keep only Z > 0
    n_dropped = (~visible).sum()
    print(f"Camera moved forward {forward_m} m: "
          f"kept {visible.sum():,} pts, dropped {n_dropped:,} pts behind camera")
    return pts_new[visible], colors[visible]


# ---------------------------------------------------------------------------
# Re-projection onto image plane
# ---------------------------------------------------------------------------

def project_to_image(pts: np.ndarray, colors: np.ndarray,
                     K: np.ndarray, H: int, W: int) -> np.ndarray:
    """
    Project a point cloud onto an H×W image using intrinsics K.

    When multiple points map to the same pixel, the closest one (smallest Z)
    wins (painter's algorithm / z-buffer).

    Parameters
    ----------
    pts    : (N, 3) float32   XYZ in the *new* camera coordinate system
    colors : (N, 3) uint8     RGB colours
    K      : (3, 3) float64   camera intrinsics (same K as original)
    H, W   : output image size

    Returns
    -------
    image  : (H, W, 3) uint8  — black where no point projects
    """
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    X, Y, Z = pts[:, 0], pts[:, 1], pts[:, 2]

    # perspective divide
    u = (fx * X / Z + cx).astype(np.float32)
    v = (fy * Y / Z + cy).astype(np.float32)

    # integer pixel coords
    ui = np.round(u).astype(np.int32)
    vi = np.round(v).astype(np.int32)

    # keep only points that land inside the image
    in_bounds = (ui >= 0) & (ui < W) & (vi >= 0) & (vi < H)
    ui, vi, Z_proj, cols = ui[in_bounds], vi[in_bounds], Z[in_bounds], colors[in_bounds]

    # z-buffer: sort far→near so nearer points overwrite farther ones
    order = np.argsort(Z_proj)[::-1]
    ui, vi, cols = ui[order], vi[order], cols[order]

    image        = np.zeros((H, W, 3),  dtype=np.uint8)
    depth_buffer = np.zeros((H, W),     dtype=np.float32)
    image[vi, ui]        = cols
    depth_buffer[vi, ui] = Z_proj[order]   # nearest point already wins

    n_projected = in_bounds.sum()
    print(f"Projected {n_projected:,} points onto {W}×{H} image "
          f"({(~in_bounds).sum():,} out of bounds)")
    return image, depth_buffer


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def process_one(image_path: Path, out_dir: Path, forward: float,
                depth_threshold: float, device: str, moge_model,
                delete_invalid: bool = False) -> str:
    """Process a single image. Returns status: 'done', 'collision', 'not_explorable'.

    If delete_invalid is True, the source image is deleted when the status is
    'collision' or 'not_explorable'.
    """
    stem = image_path.stem
    sample_dir = out_dir / stem
    sample_dir.mkdir(parents=True, exist_ok=True)

    output_render    = str(sample_dir / f"{stem}_forward{forward:.1f}m_render.png")
    output_depth     = str(sample_dir / f"{stem}_forward{forward:.1f}m_depth.png")
    output_depth_vis = str(sample_dir / f"{stem}_forward{forward:.1f}m_depth_vis.png")
    output_moge_vis  = str(sample_dir / f"{stem}_moge_depth_vis.png")

    image_rgb = load_image_rgb(image_path)
    H, W = image_rgb.shape[:2]

    depth, K, mask = run_moge(image_rgb, device, moge_model)
    cv2.imwrite(output_moge_vis, visualize_depth(depth, mask))

    # Depth quality checks (on valid pixels only)
    valid_depth = depth[mask]
    def _reject(reason):
        if delete_invalid:
            image_path.unlink(missing_ok=True)
            print(f"Deleted ({reason}): {image_path}")
        return reason

    if np.any(np.isinf(valid_depth)):
        return _reject("inf_depth")
    if valid_depth.size > 0 and valid_depth.max() > 20.0:
        return _reject("depth_too_far")
    if valid_depth.size > 0 and (valid_depth.max() - valid_depth.min()) < 0.25:
        return _reject("depth_flat")

    # Collision check
    cy0, cy1 = int(H * 0.475), int(H * 0.525)
    cx0, cx1 = int(W * 0.475), int(W * 0.525)
    roi_depth = depth[cy0:cy1, cx0:cx1]
    roi_mask  = mask[cy0:cy1, cx0:cx1]
    if roi_mask.any() and (roi_depth[roi_mask] < 1.25).any():
        if delete_invalid:
            image_path.unlink(missing_ok=True)
            print(f"Deleted (collision): {image_path}")
        return "collision"

    pts, colors = unproject(image_rgb, depth, K, mask)
    pts_moved, colors_moved = move_camera_forward(pts, colors, forward_m=forward)

    render, depth_new = project_to_image(pts_moved, colors_moved, K, H, W)
    cv2.imwrite(output_render, cv2.cvtColor(render, cv2.COLOR_RGB2BGR))
    depth_mm = (depth_new * 1000).clip(0, 65535).astype(np.uint16)
    cv2.imwrite(output_depth, depth_mm)

    valid_new = depth_new > 0
    n_visible = int(valid_new.sum())
    if n_visible > 0:
        n_near = int((depth_new[valid_new] < depth_threshold).sum())
        if (n_near / n_visible) >= 0.70:
            if delete_invalid:
                image_path.unlink(missing_ok=True)
                print(f"Deleted (not_explorable): {image_path}")
            return "not_explorable"

    cv2.imwrite(output_depth_vis, visualize_depth(depth_new, valid_new))
    return "done"


def main():
    parser = argparse.ArgumentParser(description="MoGE depth → 3-D point cloud (batch)")
    parser.add_argument("--image_dir",       required=True, help="Directory of input RGB images")
    parser.add_argument("--out_dir",         default="./data_filtering_output")
    parser.add_argument("--forward",         type=float, default=1.0,
                        help="Move camera forward by this many metres (default: 1.0)")
    parser.add_argument("--depth_threshold", type=float, default=0.5,
                        help="Depth threshold in metres for explorable check (default: 0.5)")
    parser.add_argument("--delete_invalid",  action="store_true",
                        help="Delete source images that fail collision or explorability checks")
    parser.add_argument("--device",          default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    image_dir = Path(args.image_dir)
    out_dir   = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    exts = {".jpg", ".jpeg", ".png"}
    images = sorted(p for p in image_dir.iterdir() if p.suffix.lower() in exts)
    print(f"Found {len(images)} images in {image_dir}")

    print("Loading MoGE model …")
    moge_model = MoGeModel.from_pretrained("Ruicheng/moge-vitl").to(args.device)
    moge_model.eval()

    list_paths = {
        "done":           out_dir / "explorable_images.txt",
        "collision":      out_dir / "collision_images.txt",
        "not_explorable": out_dir / "inexplorable_images.txt",
        "inf_depth":      out_dir / "inf_depth_images.txt",
        "depth_too_far":  out_dir / "depth_too_far_images.txt",
        "depth_flat":     out_dir / "depth_flat_images.txt",
    }
    # Load existing entries so appending is idempotent on resume
    existing = {k: set(p.read_text().splitlines()) if p.exists() else set()
                for k, p in list_paths.items()}

    n_done = n_skipped = n_collision = n_not_explorable = n_error = 0
    pbar = tqdm(images, unit="image")
    open_files = {k: p.open("a") for k, p in list_paths.items()}
    try:
        for image_path in pbar:
            stem = image_path.stem
            done_flag = out_dir / stem / ".done"
            if done_flag.exists():
                n_skipped += 1
                pbar.set_postfix(done=n_done, skip=n_skipped, coll=n_collision,
                                 no_expl=n_not_explorable, err=n_error)
                continue
            try:
                status = process_one(image_path, out_dir, args.forward,
                                     args.depth_threshold, args.device, moge_model,
                                     delete_invalid=args.delete_invalid)
            except Exception as e:
                tqdm.write(f"ERROR [{stem}]: {e}")
                n_error += 1
                pbar.set_postfix(done=n_done, skip=n_skipped, coll=n_collision,
                                 no_expl=n_not_explorable, err=n_error)
                continue

            if status == "collision":
                n_collision += 1
            elif status == "not_explorable":
                n_not_explorable += 1
            else:
                done_flag.touch()
                n_done += 1

            if stem not in existing[status]:
                open_files[status].write(stem + "\n")
                open_files[status].flush()
                existing[status].add(stem)

            pbar.set_postfix(done=n_done, skip=n_skipped, coll=n_collision,
                             no_expl=n_not_explorable, err=n_error)
    finally:
        for f in open_files.values():
            f.close()

    print(f"\nFinished. done={n_done}  skipped={n_skipped}  "
          f"collision={n_collision}  not_explorable={n_not_explorable}  errors={n_error}")
    for k, p in list_paths.items():
        print(f"  {k:>15} → {p}")
    print(f"Outputs in: {out_dir}")


if __name__ == "__main__":
    main()
