# Per-pixel Variance Across Seeded Videos

This program computes **per-pixel variance** across multiple generated videos (different seeds, same settings) at a chosen frame index.

You can compute variance:
- **Directly on the chosen frame** (`frame_t`).
- After **warping frame t back to frame 0 coordinates** (`forward`, `backward`, `hybrid`).

---

## Expected folder layout

```
your_project/
├─ compute_per_pixel_variance.py
├─ visualize_variance.py
├─ run.sh
├─ videos/
│  ├─ seed0.mp4
│  ├─ seed1.mp4
│  └─ ...
└─ depth_info/
   └─ camera_data.npz
```

Notes:
- All video files inside `videos/` are loaded (sorted by filename).
- All videos must have the **same resolution**.

---

## Required `camera_data.npz` format (warping modes only)

For modes `forward`, `backward`, `hybrid`, the `.npz` file must contain:

- `w2c`        : `(T, 4, 4)`
- `K`          : `(T, 3, 3)`
- `depth0`     : `(H, W)`
- `depth_all`  : `(N, T, H, W)`  *(N must match number of videos in `videos/`)*
- `mask_all`   : `(N, T, H, W)`  *(from moge; 1 = reliable depth, 0 = invalid depth)*
- `height`     : scalar int
- `width`      : scalar int

`frame_t` mode does **not** use these arrays, but the pipeline keeps the same interface.

---

## Installation

### 1) Install `requirements.txt` in your environment
```bash
pip install -r requirements.txt
```

### 2) Make the runner executable
```bash
chmod +x run.sh
```

---

## Modes

Choose via `--mode` (or the $4^\text{th}$ positional argument in `run.sh`):

### 1) `frame_t`
- Visualize per-pixel variance across videos on the same resolution as the chosen frame $t$.
- No warping / no projection.
- No depth needed.
- Dense output.

### 2) `forward`
- Visualize per-pixel variance across videos on the same resolution as the first frame $0$.
- Warp from **frame 0 $\to$ frame t** using `depth0`.
- Only covers what was visible in frame 0 (can miss newly visible content).

### 3) `backward`
- Visualize per-pixel variance across videos on the same resolution as the first frame $0$.
- Warp from **frame t $\to$ frame 0** using each video’s `depth_all[:, t]` and `mask_all[:, t]`.
- Uses splatting; many-to-one mappings are averaged.
- Coverage depends on depth reliability and geometry consistency.

### 4) `hybrid`
- Uses backward wherever it exists; otherwise falls back to forward.
- Useful when backward has holes but forward is stable on frame-0-visible regions.

---

## Command-line usage

### Compute
```bash
python compute_per_pixel_variance.py \
  --videos_dir videos \
  --camera_npz depth_info/camera_data.npz \
  --frame_index 96 \
  --mode frame_t \
  --output_npz out.npz
```

### Visualize
```bash
python visualize_variance.py \
  --result_npz out.npz \
  --save_path out.png \
  --min_count 2 \
  --print_stats
```

`--min_count` controls masking in the visualization:
- Variance is only shown where `valid_counts >= min_count`.
- For warped modes, this avoids showing “variance” computed from 0 or 1 sample.

---

## `run.sh` usage

```bash
./run.sh [videos_dir] [camera_npz] [frame_index] [mode] [output_npz] [output_png]
```

Examples:
```bash
# frame_t variance at t=0.
./run.sh videos depth_info/camera_data.npz 0 frame_t

# backward warp variance at t=48.
./run.sh videos depth_info/camera_data.npz 48 backward

# hybrid warp variance at t=96.
./run.sh videos depth_info/camera_data.npz 96 hybrid
```

---

## Output format (`.npz`)

The result `.npz` contains:

- `var_map`          : `(C, H, W)` per-pixel per-channel variance (RGB).
- `var_scalar`       : `(H, W)` per-pixel variance (mean over RGB).
- `valid_counts`     : `(H, W)` number of contributing videos per pixel.
- `intersection_mask`: `(H, W)` 1 where all N contributed (for warped modes).
- `frame_index`      : scalar int.
- `video_paths`      : list of loaded video paths.
- `mode`             : string (e.g. `frame_t`).

Interpretation:
- In `frame_t` mode, `valid_counts` should equal `N` everywhere.
- In warp modes, pixels with low `valid_counts` indicate low coverage (variance may be masked out).

---

## Debugging

### Test 1: basic sanity
Run `frame_t` at a small frame index:
```bash
./run.sh videos depth_info/camera_data.npz 0 frame_t
```
Expected:
- Coverage should be constant `N`.
- Variance should be dense (no large blank regions).

### Test 2: warp coverage
Try `backward` or `hybrid` and inspect `*_coverage.png`:
```bash
./run.sh videos depth_info/camera_data.npz 96 backward
```
If coverage is low, blank variance regions are expected.

### Quick numeric check
```bash
python - << 'PY'
import numpy as np
d = np.load("out.npz")
vc = d["valid_counts"]
print("mode:", d["mode"])
print("valid_counts min/max:", vc.min(), vc.max())
print("pixels with vc>=2:", (vc>=2).sum(), "/", vc.size)
PY
```

