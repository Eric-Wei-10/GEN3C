# Multi-Trajectory Per-Pixel Variance (Warp-to-Frame0)

This program computes **per-pixel variance across multiple videos** for a chosen frame index `t`, and (optionally) **projects the result onto the input frame (frame 0)** using camera intrinsics/extrinsics and depth.

It supports:
- **Single trajectory mode**: compute variance across videos inside one trajectory folder.
- **Multi trajectory mode**: compute per-trajectory variance for many trajectory folders and **combine** them onto one reference frame (frame 0) using orientation-based reliability.

The visualization script saves (in one pass):
- variance image
- coverage image
- (combined only) source-trajectory index image

---

## Input data layout

### Multi-trajectory root
Your `inputs/` directory should look like:

```
inputs/
  result_{theta}_{phi}/
    camera_data.npz
    video_{seed}.mp4
    video_{seed}.mp4
    ...
  result_{theta}_{phi}/
    camera_data.npz
    video_{seed}.mp4
    ...
  ...
```

Notes:
- Each `result_{theta}_{phi}` folder is one **trajectory**.
- All videos inside one trajectory folder share the same trajectory/camera path.
- `theta` and `phi` are in **degrees** and must match the folder name exactly.

### Video ordering matters
Videos are ordered by **numeric seed** extracted from filenames like:

```
video_0.mp4, video_1.mp4, video_2.mp4, ...
```

This must match the order used to store `depth_all[i]` and `mask_all[i]` inside `camera_data.npz`.

---

## `camera_data.npz` format

Each trajectory folder must include `camera_data.npz` with keys:

- `w2c`: `(T, 4, 4)` float32 — world-to-camera extrinsics per frame
- `K`: `(T, 3, 3)` float32 — intrinsics per frame
- `depth0`: `(H, W)` float32 — depth map for frame 0
- `depth_all`: `(N, T, H, W)` float32 — per-video depth maps
- `mask_all`: `(N, T, H, W)` float32 — per-video depth validity masks
- `height`: scalar int — image height
- `width`: scalar int — image width

Where:
- `T` = number of frames
- `N` = number of videos (seeds) in this trajectory folder
- `H, W` match video resolution

---

## Running

### `run.sh` interface

`run.sh` runs:
1) computation (`main.py`)
2) visualization (`visualize_variance.py`) to produce images

Usage (positional arguments):

```
bash run.sh [run_mode] [frame_index] [mode] [output_dir] [traj_dir_or_inputs_root] [traj_mask] [combine_policy] [sigma_deg]
```

Arguments:
1. `run_mode`: `single | multi`
2. `frame_index`: e.g. `96`
3. `mode`: `frame_t | forward | backward | hybrid`
4. `output_dir`: e.g. `outputs`
5. `traj_dir` (single) **or** `inputs_root` (multi)
6. `traj_mask`: `fill | intersection`
7. `combine_policy` (multi): `priority | soft`
8. `sigma_deg` (multi, soft only): e.g. `15`

Outputs:
- `${out_prefix}.npz`
- `${out_prefix}_variance.png`
- `${out_prefix}_coverage.png`
- `${out_prefix}_source.png` (combined only)

---

## Examples

### Single trajectory
Compute variance across seeds for one trajectory folder:

```bash
bash run.sh single 96 backward outputs ./inputs/result_-15_0 fill
```

This produces something like:
- `outputs/single_result_-15_0_backward_t96_YYYYMMDD_HHMMSS.npz`
- `outputs/single_result_-15_0_backward_t96_..._variance.png`
- `outputs/single_result_-15_0_backward_t96_..._coverage.png`

> Single mode does not output a source map.

### Multi trajectory (combine many folders)
Combine all `result_*` folders under `inputs/`:

```bash
bash run.sh multi 96 backward outputs ./inputs fill priority 15
```

This produces:
- `outputs/combined_priority_hybrid_t96_... .npz`
- `outputs/combined_priority_hybrid_t96_..._variance.png`
- `outputs/combined_priority_hybrid_t96_..._coverage.png`
- `outputs/combined_priority_hybrid_t96_..._source.png`

---

## Warp modes

- `frame_t`: variance on frame `t` directly (no projection to frame 0)
- `forward`: projects frame-0 pixels into frame `t` using `depth0`, samples frame `t` via `grid_sample`
- `backward`: uses per-video `depth_all[i,t]` to project pixels from frame `t` back to frame 0 via splatting
- `hybrid`: uses backward where available, otherwise forward

In `forward/backward/hybrid`, the variance map is defined on the **frame 0 (input frame) grid**.

---

## Coverage masks (single trajectory)

Single-trajectory outputs include:
- `valid_counts[u,v]`: number of videos that contributed to pixel `(u,v)`
- `fill_mask = (valid_counts >= 2)`  
  Pixels where variance is meaningful (unbiased variance requires at least 2 samples)
- `intersection_mask = (valid_counts == N)`  
  Pixels valid in all videos in the trajectory

The visualizer can show either fill or intersection coverage (via `traj_mask`).

---

## Combining trajectories (multi mode)

Each trajectory is assumed to start from the same initial view (same frame 0), but with different motion directions.

Folder name `result_{theta}_{phi}` supplies orientation:
- `theta` (deg): angle between the projection of motion direction onto the `x-z` plane and the `+z` axis
- `phi` (deg): angle between motion direction and the `x-z` plane (elevation)

A scalar orientation distance to `(0,0)` is computed as:

\[
d(\theta,\phi) = \arccos(\cos\phi \cos\theta)
\]

Smaller distance = “closer to forward / 0 orientation” = higher reliability.

### `traj_mask`: what counts as valid for a trajectory during combining
- `fill`: a trajectory contributes where `fill_mask == 1` (>=2 valid samples)
- `intersection`: a trajectory contributes only where `intersection_mask == 1` (all samples valid)

### Combine policies
- `priority`: per pixel, pick the **closest-to-0** trajectory that has valid coverage at that pixel
- `soft`: per pixel, blend all valid trajectories with Gaussian weights based on distance

`sigma_deg` controls how quickly weights fall off in `soft` mode.

---


## Visualization outputs

`visualize_variance.py` writes images in one pass:
- `*_variance.png`: masked variance display (outside coverage is NaN/blank)
- `*_coverage.png`: coverage mask
- `*_source.png`: (combined only) which trajectory supplied each pixel

`run.sh` automatically calls:

```bash
python visualize_variance.py \
  --result_npz "$OUTPUT_NPZ" \
  --out_prefix "$OUT_PREFIX" \
  --traj_mask "$TRAJ_MASK"
```

---

## Troubleshooting

### `BadZipFile: File is not a zip file`
`camera_data.npz` is corrupted or not actually a `.npz` zip archive. Ensure each trajectory folder contains a valid `camera_data.npz`.

---

## Repo structure

```
inputs/
  result_{theta}_{phi}/
    camera_data.npz
    video_{seed}.mp4
    ...
  ...
variance/
  __init__.py
  combine.py
  core.py
  io_load.py
  multi_T.py
  orientation.py
  single_T.py
main.py
run.sh
visualize_variance.py
```

---

## Retrieve validity mask and occlusion mask

```bash
bash run.sh multi [frame_index] forward outputs ./inputs [traj_mask] [combine_policy] [sigma_deg]
```

```bash
bash run.sh multi [frame_index] backward outputs ./inputs [traj_mask] [combine_policy] [sigma_deg]
```

1. `frame_index`: e.g. `96`
2. `traj_mask`: `fill | intersection`
3. `combine_policy`: `priority | soft`
4. `sigma_deg` (soft only): e.g. `15`

### Validity mask

```python
import numpy as np

data = np.load("outputs/combined_priority_backward_t{frame_index}_YYYYMMDD_HHMMSS.npz")
combined_mask = data["combined_mask"].astype(np.float32)
validity_mask = combined_mask > 0.5
```

### Occlusion mask

```python
import numpy as np

data = np.load("outputs/combined_priority_forward_t{frame_index}_YYYYMMDD_HHMMSS.npz")
combined_mask = data["combined_mask"].astype(np.float32)
occlusion_mask = combined_mask > 0.5
```
