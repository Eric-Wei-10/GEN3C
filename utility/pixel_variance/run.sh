#!/usr/bin/env bash
set -euo pipefail

# ------------------------------------------------------------------
# Basic config
# ------------------------------------------------------------------
ROOT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Positional arguments with defaults:
#   $1 -> videos_dir
#   $2 -> camera_npz
#   $3 -> frame_index
#   $4 -> mode
#   $5 -> output_npz
#   $6 -> output_png
VIDEOS_DIR="${1:-$ROOT_DIR/videos}"
CAMERA_NPZ="${2:-$ROOT_DIR/depth_info/camera_data.npz}"
FRAME_INDEX="${3:-96}"
MODE="${4:-frame_t}"
OUTPUT_NPZ="${5:-variance_${MODE}_t${FRAME_INDEX}.npz}"
OUTPUT_PNG="${6:-variance_${MODE}_t${FRAME_INDEX}.png}"

echo "[INFO] Using:"
echo "  videos_dir  = ${VIDEOS_DIR}"
echo "  camera_npz  = ${CAMERA_NPZ}"
echo "  frame_index = ${FRAME_INDEX}"
echo "  mode        = ${MODE}"
echo "  output_npz  = ${OUTPUT_NPZ}"
echo "  output_png  = ${OUTPUT_PNG}"

# ------------------------------------------------------------------
# Sanity checks
# ------------------------------------------------------------------
if [ ! -d "$VIDEOS_DIR" ]; then
  echo "ERROR: videos_dir '$VIDEOS_DIR' does not exist."
  exit 1
fi

if [ ! -f "$CAMERA_NPZ" ]; then
  echo "ERROR: camera_npz '$CAMERA_NPZ' does not exist."
  exit 1
fi

# ------------------------------------------------------------------
# Run variance computation
# ------------------------------------------------------------------
echo "[INFO] Running compute_per_pixel_variance.py"
python "$ROOT_DIR/compute_per_pixel_variance.py" \
  --videos_dir "$VIDEOS_DIR" \
  --camera_npz "$CAMERA_NPZ" \
  --frame_index "$FRAME_INDEX" \
  --mode "$MODE" \
  --output_npz "$OUTPUT_NPZ"

# ------------------------------------------------------------------
# Run visualization
# ------------------------------------------------------------------
echo "[INFO] Running visualize_variance.py"
python "$ROOT_DIR/visualize_variance.py" \
  --result_npz "$OUTPUT_NPZ" \
  --save_path "$OUTPUT_PNG" \
  --save_coverage

echo "[INFO] Done."
echo "  Result npz:   $OUTPUT_NPZ"
echo "  Result image: $OUTPUT_PNG"
echo "  Coverage:     ${OUTPUT_PNG%.*}_coverage.${OUTPUT_PNG##*.}"
