#!/usr/bin/env bash
set -e

# ------------------------------------------------------------------
# Basic config
# ------------------------------------------------------------------
ROOT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Positional arguments with defaults:
#   $1 -> videos_dir
#   $2 -> camera_npz
#   $3 -> frame_index
#   $4 -> output_npz
#   $5 -> output_png
VIDEOS_DIR="${1:-videos}"
CAMERA_NPZ="${2:-camera_info/test_single_image_720_400_97_10_42_camera_data.npz}"
FRAME_INDEX="${3:-0}"
OUTPUT_NPZ="${4:-variance_t${FRAME_INDEX}.npz}"
OUTPUT_PNG="${5:-variance_t${FRAME_INDEX}.png}"

echo "[INFO] Using:"
echo "  videos_dir  = ${VIDEOS_DIR}"
echo "  camera_npz  = ${CAMERA_NPZ}"
echo "  frame_index = ${FRAME_INDEX}"
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
  --output_npz "$OUTPUT_NPZ"

# ------------------------------------------------------------------
# Run visualization
# ------------------------------------------------------------------
echo "[INFO] Running visualize_variance.py"
python "$ROOT_DIR/visualize_variance.py" \
  --result_npz "$OUTPUT_NPZ" \
  --save_path "$OUTPUT_PNG"

echo "[INFO] Done."
echo "  Result npz:   $OUTPUT_NPZ"
echo "  Result image: $OUTPUT_PNG"
