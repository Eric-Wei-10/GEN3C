#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

PY="${PYTHON_BIN:-}"
if [ -z "${PY}" ]; then
  if command -v python >/dev/null 2>&1; then
    PY="python"
  elif command -v python3 >/dev/null 2>&1; then
    PY="python3"
  else
    echo "ERROR: Neither 'python' nor 'python3' found. Activate your conda env."
    exit 1
  fi
fi

# Args:
# 1 run_mode: single|multi
# 2 frame_index
# 3 mode: frame_t|forward|backward|hybrid
# 4 output_dir
# 5 traj_dir (single) OR inputs_root (multi)
# 6 traj_mask: fill|intersection
# 7 channel_type: rgb|dino                  (optional; default rgb)
# 8 combine_policy (multi): priority|soft   (optional; default priority)
# 9 sigma_deg (multi): e.g. 15              (optional; default 15)

RUN_MODE="${1:-multi}"
FRAME_INDEX="${2:-96}"
MODE="${3:-backward}"
OUTPUT_DIR="${4:-$ROOT_DIR/outputs}"
ARG5="${5:-$ROOT_DIR/inputs}"
TRAJ_MASK="${6:-}"
CHANNEL_TYPE="${7:-rgb}"
COMBINE_POLICY="${8:-priority}"
SIGMA_DEG="${9:-15}"

if [ -z "${TRAJ_MASK}" ]; then
  echo "ERROR: traj_mask is required."
  echo ""
  echo "Single examples:"
  echo "  bash run.sh single 96 backward outputs ./inputs/result_-15_0 fill rgb"
  echo "  bash run.sh single 96 backward outputs ./inputs/result_-15_0 fill dino"
  echo ""
  echo "Multi examples:"
  echo "  bash run.sh multi 96 backward outputs ./inputs fill rgb priority 15"
  echo "  bash run.sh multi 96 backward outputs ./inputs fill dino priority 15"
  exit 1
fi

if [ "${TRAJ_MASK}" != "fill" ] && [ "${TRAJ_MASK}" != "intersection" ]; then
  echo "ERROR: traj_mask must be 'fill' or 'intersection', got: ${TRAJ_MASK}"
  exit 1
fi

if [ "${CHANNEL_TYPE}" != "rgb" ] && [ "${CHANNEL_TYPE}" != "dino" ]; then
  echo "ERROR: channel_type must be 'rgb' or 'dino', got: ${CHANNEL_TYPE}"
  exit 1
fi

mkdir -p "$OUTPUT_DIR"
timestamp="$(date +%Y%m%d_%H%M%S)"

if [ "$RUN_MODE" = "single" ]; then
  TRAJ_DIR="$ARG5"
  [ -d "$TRAJ_DIR" ] || { echo "ERROR: traj_dir '$TRAJ_DIR' does not exist."; exit 1; }
  out_base="single_$(basename "$TRAJ_DIR")_${MODE}_t${FRAME_INDEX}_${timestamp}"
  OUTPUT_NPZ="$OUTPUT_DIR/${out_base}.npz"
else
  INPUTS_ROOT="$ARG5"
  [ -d "$INPUTS_ROOT" ] || { echo "ERROR: inputs_root '$INPUTS_ROOT' does not exist."; exit 1; }
  out_base="combined_${COMBINE_POLICY}_${MODE}_t${FRAME_INDEX}_${timestamp}"
  OUTPUT_NPZ="$OUTPUT_DIR/${out_base}.npz"
fi

OUT_PREFIX="$OUTPUT_DIR/${out_base}"

echo "[INFO] Using:"
echo "  ROOT_DIR       = ${ROOT_DIR}"
echo "  PY             = ${PY}"
echo "  RUN_MODE       = ${RUN_MODE}"
echo "  frame_index    = ${FRAME_INDEX}"
echo "  mode           = ${MODE}"
echo "  traj_mask      = ${TRAJ_MASK}"
echo "  channel_type   = ${CHANNEL_TYPE}"
echo "  output_npz     = ${OUTPUT_NPZ}"
echo "  out_prefix     = ${OUT_PREFIX}"

if [ "$RUN_MODE" = "single" ]; then
  echo "[INFO] Running single-trajectory variance"
  "$PY" "$ROOT_DIR/main.py" \
    --traj_dir "$TRAJ_DIR" \
    --frame_index "$FRAME_INDEX" \
    --mode "$MODE" \
    --channel_type "$CHANNEL_TYPE" \
    --output_npz "$OUTPUT_NPZ"
else
  echo "[INFO] Running multi-trajectory variance"
  echo "  combine_policy = ${COMBINE_POLICY}"
  echo "  sigma_deg      = ${SIGMA_DEG}"
  "$PY" "$ROOT_DIR/main.py" \
    --inputs_root "$INPUTS_ROOT" \
    --frame_index "$FRAME_INDEX" \
    --mode "$MODE" \
    --traj_mask "$TRAJ_MASK" \
    --combine_policy "$COMBINE_POLICY" \
    --sigma_deg "$SIGMA_DEG" \
    --channel_type "$CHANNEL_TYPE" \
    --save_per_trajectory \
    --output_npz "$OUTPUT_NPZ"
fi

echo "[INFO] Visualizing (variance + coverage + source in one pass)"
"$PY" "$ROOT_DIR/visualize_variance.py" \
  --result_npz "$OUTPUT_NPZ" \
  --out_prefix "$OUT_PREFIX" \
  --traj_mask "$TRAJ_MASK"

echo "[INFO] Done."
echo "  NPZ:      $OUTPUT_NPZ"
echo "  Variance: ${OUT_PREFIX}_variance.png"
echo "  Coverage: ${OUT_PREFIX}_coverage.png"
echo "  Source:   ${OUT_PREFIX}_source.png (combined only)"
