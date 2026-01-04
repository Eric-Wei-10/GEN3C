#!/bin/bash

WIDTH=720
HEIGHT=544
NUM_STEPS=10
NUM_SEEDS=4
FPS=24
NUM_FRAMES=121 # 24 * 5 + 1
AGENT_SPEED=0.5 # m/s, not faster than 0.5 
VIDEO_LENGTH=$(( (${NUM_FRAMES} - 1) / ${FPS} ))
INPUT_IMAGE=/cluster/project/cvg/students/shangwu/GEN3C/assets/diffusion/dataset_all/dataset_shard_2

echo "Starting video generation with parameters:"
echo "  NUM_SEEDS: ${NUM_SEEDS}"
echo "  WIDTH: ${WIDTH}"
echo "  HEIGHT: ${HEIGHT}"
echo "  NUM_FRAMES: ${NUM_FRAMES}"
echo "  NUM_STEPS: ${NUM_STEPS}"
echo "  VIDEO_LENGTH: ${VIDEO_LENGTH} seconds"


OUT_DIR=outputs/dataset_all
mkdir -p "${OUT_DIR}"
source ~/miniconda3/etc/profile.d/conda.sh
conda activate /cluster/project/cvg/students/shangwu/gen3c_env

CUDA_HOME=$CONDA_PREFIX PYTHONPATH=$(pwd) python cosmos_predict1/diffusion/inference/gen3c_frontier.py \
    --checkpoint_dir checkpoints \
    --input_image ${INPUT_IMAGE} \
    --height ${HEIGHT} \
    --width ${WIDTH} \
    --fps ${FPS} \
    --num_video_frames ${NUM_FRAMES} \
    --num_steps ${NUM_STEPS} \
    --video_save_folder "${OUT_DIR}" \
    --agent_speed ${AGENT_SPEED} \
    --sample_angle 15 \
    --num_seeds ${NUM_SEEDS} \

# uncomment to enable model offloading options
# not necessary for vram > 40GB (a100)

# --offload_diffusion_transformer \
# --offload_tokenizer \
# --offload_text_encoder_model \
# --offload_prompt_upsampler \
# --offload_guardrail_models \
# --disable_guardrail \
# --disable_prompt_encoder
