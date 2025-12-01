#!/bin/bash

WIDTH=848
HEIGHT=480
NUM_FRAMES=121
NUM_STEPS=15

VIDEO_SAVE_NAME="test_single_image_${WIDTH}_${HEIGHT}_${NUM_FRAMES}_${NUM_STEPS}"

CUDA_HOME=$CONDA_PREFIX PYTHONPATH=$(pwd) python cosmos_predict1/diffusion/inference/gen3c_single_image.py \
--checkpoint_dir checkpoints \
--input_image_path assets/diffusion/000000.png \
--video_save_name ${VIDEO_SAVE_NAME} \
--guidance 1 \
--foreground_masking \
--width ${WIDTH} \
--height ${HEIGHT} \
--num_video_frames ${NUM_FRAMES} \
--num_steps ${NUM_STEPS}
# --offload_diffusion_transformer \
# --offload_tokenizer \
# --offload_text_encoder_model \
# --offload_prompt_upsampler \
# --offload_guardrail_models \
# --disable_guardrail \
# --disable_prompt_encoder
