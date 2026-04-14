#!/bin/bash -l

#SBATCH --job-name=random_pose
#SBATCH --output=job_%j.out
#SBATCH --error=job_%j.err
#SBATCH --mem-per-cpu=16g
#SBATCH --ntasks=1
#SBATCH --time=24:00:00
#SBATCH --gpus=rtx_4090:1
#SBATCH --account=ls_polle

module load eth_proxy

source ~/miniconda3/etc/profile.d/conda.sh
conda activate /cluster/project/cvg/students/shangwu/gen3c_env

cd /cluster/project/cvg/students/shangwu/GEN3C/preprocess

python3 random_pose.py \
    --n 100 \
    --seed 42 \
    --glb ../../FrontierNet/eval_data/mesh/000000-kfPV7w3FaU5.glb \
    --output ../dummy_render \
    --filter \
    --device cuda