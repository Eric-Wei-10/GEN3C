#!/bin/bash -l
#SBATCH --job-name=sun1_40g
#SBATCH --output=job_%j.out
#SBATCH --error=job_%j.err
#SBATCH --mem-per-cpu=48g
#SBATCH --ntasks=1
#SBATCH --time=20:20:00
#SBATCH --gpus=a100-pcie-40gb:1


module load stack/2024-06
module load cuda/12.4
conda activate /cluster/project/cvg/students/shangwu/gen3c_env

cd /cluster/project/cvg/students/shangwu/GEN3C
export PYTHONPATH=/cluster/project/cvg/students/shangwu/GEN3C:$PYTHONPATH

bash generate_frontiers.sh
