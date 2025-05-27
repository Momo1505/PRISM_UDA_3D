#!/bin/bash

# +------------------------------------------------------------------------------------+ #
# |                                  SLURM PARAMETERS                                  | #
# +------------------------------------------------------------------------------------+ #

#SBATCH -p grantgpu -A g2024a219g
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --constraint="gpua100"
#SBATCH --mail-user="mouhamed.sow@unistra.fr"
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH -o jobs/LW4toI3_mic_hrda.out

# +------------------------------------------------------------------------------------+ #
# |                                ENVIRONNEMENT SET UP                                | #
# +------------------------------------------------------------------------------------+ #

module load python/python-3.8.18
module load cuda/cuda-11.2
source ~/venv/prism-uda/bin/activate
python --version

cd ~/domain_adaptation

# +------------------------------------------------------------------------------------+ #
# |                                 RUN PYTHON SCRIPT                                  | #
# +------------------------------------------------------------------------------------+ #

hostname=$(hostname)
gpu_name=$(nvidia-smi --query-gpu=name --format=csv,noheader --id=0)
vram_total=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader --id=0)

printf "
================================== HPC NODE ===================================
Hostname.........: $hostname
GPU name.........: $gpu_name
GPU VRAM.........: $vram_total
===============================================================================

"

export CUDA_LAUNCH_BLOCKING=1

echo 'START'
python run_experiments.py --config configs/mic/LW4toI3_mic_hrda.py
echo 'END'


