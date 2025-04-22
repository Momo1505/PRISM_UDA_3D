#!/bin/bash

# +------------------------------------------------------------------------------------+ #
# |                                  SLURM PARAMETERS                                  | #
# +------------------------------------------------------------------------------------+ #

#SBATCH -p publicgpu
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --mail-user="mouhamed.sow@unistra.fr"
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH -o GTA_to_Cityscapes_res.out

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
echo 'START'

export CUDA_LAUNCH_BLOCKING=1

TEST_ROOT=work_dirs/local-basic/250419_1741_GTAtoCityscapes_21fb8
CONFIG_FILE="${TEST_ROOT}/*${TEST_ROOT: -1}.py"
CHECKPOINT_FILE="${TEST_ROOT}/latest.pth"
SHOW_DIR="${TEST_ROOT}/preds"

echo 'Config File:' $CONFIG_FILE
echo 'Checkpoint File:' $CHECKPOINT_FILE
echo 'Predictions Output Directory:' $SHOW_DIR
python -m tools.test ${CONFIG_FILE} ${CHECKPOINT_FILE} --eval mIoU --show-dir ${SHOW_DIR} --opacity 1
echo 'END'