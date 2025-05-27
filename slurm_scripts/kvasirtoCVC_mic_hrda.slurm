#!/bin/bash
#SBATCH -p grantgpu -A g2024a219g
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --constraint="gpua100"
#SBATCH --mail-user="mouhamed.sow@unistra.fr"
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH -o jobs/kvasirtoCVC_mic_hrda.out
hostname
module load python/python-3.8.18
source ~/venv/prism-uda/bin/activate
cd ~/domain_adaptation
python --version
echo 'START'
python run_experiments.py --config configs/mic/kvasirtoCVC_mic_hrda.py
echo 'END'
