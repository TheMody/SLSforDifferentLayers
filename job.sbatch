#!/bin/bash 
#SBATCH --gres=gpu:1
#SBATCH --time=96:00:00
#SBATCH --job-name=SLSlong
#SBATCH --output=SLSlong.log



eval "$(conda shell.bash hook)"
conda activate sls3

python run_multiple_img.py
