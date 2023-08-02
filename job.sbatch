#!/bin/bash 
#SBATCH --gres=gpu:1
#SBATCH --time=96:00:00
#SBATCH --job-name=SLSc
#SBATCH --output=SLSc.log



eval "$(conda shell.bash hook)"
conda activate sls3

python run_multiple.py
