#!/bin/bash 
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --job-name=SLSSGDmnist
#SBATCH --output=SLSSGDmnist.log



eval "$(conda shell.bash hook)"
conda activate sls3

python run_multiple_img.py
