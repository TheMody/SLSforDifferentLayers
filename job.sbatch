#!/bin/bash 
#SBATCH --gres=gpu:1
#SBATCH --time=120:00:00
#SBATCH --job-name=SLScifar10
#SBATCH --output=SLScifar10.log
#SBATCH --mail-user=pkenneweg@techfak.uni-bielefeld.de
#SBATCH --mail-type=END


eval "$(conda shell.bash hook)"
conda activate sls3

python run_multiple_img.py
