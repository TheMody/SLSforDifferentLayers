#!/bin/bash 
#SBATCH --gres=gpu:1
#SBATCH --time=120:00:00
#SBATCH --job-name=SLSiamgenetadam
#SBATCH --output=SLSimagenetadam.log
#SBATCH --mail-user=pkenneweg@techfak.de


eval "$(conda shell.bash hook)"
conda activate sls3

python run_multiple_img.py
