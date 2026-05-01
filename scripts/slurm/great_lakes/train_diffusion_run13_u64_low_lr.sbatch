#!/bin/bash
#SBATCH --job-name=cosmodiff_run13_lr
#SBATCH --partition=spgpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48gb
#SBATCH --time=01:00:00
#SBATCH -o /home/jiamingp/Diffusion_model/logs/run13_u64_low_lr_train_%j.out
#SBATCH -e /home/jiamingp/Diffusion_model/logs/run13_u64_low_lr_train_%j.err
#SBATCH -A huterer0

set -euo pipefail

cd /home/jiamingp/Diffusion_model

export PYTHONPATH=/home/jiamingp/Diffusion_model/cosmo_diffusion:${PYTHONPATH:-}

mkdir -p /home/jiamingp/Diffusion_model/logs
mkdir -p /scratch/huterer_root/huterer0/jiamingp/saved_runs/run13_u64_low_lr_checkpoints

python /home/jiamingp/Diffusion_model/cosmo_diffusion/scripts/cosmodiff_train.py --config /home/jiamingp/Diffusion_model/configs/great_lakes/config_run13_u64_low_lr.yaml
