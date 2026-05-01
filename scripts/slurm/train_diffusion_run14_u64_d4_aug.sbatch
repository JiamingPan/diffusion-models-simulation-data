#!/bin/bash
#SBATCH --job-name=cosmodiff_run14_d4
#SBATCH --partition=spgpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48gb
#SBATCH --time=01:00:00
#SBATCH -o /home/jiamingp/Diffusion_model/logs/run14_u64_d4_aug_train_%j.out
#SBATCH -e /home/jiamingp/Diffusion_model/logs/run14_u64_d4_aug_train_%j.err
#SBATCH -A huterer0

set -euo pipefail

cd /home/jiamingp/Diffusion_model

export PYTHONPATH=/home/jiamingp/Diffusion_model/cosmo_diffusion:${PYTHONPATH:-}

mkdir -p /home/jiamingp/Diffusion_model/logs
mkdir -p /scratch/huterer_root/huterer0/jiamingp/saved_runs/run14_u64_d4_aug_checkpoints

python /home/jiamingp/Diffusion_model/cosmo_diffusion/scripts/cosmodiff_train.py --config /home/jiamingp/Diffusion_model/configs_saved/config_run14_u64_d4_aug.yaml
