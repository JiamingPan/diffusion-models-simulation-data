#!/bin/bash
#SBATCH --job-name=cosmodiff_dit
#SBATCH --partition=spgpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64gb
#SBATCH --time=04:00:00
#SBATCH -o /home/jiamingp/Diffusion_model/logs/dit_train_%j.out
#SBATCH -e /home/jiamingp/Diffusion_model/logs/dit_train_%j.err
#SBATCH -A huterer0

set -euo pipefail

PROJECT_DIR=${PROJECT_DIR:-/home/jiamingp/Diffusion_model}
CONFIG_PATH=${CONFIG_PATH:-${PROJECT_DIR}/configs_saved/config_dit_small.yaml}

cd "${PROJECT_DIR}"

export PYTHONPATH="${PROJECT_DIR}/cosmo_diffusion:${PYTHONPATH:-}"

mkdir -p "${PROJECT_DIR}/logs"

python "${PROJECT_DIR}/cosmo_diffusion/scripts/cosmodiff_train.py" --config "${CONFIG_PATH}"
