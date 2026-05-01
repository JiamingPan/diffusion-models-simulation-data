#!/bin/bash
#SBATCH --job-name=cosmodiff_train
#SBATCH --partition=<partition>
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64gb
#SBATCH --time=08:00:00
#SBATCH -o logs/%x_%j.out
#SBATCH -e logs/%x_%j.err
#SBATCH -A <account>

set -euo pipefail

PROJECT_DIR=/path/to/diffusion-models-simulation-data
COSMODIFF_DIR=/path/to/cosmo_diffusion
CONFIG_PATH=${PROJECT_DIR}/configs/templates/u64_lh_template.yaml

cd "${PROJECT_DIR}"

export PYTHONPATH="${COSMODIFF_DIR}:${PROJECT_DIR}:${PYTHONPATH:-}"

mkdir -p logs

python "${COSMODIFF_DIR}/scripts/cosmodiff_train.py" --config "${CONFIG_PATH}"
