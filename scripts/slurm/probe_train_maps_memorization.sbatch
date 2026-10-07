#!/bin/bash
#SBATCH --job-name=probe_train_memo
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32gb
#SBATCH --time=01:00:00
#SBATCH -o /home/jiamingp/diffusion_models_repo/logs/probe_train_memo-%j.out
#SBATCH -e /home/jiamingp/diffusion_models_repo/logs/probe_train_memo-%j.err
#SBATCH -A huterer2
# Frozen VGG probe on the N = 64/256/1024 training maps (memorization-regime diagnostic). CPU only.
set -euo pipefail
PROJECT_DIR=/home/jiamingp/diffusion_models_repo
cd "${PROJECT_DIR}"
source /home/jiamingp/venvs/cosmodiff_nf_class/bin/activate
export PYTHONNOUSERSITE=1
export PYTHONPATH="${PROJECT_DIR}:${PROJECT_DIR}/scripts"
export MPLCONFIGDIR="${PROJECT_DIR}/results/cache/matplotlib"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}" MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
OUT_DIR=${PROJECT_DIR}/results/memorization_regime_diagnostics/probe_on_train_maps_$(date +%Y%m%d_%H%M%S)
python scripts/probe_train_maps_memorization.py --out-dir "${OUT_DIR}" --device cpu
