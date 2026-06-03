#!/bin/bash
#SBATCH --job-name=nf_bias_encoder
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120gb
#SBATCH --time=04:00:00
#SBATCH -o /home/jiamingp/diffusion_models_repo/logs/nf_conditional_bias_probe/encoder_%j.out
#SBATCH -e /home/jiamingp/diffusion_models_repo/logs/nf_conditional_bias_probe/encoder_%j.err
#SBATCH -A huterer2

set -euo pipefail

PROJECT_DIR=${PROJECT_DIR:-/home/jiamingp/diffusion_models_repo}
VENV_PATH=${VENV_PATH:-/home/jiamingp/venvs/cosmodiff_nf}
PYTHON_BIN=${PYTHON_BIN:-python}
PCA_COMPONENTS=${PCA_COMPONENTS:-8192}
PCA_TARGET_VARIANCE=${PCA_TARGET_VARIANCE:-0.98}
PCA_FIT_SLICES=${PCA_FIT_SLICES:-16384}
HEAD_TRAIN_SLICES=${HEAD_TRAIN_SLICES:-16384}
HEAD_VAL_SLICES=${HEAD_VAL_SLICES:-4096}
RIDGE_ALPHA=${RIDGE_ALPHA:-10.0}
REUSE_EXISTING_PCA=${REUSE_EXISTING_PCA:-0}

cd "${PROJECT_DIR}"
mkdir -p logs/nf_conditional_bias_probe results/cache/matplotlib
if [[ -f "${VENV_PATH}/bin/activate" ]]; then
  source "${VENV_PATH}/bin/activate"
fi

export PYTHONPATH="${PROJECT_DIR}:${PYTHONPATH:-}"
export MPLCONFIGDIR="${PROJECT_DIR}/results/cache/matplotlib"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

reuse_args=()
if [[ "${REUSE_EXISTING_PCA}" == "1" ]]; then
  reuse_args+=(--reuse-existing-pca)
fi

"${PYTHON_BIN}" scripts/train_nf_conditional_bias_encoder.py \
  --project-dir "${PROJECT_DIR}" \
  --pca-components "${PCA_COMPONENTS}" \
  --pca-target-variance "${PCA_TARGET_VARIANCE}" \
  --pca-fit-slices "${PCA_FIT_SLICES}" \
  --head-train-slices "${HEAD_TRAIN_SLICES}" \
  --head-val-slices "${HEAD_VAL_SLICES}" \
  --ridge-alpha "${RIDGE_ALPHA}" \
  "${reuse_args[@]}"
