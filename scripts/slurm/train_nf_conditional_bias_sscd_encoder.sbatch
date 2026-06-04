#!/bin/bash
#SBATCH --job-name=nf_bias_sscdenc
#SBATCH --partition=spgpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=80gb
#SBATCH --time=04:00:00
#SBATCH --gres=gpu:1
#SBATCH -o /home/jiamingp/diffusion_models_repo/logs/nf_conditional_bias_probe/sscd_encoder_%j.out
#SBATCH -e /home/jiamingp/diffusion_models_repo/logs/nf_conditional_bias_probe/sscd_encoder_%j.err
#SBATCH -A huterer2

set -euo pipefail

PROJECT_DIR=${PROJECT_DIR:-/home/jiamingp/diffusion_models_repo}
VENV_PATH=${VENV_PATH:-/home/jiamingp/venvs/cosmodiff_nf}
PYTHON_BIN=${PYTHON_BIN:-python}
SSCD_PATH=${SSCD_PATH:-${HOME}/.cache/torch/hub/sscd_disc_mixup.torchscript.pt}
SSCD_DEVICE=${SSCD_DEVICE:-auto}
SSCD_IMAGE_SIZE=${SSCD_IMAGE_SIZE:-320}
SSCD_RENDER_MODE=${SSCD_RENDER_MODE:-fixed}
SSCD_BATCH_SIZE=${SSCD_BATCH_SIZE:-16}
HEAD_TRAIN_SLICES=${HEAD_TRAIN_SLICES:-16384}
HEAD_VAL_SLICES=${HEAD_VAL_SLICES:-4096}
RIDGE_ALPHA=${RIDGE_ALPHA:-10.0}

cd "${PROJECT_DIR}"
mkdir -p logs/nf_conditional_bias_probe results/cache/matplotlib
if [[ -f "${VENV_PATH}/bin/activate" ]]; then
  source "${VENV_PATH}/bin/activate"
fi

export PYTHONPATH="${PROJECT_DIR}:${PYTHONPATH:-}"
export MPLCONFIGDIR="${PROJECT_DIR}/results/cache/matplotlib"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"

"${PYTHON_BIN}" scripts/train_nf_conditional_bias_sscd_encoder.py \
  --project-dir "${PROJECT_DIR}" \
  --sscd-path "${SSCD_PATH}" \
  --device "${SSCD_DEVICE}" \
  --image-size "${SSCD_IMAGE_SIZE}" \
  --render-mode "${SSCD_RENDER_MODE}" \
  --embedding-batch-size "${SSCD_BATCH_SIZE}" \
  --head-train-slices "${HEAD_TRAIN_SLICES}" \
  --head-val-slices "${HEAD_VAL_SLICES}" \
  --ridge-alpha "${RIDGE_ALPHA}"
