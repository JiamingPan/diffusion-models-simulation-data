#!/bin/bash
#SBATCH --job-name=nf_gen_pca_nn
#SBATCH --partition=spgpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=160gb
#SBATCH --time=06:00:00
#SBATCH -o /home/jiamingp/diffusion_models_repo/logs/nf_generalize_nick_data/pca_full_nn_%j.out
#SBATCH -e /home/jiamingp/diffusion_models_repo/logs/nf_generalize_nick_data/pca_full_nn_%j.err
#SBATCH -A huterer0

set -euo pipefail

PROJECT_DIR=${PROJECT_DIR:-/home/jiamingp/diffusion_models_repo}
VENV_PATH=${VENV_PATH:-/home/jiamingp/venvs/cosmodiff_nf}
PYTHON_BIN=${PYTHON_BIN:-python}

PCA_COMPONENTS=${PCA_COMPONENTS:-32}
PCA_FIT_MAX_SLICES=${PCA_FIT_MAX_SLICES:-4096}
MAX_GENERATED=${MAX_GENERATED:-512}
VAL_RAW_PER_SOURCE=${VAL_RAW_PER_SOURCE:-8}
MAX_VAL_SLICES=${MAX_VAL_SLICES:-512}
EMBEDDING_BATCH_SIZE=${EMBEDDING_BATCH_SIZE:-512}
SIMILARITY_BATCH_SIZE=${SIMILARITY_BATCH_SIZE:-512}

cd "${PROJECT_DIR}"
mkdir -p logs/nf_generalize_nick_data results/cache/matplotlib
if [[ -f "${VENV_PATH}/bin/activate" ]]; then
  source "${VENV_PATH}/bin/activate"
fi

export PYTHONPATH="${PROJECT_DIR}:${PYTHONPATH:-}"
export MPLCONFIGDIR="${PROJECT_DIR}/results/cache/matplotlib"

"${PYTHON_BIN}" scripts/compute_nf_generalize_pca_full_nn.py \
  --project-dir "${PROJECT_DIR}" \
  --pca-components "${PCA_COMPONENTS}" \
  --pca-fit-max-slices "${PCA_FIT_MAX_SLICES}" \
  --max-generated "${MAX_GENERATED}" \
  --val-raw-per-source "${VAL_RAW_PER_SOURCE}" \
  --max-val-slices "${MAX_VAL_SLICES}" \
  --embedding-batch-size "${EMBEDDING_BATCH_SIZE}" \
  --similarity-batch-size "${SIMILARITY_BATCH_SIZE}"
