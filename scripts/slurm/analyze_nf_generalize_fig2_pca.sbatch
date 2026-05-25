#!/bin/bash
#SBATCH --job-name=nf_fig2_pca
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80gb
#SBATCH --time=02:00:00
#SBATCH -o /home/jiamingp/diffusion_models_repo/logs/nf_generalize_fig2/pca_full_nn_%j.out
#SBATCH -e /home/jiamingp/diffusion_models_repo/logs/nf_generalize_fig2/pca_full_nn_%j.err
#SBATCH -A huterer0

set -euo pipefail

PROJECT_DIR=${PROJECT_DIR:-/home/jiamingp/diffusion_models_repo}
VENV_PATH=${VENV_PATH:-/home/jiamingp/venvs/cosmodiff_nf}
PYTHON_BIN=${PYTHON_BIN:-python}

PCA_COMPONENTS=${PCA_COMPONENTS:-32}
PCA_FIT_MAX_SLICES=${PCA_FIT_MAX_SLICES:-4096}
PCA_TARGET_VARIANCE=${PCA_TARGET_VARIANCE:-0}
MAX_GENERATED=${MAX_GENERATED:-512}
VAL_RAW_PER_SOURCE=${VAL_RAW_PER_SOURCE:-8}
MAX_VAL_SLICES=${MAX_VAL_SLICES:-512}
SIMILARITY_BATCH_SIZE=${SIMILARITY_BATCH_SIZE:-512}
FIXED_THRESHOLDS=${FIXED_THRESHOLDS:-0.5,0.6,0.7,0.8,0.9,0.95}
OUT_PREFIX=${OUT_PREFIX:-nf_generalize_fig2_pca_full_nn}

cd "${PROJECT_DIR}"
mkdir -p logs/nf_generalize_fig2 results/cache/matplotlib
if [[ -f "${VENV_PATH}/bin/activate" ]]; then
  source "${VENV_PATH}/bin/activate"
fi

export PYTHONPATH="${PROJECT_DIR}:${PYTHONPATH:-}"
export MPLCONFIGDIR="${PROJECT_DIR}/results/cache/matplotlib"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

"${PYTHON_BIN}" scripts/compute_nf_generalize_pca_full_nn.py \
  --project-dir "${PROJECT_DIR}" \
  --manifest "${PROJECT_DIR}/local/nf_generalize_fig2/manifest.json" \
  --output-dir "${PROJECT_DIR}/results/nf_generalize_fig2/quickcheck" \
  --table-dir "${PROJECT_DIR}/results/nf_generalize_fig2/tables" \
  --out-prefix "${OUT_PREFIX}" \
  --pca-components "${PCA_COMPONENTS}" \
  --pca-fit-max-slices "${PCA_FIT_MAX_SLICES}" \
  --pca-target-variance "${PCA_TARGET_VARIANCE}" \
  --max-generated "${MAX_GENERATED}" \
  --val-raw-per-source "${VAL_RAW_PER_SOURCE}" \
  --max-val-slices "${MAX_VAL_SLICES}" \
  --similarity-batch-size "${SIMILARITY_BATCH_SIZE}" \
  --fixed-similarity-thresholds "${FIXED_THRESHOLDS}" \
  --skip-missing-samples
