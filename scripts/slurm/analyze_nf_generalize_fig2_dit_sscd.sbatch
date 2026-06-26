#!/bin/bash
#SBATCH --job-name=nf_dit_sscd
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80gb
#SBATCH --time=04:00:00
#SBATCH -o /home/jiamingp/diffusion_models_repo/logs/nf_generalize_fig2_dit/sscd_full_nn_%j.out
#SBATCH -e /home/jiamingp/diffusion_models_repo/logs/nf_generalize_fig2_dit/sscd_full_nn_%j.err
#SBATCH -A huterer0

set -euo pipefail

PROJECT_DIR=${PROJECT_DIR:-/home/jiamingp/diffusion_models_repo}
VENV_PATH=${VENV_PATH:-/home/jiamingp/venvs/cosmodiff_nf}
PYTHON_BIN=${PYTHON_BIN:-python}

SSCD_PATH=${SSCD_PATH:-/home/jiamingp/.cache/torch/hub/sscd_disc_mixup.torchscript.pt}
SSCD_DEVICE=${SSCD_DEVICE:-cpu}
SSCD_IMAGE_SIZE=${SSCD_IMAGE_SIZE:-320}
SSCD_BATCH_SIZE=${SSCD_BATCH_SIZE:-16}
SIMILARITY_BATCH_SIZE=${SIMILARITY_BATCH_SIZE:-512}
MAX_GENERATED=${MAX_GENERATED:-512}
VAL_RAW_PER_SOURCE=${VAL_RAW_PER_SOURCE:-8}
MAX_VAL_SLICES=${MAX_VAL_SLICES:-512}
SSCD_THRESHOLD=${SSCD_THRESHOLD:-0.6}
FIXED_THRESHOLDS=${FIXED_THRESHOLDS:-0.5,0.6,0.7,0.8,0.9,0.95}
SAMPLE_LABEL=${SAMPLE_LABEL:-dpm50}
OUT_PREFIX=${OUT_PREFIX:-nf_generalize_fig2_dit_sscd_full_nn}

cd "${PROJECT_DIR}"
mkdir -p logs/nf_generalize_fig2_dit results/cache/matplotlib
if [[ -f "${VENV_PATH}/bin/activate" ]]; then
  source "${VENV_PATH}/bin/activate"
fi

export PYTHONPATH="${PROJECT_DIR}:${PYTHONPATH:-}"
export MPLCONFIGDIR="${PROJECT_DIR}/results/cache/matplotlib"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

"${PYTHON_BIN}" scripts/compute_nf_generalize_sscd_full_nn.py \
  --project-dir "${PROJECT_DIR}" \
  --manifest "${PROJECT_DIR}/local/nf_generalize_fig2_dit/manifest.json" \
  --output-dir "${PROJECT_DIR}/results/nf_generalize_fig2_dit/quickcheck" \
  --table-dir "${PROJECT_DIR}/results/nf_generalize_fig2_dit/tables" \
  --cache-dir "${PROJECT_DIR}/results/nf_generalize_fig2_dit/cache/sscd_full_nn" \
  --out-prefix "${OUT_PREFIX}" \
  --sample-label "${SAMPLE_LABEL}" \
  --sscd-path "${SSCD_PATH}" \
  --device "${SSCD_DEVICE}" \
  --image-size "${SSCD_IMAGE_SIZE}" \
  --embedding-batch-size "${SSCD_BATCH_SIZE}" \
  --similarity-batch-size "${SIMILARITY_BATCH_SIZE}" \
  --max-generated "${MAX_GENERATED}" \
  --val-raw-per-source "${VAL_RAW_PER_SOURCE}" \
  --max-val-slices "${MAX_VAL_SLICES}" \
  --threshold "${SSCD_THRESHOLD}" \
  --fixed-similarity-thresholds "${FIXED_THRESHOLDS}" \
  --skip-missing-samples
