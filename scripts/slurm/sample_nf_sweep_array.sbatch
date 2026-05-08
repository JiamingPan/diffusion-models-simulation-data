#!/bin/bash
#SBATCH --job-name=nf_sweep_sample
#SBATCH --partition=spgpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=80gb
#SBATCH --time=08:00:00
#SBATCH --array=0-95%4
#SBATCH -o /home/jiamingp/diffusion_models_repo/logs/nf_sweep/sample_%A_%a.out
#SBATCH -e /home/jiamingp/diffusion_models_repo/logs/nf_sweep/sample_%A_%a.err
#SBATCH -A huterer0

set -euo pipefail

PROJECT_DIR=${PROJECT_DIR:-/home/jiamingp/diffusion_models_repo}
INHERITED_COSMODIFF_DIR=${COSMODIFF_DIR:-}
PREFERRED_COSMODIFF_DIR=/home/jiamingp/Diffusion_model/cosmo_diffusion_normalization_fixes_git
LEGACY_COSMODIFF_DIR=/home/jiamingp/Diffusion_model/cosmo_diffusion_normalization_fixes
if [[ -n "${COSMODIFF_DIR_OVERRIDE:-}" ]]; then
  COSMODIFF_DIR=${COSMODIFF_DIR_OVERRIDE}
elif [[ -f "${PREFERRED_COSMODIFF_DIR}/scripts/cosmodiff_sample.py" ]]; then
  COSMODIFF_DIR=${PREFERRED_COSMODIFF_DIR}
else
  COSMODIFF_DIR=${LEGACY_COSMODIFF_DIR}
fi
if [[ -n "${INHERITED_COSMODIFF_DIR}" && -z "${COSMODIFF_DIR_OVERRIDE:-}" && "${INHERITED_COSMODIFF_DIR}" != "${COSMODIFF_DIR}" ]]; then
  echo "Ignoring inherited COSMODIFF_DIR=${INHERITED_COSMODIFF_DIR}; using ${COSMODIFF_DIR}" >&2
fi
VENV_PATH=${VENV_PATH:-/home/jiamingp/venvs/cosmodiff_nf}
CHECKPOINT_ROOT=${CHECKPOINT_ROOT:-/scratch/huterer_root/huterer0/jiamingp/saved_runs/nf_sweep}
SAMPLE_ROOT=${SAMPLE_ROOT:-${PROJECT_DIR}/results/nf_sweep/samples}
PYTHON_BIN=${PYTHON_BIN:-python}
NUM_SAMPLES=${NUM_SAMPLES:-64}
BATCH_SIZE=${BATCH_SIZE:-8}
SEED=${SEED:-123}
OVERWRITE=${OVERWRITE:-0}
DDIM_THINNING=${DDIM_THINNING:-}
REQUIRED_COSMODIFF_BRANCH=${REQUIRED_COSMODIFF_BRANCH:-normalization_fixes}
ALLOW_COSMODIFF_BRANCH_MISMATCH=${ALLOW_COSMODIFF_BRANCH_MISMATCH:-0}

RUNS=(
  nf_sweep_u64_n500_e100_base_ema
  nf_sweep_u64_n500_e100_zero_snr
  nf_sweep_u64_n500_e100_beta_linear
  nf_sweep_u64_n500_e100_beta_cosine
  nf_sweep_u64_n500_e100_pred_eps
  nf_sweep_u64_n500_e100_sigma_edm
  nf_sweep_u64_n500_e100_sigma_edm2
  nf_sweep_u64_n500_e100_minsnr5
  nf_sweep_u128_n500_e100_base_ema
  nf_sweep_u128_n500_e100_zero_snr
  nf_sweep_u128_n500_e100_beta_linear
  nf_sweep_u128_n500_e100_beta_cosine
  nf_sweep_u128_n500_e100_pred_eps
  nf_sweep_u128_n500_e100_sigma_edm
  nf_sweep_u128_n500_e100_sigma_edm2
  nf_sweep_u128_n500_e100_minsnr5
)

EMA_TARGETS=(raw 0.03 0.05 0.08 0.13 0.20)
EMA_LABELS=(raw ema0p03 ema0p05 ema0p08 ema0p13 ema0p20)

N_EMA=${#EMA_TARGETS[@]}
RUN_INDEX=$((SLURM_ARRAY_TASK_ID / N_EMA))
EMA_INDEX=$((SLURM_ARRAY_TASK_ID % N_EMA))

RUN_NAME=${RUNS[$RUN_INDEX]}
EMA_TARGET=${EMA_TARGETS[$EMA_INDEX]}
EMA_LABEL=${EMA_LABELS[$EMA_INDEX]}
CHECKPOINT_DIR=${CHECKPOINT_ROOT}/${RUN_NAME}_checkpoints
OUTPUT_PATH=${SAMPLE_ROOT}/${RUN_NAME}_seed${SEED}_${EMA_LABEL}.npy

cd "${PROJECT_DIR}"
mkdir -p logs/nf_sweep "${SAMPLE_ROOT}"
if [[ -f "${VENV_PATH}/bin/activate" ]]; then
  source "${VENV_PATH}/bin/activate"
fi
export PYTHONPATH="${COSMODIFF_DIR}:${PROJECT_DIR}:${PYTHONPATH:-}"
export COSMODIFF_DIR

if [[ ! -d "${CHECKPOINT_DIR}" ]]; then
  echo "Missing checkpoint directory: ${CHECKPOINT_DIR}" >&2
  exit 1
fi

if [[ ! -f "${COSMODIFF_DIR}/scripts/cosmodiff_sample.py" ]]; then
  echo "Missing normalization-fix cosmo_diffusion sampler: ${COSMODIFF_DIR}" >&2
  exit 1
fi

if [[ -d "${COSMODIFF_DIR}/.git" ]]; then
  COSMODIFF_BRANCH=$(git -C "${COSMODIFF_DIR}" rev-parse --abbrev-ref HEAD)
  COSMODIFF_COMMIT=$(git -C "${COSMODIFF_DIR}" rev-parse --short HEAD)
  echo "cosmodiff git branch: ${COSMODIFF_BRANCH}"
  echo "cosmodiff git commit: ${COSMODIFF_COMMIT}"
  if [[ -n "${REQUIRED_COSMODIFF_BRANCH}" && "${COSMODIFF_BRANCH}" != "${REQUIRED_COSMODIFF_BRANCH}" && "${ALLOW_COSMODIFF_BRANCH_MISMATCH}" != "1" ]]; then
    echo "Wrong cosmo_diffusion branch: expected ${REQUIRED_COSMODIFF_BRANCH}, got ${COSMODIFF_BRANCH}" >&2
    echo "Set ALLOW_COSMODIFF_BRANCH_MISMATCH=1 only if you intentionally use a detached/test checkout." >&2
    exit 1
  fi
else
  echo "Cannot verify cosmo_diffusion git branch because ${COSMODIFF_DIR}/.git is missing." >&2
  echo "Continuing with Python import and feature checks for this non-git checkout." >&2
fi

"${PYTHON_BIN}" - <<'PY'
import inspect
import os
from pathlib import Path
from cosmodiff import optim

expected_root = Path(os.environ["COSMODIFF_DIR"]).resolve()
optim_path = Path(inspect.getsourcefile(optim)).resolve()
if expected_root not in optim_path.parents:
    raise SystemExit(
        f"Imported cosmodiff from {optim_path}, expected it under {expected_root}"
    )
print("cosmodiff sampling support:", optim_path)
PY

if [[ -f "${OUTPUT_PATH}" && "${OVERWRITE}" != "1" ]]; then
  echo "Skipping existing sample file: ${OUTPUT_PATH}"
  exit 0
fi

echo "Sampling ${RUN_NAME}"
echo "  checkpoint root: ${CHECKPOINT_DIR}"
echo "  output:          ${OUTPUT_PATH}"
echo "  samples:         ${NUM_SAMPLES}"
echo "  batch:           ${BATCH_SIZE}"
echo "  ema target:      ${EMA_TARGET}"
echo "  ddim thinning:   ${DDIM_THINNING:-none}"
echo "  cosmodiff:       ${COSMODIFF_DIR}"

CMD=(
  "${PYTHON_BIN}" "${COSMODIFF_DIR}/scripts/cosmodiff_sample.py"
  --output_dir "${CHECKPOINT_DIR}"
  --output "${OUTPUT_PATH}"
  --n_samples "${NUM_SAMPLES}"
  --batch_size "${BATCH_SIZE}"
  --image_shape 1 128 128
  --seed "${SEED}"
  --device cuda
  --verbose
)

if [[ "${EMA_TARGET}" != "raw" ]]; then
  CMD+=(--ema_sigma_rel "${EMA_TARGET}")
fi

if [[ -n "${DDIM_THINNING}" ]]; then
  CMD+=(--ddim_thinning "${DDIM_THINNING}")
fi

"${CMD[@]}"
