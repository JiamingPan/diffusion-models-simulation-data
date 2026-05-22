#!/bin/bash
#SBATCH --job-name=nf_cond_sample
#SBATCH --partition=spgpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=80gb
#SBATCH --time=04:00:00
#SBATCH -o /home/jiamingp/diffusion_models_repo/logs/nf_conditional_u128/sample_%j.out
#SBATCH -e /home/jiamingp/diffusion_models_repo/logs/nf_conditional_u128/sample_%j.err
#SBATCH -A huterer0

set -euo pipefail

PROJECT_DIR=${PROJECT_DIR:-/home/jiamingp/diffusion_models_repo}
INHERITED_COSMODIFF_DIR=${COSMODIFF_DIR:-}
PREFERRED_COSMODIFF_DIR=/home/jiamingp/Diffusion_model/cosmo_diffusion_main
if [[ -n "${COSMODIFF_DIR_OVERRIDE:-}" ]]; then
  COSMODIFF_DIR=${COSMODIFF_DIR_OVERRIDE}
else
  COSMODIFF_DIR=${PREFERRED_COSMODIFF_DIR}
fi
if [[ -n "${INHERITED_COSMODIFF_DIR}" && -z "${COSMODIFF_DIR_OVERRIDE:-}" && "${INHERITED_COSMODIFF_DIR}" != "${COSMODIFF_DIR}" ]]; then
  echo "Ignoring inherited COSMODIFF_DIR=${INHERITED_COSMODIFF_DIR}; using ${COSMODIFF_DIR}" >&2
fi
VENV_PATH=${VENV_PATH:-/home/jiamingp/venvs/cosmodiff_nf}
PYTHON_BIN=${PYTHON_BIN:-python}
SEED=${SEED:-123}
NUM_SAMPLES=${NUM_SAMPLES:-512}
BATCH_SIZE=${BATCH_SIZE:-8}
OVERWRITE=${OVERWRITE:-0}
REQUIRED_COSMODIFF_BRANCH=${REQUIRED_COSMODIFF_BRANCH:-main}

cd "${PROJECT_DIR}"
mkdir -p logs/nf_conditional_u128 results/nf_conditional_u128/samples
if [[ -f "${VENV_PATH}/bin/activate" ]]; then
  source "${VENV_PATH}/bin/activate"
fi

RUN_NAME=$("${PYTHON_BIN}" scripts/prepare_nf_conditional_u128_config.py --project-dir "${PROJECT_DIR}" --print-runs)
CONFIG_PATH="${PROJECT_DIR}/local/nf_conditional_u128/configs/${RUN_NAME}.yaml"
CHECKPOINT_DIR="/scratch/huterer_root/huterer0/jiamingp/saved_runs/nf_conditional_u128/${RUN_NAME}_checkpoints"
OUTPUT_PATH="${PROJECT_DIR}/results/nf_conditional_u128/samples/${RUN_NAME}_seed${SEED}_raw_conditional.npz"

STUB_ROOT="${PROJECT_DIR}/results/cache/python_stubs"
mkdir -p "${STUB_ROOT}/sklearn/metrics"
printf 'from . import metrics\n' > "${STUB_ROOT}/sklearn/__init__.py"
printf "def roc_curve(*args, **kwargs):\n    raise RuntimeError('sklearn.metrics.roc_curve is stubbed for cosmodiff sampling')\n" > "${STUB_ROOT}/sklearn/metrics/__init__.py"
export COSMODIFF_STUB_SKLEARN=1
export TORCHDYNAMO_DISABLE=1
export PYTHONPATH="${STUB_ROOT}:${COSMODIFF_DIR}:${PROJECT_DIR}:${PYTHONPATH:-}"
export COSMODIFF_DIR

if [[ ! -f "${CONFIG_PATH}" ]]; then
  echo "Missing config: ${CONFIG_PATH}" >&2
  exit 1
fi
if [[ ! -d "${CHECKPOINT_DIR}" ]]; then
  echo "Missing checkpoint directory: ${CHECKPOINT_DIR}" >&2
  exit 1
fi
if [[ ! -f "${COSMODIFF_DIR}/scripts/cosmodiff_sample.py" ]]; then
  echo "Missing main cosmo_diffusion sampler: ${COSMODIFF_DIR}" >&2
  exit 1
fi

PATCH_LOCK="${COSMODIFF_DIR}/.codex_sampling_patch.lock"
while ! mkdir "${PATCH_LOCK}" 2>/dev/null; do
  sleep 1
done
trap 'rmdir "${PATCH_LOCK}" 2>/dev/null || true' EXIT
"${PYTHON_BIN}" "${PROJECT_DIR}/scripts/patch_cosmodiff_sampling_compat.py" "${COSMODIFF_DIR}"
rmdir "${PATCH_LOCK}" 2>/dev/null || true
trap - EXIT

COSMODIFF_BRANCH=$(git -C "${COSMODIFF_DIR}" rev-parse --abbrev-ref HEAD)
COSMODIFF_COMMIT=$(git -C "${COSMODIFF_DIR}" rev-parse --short HEAD)
echo "cosmodiff git branch: ${COSMODIFF_BRANCH}"
echo "cosmodiff git commit: ${COSMODIFF_COMMIT}"
if [[ "${COSMODIFF_BRANCH}" != "${REQUIRED_COSMODIFF_BRANCH}" ]]; then
  echo "Wrong cosmo_diffusion branch: expected ${REQUIRED_COSMODIFF_BRANCH}, got ${COSMODIFF_BRANCH}" >&2
  exit 1
fi

if [[ -f "${OUTPUT_PATH}" && "${OVERWRITE}" != "1" ]]; then
  echo "Skipping existing sample file: ${OUTPUT_PATH}"
  exit 0
fi

echo "Sampling conditional ${RUN_NAME}"
echo "  checkpoint root: ${CHECKPOINT_DIR}"
echo "  config:          ${CONFIG_PATH}"
echo "  output:          ${OUTPUT_PATH}"
echo "  samples:         ${NUM_SAMPLES}"
echo "  batch:           ${BATCH_SIZE}"
echo "  cosmodiff:       ${COSMODIFF_DIR}"

"${PYTHON_BIN}" "${COSMODIFF_DIR}/scripts/cosmodiff_sample.py" \
  --config "${CONFIG_PATH}" \
  --filepath "${OUTPUT_PATH}" \
  --n_samples "${NUM_SAMPLES}" \
  --batch_size "${BATCH_SIZE}" \
  --seed "${SEED}" \
  --device cuda \
  --verbose
