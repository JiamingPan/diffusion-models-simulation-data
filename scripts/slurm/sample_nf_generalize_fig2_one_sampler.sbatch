#!/bin/bash
#SBATCH --job-name=nf_fig2_sampler
#SBATCH --partition=spgpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=80gb
#SBATCH --time=04:00:00
#SBATCH -o /home/jiamingp/diffusion_models_repo/logs/nf_generalize_fig2/sample_sampler_%j.out
#SBATCH -e /home/jiamingp/diffusion_models_repo/logs/nf_generalize_fig2/sample_sampler_%j.err
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

VENV_PATH=${VENV_PATH:-/home/jiamingp/venvs/cosmodiff_nf_class}
BASE_VENV_PATH=${BASE_VENV_PATH:-/home/jiamingp/venvs/cosmodiff_nf}
CHECKPOINT_ROOT=${CHECKPOINT_ROOT:-/scratch/huterer_root/huterer0/jiamingp/saved_runs/nf_generalize_fig2}
CONFIG_DIR=${CONFIG_DIR:-${PROJECT_DIR}/local/nf_generalize_fig2/configs}
SAMPLE_ROOT=${SAMPLE_ROOT:-${PROJECT_DIR}/results/nf_generalize_fig2/samples}
PYTHON_BIN=${PYTHON_BIN:-python}

RUN_NAME=${RUN_NAME:-nf_fig2_u64_d2p15_noaug_200k}
SAMPLE_LABEL=${SAMPLE_LABEL:-dpm50}
SAMPLER_CLASS=${SAMPLER_CLASS:-DPMSolverMultistepScheduler}
SAMPLER_STEPS=${SAMPLER_STEPS:-50}
NUM_SAMPLES=${NUM_SAMPLES:-512}
BATCH_SIZE=${BATCH_SIZE:-8}
IMAGE_SIZE=${IMAGE_SIZE:-128}
SEED=${SEED:-123}
OVERWRITE=${OVERWRITE:-0}
REQUIRED_COSMODIFF_BRANCH=${REQUIRED_COSMODIFF_BRANCH:-main}

cd "${PROJECT_DIR}"
mkdir -p logs/nf_generalize_fig2 "${SAMPLE_ROOT}"
if [[ -f "${VENV_PATH}/bin/activate" ]]; then
  source "${VENV_PATH}/bin/activate"
fi
source "${PROJECT_DIR}/scripts/nf_class_conditional_pythonpath.sh"
set_nf_class_conditional_pythonpath "${VENV_PATH}/bin/python" "${BASE_VENV_PATH}"

"${PYTHON_BIN}" scripts/prepare_nf_generalize_fig2_configs.py --project-dir "${PROJECT_DIR}"

CONFIG_PATH=${CONFIG_DIR}/${RUN_NAME}.yaml
CHECKPOINT_DIR=${CHECKPOINT_ROOT}/${RUN_NAME}_checkpoints
CHECKPOINT_PATH=${CHECKPOINT_PATH:-${CHECKPOINT_DIR}}
OUTPUT_PATH=${SAMPLE_ROOT}/${RUN_NAME}_seed${SEED}_${SAMPLE_LABEL}.npz

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

if [[ ! -e "${CHECKPOINT_PATH}" ]]; then
  echo "Missing checkpoint path: ${CHECKPOINT_PATH}" >&2
  exit 1
fi

if [[ ! -f "${PROJECT_DIR}/scripts/sample_cosmodiff.py" ]]; then
  echo "Missing local sampling wrapper: ${PROJECT_DIR}/scripts/sample_cosmodiff.py" >&2
  exit 1
fi

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

echo "Sampling Fig. 2 run ${RUN_NAME}"
echo "  checkpoint input: ${CHECKPOINT_PATH}"
echo "  config:          ${CONFIG_PATH}"
echo "  output:          ${OUTPUT_PATH}"
echo "  samples:         ${NUM_SAMPLES}"
echo "  batch:           ${BATCH_SIZE}"
echo "  sampler:         ${SAMPLE_LABEL} (${SAMPLER_CLASS}, ${SAMPLER_STEPS} steps)"
echo "  cosmodiff:       ${COSMODIFF_DIR}"
echo "  venv:            ${VENV_PATH}"

"${PYTHON_BIN}" "${PROJECT_DIR}/scripts/sample_cosmodiff.py" \
  --checkpoint "${CHECKPOINT_PATH}" \
  --config "${CONFIG_PATH}" \
  --output "${OUTPUT_PATH}" \
  --num-samples 1 \
  --batch-size 1 \
  --image-size "${IMAGE_SIZE}" \
  --scheduler "${SAMPLER_CLASS}" \
  --num-steps "${SAMPLER_STEPS}" \
  --seed "${SEED}" \
  --device cpu \
  --preflight-only

"${PYTHON_BIN}" "${PROJECT_DIR}/scripts/sample_cosmodiff.py" \
  --checkpoint "${CHECKPOINT_PATH}" \
  --config "${CONFIG_PATH}" \
  --output "${OUTPUT_PATH}" \
  --num-samples "${NUM_SAMPLES}" \
  --batch-size "${BATCH_SIZE}" \
  --image-size "${IMAGE_SIZE}" \
  --scheduler "${SAMPLER_CLASS}" \
  --num-steps "${SAMPLER_STEPS}" \
  --seed "${SEED}" \
  --device cuda
