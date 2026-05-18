#!/bin/bash
#SBATCH --job-name=nf_n64_mem_train
#SBATCH --partition=spgpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=80gb
#SBATCH --time=08:00:00
#SBATCH -o /home/jiamingp/diffusion_models_repo/logs/nf_generalize_n64_memorize/train_%j.out
#SBATCH -e /home/jiamingp/diffusion_models_repo/logs/nf_generalize_n64_memorize/train_%j.err
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
CONFIG_DIR=${CONFIG_DIR:-${PROJECT_DIR}/local/nf_generalize_n64_memorize/configs}
CHECKPOINT_ROOT=${CHECKPOINT_ROOT:-/scratch/huterer_root/huterer0/jiamingp/saved_runs/nf_generalize_n64_memorize}
PYTHON_BIN=${PYTHON_BIN:-python}
SKIP_IF_FINAL_CHECKPOINT=${SKIP_IF_FINAL_CHECKPOINT:-0}
REQUIRED_COSMODIFF_BRANCH=${REQUIRED_COSMODIFF_BRANCH:-main}

cd "${PROJECT_DIR}"
mkdir -p logs/nf_generalize_n64_memorize
if [[ -f "${VENV_PATH}/bin/activate" ]]; then
  source "${VENV_PATH}/bin/activate"
fi

RUN_NAME=$("${PYTHON_BIN}" scripts/prepare_nf_generalize_n64_memorize_config.py --project-dir "${PROJECT_DIR}" --print-runs)
CONFIG_PATH=${CONFIG_DIR}/${RUN_NAME}.yaml
CHECKPOINT_DIR=${CHECKPOINT_ROOT}/${RUN_NAME}_checkpoints
mkdir -p "${CHECKPOINT_DIR}"

STUB_ROOT="${PROJECT_DIR}/results/cache/python_stubs"
mkdir -p "${STUB_ROOT}/sklearn/metrics"
printf 'from . import metrics\n' > "${STUB_ROOT}/sklearn/__init__.py"
printf "def roc_curve(*args, **kwargs):\n    raise RuntimeError('sklearn.metrics.roc_curve is stubbed for cosmodiff training')\n" > "${STUB_ROOT}/sklearn/metrics/__init__.py"
export COSMODIFF_STUB_SKLEARN=1
export TORCHDYNAMO_DISABLE=1
export PYTHONPATH="${STUB_ROOT}:${COSMODIFF_DIR}:${PROJECT_DIR}:${PYTHONPATH:-}"
export COSMODIFF_DIR

if [[ ! -f "${CONFIG_PATH}" ]]; then
  echo "Missing config: ${CONFIG_PATH}" >&2
  echo "Run first: python scripts/prepare_nf_generalize_n64_memorize_config.py --project-dir ${PROJECT_DIR}" >&2
  exit 1
fi

if [[ ! -f "${COSMODIFF_DIR}/scripts/cosmodiff_train.py" ]]; then
  echo "Missing main cosmo_diffusion checkout: ${COSMODIFF_DIR}" >&2
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

"${PYTHON_BIN}" scripts/prepare_nf_generalize_n64_memorize_config.py \
  --project-dir "${PROJECT_DIR}" \
  --check-only

if [[ "${SKIP_IF_FINAL_CHECKPOINT}" == "1" ]] && [[ -d "${CHECKPOINT_DIR}/checkpoint-epoch-25000" ]]; then
  echo "Skipping ${RUN_NAME}; final checkpoint already exists under ${CHECKPOINT_DIR}"
  exit 0
fi

echo "Training ${RUN_NAME}"
echo "  config:      ${CONFIG_PATH}"
echo "  checkpoints: ${CHECKPOINT_DIR}"
echo "  target:      N=64, about 50k optimizer updates"
echo "  cosmodiff:   ${COSMODIFF_DIR}"
echo "  python:      ${PYTHON_BIN}"

"${PYTHON_BIN}" scripts/train_cosmodiff.py \
  --config "${CONFIG_PATH}" \
  --cosmodiff-train "${COSMODIFF_DIR}/scripts/cosmodiff_train.py"
