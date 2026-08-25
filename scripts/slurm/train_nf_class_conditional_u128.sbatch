#!/bin/bash
#SBATCH --job-name=nf_cls_u128
#SBATCH --partition=spgpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=80gb
#SBATCH --time=24:00:00
#SBATCH -o /home/jiamingp/diffusion_models_repo/logs/nf_class_conditional_u128/train_%j.out
#SBATCH -e /home/jiamingp/diffusion_models_repo/logs/nf_class_conditional_u128/train_%j.err
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
PYTHON_BIN=${PYTHON_BIN:-python}
REQUIRED_COSMODIFF_BRANCH=${REQUIRED_COSMODIFF_BRANCH:-main}

cd "${PROJECT_DIR}"
source "${PROJECT_DIR}/scripts/nf_class_conditional_pythonpath.sh"
mkdir -p logs/nf_class_conditional_u128
if [[ -f "${VENV_PATH}/bin/activate" ]]; then
  source "${VENV_PATH}/bin/activate"
else
  echo "Missing class-conditional env: ${VENV_PATH}" >&2
  echo "Run first: bash scripts/setup_nf_class_conditional_env.sh" >&2
  exit 1
fi

set_nf_class_conditional_pythonpath "${PYTHON_BIN}" "${BASE_VENV_PATH}"

RUN_NAME=$("${PYTHON_BIN}" scripts/prepare_nf_class_conditional_u128_config.py --project-dir "${PROJECT_DIR}" --print-runs)
CONFIG_PATH="${PROJECT_DIR}/local/nf_class_conditional_u128/configs/${RUN_NAME}.yaml"
export CONFIG_PATH
"${PYTHON_BIN}" scripts/prepare_nf_class_conditional_u128_config.py --project-dir "${PROJECT_DIR}"

STUB_ROOT="${PROJECT_DIR}/results/cache/python_stubs"
mkdir -p "${STUB_ROOT}/sklearn/metrics"
printf 'from . import metrics\n' > "${STUB_ROOT}/sklearn/__init__.py"
printf "def roc_curve(*args, **kwargs):\n    raise RuntimeError('sklearn.metrics.roc_curve is stubbed for cosmodiff training')\n" > "${STUB_ROOT}/sklearn/metrics/__init__.py"
"${PYTHON_BIN}" "${PROJECT_DIR}/scripts/write_diffusers_runtime_sitecustomize.py" "${STUB_ROOT}/sitecustomize.py"
export COSMODIFF_STUB_SKLEARN=1
export TORCHDYNAMO_DISABLE=1
export PYTHONPATH="${STUB_ROOT}:${COSMODIFF_DIR}:${PROJECT_DIR}:${PYTHONPATH:-}"
export COSMODIFF_DIR

if [[ ! -f "${CONFIG_PATH}" ]]; then
  echo "Missing config: ${CONFIG_PATH}" >&2
  echo "Run first: python scripts/prepare_nf_class_conditional_u128_config.py --project-dir ${PROJECT_DIR}" >&2
  exit 1
fi

if [[ ! -f "${COSMODIFF_DIR}/scripts/cosmodiff_train.py" ]]; then
  echo "Missing main cosmo_diffusion checkout: ${COSMODIFF_DIR}" >&2
  exit 1
fi

PATCH_LOCK="${COSMODIFF_DIR}/.codex_conditional_labels_patch.lock"
while ! mkdir "${PATCH_LOCK}" 2>/dev/null; do
  sleep 1
done
trap 'rmdir "${PATCH_LOCK}" 2>/dev/null || true' EXIT
"${PYTHON_BIN}" "${PROJECT_DIR}/scripts/patch_cosmodiff_package_metadata.py" "${COSMODIFF_DIR}"
"${PYTHON_BIN}" "${PROJECT_DIR}/scripts/patch_cosmodiff_safe_log.py" "${COSMODIFF_DIR}"
"${PYTHON_BIN}" "${PROJECT_DIR}/scripts/patch_cosmodiff_continuous_labels.py" "${COSMODIFF_DIR}"
"${PYTHON_BIN}" "${PROJECT_DIR}/scripts/patch_cosmodiff_multinorm_labels.py" "${COSMODIFF_DIR}"
"${PYTHON_BIN}" "${PROJECT_DIR}/scripts/patch_cosmodiff_nonfinite_loss_guard.py" "${COSMODIFF_DIR}"
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

"${PYTHON_BIN}" scripts/check_nf_class_conditional_u128_runtime.py \
  --project-dir "${PROJECT_DIR}" \
  --cosmodiff-dir "${COSMODIFF_DIR}" \
  --config "${CONFIG_PATH}"

"${PYTHON_BIN}" scripts/prepare_nf_class_conditional_u128_config.py \
  --project-dir "${PROJECT_DIR}" \
  --check-only

echo "Training discrete class-conditional ${RUN_NAME}"
echo "  config:    ${CONFIG_PATH}"
echo "  cosmodiff: ${COSMODIFF_DIR}"
echo "  venv:      ${VENV_PATH}"
echo "  python:    ${PYTHON_BIN}"

"${PYTHON_BIN}" scripts/train_cosmodiff.py \
  --config "${CONFIG_PATH}" \
  --cosmodiff-train "${COSMODIFF_DIR}/scripts/cosmodiff_train.py"
