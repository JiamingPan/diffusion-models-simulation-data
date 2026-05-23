#!/bin/bash
set -euo pipefail

PROJECT_DIR=${PROJECT_DIR:-/home/jiamingp/diffusion_models_repo}
COSMODIFF_DIR=${COSMODIFF_DIR:-/home/jiamingp/Diffusion_model/cosmo_diffusion_main}
VENV_PATH=${VENV_PATH:-/home/jiamingp/venvs/cosmodiff_nf}
PYTHON_BIN=${PYTHON_BIN:-python}

cd "${PROJECT_DIR}"
if [[ -f "${VENV_PATH}/bin/activate" ]]; then
  source "${VENV_PATH}/bin/activate"
fi

RUN_NAME=$("${PYTHON_BIN}" scripts/prepare_nf_class_conditional_u128_config.py --project-dir "${PROJECT_DIR}" --print-runs)
CONFIG_PATH="${PROJECT_DIR}/local/nf_class_conditional_u128/configs/${RUN_NAME}.yaml"

STUB_ROOT="${PROJECT_DIR}/results/cache/python_stubs"
mkdir -p "${STUB_ROOT}/sklearn/metrics"
printf 'from . import metrics\n' > "${STUB_ROOT}/sklearn/__init__.py"
printf "def roc_curve(*args, **kwargs):\n    raise RuntimeError('sklearn.metrics.roc_curve is stubbed for cosmodiff preflight')\n" > "${STUB_ROOT}/sklearn/metrics/__init__.py"
export COSMODIFF_STUB_SKLEARN=1
export TORCHDYNAMO_DISABLE=1
export PYTHONPATH="${STUB_ROOT}:${COSMODIFF_DIR}:${PROJECT_DIR}:${PYTHONPATH:-}"
export COSMODIFF_DIR

if [[ ! -f "${CONFIG_PATH}" ]]; then
  echo "Missing config: ${CONFIG_PATH}" >&2
  echo "Run: python scripts/prepare_nf_class_conditional_u128_config.py --project-dir ${PROJECT_DIR}" >&2
  exit 1
fi

if [[ ! -f "${COSMODIFF_DIR}/scripts/cosmodiff_train.py" ]]; then
  echo "Missing cosmo_diffusion checkout: ${COSMODIFF_DIR}" >&2
  exit 1
fi

"${PYTHON_BIN}" "${PROJECT_DIR}/scripts/patch_cosmodiff_continuous_labels.py" "${COSMODIFF_DIR}"

echo "project:    ${PROJECT_DIR}"
echo "cosmodiff:  ${COSMODIFF_DIR}"
echo "config:     ${CONFIG_PATH}"
echo "python:     $(${PYTHON_BIN} -c 'import sys; print(sys.executable)')"
echo "repo head:  $(git -C "${PROJECT_DIR}" rev-parse --short HEAD)"
echo "cosmodiff:  $(git -C "${COSMODIFF_DIR}" rev-parse --abbrev-ref HEAD) $(git -C "${COSMODIFF_DIR}" rev-parse --short HEAD)"

"${PYTHON_BIN}" scripts/check_nf_class_conditional_u128_runtime.py \
  --project-dir "${PROJECT_DIR}" \
  --cosmodiff-dir "${COSMODIFF_DIR}" \
  --config "${CONFIG_PATH}"
