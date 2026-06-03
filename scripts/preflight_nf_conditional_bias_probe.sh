#!/bin/bash
# Login-node preflight for the continuous HI bias-probe training configs.

set -euo pipefail
trap 'rc=$?; echo "[error] bias preflight failed at line ${LINENO}: ${BASH_COMMAND} (exit ${rc})" >&2' ERR

PROJECT_DIR=${PROJECT_DIR:-/home/jiamingp/diffusion_models_repo}
COSMODIFF_DIR=${COSMODIFF_DIR_OVERRIDE:-/home/jiamingp/Diffusion_model/cosmo_diffusion_main}
VENV_PATH=${VENV_PATH:-/home/jiamingp/venvs/cosmodiff_nf}
PYTHON_BIN=${PYTHON_BIN:-python}

cd "${PROJECT_DIR}"
mkdir -p logs/nf_conditional_bias_probe results/cache/python_stubs
if [[ -f "${VENV_PATH}/bin/activate" ]]; then
  source "${VENV_PATH}/bin/activate"
fi

STUB_ROOT="${PROJECT_DIR}/results/cache/python_stubs/manual_bias_preflight_${USER:-user}_$$"
mkdir -p "${STUB_ROOT}/sklearn/metrics"
printf 'from . import metrics\n' > "${STUB_ROOT}/sklearn/__init__.py"
printf "def roc_curve(*args, **kwargs):\n    raise RuntimeError('sklearn.metrics.roc_curve is stubbed for cosmodiff preflight')\n" > "${STUB_ROOT}/sklearn/metrics/__init__.py"
"${PYTHON_BIN}" scripts/write_diffusers_runtime_sitecustomize.py "${STUB_ROOT}/sitecustomize.py"

export PYTHONPATH="${STUB_ROOT}:${COSMODIFF_DIR}:${PROJECT_DIR}:${PYTHONPATH:-}"
export COSMODIFF_STUB_SKLEARN=1
export COSMODIFF_DIR
export TORCHDYNAMO_DISABLE=1

echo "[preflight-shell] python=$("${PYTHON_BIN}" -c 'import sys; print(sys.executable)')"
echo "[preflight-shell] project=${PROJECT_DIR}"
echo "[preflight-shell] cosmodiff=${COSMODIFF_DIR}"

for dataset_size in 128 16384; do
  run_name=$("${PYTHON_BIN}" scripts/prepare_nf_conditional_bias_probe_configs.py \
    --project-dir "${PROJECT_DIR}" \
    --dataset-sizes "${dataset_size}" \
    --print-runs)
  config_path="${PROJECT_DIR}/local/nf_conditional_bias_probe/configs/${run_name}.yaml"
  "${PYTHON_BIN}" scripts/check_nf_conditional_bias_probe_runtime.py \
    --config "${config_path}" \
    --cosmodiff-dir "${COSMODIFF_DIR}"
done

echo "[preflight-shell] nf_conditional_bias_probe checks passed for N=128 and N=16384."
