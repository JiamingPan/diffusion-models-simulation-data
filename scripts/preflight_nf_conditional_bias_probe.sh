#!/bin/bash
# Login-node preflight for the continuous HI bias-probe training configs.

set -euo pipefail
trap 'rc=$?; echo "[error] bias preflight failed at line ${LINENO}: ${BASH_COMMAND} (exit ${rc})" >&2' ERR

PROJECT_DIR=${PROJECT_DIR:-/home/jiamingp/diffusion_models_repo}
COSMODIFF_DIR=${COSMODIFF_DIR_OVERRIDE:-/home/jiamingp/Diffusion_model/cosmo_diffusion_main}
VENV_PATH=${VENV_PATH:-/home/jiamingp/venvs/cosmodiff_nf_class}
BASE_VENV_PATH=${BASE_VENV_PATH:-/home/jiamingp/venvs/cosmodiff_nf}
PYTHON_BIN=${PYTHON_BIN:-python}

cd "${PROJECT_DIR}"
source "${PROJECT_DIR}/scripts/nf_class_conditional_pythonpath.sh"
mkdir -p logs/nf_conditional_bias_probe results/cache/python_stubs
if [[ -f "${VENV_PATH}/bin/activate" ]]; then
  source "${VENV_PATH}/bin/activate"
else
  echo "Missing CUDA-capable class env: ${VENV_PATH}" >&2
  echo "Run first: bash scripts/setup_nf_class_conditional_env.sh" >&2
  exit 1
fi
set_nf_class_conditional_pythonpath "${PYTHON_BIN}" "${BASE_VENV_PATH}"

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

PATCH_LOCK="${COSMODIFF_DIR}/.codex_bias_preflight_patch.lock"
while ! mkdir "${PATCH_LOCK}" 2>/dev/null; do
  sleep 1
done
trap 'rmdir "${PATCH_LOCK}" 2>/dev/null || true' EXIT
"${PYTHON_BIN}" "${PROJECT_DIR}/scripts/patch_cosmodiff_continuous_labels.py" "${COSMODIFF_DIR}"
"${PYTHON_BIN}" "${PROJECT_DIR}/scripts/patch_cosmodiff_direct_unet_checkpoint.py" "${COSMODIFF_DIR}"
"${PYTHON_BIN}" "${PROJECT_DIR}/scripts/patch_cosmodiff_numpy_compat.py" "${COSMODIFF_DIR}"
rmdir "${PATCH_LOCK}" 2>/dev/null || true
trap - EXIT
trap 'rc=$?; echo "[error] bias preflight failed at line ${LINENO}: ${BASH_COMMAND} (exit ${rc})" >&2' ERR

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
