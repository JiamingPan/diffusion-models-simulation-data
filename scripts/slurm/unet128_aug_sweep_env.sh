#!/bin/bash
# Shared runtime for the width-128 augmentation sweep; mirrors
# train_nf_generalize_fig2_array.sbatch (same venv, cosmodiff checkout, stubs).
set -euo pipefail

PROJECT_DIR=${PROJECT_DIR:-/home/jiamingp/diffusion_models_repo}
COSMODIFF_DIR=/home/jiamingp/Diffusion_model/cosmo_diffusion_main
VENV_PATH=${VENV_PATH:-/home/jiamingp/venvs/cosmodiff_nf_class}
BASE_VENV_PATH=${BASE_VENV_PATH:-/home/jiamingp/venvs/cosmodiff_nf}
PYTHON_BIN=${PYTHON_BIN:-python}
SWEEP_LOCAL=${PROJECT_DIR}/local/unet128_aug_sweep

cd "${PROJECT_DIR}"
mkdir -p logs/unet128_aug_sweep
source "${VENV_PATH}/bin/activate"
source "${PROJECT_DIR}/scripts/nf_class_conditional_pythonpath.sh"
set_nf_class_conditional_pythonpath "${VENV_PATH}/bin/python" "${BASE_VENV_PATH}"

# Per-job stub dir (as other launchers do); never rewrite the shared python_stubs/sitecustomize.py.
STUB_ROOT="${PROJECT_DIR}/results/cache/python_stubs/${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID:-manual_$$}}${SLURM_ARRAY_TASK_ID:+_${SLURM_ARRAY_TASK_ID}}"
mkdir -p "${STUB_ROOT}/sklearn/metrics"
printf 'from . import metrics\n' > "${STUB_ROOT}/sklearn/__init__.py"
printf "def roc_curve(*args, **kwargs):\n    raise RuntimeError('sklearn.metrics.roc_curve is stubbed for cosmodiff')\n" > "${STUB_ROOT}/sklearn/metrics/__init__.py"
"${PYTHON_BIN}" scripts/write_diffusers_runtime_sitecustomize.py "${STUB_ROOT}/sitecustomize.py"
export COSMODIFF_STUB_SKLEARN=1
export TORCHDYNAMO_DISABLE=1
export PYTHONPATH="${STUB_ROOT}:${COSMODIFF_DIR}:${PROJECT_DIR}:${PYTHONPATH:-}"
export MPLCONFIGDIR="${PROJECT_DIR}/results/cache/matplotlib"
export COSMODIFF_DIR

COSMODIFF_BRANCH=$(git -C "${COSMODIFF_DIR}" rev-parse --abbrev-ref HEAD)
echo "cosmodiff ${COSMODIFF_BRANCH} $(git -C "${COSMODIFF_DIR}" rev-parse --short HEAD)"
[[ "${COSMODIFF_BRANCH}" == main ]] || { echo "Wrong cosmo_diffusion branch: ${COSMODIFF_BRANCH}" >&2; exit 1; }

# Field of runs.json for array task $2 of arm $1.
run_field() {
  "${PYTHON_BIN}" - "$1" "$2" "$3" <<'PY'
import json, os, sys
arm, task, key = sys.argv[1], int(sys.argv[2]), sys.argv[3]
runs = [r for r in json.load(open(os.path.join(os.environ["SWEEP_LOCAL"], "runs.json"))) if r["arm_tag"] == arm]
print(runs[task][key])
PY
}
export SWEEP_LOCAL
