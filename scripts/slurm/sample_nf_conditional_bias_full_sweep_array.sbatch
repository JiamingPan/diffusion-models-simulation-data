#!/bin/bash
#SBATCH --job-name=nf_calfsamp
#SBATCH --partition=spgpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=80gb
#SBATCH --time=08:00:00
#SBATCH -o /home/jiamingp/diffusion_models_repo/logs/nf_conditional_bias_fresh_full_sweep_200k/sample_%A_%a.out
#SBATCH -e /home/jiamingp/diffusion_models_repo/logs/nf_conditional_bias_fresh_full_sweep_200k/sample_%A_%a.err
#SBATCH -A huterer2

set -euo pipefail
PROJECT_DIR=${PROJECT_DIR:-/home/jiamingp/diffusion_models_repo}
COSMODIFF_DIR=${COSMODIFF_DIR_OVERRIDE:-/home/jiamingp/Diffusion_model/cosmo_diffusion_main}
VENV_PATH=${VENV_PATH:-/home/jiamingp/venvs/cosmodiff_nf_class}
BASE_VENV_PATH=${BASE_VENV_PATH:-/home/jiamingp/venvs/cosmodiff_nf}
PYTHON_BIN=${PYTHON_BIN:-python}

cd "${PROJECT_DIR}"
source "${PROJECT_DIR}/scripts/nf_class_conditional_pythonpath.sh"
mkdir -p logs/nf_conditional_bias_fresh_full_sweep_200k results/nf_conditional_bias_fresh_full_sweep_200k/samples results/cache/python_stubs
source "${VENV_PATH}/bin/activate"
set_nf_class_conditional_pythonpath "${PYTHON_BIN}" "${BASE_VENV_PATH}"

mapfile -t RUNS < <("${PYTHON_BIN}" scripts/prepare_nf_conditional_bias_full_sweep_configs.py --print-runs)
TASK_ID=${SLURM_ARRAY_TASK_ID:-0}
RUN_NAME=${RUNS[$TASK_ID]}

STUB_ROOT="${PROJECT_DIR}/results/cache/python_stubs/${SLURM_JOB_ID:-manual_calfsamp}_${TASK_ID}"
mkdir -p "${STUB_ROOT}/sklearn/metrics"
printf 'from . import metrics\n' > "${STUB_ROOT}/sklearn/__init__.py"
printf "def roc_curve(*args, **kwargs):\n    raise RuntimeError('sklearn.metrics.roc_curve is stubbed')\n" > "${STUB_ROOT}/sklearn/metrics/__init__.py"
"${PYTHON_BIN}" scripts/write_diffusers_runtime_sitecustomize.py "${STUB_ROOT}/sitecustomize.py"
export PYTHONPATH="${STUB_ROOT}:${COSMODIFF_DIR}:${PROJECT_DIR}:${PROJECT_DIR}/scripts:${PYTHONPATH:-}"
export COSMODIFF_DIR
export TORCHDYNAMO_DISABLE=1

PATCH_LOCK="${COSMODIFF_DIR}/.codex_conditional_full_sample_patch.lock"
while ! mkdir "${PATCH_LOCK}" 2>/dev/null; do sleep 1; done
trap 'rmdir "${PATCH_LOCK}" 2>/dev/null || true' EXIT
"${PYTHON_BIN}" scripts/patch_cosmodiff_package_metadata.py "${COSMODIFF_DIR}"
"${PYTHON_BIN}" scripts/patch_cosmodiff_continuous_labels.py "${COSMODIFF_DIR}"
"${PYTHON_BIN}" scripts/patch_cosmodiff_direct_unet_checkpoint.py "${COSMODIFF_DIR}"
"${PYTHON_BIN}" scripts/patch_cosmodiff_numpy_compat.py "${COSMODIFF_DIR}"
"${PYTHON_BIN}" scripts/patch_cosmodiff_sampling_compat.py "${COSMODIFF_DIR}"
rmdir "${PATCH_LOCK}" 2>/dev/null || true
trap - EXIT

"${PYTHON_BIN}" scripts/sample_nf_conditional_bias_probe.py \
  --project-dir "${PROJECT_DIR}" \
  --manifest "${PROJECT_DIR}/local/nf_conditional_bias_fresh_full_sweep_200k/manifest.json" \
  --run-name "${RUN_NAME}" \
  --seed 123 \
  --samples-per-cosmology 64 \
  --batch-size 8 \
  --scheduler DPMSolverMultistepScheduler \
  --num-steps 50 \
  --device cuda \
  --overwrite
