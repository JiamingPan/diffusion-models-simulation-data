#!/bin/bash
#SBATCH --job-name=nf_calftrain
#SBATCH --partition=spgpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=80gb
#SBATCH --time=48:00:00
#SBATCH -o /home/jiamingp/diffusion_models_repo/logs/nf_conditional_bias_fresh_full_sweep_200k/train_%A_%a.out
#SBATCH -e /home/jiamingp/diffusion_models_repo/logs/nf_conditional_bias_fresh_full_sweep_200k/train_%A_%a.err
#SBATCH -A huterer2

set -euo pipefail
trap 'rc=$?; echo "[error] training failed at line ${LINENO}: ${BASH_COMMAND} (exit ${rc})" >&2' ERR

PROJECT_DIR=${PROJECT_DIR:-/home/jiamingp/diffusion_models_repo}
COSMODIFF_DIR=${COSMODIFF_DIR_OVERRIDE:-/home/jiamingp/Diffusion_model/cosmo_diffusion_main}
VENV_PATH=${VENV_PATH:-/home/jiamingp/venvs/cosmodiff_nf_class}
BASE_VENV_PATH=${BASE_VENV_PATH:-/home/jiamingp/venvs/cosmodiff_nf}
PYTHON_BIN=${PYTHON_BIN:-python}
TRAINING_SEED=${TRAINING_SEED:-123}

cd "${PROJECT_DIR}"
source "${PROJECT_DIR}/scripts/nf_class_conditional_pythonpath.sh"
mkdir -p logs/nf_conditional_bias_fresh_full_sweep_200k results/cache/python_stubs
if [[ ! -f "${VENV_PATH}/bin/activate" ]]; then
  echo "Missing CUDA training environment: ${VENV_PATH}" >&2
  exit 1
fi
source "${VENV_PATH}/bin/activate"
set_nf_class_conditional_pythonpath "${PYTHON_BIN}" "${BASE_VENV_PATH}"

mapfile -t RUNS < <("${PYTHON_BIN}" scripts/prepare_nf_conditional_bias_full_sweep_configs.py --print-train-runs)
TASK_ID=${SLURM_ARRAY_TASK_ID:-0}
if (( TASK_ID < 0 || TASK_ID >= ${#RUNS[@]} )); then
  echo "Array index ${TASK_ID} outside run count ${#RUNS[@]}" >&2
  exit 1
fi
RUN_NAME=${RUNS[$TASK_ID]}
CONFIG_PATH="${PROJECT_DIR}/local/nf_conditional_bias_fresh_full_sweep_200k/configs/${RUN_NAME}.yaml"

STUB_ROOT="${PROJECT_DIR}/results/cache/python_stubs/${SLURM_JOB_ID:-manual_calf}_${TASK_ID}"
mkdir -p "${STUB_ROOT}/sklearn/metrics"
printf 'from . import metrics\n' > "${STUB_ROOT}/sklearn/__init__.py"
printf "def roc_curve(*args, **kwargs):\n    raise RuntimeError('sklearn.metrics.roc_curve is stubbed')\n" > "${STUB_ROOT}/sklearn/metrics/__init__.py"
"${PYTHON_BIN}" scripts/write_diffusers_runtime_sitecustomize.py "${STUB_ROOT}/sitecustomize.py"
cat >> "${STUB_ROOT}/sitecustomize.py" <<'PY'
import os
import random

import numpy as np
import torch

_cosmodiff_seed = int(os.environ["COSMODIFF_TRAIN_SEED"])
random.seed(_cosmodiff_seed)
np.random.seed(_cosmodiff_seed)
torch.manual_seed(_cosmodiff_seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(_cosmodiff_seed)
PY
export PYTHONPATH="${STUB_ROOT}:${COSMODIFF_DIR}:${PROJECT_DIR}:${PROJECT_DIR}/scripts:${PYTHONPATH:-}"
export COSMODIFF_TRAIN_SEED="${TRAINING_SEED}"
export PYTHONHASHSEED="${TRAINING_SEED}"
export COSMODIFF_STUB_SKLEARN=1
export COSMODIFF_DIR
export TORCHDYNAMO_DISABLE=1

PATCH_LOCK="${COSMODIFF_DIR}/.codex_conditional_full_sweep_patch.lock"
while ! mkdir "${PATCH_LOCK}" 2>/dev/null; do sleep 1; done
trap 'rmdir "${PATCH_LOCK}" 2>/dev/null || true' EXIT
"${PYTHON_BIN}" scripts/patch_cosmodiff_continuous_labels.py "${COSMODIFF_DIR}"
"${PYTHON_BIN}" scripts/patch_cosmodiff_direct_unet_checkpoint.py "${COSMODIFF_DIR}"
"${PYTHON_BIN}" scripts/patch_cosmodiff_numpy_compat.py "${COSMODIFF_DIR}"
rmdir "${PATCH_LOCK}" 2>/dev/null || true
trap - EXIT

"${PYTHON_BIN}" scripts/check_nf_conditional_bias_probe_runtime.py \
  --config "${CONFIG_PATH}" \
  --cosmodiff-dir "${COSMODIFF_DIR}" \
  --require-cuda
echo "Training ${RUN_NAME} from a clean initialization"
"${PYTHON_BIN}" scripts/train_cosmodiff.py \
  --config "${CONFIG_PATH}" \
  --cosmodiff-train "${COSMODIFF_DIR}/scripts/cosmodiff_train.py"
