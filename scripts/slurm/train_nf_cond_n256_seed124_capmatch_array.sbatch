#!/bin/bash
#SBATCH --job-name=n256ctl_train
#SBATCH --partition=spgpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16gb
#SBATCH --time=15:00:00
#SBATCH --array=0-1%2
#SBATCH --no-requeue
#SBATCH -o /home/jiamingp/diffusion_models_repo/logs/nf_cond_n256_controls/train_%A_%a.out
#SBATCH -e /home/jiamingp/diffusion_models_repo/logs/nf_cond_n256_controls/train_%A_%a.err
#SBATCH -A huterer2
# Two N = 256 controls of nf_cond_omsigc_hi_u128_d2p08_n256_fresh200k (scripts/prepare_nf_cond_capmatch_seed_n256.py):
#   task 0  ledger nf_cond_omsigc_n256_seed124       baseline yaml (new output_dir), TRAINING_SEED 124
#   task 1  ledger nf_condmech_n256_capmatch_tembss  baseline + projection class embedding + scale_shift, seed 123
# Runtime mirrors train_conditioning_mechanism_n256_array.sbatch (venv, sitecustomize seed + timeemb routing patch,
# cosmodiff checkout and patches). 16 GB / 15 h as the condmech trainings after 2026-10-08 (docs/decisions.md;
# 63381901 tasks took 11:29-12:10, MaxRSS <= 1.3 GB). The seed comes from the prepare script, not the environment.
# Submit one task per ledger run (--array=0 or --array=1) with RUN_NAME=<ledger run>; the task refuses a mismatch.

set -euo pipefail
trap 'rc=$?; echo "[error] training failed at line ${LINENO}: ${BASH_COMMAND} (exit ${rc})" >&2' ERR

PROJECT_DIR=/home/jiamingp/diffusion_models_repo
COSMODIFF_DIR=/home/jiamingp/Diffusion_model/cosmo_diffusion_main
VENV_PATH=/home/jiamingp/venvs/cosmodiff_nf_class
BASE_VENV_PATH=/home/jiamingp/venvs/cosmodiff_nf
PYTHON_BIN=python

cd "${PROJECT_DIR}"
source "${PROJECT_DIR}/scripts/nf_class_conditional_pythonpath.sh"
mkdir -p logs/nf_cond_n256_controls results/cache/python_stubs
source "${VENV_PATH}/bin/activate"
set_nf_class_conditional_pythonpath "${PYTHON_BIN}" "${BASE_VENV_PATH}"

mapfile -t RUNS < <("${PYTHON_BIN}" scripts/prepare_nf_cond_capmatch_seed_n256.py --print-train-runs)
TASK_ID=${SLURM_ARRAY_TASK_ID:-0}
if (( TASK_ID < 0 || TASK_ID >= ${#RUNS[@]} )); then
  echo "Array index ${TASK_ID} outside run count ${#RUNS[@]}" >&2
  exit 1
fi
read -r MODEL_RUN STUDY TRAINING_SEED LEDGER_RUN <<< "${RUNS[$TASK_ID]}"
if [[ -n "${RUN_NAME:-}" && "${RUN_NAME}" != "${LEDGER_RUN}" ]]; then
  echo "RUN_NAME=${RUN_NAME} does not match task ${TASK_ID} (ledger ${LEDGER_RUN})" >&2
  exit 1
fi
CONFIG_PATH="${PROJECT_DIR}/local/${STUDY}/configs/${MODEL_RUN}.yaml"
OUTPUT_DIR=$("${PYTHON_BIN}" -c "import sys, yaml; print(yaml.safe_load(open(sys.argv[1]))['io']['output_dir'])" "${CONFIG_PATH}")
if compgen -G "${OUTPUT_DIR}/checkpoint-epoch-*" > /dev/null; then
  echo "${OUTPUT_DIR} already holds checkpoints; refusing to train over them" >&2
  exit 1
fi
echo "task ${TASK_ID}: ${MODEL_RUN} (ledger ${LEDGER_RUN}) seed ${TRAINING_SEED} config ${CONFIG_PATH}"
sha256sum "${CONFIG_PATH}"

STUB_ROOT="${PROJECT_DIR}/results/cache/python_stubs/${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID:-manual_n256ctl}}_${TASK_ID}"
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

import simdiff_eval.timeemb_conditioning as _timeemb
_timeemb.install()
PY
export PYTHONPATH="${STUB_ROOT}:${COSMODIFF_DIR}:${PROJECT_DIR}:${PROJECT_DIR}/scripts:${PYTHONPATH:-}"
export COSMODIFF_TRAIN_SEED="${TRAINING_SEED}"
export PYTHONHASHSEED="${TRAINING_SEED}"
export COSMODIFF_STUB_SKLEARN=1
export COSMODIFF_DIR
export TORCHDYNAMO_DISABLE=1

PATCH_LOCK="${COSMODIFF_DIR}/.codex_conditional_omsc_patch.lock"
while ! mkdir "${PATCH_LOCK}" 2>/dev/null; do sleep 1; done
trap 'rmdir "${PATCH_LOCK}" 2>/dev/null || true' EXIT
"${PYTHON_BIN}" scripts/patch_cosmodiff_continuous_labels.py "${COSMODIFF_DIR}"
"${PYTHON_BIN}" scripts/patch_cosmodiff_direct_unet_checkpoint.py "${COSMODIFF_DIR}"
"${PYTHON_BIN}" scripts/patch_cosmodiff_numpy_compat.py "${COSMODIFF_DIR}"
rmdir "${PATCH_LOCK}" 2>/dev/null || true
trap - EXIT

echo "Training ${MODEL_RUN} from a clean initialization"
"${PYTHON_BIN}" scripts/train_cosmodiff.py \
  --config "${CONFIG_PATH}" \
  --cosmodiff-train "${COSMODIFF_DIR}/scripts/cosmodiff_train.py"
