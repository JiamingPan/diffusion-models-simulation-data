#!/bin/bash
#SBATCH --job-name=nf_bias_cont
#SBATCH --partition=spgpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=80gb
#SBATCH --time=18:00:00
#SBATCH --array=0-1%2
#SBATCH -o /home/jiamingp/diffusion_models_repo/logs/nf_conditional_bias_probe_continue/train_%A_%a.out
#SBATCH -e /home/jiamingp/diffusion_models_repo/logs/nf_conditional_bias_probe_continue/train_%A_%a.err
#SBATCH -A huterer2

set -euo pipefail
trap 'rc=$?; echo "[error] ${SLURM_JOB_NAME:-nf_bias_cont} failed at line ${LINENO}: ${BASH_COMMAND} (exit ${rc})" >&2' ERR

PROJECT_DIR=${PROJECT_DIR:-/home/jiamingp/diffusion_models_repo}
COSMODIFF_DIR=${COSMODIFF_DIR_OVERRIDE:-/home/jiamingp/Diffusion_model/cosmo_diffusion_main}
VENV_PATH=${VENV_PATH:-/home/jiamingp/venvs/cosmodiff_nf_class}
BASE_VENV_PATH=${BASE_VENV_PATH:-/home/jiamingp/venvs/cosmodiff_nf}
PYTHON_BIN=${PYTHON_BIN:-python}
CONTINUE_SWEEP_NAME=${CONTINUE_SWEEP_NAME:-nf_conditional_bias_probe_continue}
ADDITIONAL_UPDATES=${ADDITIONAL_UPDATES:-300000}
CHECKPOINT_EVERY_UPDATES=${CHECKPOINT_EVERY_UPDATES:-20000}
BIAS_RUN_PREFLIGHT=${BIAS_RUN_PREFLIGHT:-1}
BIAS_PREFLIGHT_ONLY=${BIAS_PREFLIGHT_ONLY:-0}
REQUIRED_COSMODIFF_BRANCH=${REQUIRED_COSMODIFF_BRANCH:-main}

cd "${PROJECT_DIR}"
source "${PROJECT_DIR}/scripts/nf_class_conditional_pythonpath.sh"
mkdir -p logs/nf_conditional_bias_probe_continue results/cache/python_stubs
if [[ -f "${VENV_PATH}/bin/activate" ]]; then
  source "${VENV_PATH}/bin/activate"
else
  echo "Missing CUDA training env: ${VENV_PATH}" >&2
  echo "Run first: bash scripts/setup_nf_class_conditional_env.sh" >&2
  exit 1
fi
set_nf_class_conditional_pythonpath "${PYTHON_BIN}" "${BASE_VENV_PATH}"

mapfile -t RUN_NAMES < <(
  "${PYTHON_BIN}" scripts/prepare_nf_conditional_bias_probe_continue_configs.py \
    --project-dir "${PROJECT_DIR}" \
    --continue-sweep-name "${CONTINUE_SWEEP_NAME}" \
    --additional-updates "${ADDITIONAL_UPDATES}" \
    --checkpoint-every-updates "${CHECKPOINT_EVERY_UPDATES}" \
    --print-runs
)

TASK_ID=${SLURM_ARRAY_TASK_ID:-0}
if (( TASK_ID < 0 || TASK_ID >= ${#RUN_NAMES[@]} )); then
  echo "Invalid task ${TASK_ID}; valid range is 0-$(( ${#RUN_NAMES[@]} - 1 ))" >&2
  exit 1
fi
RUN_NAME=${RUN_NAMES[$TASK_ID]}
CONFIG_PATH="${PROJECT_DIR}/local/${CONTINUE_SWEEP_NAME}/configs/${RUN_NAME}.yaml"
export CONFIG_PATH

STUB_ROOT="${PROJECT_DIR}/results/cache/python_stubs/${SLURM_JOB_ID:-manual_bias_continue}"
mkdir -p "${STUB_ROOT}/sklearn/metrics"
printf 'from . import metrics\n' > "${STUB_ROOT}/sklearn/__init__.py"
printf "def roc_curve(*args, **kwargs):\n    raise RuntimeError('sklearn.metrics.roc_curve is stubbed for cosmodiff training')\n" > "${STUB_ROOT}/sklearn/metrics/__init__.py"
"${PYTHON_BIN}" scripts/write_diffusers_runtime_sitecustomize.py "${STUB_ROOT}/sitecustomize.py"

export PYTHONPATH="${STUB_ROOT}:${COSMODIFF_DIR}:${PROJECT_DIR}:${PYTHONPATH:-}"
export COSMODIFF_STUB_SKLEARN=1
export COSMODIFF_DIR
export TORCHDYNAMO_DISABLE=1

if [[ ! -f "${COSMODIFF_DIR}/scripts/cosmodiff_train.py" ]]; then
  echo "Missing cosmodiff_train.py under ${COSMODIFF_DIR}" >&2
  exit 1
fi

PATCH_LOCK="${COSMODIFF_DIR}/.codex_conditional_bias_continue_patch.lock"
while ! mkdir "${PATCH_LOCK}" 2>/dev/null; do
  sleep 1
done
trap 'rmdir "${PATCH_LOCK}" 2>/dev/null || true' EXIT
"${PYTHON_BIN}" "${PROJECT_DIR}/scripts/patch_cosmodiff_continuous_labels.py" "${COSMODIFF_DIR}"
"${PYTHON_BIN}" "${PROJECT_DIR}/scripts/patch_cosmodiff_direct_unet_checkpoint.py" "${COSMODIFF_DIR}"
"${PYTHON_BIN}" "${PROJECT_DIR}/scripts/patch_cosmodiff_numpy_compat.py" "${COSMODIFF_DIR}"
"${PYTHON_BIN}" "${PROJECT_DIR}/scripts/patch_cosmodiff_nonfinite_loss_guard.py" "${COSMODIFF_DIR}"
rmdir "${PATCH_LOCK}" 2>/dev/null || true
trap - EXIT
trap 'rc=$?; echo "[error] ${SLURM_JOB_NAME:-nf_bias_cont} failed at line ${LINENO}: ${BASH_COMMAND} (exit ${rc})" >&2' ERR

COSMODIFF_BRANCH=$(git -C "${COSMODIFF_DIR}" rev-parse --abbrev-ref HEAD)
COSMODIFF_COMMIT=$(git -C "${COSMODIFF_DIR}" rev-parse --short HEAD)
echo "cosmodiff git branch: ${COSMODIFF_BRANCH}"
echo "cosmodiff git commit: ${COSMODIFF_COMMIT}"
if [[ "${COSMODIFF_BRANCH}" != "${REQUIRED_COSMODIFF_BRANCH}" ]]; then
  echo "Wrong cosmo_diffusion branch: expected ${REQUIRED_COSMODIFF_BRANCH}, got ${COSMODIFF_BRANCH}" >&2
  exit 1
fi

"${PYTHON_BIN}" scripts/prepare_nf_conditional_bias_probe_continue_configs.py \
  --project-dir "${PROJECT_DIR}" \
  --continue-sweep-name "${CONTINUE_SWEEP_NAME}" \
  --additional-updates "${ADDITIONAL_UPDATES}" \
  --checkpoint-every-updates "${CHECKPOINT_EVERY_UPDATES}"
"${PYTHON_BIN}" scripts/prepare_nf_conditional_bias_probe_continue_configs.py \
  --project-dir "${PROJECT_DIR}" \
  --continue-sweep-name "${CONTINUE_SWEEP_NAME}" \
  --additional-updates "${ADDITIONAL_UPDATES}" \
  --checkpoint-every-updates "${CHECKPOINT_EVERY_UPDATES}" \
  --check-only

if [[ ! -f "${CONFIG_PATH}" ]]; then
  echo "Missing continuation config: ${CONFIG_PATH}" >&2
  exit 1
fi

CHECKPOINT_DIR=$("${PYTHON_BIN}" - <<'PY'
import os
import yaml
with open(os.environ["CONFIG_PATH"]) as f:
    cfg = yaml.safe_load(f)
print(cfg["io"]["output_dir"])
PY
)
if [[ ! -d "${CHECKPOINT_DIR}" ]]; then
  echo "Missing checkpoint directory: ${CHECKPOINT_DIR}" >&2
  exit 1
fi
if ! compgen -G "${CHECKPOINT_DIR}/checkpoint-epoch-*" > /dev/null; then
  echo "No checkpoint found under ${CHECKPOINT_DIR}; refusing to start a continuation from scratch." >&2
  exit 1
fi

if [[ "${BIAS_RUN_PREFLIGHT}" == "1" ]]; then
  "${PYTHON_BIN}" scripts/check_nf_conditional_bias_probe_runtime.py \
    --config "${CONFIG_PATH}" \
    --cosmodiff-dir "${COSMODIFF_DIR}" \
    --require-cuda
fi
if [[ "${BIAS_PREFLIGHT_ONLY}" == "1" ]]; then
  echo "BIAS_PREFLIGHT_ONLY=1; exiting before training."
  exit 0
fi

echo "Continuing ${RUN_NAME}"
echo "config: ${CONFIG_PATH}"
echo "checkpoints: ${CHECKPOINT_DIR}"
echo "additional target: about ${ADDITIONAL_UPDATES} optimizer updates"
echo "checkpoint every: about ${CHECKPOINT_EVERY_UPDATES} optimizer updates"
echo "requested walltime: 18:00:00"
echo "cosmodiff: ${COSMODIFF_DIR}"

"${PYTHON_BIN}" scripts/train_cosmodiff.py \
  --config "${CONFIG_PATH}" \
  --cosmodiff-train "${COSMODIFF_DIR}/scripts/cosmodiff_train.py"
