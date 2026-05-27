#!/bin/bash
#SBATCH --job-name=nf_cond_continue
#SBATCH --partition=spgpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=80gb
#SBATCH --time=24:00:00
#SBATCH --array=0-1%1
#SBATCH -o /home/jiamingp/diffusion_models_repo/logs/nf_conditional_u128_continue400k/train_%A_%a.out
#SBATCH -e /home/jiamingp/diffusion_models_repo/logs/nf_conditional_u128_continue400k/train_%A_%a.err
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
ADDITIONAL_UPDATES=${ADDITIONAL_UPDATES:-400000}
CONTINUE_SWEEP_NAME=${CONTINUE_SWEEP_NAME:-nf_conditional_u128_continue400k}
CONT_CHECKPOINT_ROOT=${CONT_CHECKPOINT_ROOT:-/scratch/huterer_root/huterer0/jiamingp/saved_runs/nf_conditional_u128}
CLASS_CHECKPOINT_ROOT=${CLASS_CHECKPOINT_ROOT:-/scratch/huterer_root/huterer0/jiamingp/saved_runs/nf_class_conditional_u128}
REQUIRED_COSMODIFF_BRANCH=${REQUIRED_COSMODIFF_BRANCH:-main}

cd "${PROJECT_DIR}"
source "${PROJECT_DIR}/scripts/nf_class_conditional_pythonpath.sh"
mkdir -p logs/nf_conditional_u128_continue400k

if [[ -f "${VENV_PATH}/bin/activate" ]]; then
  source "${VENV_PATH}/bin/activate"
else
  echo "Missing conditional env: ${VENV_PATH}" >&2
  echo "Run first: bash scripts/setup_nf_class_conditional_env.sh" >&2
  exit 1
fi

set_nf_class_conditional_pythonpath "${PYTHON_BIN}" "${BASE_VENV_PATH}"

mapfile -t RUN_LINES < <(
  "${PYTHON_BIN}" scripts/prepare_nf_conditional_u128_continue_configs.py \
    --project-dir "${PROJECT_DIR}" \
    --additional-updates "${ADDITIONAL_UPDATES}" \
    --print-runs
)

TASK_ID=${SLURM_ARRAY_TASK_ID:-0}
if (( TASK_ID < 0 || TASK_ID >= ${#RUN_LINES[@]} )); then
  echo "Invalid task ${TASK_ID}; valid range is 0-$(( ${#RUN_LINES[@]} - 1 ))" >&2
  exit 1
fi

RUN_LINE=${RUN_LINES[$TASK_ID]}
RUN_KIND=${RUN_LINE%%$'\t'*}
RUN_NAME=${RUN_LINE#*$'\t'}
CONFIG_PATH="${PROJECT_DIR}/local/${CONTINUE_SWEEP_NAME}/configs/${RUN_NAME}.yaml"
if [[ "${RUN_KIND}" == "continuous" ]]; then
  CHECKPOINT_DIR="${CONT_CHECKPOINT_ROOT}/${RUN_NAME}_checkpoints"
else
  CHECKPOINT_DIR="${CLASS_CHECKPOINT_ROOT}/${RUN_NAME}_checkpoints"
fi
export CONFIG_PATH

STUB_ROOT="${PROJECT_DIR}/results/cache/python_stubs"
mkdir -p "${STUB_ROOT}/sklearn/metrics"
printf 'from . import metrics\n' > "${STUB_ROOT}/sklearn/__init__.py"
printf "def roc_curve(*args, **kwargs):\n    raise RuntimeError('sklearn.metrics.roc_curve is stubbed for cosmodiff training')\n" > "${STUB_ROOT}/sklearn/metrics/__init__.py"
"${PYTHON_BIN}" scripts/write_diffusers_runtime_sitecustomize.py "${STUB_ROOT}/sitecustomize.py"
export COSMODIFF_STUB_SKLEARN=1
export TORCHDYNAMO_DISABLE=1
export PYTHONPATH="${STUB_ROOT}:${COSMODIFF_DIR}:${PROJECT_DIR}:${PYTHONPATH:-}"
export COSMODIFF_DIR

if [[ ! -f "${COSMODIFF_DIR}/scripts/cosmodiff_train.py" ]]; then
  echo "Missing main cosmo_diffusion checkout: ${COSMODIFF_DIR}" >&2
  exit 1
fi

PATCH_LOCK="${COSMODIFF_DIR}/.codex_conditional_continue_patch.lock"
while ! mkdir "${PATCH_LOCK}" 2>/dev/null; do
  sleep 1
done
trap 'rmdir "${PATCH_LOCK}" 2>/dev/null || true' EXIT
"${PYTHON_BIN}" "${PROJECT_DIR}/scripts/patch_cosmodiff_package_metadata.py" "${COSMODIFF_DIR}"
"${PYTHON_BIN}" "${PROJECT_DIR}/scripts/patch_cosmodiff_direct_unet_checkpoint.py" "${COSMODIFF_DIR}"
"${PYTHON_BIN}" "${PROJECT_DIR}/scripts/patch_cosmodiff_safe_log.py" "${COSMODIFF_DIR}"
"${PYTHON_BIN}" "${PROJECT_DIR}/scripts/patch_cosmodiff_continuous_labels.py" "${COSMODIFF_DIR}"
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

# Refresh the base labels/configs, then write the continuation configs.
"${PYTHON_BIN}" scripts/prepare_nf_conditional_u128_config.py --project-dir "${PROJECT_DIR}"
"${PYTHON_BIN}" scripts/prepare_nf_class_conditional_u128_config.py --project-dir "${PROJECT_DIR}"
"${PYTHON_BIN}" scripts/prepare_nf_conditional_u128_continue_configs.py \
  --project-dir "${PROJECT_DIR}" \
  --additional-updates "${ADDITIONAL_UPDATES}"
"${PYTHON_BIN}" scripts/prepare_nf_conditional_u128_continue_configs.py \
  --project-dir "${PROJECT_DIR}" \
  --additional-updates "${ADDITIONAL_UPDATES}" \
  --run-kind "${RUN_KIND}" \
  --check-only

if [[ ! -f "${CONFIG_PATH}" ]]; then
  echo "Missing continuation config: ${CONFIG_PATH}" >&2
  exit 1
fi
if [[ ! -d "${CHECKPOINT_DIR}" ]]; then
  echo "Missing checkpoint directory: ${CHECKPOINT_DIR}" >&2
  exit 1
fi
if ! compgen -G "${CHECKPOINT_DIR}/checkpoint-epoch-*" > /dev/null; then
  echo "No checkpoint found under ${CHECKPOINT_DIR}; refusing to start a continuation from scratch." >&2
  exit 1
fi

"${PYTHON_BIN}" - <<'PY'
import inspect
import os
from pathlib import Path

import torch
import yaml
from cosmodiff import optim, utils

expected_root = Path(os.environ["COSMODIFF_DIR"]).resolve()
optim_path = Path(inspect.getsourcefile(optim)).resolve()
utils_path = Path(inspect.getsourcefile(utils)).resolve()
if expected_root not in optim_path.parents:
    raise SystemExit(f"Imported cosmodiff from {optim_path}, expected under {expected_root}")

required_train = {
    "conditioning",
    "cfg_dropout",
    "ema_sigma_rels",
    "ema_update_every",
    "ema_burn_in",
    "min_snr_gamma",
    "sigma_log_normal",
}
missing = required_train.difference(inspect.signature(optim.train).parameters)
if missing:
    raise SystemExit(f"cosmodiff.optim.train is too old; missing {sorted(missing)}")

with open(os.environ["CONFIG_PATH"]) as f:
    config = yaml.safe_load(f)
data_out = utils.parse_config_data(config)
dataset = data_out["data"] if isinstance(data_out, dict) else data_out
sample = dataset[0]
labels = sample.get("labels")
if labels is None:
    raise SystemExit("Conditional dataset did not return labels.")
conditioning = config["train"].get("conditioning")
if conditioning == "continuous":
    if not torch.is_floating_point(labels):
        raise SystemExit(f"Continuous labels must be floating point, got dtype {labels.dtype}.")
    if labels.numel() != 6:
        raise SystemExit(f"Expected 6 continuous labels, got shape {tuple(labels.shape)}.")
elif conditioning == "discrete":
    if torch.is_floating_point(labels):
        raise SystemExit(f"Discrete labels should be integer-like, got dtype {labels.dtype}.")
else:
    raise SystemExit(f"Unexpected conditioning mode: {conditioning}")

print("cosmodiff conditional support:", optim_path)
print("cosmodiff data support:", utils_path)
print("conditioning:", conditioning)
print("first label:", labels.tolist() if hasattr(labels, "tolist") else labels)
PY

echo "Continuing conditional run ${RUN_NAME}"
echo "  kind:               ${RUN_KIND}"
echo "  config:             ${CONFIG_PATH}"
echo "  checkpoints:        ${CHECKPOINT_DIR}"
echo "  additional target:  about ${ADDITIONAL_UPDATES} optimizer updates"
echo "  walltime:           24:00:00"
echo "  cosmodiff:          ${COSMODIFF_DIR}"
echo "  venv:               ${VENV_PATH}"
echo "  python:             ${PYTHON_BIN}"

"${PYTHON_BIN}" scripts/train_cosmodiff.py \
  --config "${CONFIG_PATH}" \
  --cosmodiff-train "${COSMODIFF_DIR}/scripts/cosmodiff_train.py"
