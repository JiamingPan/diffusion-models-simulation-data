#!/bin/bash
#SBATCH --job-name=nf_cond_u128
#SBATCH --partition=spgpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=80gb
#SBATCH --time=24:00:00
#SBATCH -o /home/jiamingp/diffusion_models_repo/logs/nf_conditional_u128/train_%j.out
#SBATCH -e /home/jiamingp/diffusion_models_repo/logs/nf_conditional_u128/train_%j.err
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
VENV_PATH=${VENV_PATH:-/home/jiamingp/venvs/cosmodiff_nf}
PYTHON_BIN=${PYTHON_BIN:-python}
REQUIRED_COSMODIFF_BRANCH=${REQUIRED_COSMODIFF_BRANCH:-main}

cd "${PROJECT_DIR}"
mkdir -p logs/nf_conditional_u128
if [[ -f "${VENV_PATH}/bin/activate" ]]; then
  source "${VENV_PATH}/bin/activate"
fi

RUN_NAME=$("${PYTHON_BIN}" scripts/prepare_nf_conditional_u128_config.py --project-dir "${PROJECT_DIR}" --print-runs)
CONFIG_PATH="${PROJECT_DIR}/local/nf_conditional_u128/configs/${RUN_NAME}.yaml"
export CONFIG_PATH

STUB_ROOT="${PROJECT_DIR}/results/cache/python_stubs"
mkdir -p "${STUB_ROOT}/sklearn/metrics"
printf 'from . import metrics\n' > "${STUB_ROOT}/sklearn/__init__.py"
printf "def roc_curve(*args, **kwargs):\n    raise RuntimeError('sklearn.metrics.roc_curve is stubbed for cosmodiff training')\n" > "${STUB_ROOT}/sklearn/metrics/__init__.py"
export COSMODIFF_STUB_SKLEARN=1
export TORCHDYNAMO_DISABLE=1
export PYTHONPATH="${STUB_ROOT}:${COSMODIFF_DIR}:${PROJECT_DIR}:${PYTHONPATH:-}"
export COSMODIFF_DIR

if [[ ! -f "${CONFIG_PATH}" ]]; then
  echo "Missing config: ${CONFIG_PATH}" >&2
  echo "Run first: python scripts/prepare_nf_conditional_u128_config.py --project-dir ${PROJECT_DIR}" >&2
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
"${PYTHON_BIN}" "${PROJECT_DIR}/scripts/patch_cosmodiff_continuous_labels.py" "${COSMODIFF_DIR}"
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

"${PYTHON_BIN}" - <<'PY'
import inspect
import os
import yaml
import torch
from pathlib import Path
from simdiff_eval.torch_compat import install_torch_backend_compat
install_torch_backend_compat(entry_point="train_nf_conditional_u128")
from cosmodiff import optim, utils

expected_root = Path(os.environ["COSMODIFF_DIR"]).resolve()
optim_path = Path(inspect.getsourcefile(optim)).resolve()
utils_path = Path(inspect.getsourcefile(utils)).resolve()
if expected_root not in optim_path.parents:
    raise SystemExit(f"Imported cosmodiff from {optim_path}, expected under {expected_root}")

required_train = {"conditioning", "cfg_dropout", "ema_sigma_rels", "ema_update_every", "ema_burn_in", "min_snr_gamma", "sigma_log_normal"}
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
if not torch.is_floating_point(labels):
    raise SystemExit(f"Continuous labels must be floating point, got dtype {labels.dtype}.")
if labels.numel() != 6:
    raise SystemExit(f"Expected 6 conditioning parameters, got shape {tuple(labels.shape)}.")

print("cosmodiff conditional support:", optim_path)
print("cosmodiff data support:", utils_path)
print("first normalized label:", labels.tolist())
PY

"${PYTHON_BIN}" scripts/prepare_nf_conditional_u128_config.py \
  --project-dir "${PROJECT_DIR}" \
  --check-only

echo "Training conditional ${RUN_NAME}"
echo "  config:    ${CONFIG_PATH}"
echo "  cosmodiff: ${COSMODIFF_DIR}"
echo "  python:    ${PYTHON_BIN}"

"${PYTHON_BIN}" scripts/train_cosmodiff.py \
  --config "${CONFIG_PATH}" \
  --cosmodiff-train "${COSMODIFF_DIR}/scripts/cosmodiff_train.py"
