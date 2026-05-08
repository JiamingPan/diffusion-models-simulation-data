#!/bin/bash
#SBATCH --job-name=nf_sweep_train
#SBATCH --partition=spgpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=80gb
#SBATCH --time=24:00:00
#SBATCH --array=0-15%4
#SBATCH -o /home/jiamingp/diffusion_models_repo/logs/nf_sweep/train_%A_%a.out
#SBATCH -e /home/jiamingp/diffusion_models_repo/logs/nf_sweep/train_%A_%a.err
#SBATCH -A huterer0

set -euo pipefail

PROJECT_DIR=${PROJECT_DIR:-/home/jiamingp/diffusion_models_repo}
COSMODIFF_DIR=${COSMODIFF_DIR:-/home/jiamingp/Diffusion_model/cosmo_diffusion_normalization_fixes}
VENV_PATH=${VENV_PATH:-/home/jiamingp/venvs/cosmodiff_nf}
CONFIG_DIR=${CONFIG_DIR:-${PROJECT_DIR}/local/nf_sweep/configs}
CHECKPOINT_ROOT=${CHECKPOINT_ROOT:-/scratch/huterer_root/huterer0/jiamingp/saved_runs/nf_sweep}
PYTHON_BIN=${PYTHON_BIN:-python}
SKIP_IF_ANY_CHECKPOINT=${SKIP_IF_ANY_CHECKPOINT:-0}
REQUIRED_COSMODIFF_BRANCH=${REQUIRED_COSMODIFF_BRANCH:-normalization_fixes}
ALLOW_COSMODIFF_BRANCH_MISMATCH=${ALLOW_COSMODIFF_BRANCH_MISMATCH:-0}

RUNS=(
  nf_sweep_u64_n500_e100_base_ema
  nf_sweep_u64_n500_e100_zero_snr
  nf_sweep_u64_n500_e100_beta_linear
  nf_sweep_u64_n500_e100_beta_cosine
  nf_sweep_u64_n500_e100_pred_eps
  nf_sweep_u64_n500_e100_sigma_edm
  nf_sweep_u64_n500_e100_sigma_edm2
  nf_sweep_u64_n500_e100_minsnr5
  nf_sweep_u128_n500_e100_base_ema
  nf_sweep_u128_n500_e100_zero_snr
  nf_sweep_u128_n500_e100_beta_linear
  nf_sweep_u128_n500_e100_beta_cosine
  nf_sweep_u128_n500_e100_pred_eps
  nf_sweep_u128_n500_e100_sigma_edm
  nf_sweep_u128_n500_e100_sigma_edm2
  nf_sweep_u128_n500_e100_minsnr5
)

RUN_NAME=${RUNS[$SLURM_ARRAY_TASK_ID]}
CONFIG_PATH=${CONFIG_DIR}/${RUN_NAME}.yaml
CHECKPOINT_DIR=${CHECKPOINT_ROOT}/${RUN_NAME}_checkpoints

cd "${PROJECT_DIR}"
mkdir -p logs/nf_sweep "${CHECKPOINT_DIR}"
if [[ -f "${VENV_PATH}/bin/activate" ]]; then
  source "${VENV_PATH}/bin/activate"
fi
export PYTHONPATH="${COSMODIFF_DIR}:${PROJECT_DIR}:${PYTHONPATH:-}"
export COSMODIFF_DIR

if [[ ! -f "${CONFIG_PATH}" ]]; then
  echo "Missing config: ${CONFIG_PATH}" >&2
  echo "Run first: python scripts/prepare_nf_sweep_configs.py --project-dir ${PROJECT_DIR}" >&2
  exit 1
fi

if [[ ! -f "${COSMODIFF_DIR}/scripts/cosmodiff_train.py" ]]; then
  echo "Missing normalization-fix cosmo_diffusion checkout: ${COSMODIFF_DIR}" >&2
  exit 1
fi

if [[ -d "${COSMODIFF_DIR}/.git" ]]; then
  COSMODIFF_BRANCH=$(git -C "${COSMODIFF_DIR}" rev-parse --abbrev-ref HEAD)
  COSMODIFF_COMMIT=$(git -C "${COSMODIFF_DIR}" rev-parse --short HEAD)
  echo "cosmodiff git branch: ${COSMODIFF_BRANCH}"
  echo "cosmodiff git commit: ${COSMODIFF_COMMIT}"
  if [[ -n "${REQUIRED_COSMODIFF_BRANCH}" && "${COSMODIFF_BRANCH}" != "${REQUIRED_COSMODIFF_BRANCH}" && "${ALLOW_COSMODIFF_BRANCH_MISMATCH}" != "1" ]]; then
    echo "Wrong cosmo_diffusion branch: expected ${REQUIRED_COSMODIFF_BRANCH}, got ${COSMODIFF_BRANCH}" >&2
    echo "Set ALLOW_COSMODIFF_BRANCH_MISMATCH=1 only if you intentionally use a detached/test checkout." >&2
    exit 1
  fi
else
  echo "Cannot verify cosmo_diffusion git branch because ${COSMODIFF_DIR}/.git is missing." >&2
  echo "Continuing with Python import and feature checks for this non-git checkout." >&2
fi

"${PYTHON_BIN}" - <<'PY'
import inspect
import os
from pathlib import Path
from diffusers import DDPMScheduler
from cosmodiff import optim
from cosmodiff.transform import Normalization

expected_root = Path(os.environ["COSMODIFF_DIR"]).resolve()
optim_path = Path(inspect.getsourcefile(optim)).resolve()
if expected_root not in optim_path.parents:
    raise SystemExit(
        f"Imported cosmodiff from {optim_path}, expected it under {expected_root}"
    )

required = {"ema_sigma_rels", "ema_update_every", "ema_burn_in", "min_snr_gamma", "sigma_log_normal"}
missing = required.difference(inspect.signature(optim.train).parameters)
if missing:
    raise SystemExit(f"cosmodiff.optim.train is too old; missing {sorted(missing)}")

DPM = DDPMScheduler(
    num_train_timesteps=500,
    beta_schedule="sigmoid",
    prediction_type="v_prediction",
    rescale_betas_zero_snr=True,
    timestep_spacing="trailing",
)
if DPM.config.prediction_type != "v_prediction":
    raise SystemExit("DDPMScheduler did not retain v_prediction config")

try:
    import ema_pytorch  # noqa: F401
except Exception as exc:
    raise SystemExit(f"ema_pytorch is required for this sweep: {exc}") from exc

print("cosmodiff sweep support:", inspect.getsourcefile(optim.train))
print("normalization class:", Normalization)
PY

"${PYTHON_BIN}" scripts/prepare_nf_sweep_configs.py \
  --project-dir "${PROJECT_DIR}" \
  --check-only \
  --run-name "${RUN_NAME}"

if [[ "${SKIP_IF_ANY_CHECKPOINT}" == "1" ]] && compgen -G "${CHECKPOINT_DIR}/checkpoint-*" > /dev/null; then
  echo "Skipping ${RUN_NAME}; checkpoint already exists under ${CHECKPOINT_DIR}"
  exit 0
fi

echo "Training ${RUN_NAME}"
echo "  config:      ${CONFIG_PATH}"
echo "  checkpoints: ${CHECKPOINT_DIR}"
echo "  cosmodiff:   ${COSMODIFF_DIR}"
echo "  python:      ${PYTHON_BIN}"

"${PYTHON_BIN}" scripts/train_cosmodiff.py \
  --config "${CONFIG_PATH}" \
  --cosmodiff-train "${COSMODIFF_DIR}/scripts/cosmodiff_train.py"
