#!/bin/bash
#SBATCH --job-name=repro_train_u64_u128
#SBATCH --partition=spgpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=80gb
#SBATCH --time=24:00:00
#SBATCH --array=0-19%4
#SBATCH -o /home/jiamingp/diffusion_models_repo/logs/fig1_lh/train_%A_%a.out
#SBATCH -e /home/jiamingp/diffusion_models_repo/logs/fig1_lh/train_%A_%a.err
#SBATCH -A huterer0

set -euo pipefail

PROJECT_DIR=${PROJECT_DIR:-/home/jiamingp/diffusion_models_repo}
COSMODIFF_DIR=${COSMODIFF_DIR:-/home/jiamingp/Diffusion_model/cosmo_diffusion_normalization_fixes}
VENV_PATH=${VENV_PATH:-/home/jiamingp/venvs/cosmodiff_nf}
CONFIG_DIR=${CONFIG_DIR:-${PROJECT_DIR}/local/fig1_lh/configs}
CHECKPOINT_ROOT=${CHECKPOINT_ROOT:-/scratch/huterer_root/huterer0/jiamingp/saved_runs/fig1_lh}
PYTHON_BIN=${PYTHON_BIN:-python}
SKIP_IF_ANY_CHECKPOINT=${SKIP_IF_ANY_CHECKPOINT:-0}

RUNS=(
  fig1_lh_u64_d2p06
  fig1_lh_u64_d2p07
  fig1_lh_u64_d2p08
  fig1_lh_u64_d2p09
  fig1_lh_u64_d2p10
  fig1_lh_u64_d2p11
  fig1_lh_u64_d2p12
  fig1_lh_u64_d2p13
  fig1_lh_u64_d2p14
  fig1_lh_u64_d2p15_full
  fig1_lh_u128_d2p06
  fig1_lh_u128_d2p07
  fig1_lh_u128_d2p08
  fig1_lh_u128_d2p09
  fig1_lh_u128_d2p10
  fig1_lh_u128_d2p11
  fig1_lh_u128_d2p12
  fig1_lh_u128_d2p13
  fig1_lh_u128_d2p14
  fig1_lh_u128_d2p15_full
)

RUN_NAME=${RUNS[$SLURM_ARRAY_TASK_ID]}
CONFIG_PATH=${CONFIG_DIR}/${RUN_NAME}.yaml
CHECKPOINT_DIR=${CHECKPOINT_ROOT}/${RUN_NAME}_checkpoints

cd "${PROJECT_DIR}"
mkdir -p logs/fig1_lh "${CHECKPOINT_DIR}"
if [[ -f "${VENV_PATH}/bin/activate" ]]; then
  source "${VENV_PATH}/bin/activate"
fi
export PYTHONPATH="${COSMODIFF_DIR}:${PROJECT_DIR}:${PYTHONPATH:-}"
export COSMODIFF_DIR

if [[ ! -f "${CONFIG_PATH}" ]]; then
  echo "Missing config: ${CONFIG_PATH}" >&2
  exit 1
fi

if [[ ! -f "${COSMODIFF_DIR}/scripts/cosmodiff_train.py" ]]; then
  echo "Missing normalization-fix cosmo_diffusion checkout: ${COSMODIFF_DIR}" >&2
  exit 1
fi

"${PYTHON_BIN}" - <<'PY'
import sys
from pathlib import Path
from simdiff_eval.torch_compat import install_torch_backend_compat
install_torch_backend_compat(entry_point="train_repro_u64_u128_array")
from cosmodiff import utils
text = Path(utils.__file__).read_text()
if "tanh_norm" not in text and "self.method in ['tanh']" not in text:
    raise SystemExit(f"cosmodiff at {utils.__file__} does not support tanh normalization")
print("cosmodiff tanh support:", utils.__file__)
PY

"${PYTHON_BIN}" scripts/prepare_repro_u64_u128_configs.py \
  --project-dir "${PROJECT_DIR}" \
  --check-only \
  --run-name "${RUN_NAME}"

if [[ "${SKIP_IF_ANY_CHECKPOINT}" == "1" ]] && compgen -G "${CHECKPOINT_DIR}/checkpoint-*" > /dev/null; then
  echo "Skipping ${RUN_NAME}; checkpoint already exists under ${CHECKPOINT_DIR}"
  exit 0
fi

echo "Training ${RUN_NAME}"
echo "  config:     ${CONFIG_PATH}"
echo "  checkpoints:${CHECKPOINT_DIR}"
echo "  python:     ${PYTHON_BIN}"

"${PYTHON_BIN}" scripts/train_cosmodiff.py \
  --config "${CONFIG_PATH}" \
  --cosmodiff-train "${COSMODIFF_DIR}/scripts/cosmodiff_train.py"
