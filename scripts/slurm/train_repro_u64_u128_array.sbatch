#!/bin/bash
#SBATCH --job-name=repro_train_u64_u128
#SBATCH --partition=spgpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=80gb
#SBATCH --time=48:00:00
#SBATCH --array=0-19%4
#SBATCH -o /home/jiamingp/diffusion_models_repo/logs/fig1_lh/train_%A_%a.out
#SBATCH -e /home/jiamingp/diffusion_models_repo/logs/fig1_lh/train_%A_%a.err
#SBATCH -A huterer0

set -euo pipefail

PROJECT_DIR=${PROJECT_DIR:-/home/jiamingp/diffusion_models_repo}
COSMODIFF_DIR=${COSMODIFF_DIR:-${PROJECT_DIR}/cosmo_diffusion}
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
export PYTHONPATH="${COSMODIFF_DIR}:${PROJECT_DIR}:${PYTHONPATH:-}"

if [[ ! -f "${CONFIG_PATH}" ]]; then
  echo "Missing config: ${CONFIG_PATH}" >&2
  exit 1
fi

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
