#!/bin/bash
#SBATCH --job-name=nf_u64_sample64
#SBATCH --partition=spgpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=80gb
#SBATCH --time=02:00:00
#SBATCH -o /home/jiamingp/diffusion_models_repo/logs/normalization_fixes/nf_u64_sample64_%j.out
#SBATCH -e /home/jiamingp/diffusion_models_repo/logs/normalization_fixes/nf_u64_sample64_%j.err
#SBATCH -A huterer0

set -euo pipefail

PROJECT_DIR=/home/jiamingp/diffusion_models_repo
COSMODIFF_DIR=${PROJECT_DIR}/cosmo_diffusion
CHECKPOINT_ROOT=/scratch/huterer_root/huterer0/jiamingp/saved_runs/normalization_fixes
CONFIG_DIR=${PROJECT_DIR}/local/normalization_fixes/configs
OUT_DIR=${PROJECT_DIR}/results/tables/samples
PYTHON_BIN=/sw/pkgs/arc/python3.10-anaconda/2023.03/bin/python

RUN_BASE=nf_u64_n500_e100
N_SAMPLES=64
BATCH_SIZE=8
SEED=123

cd "${PROJECT_DIR}"

mkdir -p "${OUT_DIR}" "${PROJECT_DIR}/logs/normalization_fixes"
export PYTHONPATH="${COSMODIFF_DIR}:${PROJECT_DIR}:${PYTHONPATH:-}"

"${PYTHON_BIN}" scripts/sample_normalization_fixes.py \
  --checkpoint-root "${CHECKPOINT_ROOT}" \
  --config-dir "${CONFIG_DIR}" \
  --output-dir "${OUT_DIR}" \
  --run-base "${RUN_BASE}" \
  --num-samples "${N_SAMPLES}" \
  --batch-size "${BATCH_SIZE}" \
  --seed "${SEED}" \
  --device cuda \
  --python-bin "${PYTHON_BIN}"
