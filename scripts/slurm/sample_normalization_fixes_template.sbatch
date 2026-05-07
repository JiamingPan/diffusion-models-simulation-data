#!/bin/bash
#SBATCH --job-name=nf_sample
#SBATCH --partition=<partition>
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64gb
#SBATCH --time=02:00:00
#SBATCH -o logs/normalization_fixes/%x_%j.out
#SBATCH -e logs/normalization_fixes/%x_%j.err
#SBATCH -A <account>

set -euo pipefail

# Copy this template into local/ and edit the paths/account/partition there.
# local/ is ignored by git.
PROJECT_DIR=${PROJECT_DIR:-/path/to/diffusion-models-simulation-data}
COSMODIFF_DIR=${COSMODIFF_DIR:-${PROJECT_DIR}/cosmo_diffusion}
CHECKPOINT_ROOT=${CHECKPOINT_ROOT:-/path/to/saved_runs/normalization_fixes}
CONFIG_DIR=${CONFIG_DIR:-${PROJECT_DIR}/local/normalization_fixes/configs}
PYTHON_BIN=${PYTHON_BIN:-python}

# Smoke test defaults. For final evaluation, set NUM_SAMPLES=512 and
# BATCH_SIZE=16 or the largest value that fits on the GPU.
NUM_SAMPLES=${NUM_SAMPLES:-1}
BATCH_SIZE=${BATCH_SIZE:-1}
SEED=${SEED:-123}
RUN_BASE=${RUN_BASE:-}

cd "${PROJECT_DIR}"
mkdir -p logs/normalization_fixes results/tables/samples

export PYTHONPATH="${COSMODIFF_DIR}:${PROJECT_DIR}:${PYTHONPATH:-}"

RUN_BASE_ARGS=()
if [[ -n "${RUN_BASE}" ]]; then
  RUN_BASE_ARGS+=(--run-base "${RUN_BASE}")
fi

"${PYTHON_BIN}" scripts/sample_normalization_fixes.py \
  --checkpoint-root "${CHECKPOINT_ROOT}" \
  --config-dir "${CONFIG_DIR}" \
  --output-dir "${PROJECT_DIR}/results/tables/samples" \
  "${RUN_BASE_ARGS[@]}" \
  --num-samples "${NUM_SAMPLES}" \
  --batch-size "${BATCH_SIZE}" \
  --seed "${SEED}" \
  --device cuda \
  --python-bin "${PYTHON_BIN}"
