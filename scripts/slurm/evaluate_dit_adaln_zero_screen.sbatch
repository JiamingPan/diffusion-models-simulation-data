#!/bin/bash
#SBATCH --job-name=dit_l16_zeval
#SBATCH --partition=spgpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=80G
#SBATCH --time=01:00:00

set -euo pipefail
: "${CODE_ROOT:?}" "${EXPECTED_COMMIT:?}" "${PLAN_PATH:?}" "${PLAN_SHA256:?}"
: "${SCREEN_ROOT:?}" "${PYTHON_BIN:?}" "${SLURM_ARRAY_TASK_ID:?}"
test "$(git -C "$CODE_ROOT" rev-parse HEAD)" = "$EXPECTED_COMMIT"
test -z "$(git -C "$CODE_ROOT" status --porcelain)"
export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1
export PYTHONPATH="$CODE_ROOT:${PYTHONPATH:-}"
"$PYTHON_BIN" "$CODE_ROOT/scripts/evaluate_dit_adaln_zero_screen_results.py" \
  --plan "$PLAN_PATH" --plan-sha256 "$PLAN_SHA256" \
  --index "$SLURM_ARRAY_TASK_ID" --eval-root "$CODE_ROOT" \
  --device cuda --out "$SCREEN_ROOT/evaluation/task_${SLURM_ARRAY_TASK_ID}.json"
