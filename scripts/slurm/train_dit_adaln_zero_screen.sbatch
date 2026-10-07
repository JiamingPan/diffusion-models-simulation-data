#!/bin/bash
#SBATCH --job-name=dit_l16_z
#SBATCH --partition=spgpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=80G
#SBATCH --time=2-00:00:00

set -euo pipefail
: "${CODE_ROOT:?}" "${EXPECTED_COMMIT:?}" "${PLAN_PATH:?}" "${PLAN_SHA256:?}"
: "${COSMODIFF_ROOT:?}" "${COSMODIFF_COMMIT:?}" "${COSMODIFF_MANIFEST:?}"
: "${COSMODIFF_MANIFEST_SHA256:?}" "${PYTHON_BIN:?}"
: "${SLURM_ARRAY_TASK_ID:?}"
test "$(git -C "$CODE_ROOT" rev-parse HEAD)" = "$EXPECTED_COMMIT"
export PYTHONHASHSEED=123 PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1
export PYTHONPATH="$COSMODIFF_ROOT:$CODE_ROOT:${PYTHONPATH:-}"
"$PYTHON_BIN" "$CODE_ROOT/scripts/verify_dit_adaln_zero_screen.py" \
  --plan "$PLAN_PATH" --plan-sha256 "$PLAN_SHA256" \
  --cosmodiff-root "$COSMODIFF_ROOT" --cosmodiff-commit "$COSMODIFF_COMMIT" \
  --cosmodiff-manifest "$COSMODIFF_MANIFEST" \
  --cosmodiff-manifest-sha256 "$COSMODIFF_MANIFEST_SHA256" \
  --index "$SLURM_ARRAY_TASK_ID" --require-data-receipt
readarray -t fields < <("$PYTHON_BIN" - "$PLAN_PATH" "$SLURM_ARRAY_TASK_ID" <<'PY'
import json,sys
r=json.load(open(sys.argv[1]))['runs'][int(sys.argv[2])]
print(r['config'])
print(r['run_name'])
PY
)
config=${fields[0]}
name=${fields[1]}
echo "TRAINING $name FROM FRESH SEED 123"
"$PYTHON_BIN" "$CODE_ROOT/scripts/train_cosmodiff_seeded.py" \
  --seed 123 --config "$config" \
  --cosmodiff-train "$COSMODIFF_ROOT/scripts/cosmodiff_train.py"
"$PYTHON_BIN" "$CODE_ROOT/scripts/finalize_dit_adaln_zero_training.py" \
  --plan "$PLAN_PATH" --plan-sha256 "$PLAN_SHA256" --index "$SLURM_ARRAY_TASK_ID"
echo "TRAINING COMMAND COMPLETED: $name"
