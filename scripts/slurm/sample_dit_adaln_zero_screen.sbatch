#!/bin/bash
#SBATCH --job-name=dit_l16_zsample
#SBATCH --partition=spgpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=80G
#SBATCH --time=01:00:00

set -euo pipefail
: "${CODE_ROOT:?}" "${EXPECTED_COMMIT:?}" "${PLAN_PATH:?}" "${PLAN_SHA256:?}"
: "${COSMODIFF_ROOT:?}" "${COSMODIFF_COMMIT:?}" "${COSMODIFF_MANIFEST:?}"
: "${COSMODIFF_MANIFEST_SHA256:?}" "${PYTHON_BIN:?}"
: "${SLURM_ARRAY_TASK_ID:?}"
test "$(git -C "$CODE_ROOT" rev-parse HEAD)" = "$EXPECTED_COMMIT"
export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1
export PYTHONPATH="$COSMODIFF_ROOT:$CODE_ROOT:${PYTHONPATH:-}"
"$PYTHON_BIN" "$CODE_ROOT/scripts/verify_dit_adaln_zero_screen.py" \
  --plan "$PLAN_PATH" --plan-sha256 "$PLAN_SHA256" \
  --cosmodiff-root "$COSMODIFF_ROOT" --cosmodiff-commit "$COSMODIFF_COMMIT" \
  --cosmodiff-manifest "$COSMODIFF_MANIFEST" \
  --cosmodiff-manifest-sha256 "$COSMODIFF_MANIFEST_SHA256" \
  --index "$SLURM_ARRAY_TASK_ID" --allow-checkpoint --require-data-receipt
readarray -t fields < <("$PYTHON_BIN" - "$PLAN_PATH" "$SLURM_ARRAY_TASK_ID" <<'PY'
import json,sys
r=json.load(open(sys.argv[1]))['runs'][int(sys.argv[2])]
print(r['config']); print(r['checkpoint_dir']); print(r['sample_path'].format(seed=123,sample_label='dpm50_n512'))
PY
)
config=${fields[0]}
checkpoint=${fields[1]}
output=${fields[2]}
if [[ -e "$output" ]]; then echo "Refusing to overwrite $output" >&2; exit 1; fi
mkdir -p "$(dirname "$output")"
"$PYTHON_BIN" "$CODE_ROOT/scripts/sample_cosmodiff.py" \
  --checkpoint "$checkpoint" --config "$config" --output "$output" \
  --num-samples 512 --batch-size 4 --image-size 128 \
  --scheduler DPMSolverMultistepScheduler --num-steps 50 \
  --seed 123 --class-label 0 --device cuda
"$PYTHON_BIN" "$CODE_ROOT/scripts/finalize_dit_adaln_zero_samples.py" \
  --plan "$PLAN_PATH" --plan-sha256 "$PLAN_SHA256" --index "$SLURM_ARRAY_TASK_ID"
