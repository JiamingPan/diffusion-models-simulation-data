#!/bin/bash
# Submit the staged L16 high-N zero-init screen after explicit approval.
set -euo pipefail
trap 'echo "STOPPED at line $LINENO. Printed job IDs, if any, may still be live; do not rerun automatically." >&2' ERR

: "${CODE_ROOT:?}" "${EXPECTED_COMMIT:?}" "${PLAN_PATH:?}" "${PLAN_SHA256:?}"
: "${COSMODIFF_ROOT:?}" "${COSMODIFF_COMMIT:?}" "${COSMODIFF_MANIFEST:?}"
: "${COSMODIFF_MANIFEST_SHA256:?}" "${PYTHON_BIN:?}" "${SCREEN_ROOT:?}"
: "${SLURM_ACCOUNT:?}" "${SCRATCH_ACCOUNT:?}"

test "$(git -C "$CODE_ROOT" rev-parse HEAD)" = "$EXPECTED_COMMIT"
test -z "$(git -C "$CODE_ROOT" status --porcelain)"
test "$(sha256sum "$PLAN_PATH" | awk '{print $1}')" = "$PLAN_SHA256"
test "$(sha256sum "$COSMODIFF_MANIFEST" | awk '{print $1}')" = "$COSMODIFF_MANIFEST_SHA256"
test ! -e "$SCREEN_ROOT/data_preflight.json"
test ! -e "$SCREEN_ROOT/launch_jobs.json"
scratch-quota "$SCRATCH_ACCOUNT" | "$PYTHON_BIN" "$CODE_ROOT/scripts/check_gl_scratch_quota.py" \
  --min-gib 1024 --min-files 5000

mkdir "$SCREEN_ROOT/launch_l16_highn_v1"
mkdir -p "$SCREEN_ROOT/logs"

data_job=$(sbatch --parsable --account="$SLURM_ACCOUNT" --partition=standard \
  --job-name=dit_l16_zdata --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=128G --time=02:00:00 \
  --no-requeue --chdir="$CODE_ROOT" --export=ALL \
  --output="$SCREEN_ROOT/logs/data_%j.out" --error="$SCREEN_ROOT/logs/data_%j.err" \
  "$CODE_ROOT/scripts/slurm/preflight_dit_adaln_zero_data.sbatch")
data_job=${data_job%%;*}
echo "DATA PREFLIGHT: $data_job"

smoke_job=$(sbatch --parsable --account="$SLURM_ACCOUNT" --partition=spgpu \
  --job-name=dit_l16_zsmoke --nodes=1 --ntasks=1 --gres=gpu:1 --cpus-per-task=4 \
  --mem=80G --time=00:20:00 --no-requeue --dependency="afterok:$data_job" \
  --kill-on-invalid-dep=yes --chdir="$CODE_ROOT" --export=ALL \
  --output="$SCREEN_ROOT/logs/smoke_%j.out" --error="$SCREEN_ROOT/logs/smoke_%j.err" \
  "$CODE_ROOT/scripts/slurm/smoke_dit_adaln_zero_screen.sbatch")
smoke_job=${smoke_job%%;*}
echo "GPU SMOKE: $smoke_job"

train_job=$(sbatch --parsable --account="$SLURM_ACCOUNT" --partition=spgpu \
  --job-name=dit_l16_ztrain --array=0-3%2 --nodes=1 --ntasks=1 --gres=gpu:1 \
  --cpus-per-task=4 --mem=80G --time=2-00:00:00 --no-requeue \
  --dependency="afterok:$smoke_job" --kill-on-invalid-dep=yes \
  --chdir="$CODE_ROOT" --export=ALL \
  --output="$SCREEN_ROOT/logs/train_%A_%a.out" --error="$SCREEN_ROOT/logs/train_%A_%a.err" \
  "$CODE_ROOT/scripts/slurm/train_dit_adaln_zero_screen.sbatch")
train_job=${train_job%%;*}
echo "TRAIN ARRAY: $train_job (0-3%2)"

sample_job=$(sbatch --parsable --account="$SLURM_ACCOUNT" --partition=spgpu \
  --job-name=dit_l16_zsample --array=0-3%2 --nodes=1 --ntasks=1 --gres=gpu:1 \
  --cpus-per-task=4 --mem=80G --time=01:00:00 --no-requeue \
  --dependency="aftercorr:$train_job" --kill-on-invalid-dep=yes \
  --chdir="$CODE_ROOT" --export=ALL \
  --output="$SCREEN_ROOT/logs/sample_%A_%a.out" --error="$SCREEN_ROOT/logs/sample_%A_%a.err" \
  "$CODE_ROOT/scripts/slurm/sample_dit_adaln_zero_screen.sbatch")
sample_job=${sample_job%%;*}
echo "SAMPLE ARRAY: $sample_job (corresponding task after each successful training task)"

"$PYTHON_BIN" - "$SCREEN_ROOT/launch_jobs.json" "$data_job" "$smoke_job" "$train_job" "$sample_job" <<'PY'
import json, sys
path, data, smoke, train, sample = sys.argv[1:]
with open(path + ".pending", "w") as handle:
    json.dump({"data_preflight": data, "gpu_smoke": smoke,
               "train_array": train, "sample_array": sample}, handle, indent=2)
    handle.write("\n")
import os
os.replace(path + ".pending", path)
PY

squeue -j "$data_job,$smoke_job,$train_job,$sample_job"
