#!/bin/bash
#SBATCH --job-name=dit_late
#SBATCH --partition=spgpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=80G
#SBATCH --time=02:00:00
#SBATCH --array=0-5%2

set -euo pipefail
: "${PROJECT_DIR:?}" "${CODE_ROOT:?}" "${EXPECTED_COMMIT:?}"
: "${COSMODIFF_PIN_ROOT:?}" "${COSMODIFF_PIN_MANIFEST:?}"
: "${EXPECTED_COSMODIFF_BASE_REVISION:?}"
PYTHON_BIN=${PYTHON_BIN:-/home/jiamingp/venvs/cosmodiff_nf_class/bin/python}
RUNTIME_CODE_ROOT=${RUNTIME_CODE_ROOT:-/scratch/huterer_root/huterer0/jiamingp/iaifi_poster_code_555f350}
RUNTIME_CODE_REVISION=555f350f82c913ff06150c96e66709719856cd83
VENV_PATH=/home/jiamingp/venvs/cosmodiff_nf_class
NUM_SAMPLES=128

test "$(git -C "$CODE_ROOT" rev-parse HEAD)" = "$EXPECTED_COMMIT"
test -z "$(git -C "$CODE_ROOT" status --porcelain)"
test "$(git -C "$RUNTIME_CODE_ROOT" rev-parse HEAD)" = "$RUNTIME_CODE_REVISION"
test -z "$(git -C "$RUNTIME_CODE_ROOT" status --porcelain)"

export PYTHONNOUSERSITE=1
export PYTHONPATH="$COSMODIFF_PIN_ROOT/seed_restart_runtime:$RUNTIME_CODE_ROOT:$COSMODIFF_PIN_ROOT"
export COSMODIFF_DIR="$COSMODIFF_PIN_ROOT"
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4

cd "$RUNTIME_CODE_ROOT"
"$PYTHON_BIN" "$RUNTIME_CODE_ROOT/scripts/verify_cosmodiff_seed_restart_runtime.py" \
  "$COSMODIFF_PIN_ROOT" --manifest "$COSMODIFF_PIN_MANIFEST" \
  --expected-base-revision "$EXPECTED_COSMODIFF_BASE_REVISION" \
  --python-bin "$PYTHON_BIN" --code-root "$RUNTIME_CODE_ROOT" \
  --expected-torch-prefix "$VENV_PATH" \
  --incompatible-python-path /home/jiamingp/venvs/cosmodiff_nf/lib/python3.10/site-packages \
  --incompatible-python-path /sw/pkgs/arc/python3.10-anaconda/2023.03/lib/python3.10/site-packages \
  --patch-script "$RUNTIME_CODE_ROOT/scripts/patch_cosmodiff_package_metadata.py" \
  --patch-script "$RUNTIME_CODE_ROOT/scripts/patch_cosmodiff_constant_label.py" \
  --patch-script "$RUNTIME_CODE_ROOT/scripts/patch_cosmodiff_dit_class_labels.py" \
  --patch-script "$RUNTIME_CODE_ROOT/scripts/patch_cosmodiff_checkpoint_state.py"

if [[ "${PREFLIGHT_ONLY:-0}" == 1 ]]; then
  for model_index in 0 2 3; do
    "$PYTHON_BIN" "$CODE_ROOT/scripts/resolve_dit_high_noise_models.py" \
      --project-dir "$PROJECT_DIR" --index "$model_index" >/dev/null
  done
  echo 'LATE-START PREFLIGHT PASSED; NO SAMPLING'
  exit 0
fi

case "${SLURM_ARRAY_TASK_ID:?}" in
  0) model_index=0; t_start=499 ;;
  1) model_index=0; t_start=399 ;;
  2) model_index=2; t_start=499 ;;
  3) model_index=2; t_start=399 ;;
  4) model_index=3; t_start=499 ;;
  5) model_index=3; t_start=399 ;;
  *) echo "invalid array task" >&2; exit 2 ;;
esac

model_json=$("$PYTHON_BIN" "$CODE_ROOT/scripts/resolve_dit_high_noise_models.py" \
  --project-dir "$PROJECT_DIR" --index "$model_index")
readarray -t model_fields < <("$PYTHON_BIN" - "$model_json" <<'PY'
import json
import sys
row = json.loads(sys.argv[1])
print(row["name"])
print(row["checkpoint"])
print(row["config"])
PY
)
model_name=${model_fields[0]}
checkpoint=${model_fields[1]}
config=${model_fields[2]}
output_dir="$PROJECT_DIR/results/dit_l16_late_start_screen"
output="$output_dir/${model_name}_t${t_start}_n${NUM_SAMPLES}.npz"
mkdir -p "$output_dir"

"$PYTHON_BIN" "$CODE_ROOT/scripts/sample_late_start.py" \
  --checkpoint "$checkpoint" \
  --config "$config" \
  --t-start "$t_start" \
  --num-samples "$NUM_SAMPLES" \
  --batch-size 8 \
  --seed 123 \
  --scheduler DPMSolverMultistepScheduler \
  --num-steps 50 \
  --class-label 0 \
  --eval-root "$RUNTIME_CODE_ROOT" \
  --output "$output"
