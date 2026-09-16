#!/bin/bash
#SBATCH --job-name=dit_hi_noise
#SBATCH --partition=spgpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=80G
#SBATCH --time=02:00:00
#SBATCH --array=0-3%2

set -euo pipefail
: "${PROJECT_DIR:?}" "${CODE_ROOT:?}" "${EXPECTED_COMMIT:?}"
: "${COSMODIFF_PIN_ROOT:?}" "${COSMODIFF_PIN_MANIFEST:?}"
: "${EXPECTED_COSMODIFF_BASE_REVISION:?}"
PYTHON_BIN=${PYTHON_BIN:-/home/jiamingp/venvs/cosmodiff_nf_class/bin/python}
VENV_PATH=/home/jiamingp/venvs/cosmodiff_nf_class
test "$(git -C "$CODE_ROOT" rev-parse HEAD)" = "$EXPECTED_COMMIT"
test -z "$(git -C "$CODE_ROOT" status --porcelain)"
export PYTHONNOUSERSITE=1
export PYTHONPATH="$COSMODIFF_PIN_ROOT/seed_restart_runtime:$CODE_ROOT:$COSMODIFF_PIN_ROOT"
export COSMODIFF_DIR="$COSMODIFF_PIN_ROOT"
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
cd "$CODE_ROOT"
"$PYTHON_BIN" scripts/verify_cosmodiff_seed_restart_runtime.py \
  "$COSMODIFF_PIN_ROOT" --manifest "$COSMODIFF_PIN_MANIFEST" \
  --expected-base-revision "$EXPECTED_COSMODIFF_BASE_REVISION" \
  --python-bin "$PYTHON_BIN" --code-root "$CODE_ROOT" \
  --expected-torch-prefix "$VENV_PATH" \
  --incompatible-python-path /home/jiamingp/venvs/cosmodiff_nf/lib/python3.10/site-packages \
  --incompatible-python-path /sw/pkgs/arc/python3.10-anaconda/2023.03/lib/python3.10/site-packages \
  --patch-script "$CODE_ROOT/scripts/patch_cosmodiff_package_metadata.py" \
  --patch-script "$CODE_ROOT/scripts/patch_cosmodiff_constant_label.py" \
  --patch-script "$CODE_ROOT/scripts/patch_cosmodiff_dit_class_labels.py" \
  --patch-script "$CODE_ROOT/scripts/patch_cosmodiff_checkpoint_state.py"

OUTPUT_DIR="$PROJECT_DIR/results/dit_l16_high_noise_diagnostic"
mkdir -p "$OUTPUT_DIR"

model_json=$("$PYTHON_BIN" scripts/resolve_dit_high_noise_models.py \
  --project-dir "$PROJECT_DIR" --index "${SLURM_ARRAY_TASK_ID:?}")
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

"$PYTHON_BIN" scripts/evaluate_dit_high_noise.py \
  --model-name "$model_name" \
  --checkpoint "$checkpoint" \
  --config "$config" \
  --output-npz "$OUTPUT_DIR/${model_name}.npz" \
  --expected-code-revision "$EXPECTED_COMMIT"
