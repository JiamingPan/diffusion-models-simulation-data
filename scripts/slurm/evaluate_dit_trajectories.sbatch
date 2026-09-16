#!/bin/bash
#SBATCH --job-name=dit_trace
#SBATCH --partition=spgpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=80G
#SBATCH --time=00:30:00

set -euo pipefail
: "${PROJECT_DIR:?}" "${CODE_ROOT:?}" "${EXPECTED_COMMIT:?}"
: "${COSMODIFF_PIN_ROOT:?}" "${COSMODIFF_PIN_MANIFEST:?}"
: "${EXPECTED_COSMODIFF_BASE_REVISION:?}" "${RUNTIME_CODE_ROOT:?}"
PYTHON_BIN=${PYTHON_BIN:-/home/jiamingp/venvs/cosmodiff_nf_class/bin/python}
VENV_PATH=/home/jiamingp/venvs/cosmodiff_nf_class
RUNTIME_CODE_REVISION=555f350f82c913ff06150c96e66709719856cd83
OUTPUT_DIR="$PROJECT_DIR/results/dit_l16_trajectory_screen"

test "$(git -C "$CODE_ROOT" rev-parse HEAD)" = "$EXPECTED_COMMIT"
test -z "$(git -C "$CODE_ROOT" status --porcelain)"
test "$(git -C "$RUNTIME_CODE_ROOT" rev-parse HEAD)" = "$RUNTIME_CODE_REVISION"
test -z "$(git -C "$RUNTIME_CODE_ROOT" status --porcelain)"
test ! -e "$OUTPUT_DIR"

export PYTHONNOUSERSITE=1
export PYTHONPATH="$COSMODIFF_PIN_ROOT/seed_restart_runtime:$RUNTIME_CODE_ROOT:$COSMODIFF_PIN_ROOT"
export COSMODIFF_DIR="$COSMODIFF_PIN_ROOT"
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export PYTHONUNBUFFERED=1

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

"$PYTHON_BIN" - <<'PY'
from PIL import Image
print(f"PIL RUNTIME PASSED: {Image.__version__}")
PY

models=()
for model_index in 0 2 3; do
  models+=("$("$PYTHON_BIN" "$CODE_ROOT/scripts/resolve_dit_high_noise_models.py" \
    --project-dir "$PROJECT_DIR" --index "$model_index")")
done
l12_json=$("$PYTHON_BIN" "$CODE_ROOT/scripts/resolve_dit_high_noise_models.py" \
  --project-dir "$PROJECT_DIR" --index 1)
l12_checkpoint=$("$PYTHON_BIN" - "$l12_json" <<'PY'
import json
import sys
print(json.loads(sys.argv[1])["checkpoint"])
PY
)
if find "$l12_checkpoint" -maxdepth 2 -type f \
  \( -name 'config.json' -o -name '*.safetensors' \
     -o -name 'pytorch_model*.bin' -o -name 'diffusion_pytorch_model*.bin' \) \
  -print -quit | grep -q .; then
  echo "ERROR: L12 artifacts have changed; reassess its inclusion before submission: $l12_checkpoint" >&2
  exit 1
fi
echo "L12 OMITTED: selected checkpoint has no loadable raw model weights: $l12_checkpoint"

if [[ "${PREFLIGHT_ONLY:-0}" == 1 ]]; then
  "$PYTHON_BIN" "$CODE_ROOT/scripts/trace_trajectories.py" --help >/dev/null
  echo 'TRAJECTORY PREFLIGHT PASSED; NO GPU PROBE'
  exit 0
fi

for model_json in "${models[@]}"; do
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
  "$PYTHON_BIN" "$CODE_ROOT/scripts/trace_trajectories.py" \
    --checkpoint "${model_fields[1]}" --config "${model_fields[2]}" \
    --label "$model_name" --eval-root "$RUNTIME_CODE_ROOT" \
    --n 32 --seed 123 --num-steps 50 --batch-size 8 --class-label 0 \
    --out-dir "$OUTPUT_DIR/$model_name"
done

"$PYTHON_BIN" "$CODE_ROOT/scripts/trace_trajectories.py" \
  --pair-files \
    "$OUTPUT_DIR/dit_l8_200k/dit_l8_200k_trajectory.npz" \
    "$OUTPUT_DIR/dit_l16_fresh300k/dit_l16_fresh300k_trajectory.npz" \
    "$OUTPUT_DIR/dit_l16_seed456_500k/dit_l16_seed456_500k_trajectory.npz" \
  --out-dir "$OUTPUT_DIR/comparison"
echo 'TRAJECTORY SCREEN COMPLETE: ALL THREE CHECKPOINTS AND PAIRED COMPARISON'
