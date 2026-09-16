#!/bin/bash
#SBATCH --job-name=dit_patchwork
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:30:00

set -euo pipefail
: "${PROJECT_DIR:?}" "${CODE_ROOT:?}" "${EXPECTED_COMMIT:?}"
: "${RUNTIME_CODE_ROOT:?}"
PYTHON_BIN=${PYTHON_BIN:-/home/jiamingp/venvs/cosmodiff_nf_class/bin/python}
OUTPUT_DIR="$PROJECT_DIR/results/patchwork"

test "$(git -C "$CODE_ROOT" rev-parse HEAD)" = "$EXPECTED_COMMIT"
test -z "$(git -C "$CODE_ROOT" status --porcelain)"
test "$(git -C "$RUNTIME_CODE_ROOT" rev-parse HEAD)" = \
  555f350f82c913ff06150c96e66709719856cd83
test -z "$(git -C "$RUNTIME_CODE_ROOT" status --porcelain)"
test ! -e "$OUTPUT_DIR"

export PYTHONNOUSERSITE=1
export PYTHONPATH="$RUNTIME_CODE_ROOT"
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8

model_json=$("$PYTHON_BIN" "$CODE_ROOT/scripts/resolve_dit_high_noise_models.py" \
  --project-dir "$PROJECT_DIR" --index 0)
config=$("$PYTHON_BIN" - "$model_json" <<'PY'
import json
import sys
print(json.loads(sys.argv[1])["config"])
PY
)

samples=(
  "$PROJECT_DIR/results/dit_l16_late_start_screen/dit_l16_fresh300k_t499_n128.npz"
  "$PROJECT_DIR/results/dit_l16_late_start_screen/dit_l16_seed456_500k_t499_n128.npz"
  "$PROJECT_DIR/results/dit_l16_late_start_screen/dit_l8_200k_t499_n128.npz"
)
for sample in "${samples[@]}"; do
  test -s "$sample"
done

if [[ "${PREFLIGHT_ONLY:-0}" == 1 ]]; then
  "$PYTHON_BIN" "$CODE_ROOT/scripts/patchwork_check.py" --help >/dev/null
  echo 'PATCHWORK PREFLIGHT PASSED; NO ANALYSIS'
  exit 0
fi

"$PYTHON_BIN" "$CODE_ROOT/scripts/patchwork_check.py" \
  --config "$config" \
  --eval-root "$RUNTIME_CODE_ROOT" \
  --samples "${samples[@]}" \
  --out-dir "$OUTPUT_DIR"
