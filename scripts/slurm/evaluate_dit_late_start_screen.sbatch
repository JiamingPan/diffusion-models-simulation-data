#!/bin/bash
#SBATCH --job-name=dit_late_eval
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00

set -euo pipefail
: "${PROJECT_DIR:?}" "${CODE_ROOT:?}" "${EXPECTED_COMMIT:?}"
: "${COSMODIFF_PIN_ROOT:?}" "${RUNTIME_CODE_ROOT:?}"
PYTHON_BIN=${PYTHON_BIN:-/home/jiamingp/venvs/cosmodiff_nf_class/bin/python}
OUTPUT_DIR="$PROJECT_DIR/results/dit_l16_late_start_screen"

test "$(git -C "$CODE_ROOT" rev-parse HEAD)" = "$EXPECTED_COMMIT"
test -z "$(git -C "$CODE_ROOT" status --porcelain)"
export PYTHONNOUSERSITE=1
export PYTHONPATH="$COSMODIFF_PIN_ROOT/seed_restart_runtime:$RUNTIME_CODE_ROOT:$COSMODIFF_PIN_ROOT"
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8

mapfile -t samples < <(find "$OUTPUT_DIR" -maxdepth 1 -type f \
  -name 'dit_*_t*_n128.npz' -print | sort)
test "${#samples[@]}" -eq 6 || {
  echo "expected 6 completed sample files, found ${#samples[@]}" >&2
  exit 1
}
model_json=$("$PYTHON_BIN" "$CODE_ROOT/scripts/resolve_dit_high_noise_models.py" \
  --project-dir "$PROJECT_DIR" --index 0)
config=$("$PYTHON_BIN" - "$model_json" <<'PY'
import json
import sys
print(json.loads(sys.argv[1])["config"])
PY
)

"$PYTHON_BIN" "$CODE_ROOT/scripts/evaluate_late_start.py" \
  --config "$config" \
  --eval-root "$RUNTIME_CODE_ROOT" \
  --samples "${samples[@]}" \
  --out "$OUTPUT_DIR/summary_n128.csv"
