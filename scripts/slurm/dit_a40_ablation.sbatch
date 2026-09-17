#!/bin/bash
#SBATCH --job-name=dit_a40_ablate
#SBATCH --partition=spgpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=80G
#SBATCH --time=5-00:00:00

set -euo pipefail
: "${PROJECT_DIR:?}" "${CODE_ROOT:?}" "${EXPECTED_COMMIT:?}"
: "${RUNTIME_CODE_ROOT:?}" "${COSMODIFF_PIN_ROOT:?}" "${COSMODIFF_PIN_MANIFEST:?}"
: "${EXPECTED_COSMODIFF_BASE_REVISION:?}" "${ABLATION_OUT_DIR:?}"
PYTHON_BIN=${PYTHON_BIN:-/home/jiamingp/venvs/cosmodiff_nf_class/bin/python}
MODE=${ABLATION_MODE:-train}
case "$MODE" in train|sample|test-a|data-preflight) ;; *) echo "Unknown mode: $MODE" >&2; exit 1 ;; esac

test "$(git -C "$CODE_ROOT" rev-parse HEAD)" = "$EXPECTED_COMMIT"
test -z "$(git -C "$CODE_ROOT" status --porcelain)"
test "$(git -C "$RUNTIME_CODE_ROOT" rev-parse HEAD)" = 555f350f82c913ff06150c96e66709719856cd83
test -z "$(git -C "$RUNTIME_CODE_ROOT" status --porcelain)"
export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1
export PYTHONPATH="$COSMODIFF_PIN_ROOT/seed_restart_runtime:$RUNTIME_CODE_ROOT:$COSMODIFF_PIN_ROOT"
export COSMODIFF_DIR="$COSMODIFF_PIN_ROOT"
export PYTHONHASHSEED=123
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
cd "$RUNTIME_CODE_ROOT"

"$PYTHON_BIN" "$RUNTIME_CODE_ROOT/scripts/verify_cosmodiff_seed_restart_runtime.py" \
  "$COSMODIFF_PIN_ROOT" --manifest "$COSMODIFF_PIN_MANIFEST" \
  --expected-base-revision "$EXPECTED_COSMODIFF_BASE_REVISION" \
  --python-bin "$PYTHON_BIN" --code-root "$RUNTIME_CODE_ROOT" \
  --expected-torch-prefix /home/jiamingp/venvs/cosmodiff_nf_class \
  --incompatible-python-path /home/jiamingp/venvs/cosmodiff_nf/lib/python3.10/site-packages \
  --incompatible-python-path /sw/pkgs/arc/python3.10-anaconda/2023.03/lib/python3.10/site-packages \
  --patch-script "$RUNTIME_CODE_ROOT/scripts/patch_cosmodiff_package_metadata.py" \
  --patch-script "$RUNTIME_CODE_ROOT/scripts/patch_cosmodiff_constant_label.py" \
  --patch-script "$RUNTIME_CODE_ROOT/scripts/patch_cosmodiff_dit_class_labels.py" \
  --patch-script "$RUNTIME_CODE_ROOT/scripts/patch_cosmodiff_checkpoint_state.py"

"$PYTHON_BIN" "$CODE_ROOT/scripts/dit_a40_ablation.py" --preflight --out-dir "$ABLATION_OUT_DIR"
if [[ "${PREFLIGHT_ONLY:-0}" == 1 ]]; then
  echo "A40 PREFLIGHT PASSED; NO TRAINING, SAMPLING, TEST A OR JOB SUBMISSION"
  exit 0
fi
if [[ "$MODE" == data-preflight ]]; then
  "$PYTHON_BIN" "$CODE_ROOT/scripts/dit_a40_ablation.py" \
    --data-preflight --out-dir "$ABLATION_OUT_DIR"
elif [[ "$MODE" == test-a ]]; then
  "$PYTHON_BIN" "$CODE_ROOT/scripts/patch_memorization_test.py" \
    --project-dir "$PROJECT_DIR" --eval-root "$RUNTIME_CODE_ROOT" \
    --out-dir "$ABLATION_OUT_DIR/test_a"
else
  : "${SLURM_ARRAY_TASK_ID:?}"
  "$PYTHON_BIN" "$CODE_ROOT/scripts/dit_a40_ablation.py" \
    "--$MODE" "$SLURM_ARRAY_TASK_ID" --out-dir "$ABLATION_OUT_DIR"
fi
