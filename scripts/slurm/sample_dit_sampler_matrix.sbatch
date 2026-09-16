#!/bin/bash
#SBATCH --job-name=dit_matrix
#SBATCH --partition=spgpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=80G
#SBATCH --time=01:00:00

set -euo pipefail
: "${PROJECT_DIR:?}" "${CODE_ROOT:?}" "${EXPECTED_COMMIT:?}"
: "${COSMODIFF_PIN_ROOT:?}" "${COSMODIFF_PIN_MANIFEST:?}"
: "${EXPECTED_COSMODIFF_BASE_REVISION:?}" "${RUNTIME_CODE_ROOT:?}"
: "${MATRIX_PLAN_SHA256:?}"
PYTHON_BIN=${PYTHON_BIN:-/home/jiamingp/venvs/cosmodiff_nf_class/bin/python}
VENV_PATH=/home/jiamingp/venvs/cosmodiff_nf_class
RUNTIME_CODE_REVISION=555f350f82c913ff06150c96e66709719856cd83
MATRIX_OUT_DIR=${MATRIX_OUT_DIR:-$PROJECT_DIR/results/dit_l16_sampler_matrix_v1}

test "$(git -C "$CODE_ROOT" rev-parse HEAD)" = "$EXPECTED_COMMIT"
test -z "$(git -C "$CODE_ROOT" status --porcelain)"
test "$(git -C "$RUNTIME_CODE_ROOT" rev-parse HEAD)" = "$RUNTIME_CODE_REVISION"
test -z "$(git -C "$RUNTIME_CODE_ROOT" status --porcelain)"

export PYTHONNOUSERSITE=1
export PYTHONPATH="$COSMODIFF_PIN_ROOT/seed_restart_runtime:$RUNTIME_CODE_ROOT:$COSMODIFF_PIN_ROOT"
export COSMODIFF_DIR="$COSMODIFF_PIN_ROOT"
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
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

if [[ "${PREFLIGHT_ONLY:-0}" == 1 ]]; then
  "$PYTHON_BIN" "$CODE_ROOT/scripts/dit_sampler_matrix.py" \
    --runtime-preflight --out-dir "$MATRIX_OUT_DIR"
  echo 'MATRIX PREFLIGHT PASSED; NO MODEL LOAD OR GPU SAMPLING OR JOB SUBMISSION'
  exit 0
fi

: "${SLURM_ARRAY_TASK_ID:?}"
"$PYTHON_BIN" "$CODE_ROOT/scripts/dit_sampler_matrix.py" \
  --task "$SLURM_ARRAY_TASK_ID" --out-dir "$MATRIX_OUT_DIR" \
  --eval-root "$RUNTIME_CODE_ROOT" --device cuda
echo "MATRIX ARRAY TASK PASSED: $SLURM_ARRAY_TASK_ID"
