#!/bin/bash
#SBATCH --job-name=dit456_sample
#SBATCH --partition=spgpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=80G
#SBATCH --time=04:00:00
#SBATCH --array=0-3%2

set -euo pipefail
: "${PROJECT_DIR:?}" "${CODE_ROOT:?}" "${EXPECTED_COMMIT:?}"
: "${COSMODIFF_PIN_ROOT:?}" "${COSMODIFF_PIN_MANIFEST:?}"
: "${EXPECTED_COSMODIFF_BASE_REVISION:?}"
PYTHON_BIN=${PYTHON_BIN:-/home/jiamingp/venvs/cosmodiff_nf_class/bin/python}
VENV_PATH=/home/jiamingp/venvs/cosmodiff_nf_class
RUNTIME_CODE_ROOT=${RUNTIME_CODE_ROOT:-/scratch/huterer_root/huterer0/jiamingp/iaifi_poster_code_555f350}
export RUNTIME_CODE_ROOT
RUNTIME_CODE_REVISION=555f350f82c913ff06150c96e66709719856cd83
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
"$PYTHON_BIN" scripts/verify_cosmodiff_seed_restart_runtime.py \
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
  echo 'RUNTIME PREFLIGHT PASSED; NO SAMPLING'
  exit 0
fi
case "${SLURM_ARRAY_TASK_ID:?}" in
  0) dataset=d2p08; checkpoint_kind=300k ;;
  1) dataset=d2p08; checkpoint_kind=500k ;;
  2) dataset=d2p10; checkpoint_kind=300k ;;
  3) dataset=d2p10; checkpoint_kind=500k ;;
  *) echo 'Invalid array task' >&2; exit 2 ;;
esac
"$PYTHON_BIN" "$CODE_ROOT/scripts/sample_dit_seed456_review.py" \
  --project-dir "$PROJECT_DIR" --code-root "$CODE_ROOT" \
  --dataset "$dataset" --checkpoint-kind "$checkpoint_kind" --execute
