#!/bin/bash
#SBATCH --job-name=dit_l16_zdata
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=02:00:00

set -euo pipefail
: "${CODE_ROOT:?}" "${EXPECTED_COMMIT:?}" "${PLAN_PATH:?}" "${PLAN_SHA256:?}"
: "${COSMODIFF_ROOT:?}" "${COSMODIFF_COMMIT:?}" "${COSMODIFF_MANIFEST:?}"
: "${COSMODIFF_MANIFEST_SHA256:?}" "${PYTHON_BIN:?}"
test "$(git -C "$CODE_ROOT" rev-parse HEAD)" = "$EXPECTED_COMMIT"
export CUDA_VISIBLE_DEVICES="" PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1
export PYTHONPATH="$COSMODIFF_ROOT:$CODE_ROOT:${PYTHONPATH:-}"
"$PYTHON_BIN" "$CODE_ROOT/scripts/verify_dit_adaln_zero_screen.py" \
  --plan "$PLAN_PATH" --plan-sha256 "$PLAN_SHA256" \
  --cosmodiff-root "$COSMODIFF_ROOT" --cosmodiff-commit "$COSMODIFF_COMMIT" \
  --cosmodiff-manifest "$COSMODIFF_MANIFEST" \
  --cosmodiff-manifest-sha256 "$COSMODIFF_MANIFEST_SHA256"
"$PYTHON_BIN" "$CODE_ROOT/scripts/check_dit_zero_screen_imports.py"
"$PYTHON_BIN" "$CODE_ROOT/scripts/preflight_dit_adaln_zero_data.py" \
  --plan "$PLAN_PATH" --plan-sha256 "$PLAN_SHA256" \
  --cosmodiff-manifest-sha256 "$COSMODIFF_MANIFEST_SHA256"
