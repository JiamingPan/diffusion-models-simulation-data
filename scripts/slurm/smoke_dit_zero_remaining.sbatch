#!/bin/bash
set -euo pipefail
: "${CODE_ROOT:?}" "${EXPECTED_COMMIT:?}" "${PLAN_PATH:?}" "${PLAN_SHA256:?}"
: "${COSMODIFF_ROOT:?}" "${COSMODIFF_COMMIT:?}" "${COSMODIFF_MANIFEST:?}"
: "${COSMODIFF_MANIFEST_SHA256:?}" "${PYTHON_BIN:?}"
test "$(git -C "$CODE_ROOT" rev-parse HEAD)" = "$EXPECTED_COMMIT"
export PYTHONHASHSEED=123 PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1
export PYTHONPATH="$COSMODIFF_ROOT:$CODE_ROOT:${PYTHONPATH:-}"
"$PYTHON_BIN" "$CODE_ROOT/scripts/verify_dit_adaln_zero_screen.py" \
  --plan "$PLAN_PATH" --plan-sha256 "$PLAN_SHA256" \
  --cosmodiff-root "$COSMODIFF_ROOT" --cosmodiff-commit "$COSMODIFF_COMMIT" \
  --cosmodiff-manifest "$COSMODIFF_MANIFEST" \
  --cosmodiff-manifest-sha256 "$COSMODIFF_MANIFEST_SHA256" --require-data-receipt
# The fixed plan begins each depth at indices 0 (L16), 5 (L12), 15 (L8).
for index in 0 5 15; do
  "$PYTHON_BIN" "$CODE_ROOT/scripts/smoke_dit_adaln_zero_screen.py" --plan "$PLAN_PATH" --index "$index"
done
