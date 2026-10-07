#!/bin/bash
# Run from an immutable checkout. Default is staging + preview, never submission.
set -euo pipefail
if [[ $# -gt 1 || ( $# -eq 1 && "$1" != --submit ) ]]; then
  echo 'Usage: bash stage_dit_zero_remaining.sh [--submit]' >&2; exit 2
fi
export CODE_ROOT
CODE_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
export EXPECTED_COMMIT
EXPECTED_COMMIT=$(git -C "$CODE_ROOT" rev-parse HEAD)
test -z "$(git -C "$CODE_ROOT" status --porcelain)"
export PYTHON_BIN=/home/jiamingp/venvs/cosmodiff_nf_class/bin/python
export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1
export SLURM_ACCOUNT=huterer2 SCRATCH_ACCOUNT=huterer_root
export SCREEN_ROOT=/scratch/huterer_root/huterer0/jiamingp/dit_zero_remaining_300k_v1
export PLAN_PATH="$SCREEN_ROOT/plan.json"
export COSMODIFF_MANIFEST=/scratch/huterer_root/huterer0/jiamingp/dit_l16_adalnzero_highn_screen_v1/cosmodiff_runtime_manifest_4973364.json
export COSMODIFF_MANIFEST_SHA256=cb4ffd8592eac6a1eac4f81c5b80d4d3982854a03e8c1bfe7222a49b667a6d0a
test "$(sha256sum "$COSMODIFF_MANIFEST" | awk '{print $1}')" = "$COSMODIFF_MANIFEST_SHA256"
export COSMODIFF_ROOT COSMODIFF_COMMIT
COSMODIFF_ROOT=$("$PYTHON_BIN" -c 'import json,sys; print(json.load(open(sys.argv[1]))["root_at_creation"])' "$COSMODIFF_MANIFEST")
COSMODIFF_COMMIT=$("$PYTHON_BIN" -c 'import json,sys; print(json.load(open(sys.argv[1]))["base_commit"])' "$COSMODIFF_MANIFEST")
if [[ ! -e "$SCREEN_ROOT" ]]; then
  "$PYTHON_BIN" "$CODE_ROOT/scripts/prepare_dit_zero_remaining.py" \
    --out-dir "$SCREEN_ROOT" \
    --checkpoint-root /scratch/huterer_root/huterer0/jiamingp/saved_runs/dit_zero_remaining_300k_v1
fi
export PLAN_SHA256
PLAN_SHA256=$(sha256sum "$PLAN_PATH" | awk '{print $1}')
echo "PLAN_SHA256=$PLAN_SHA256"
"$PYTHON_BIN" "$CODE_ROOT/scripts/launch_dit_zero_remaining.py" "$@"
