#!/bin/bash
# Shared runtime for augmentation wave 3 (ledger unet128_aug_wave3). Reuses the wave-1/2 runtime
# (same venv, cosmodiff checkout, stubs, branch check) and points wave_field at the wave-3 runs.json.
set -euo pipefail
source /home/jiamingp/diffusion_models_repo/scripts/slurm/unet128_aug_sweep_env.sh
WAVE_LOCAL=${PROJECT_DIR}/local/unet128_aug_wave3
mkdir -p "${PROJECT_DIR}/logs/unet128_aug_wave3"

# Field of wave-3 runs.json for flat array task $1 (0-17, order of prepare_unet128_aug_wave3.PLAN).
wave_field() {
  "${PYTHON_BIN}" - "$1" "$2" <<'PY'
import json, os, sys
task, key = int(sys.argv[1]), sys.argv[2]
print(json.load(open(os.path.join(os.environ["WAVE_LOCAL"], "runs.json")))[task][key])
PY
}
export WAVE_LOCAL
