#!/bin/bash
# Shared runtime for augmentation wave 2 (ledger unet128_aug_wave2). Reuses the wave-1 runtime
# (same venv, cosmodiff checkout, stubs, branch check) and points run_field at the wave-2 runs.json.
set -euo pipefail
source /home/jiamingp/diffusion_models_repo/scripts/slurm/unet128_aug_sweep_env.sh
# Wave-1 dir: data_reference.json and verification_report.json (same configs, read-only).
REF_LOCAL=${PROJECT_DIR}/local/unet128_aug_sweep
WAVE_LOCAL=${PROJECT_DIR}/local/unet128_aug_wave2
mkdir -p "${PROJECT_DIR}/logs/unet128_aug_wave2"

# Field of wave-2 runs.json for flat array task $1 (0-8: N=64 x3 arms, N=128 x3, N=256 x3).
wave_field() {
  "${PYTHON_BIN}" - "$1" "$2" <<'PY'
import json, os, sys
task, key = int(sys.argv[1]), sys.argv[2]
print(json.load(open(os.path.join(os.environ["WAVE_LOCAL"], "runs.json")))[task][key])
PY
}
export REF_LOCAL WAVE_LOCAL
