#!/bin/bash
# No-GPU preflight for the frozen probe-control evaluation.

set -euo pipefail
trap 'rc=$?; echo "[probe-preflight] failed at line ${LINENO}: ${BASH_COMMAND} (exit ${rc})" >&2' ERR

PROJECT_DIR=${PROJECT_DIR:-/home/jiamingp/diffusion_models_repo}
CODE_ROOT=${CODE_ROOT:-/scratch/huterer_root/huterer0/jiamingp/probe_controls_code_456f01a}
PYTHON_BIN=${PYTHON_BIN:-/home/jiamingp/venvs/cosmodiff_nf_class/bin/python}
TORCH_HOME=${TORCH_HOME:-/scratch/huterer_root/huterer0/jiamingp/torch_cache}
DATA_ROOT=${DATA_ROOT:-/scratch/huterer_root/huterer0/CAMELS/CMD/3d_grids/IllustrisTNG}
INCOMPATIBLE_PYTHON_PATHS=${INCOMPATIBLE_PYTHON_PATHS:-/home/jiamingp/venvs/cosmodiff_nf/lib/python3.10/site-packages}
EXPECTED_COMMIT=${EXPECTED_COMMIT:-}

if [[ -z "${EXPECTED_COMMIT}" ]]; then
  echo "EXPECTED_COMMIT must be set explicitly; refusing an unpinned checkout" >&2
  exit 1
fi

actual_commit=$(git -C "${CODE_ROOT}" rev-parse HEAD)
if [[ "${actual_commit}" != "${EXPECTED_COMMIT}" ]]; then
  echo "Expected CODE_ROOT commit ${EXPECTED_COMMIT}; found ${actual_commit}" >&2
  exit 1
fi

export PYTHONNOUSERSITE=1
source "${CODE_ROOT}/scripts/probe_controls_runtime.sh"
probe_controls_prepare_runtime "${CODE_ROOT}" "${PYTHON_BIN}" "${INCOMPATIBLE_PYTHON_PATHS}"
export PROJECT_DIR CODE_ROOT PYTHON_BIN TORCH_HOME DATA_ROOT
export TORCH_HOME

required_files=(
  "${CODE_ROOT}/scripts/evaluate_probe_transform_controls.py"
  "${CODE_ROOT}/scripts/evaluate_probe_degradation_control.py"
  "${CODE_ROOT}/simdiff_eval/probe_controls.py"
  "${CODE_ROOT}/simdiff_eval/probe_transforms.py"
  "${PROJECT_DIR}/results/nf_conditional_bias_probe/encoder/vgg_mlp_encoder.npz"
  "${PROJECT_DIR}/results/nf_conditional_bias_probe/encoder/vgg_mlp_encoder.pkl"
  "${PROJECT_DIR}/local/nf_conditional_bias_probe/manifest.json"
  "${PROJECT_DIR}/results/nf_conditional_bias_probe/samples/nf_cond_bias_hi_u128_d2p07_n128_200k_seed123_dpm50_heldout_k64.npz"
  "${PROJECT_DIR}/results/nf_conditional_bias_probe/samples/nf_cond_bias_hi_u128_d2p14_n16384_200k_seed123_dpm50_heldout_k64.npz"
  "${TORCH_HOME}/hub/checkpoints/vgg16-397923af.pth"
)
for path in "${required_files[@]}"; do
  [[ -f "${path}" ]] || { echo "Missing required file: ${path}" >&2; exit 1; }
done

cd "${CODE_ROOT}"
"${PYTHON_BIN}" -c '
import sys
from pathlib import Path
from simdiff_eval import probe_controls, probe_transforms
import torch
import torchvision
bad = set(filter(None, __import__("os").environ["PROBE_CONTROLS_INCOMPATIBLE_PATHS"].split(__import__("os").pathsep)))
assert not (bad & set(sys.path)), (bad, sys.path)
print(f"[probe-preflight] torch={torch.__version__} torchvision={torchvision.__version__}")
print(f"[probe-preflight] probe_controls={Path(probe_controls.__file__).resolve()}")
print(f"[probe-preflight] probe_transforms={Path(probe_transforms.__file__).resolve()}")
print("[probe-preflight] sanitized imports passed")
'
"${PYTHON_BIN}" "${CODE_ROOT}/scripts/evaluate_probe_transform_controls.py" --help >/dev/null
"${PYTHON_BIN}" "${CODE_ROOT}/scripts/evaluate_probe_degradation_control.py" --help >/dev/null
echo "[probe-preflight] PASSED"
