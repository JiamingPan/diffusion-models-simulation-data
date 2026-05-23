#!/bin/bash
set -euo pipefail

PROJECT_DIR=${PROJECT_DIR:-/home/jiamingp/diffusion_models_repo}
BASE_VENV_PATH=${BASE_VENV_PATH:-/home/jiamingp/venvs/cosmodiff_nf}
CLASS_VENV_PATH=${CLASS_VENV_PATH:-/home/jiamingp/venvs/cosmodiff_nf_class}
DIFFUSERS_VERSION=${DIFFUSERS_VERSION:-0.31.0}
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "${SCRIPT_DIR}/nf_class_conditional_pythonpath.sh"

if [[ ! -x "${BASE_VENV_PATH}/bin/python" ]]; then
  echo "Missing base venv python: ${BASE_VENV_PATH}/bin/python" >&2
  exit 1
fi

if [[ ! -x "${CLASS_VENV_PATH}/bin/python" ]]; then
  echo "Creating class-conditional venv: ${CLASS_VENV_PATH}"
  "${BASE_VENV_PATH}/bin/python" -m venv --system-site-packages "${CLASS_VENV_PATH}"
fi

echo "Class env:             ${CLASS_VENV_PATH}"
echo "Installing diffusers==${DIFFUSERS_VERSION} directly into the class env"
"${CLASS_VENV_PATH}/bin/python" -m pip install --upgrade pip setuptools wheel
"${CLASS_VENV_PATH}/bin/python" -m pip install --upgrade --no-deps "diffusers==${DIFFUSERS_VERSION}"

set_nf_class_conditional_pythonpath "${CLASS_VENV_PATH}/bin/python" "${BASE_VENV_PATH}"
echo "Class package path:    ${NF_CLASS_CONDITIONAL_CLASS_SITES}"
echo "Base package fallback: ${NF_CLASS_CONDITIONAL_BASE_SITES}"
echo "PYTHONPATH:            ${PYTHONPATH:-<empty>}"

DIFFUSERS_VERSION="${DIFFUSERS_VERSION}" "${CLASS_VENV_PATH}/bin/python" - <<'PY'
import os
import sys

expected = os.environ["DIFFUSERS_VERSION"]

import torch
import diffusers
from diffusers import DDPMScheduler, UNet2DModel

print("python:", sys.executable)
print("torch:", torch.__version__, torch.__file__)
print("diffusers:", diffusers.__version__, diffusers.__file__)
print("sys.path[:5]:", sys.path[:5])

if diffusers.__version__ != expected:
    raise SystemExit(f"Expected diffusers=={expected}, got {diffusers.__version__}")

scheduler = DDPMScheduler(
    num_train_timesteps=500,
    beta_schedule="squaredcos_cap_v2",
    prediction_type="v_prediction",
    rescale_betas_zero_snr=True,
)
print("scheduler:", scheduler.__class__.__name__, scheduler.config.prediction_type)

model = UNet2DModel(
    sample_size=32,
    in_channels=1,
    out_channels=1,
    layers_per_block=1,
    block_out_channels=(16, 32),
    down_block_types=("DownBlock2D", "DownBlock2D"),
    up_block_types=("UpBlock2D", "UpBlock2D"),
    num_class_embeds=7,
    norm_num_groups=8,
).cpu().eval()
x = torch.zeros((2, 1, 32, 32), dtype=torch.float32)
t = torch.zeros((2,), dtype=torch.long)
labels = torch.tensor([0, 1], dtype=torch.long)
with torch.no_grad():
    y = model(x, timestep=t, class_labels=labels, return_dict=False)[0]
if tuple(y.shape) != tuple(x.shape):
    raise SystemExit(f"UNet class-label forward shape mismatch: {tuple(y.shape)} vs {tuple(x.shape)}")
print("class-label forward: ok", tuple(y.shape))
PY

echo
echo "Runtime env is ready."
echo "Next checks:"
echo "  bash scripts/preflight_nf_class_conditional_u128.sh"
echo "Then submit:"
echo "  sbatch -A huterer0 scripts/slurm/train_nf_class_conditional_u128.sbatch"
