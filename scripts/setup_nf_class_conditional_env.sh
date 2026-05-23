#!/bin/bash
set -euo pipefail

PROJECT_DIR=${PROJECT_DIR:-/home/jiamingp/diffusion_models_repo}
BASE_VENV_PATH=${BASE_VENV_PATH:-/home/jiamingp/venvs/cosmodiff_nf}
CLASS_VENV_PATH=${CLASS_VENV_PATH:-/home/jiamingp/venvs/cosmodiff_nf_class}
DIFFUSERS_VERSION=${DIFFUSERS_VERSION:-0.38.0}
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

from contextlib import nullcontext
from types import ModuleType, SimpleNamespace


class _OptionalDeviceStub:
    def is_available(self): return False
    def device_count(self): return 0
    def empty_cache(self): return None
    def _is_compiled(self): return False
    def current_device(self): return 0
    def set_device(self, *args, **kwargs): return None
    def synchronize(self, *args, **kwargs): return None
    def manual_seed(self, *args, **kwargs): return None
    def manual_seed_all(self, *args, **kwargs): return None
    def seed(self, *args, **kwargs): return 0
    def initial_seed(self, *args, **kwargs): return 0
    def get_rng_state(self, *args, **kwargs): return None
    def set_rng_state(self, *args, **kwargs): return None
    def is_built(self, *args, **kwargs): return False
    def current_stream(self, *args, **kwargs): return None
    def stream(self, *args, **kwargs): return nullcontext()
    def device(self, *args, **kwargs): return nullcontext()
    def memory_allocated(self, *args, **kwargs): return 0
    def max_memory_allocated(self, *args, **kwargs): return 0
    def reset_peak_memory_stats(self, *args, **kwargs): return None
    def get_device_name(self, *args, **kwargs): return "optional-device-unavailable"
    def get_device_properties(self, *args, **kwargs): return None
    def __getattr__(self, name):
        def missing(*args, **kwargs):
            if name.startswith("is_"):
                return False
            return None
        return missing


_stub = _OptionalDeviceStub()
_required = ("empty_cache", "is_available", "device_count", "manual_seed")
for _backend in ("xpu", "mps"):
    _existing = getattr(torch, _backend, None)
    if _existing is None or any(not hasattr(_existing, _name) for _name in _required):
        setattr(torch, _backend, _stub)
        continue
    for _name in dir(_stub):
        if _name.startswith("__"):
            continue
        if not hasattr(_existing, _name):
            setattr(_existing, _name, getattr(_stub, _name))
for _name in (
    "float8_e4m3fn",
    "float8_e4m3fnuz",
    "float8_e5m2",
    "float8_e5m2fnuz",
    "float8_e8m0fnu",
    "float4_e2m1fn_x2",
):
    if not hasattr(torch, _name):
        setattr(torch, _name, torch.float16)
for _bits in range(1, 8):
    _name = f"uint{_bits}"
    if not hasattr(torch, _name):
        setattr(torch, _name, torch.uint8)
if hasattr(torch, "distributed") and not hasattr(torch.distributed, "device_mesh"):
    torch.distributed.device_mesh = SimpleNamespace(DeviceMesh=object)
if hasattr(torch, "distributed") and "torch.distributed._functional_collectives" not in sys.modules:
    funcol = ModuleType("torch.distributed._functional_collectives")

    class AsyncCollectiveTensor:
        pass

    def _identity_collective(tensor, *args, **kwargs):
        return tensor

    funcol.AsyncCollectiveTensor = AsyncCollectiveTensor
    funcol.all_to_all_single = _identity_collective
    funcol.all_gather_tensor = _identity_collective
    funcol.permute_tensor = _identity_collective
    sys.modules["torch.distributed._functional_collectives"] = funcol
    torch.distributed._functional_collectives = funcol

import diffusers
from diffusers import AutoModel, DDPMScheduler, UNet2DModel

print("python:", sys.executable)
print("torch:", torch.__version__, torch.__file__)
print("diffusers:", diffusers.__version__, diffusers.__file__)
print("AutoModel:", AutoModel)
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
