#!/bin/bash
set -euo pipefail

PROJECT_DIR=${PROJECT_DIR:-/home/jiamingp/diffusion_models_repo}
BASE_VENV_PATH=${BASE_VENV_PATH:-/home/jiamingp/venvs/cosmodiff_nf}
CLASS_VENV_PATH=${CLASS_VENV_PATH:-/home/jiamingp/venvs/cosmodiff_nf_class}
DIFFUSERS_VERSION=${DIFFUSERS_VERSION:-0.38.0}
INSTALL_TORCH=${INSTALL_TORCH:-1}
TORCH_VERSION=${TORCH_VERSION:-2.1.2+cu118}
TORCH_CUDA_INDEX_URL=${TORCH_CUDA_INDEX_URL:-https://download.pytorch.org/whl/cu118}
NUMPY_VERSION=${NUMPY_VERSION:-1.26.4}
SCIPY_VERSION=${SCIPY_VERSION:-1.11.4}
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROJECT_DIR=$(cd "${SCRIPT_DIR}/.." && pwd)
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
if [[ "${INSTALL_TORCH}" != "0" ]]; then
  echo "Installing CUDA PyTorch ${TORCH_VERSION} directly into the class env"
fi
echo "Installing numpy==${NUMPY_VERSION} and scipy==${SCIPY_VERSION} directly into the class env"
echo "Installing diffusers==${DIFFUSERS_VERSION} directly into the class env"
"${CLASS_VENV_PATH}/bin/python" -m pip install --upgrade pip setuptools wheel
if [[ "${INSTALL_TORCH}" != "0" ]]; then
  "${CLASS_VENV_PATH}/bin/python" -m pip install --upgrade --extra-index-url "${TORCH_CUDA_INDEX_URL}" "torch==${TORCH_VERSION}"
fi
"${CLASS_VENV_PATH}/bin/python" -m pip install --upgrade --only-binary=:all: --ignore-installed "numpy==${NUMPY_VERSION}" "scipy==${SCIPY_VERSION}"
"${CLASS_VENV_PATH}/bin/python" -m pip install --upgrade --no-deps "diffusers==${DIFFUSERS_VERSION}"

set_nf_class_conditional_pythonpath "${CLASS_VENV_PATH}/bin/python" "${BASE_VENV_PATH}"
export PYTHONPATH="${PROJECT_DIR}:${PYTHONPATH:-}"
echo "Class package path:    ${NF_CLASS_CONDITIONAL_CLASS_SITES}"
echo "Base package fallback: ${NF_CLASS_CONDITIONAL_BASE_SITES}"
echo "PYTHONPATH:            ${PYTHONPATH:-<empty>}"

DIFFUSERS_VERSION="${DIFFUSERS_VERSION}" "${CLASS_VENV_PATH}/bin/python" - <<'PY'
import os
import sys
from pathlib import Path

expected = os.environ["DIFFUSERS_VERSION"]

from importlib.machinery import ModuleSpec
from types import ModuleType
from simdiff_eval.torch_compat import install_torch_backend_compat

torch = install_torch_backend_compat(entry_point="setup_nf_class_conditional_env")

sklearn = ModuleType("sklearn")
sklearn_metrics = ModuleType("sklearn.metrics")
sklearn.__path__ = []
sklearn.__spec__ = ModuleSpec("sklearn", loader=None, is_package=True)
sklearn.__spec__.submodule_search_locations = []
sklearn_metrics.__spec__ = ModuleSpec("sklearn.metrics", loader=None)

def _stubbed_roc_curve(*args, **kwargs):
    raise RuntimeError("sklearn.metrics.roc_curve is stubbed for class-env setup validation")

sklearn_metrics.roc_curve = _stubbed_roc_curve
sklearn.metrics = sklearn_metrics
sys.modules["sklearn"] = sklearn
sys.modules["sklearn.metrics"] = sklearn_metrics

import diffusers
from diffusers import AutoModel, DDPMScheduler, UNet2DModel
import numpy
import scipy
import scipy.stats

print("python:", sys.executable)
print("torch:", torch.__version__, torch.__file__)
print("torch_cuda_build:", torch.version.cuda)
print("torch_cuda_available:", torch.cuda.is_available())
print("numpy:", numpy.__version__, numpy.__file__)
print("scipy:", scipy.__version__, scipy.__file__)
print("diffusers:", diffusers.__version__, diffusers.__file__)
print("AutoModel:", AutoModel)
print("sys.path[:5]:", sys.path[:5])

if diffusers.__version__ != expected:
    raise SystemExit(f"Expected diffusers=={expected}, got {diffusers.__version__}")
if torch.version.cuda is None:
    raise SystemExit(
        "Class env is importing a CPU-only PyTorch build. "
        "Install a CUDA wheel in the class env before submitting GPU training."
    )
class_prefix = Path(sys.prefix).resolve()
for name, module in (("numpy", numpy), ("scipy", scipy)):
    module_path = Path(module.__file__).resolve()
    if not module_path.is_relative_to(class_prefix):
        raise SystemExit(
            f"{name} is being imported from {module_path}, outside the class env {class_prefix}. "
            "Re-run setup so the class env shadows the Great Lakes Anaconda binary packages."
        )

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
