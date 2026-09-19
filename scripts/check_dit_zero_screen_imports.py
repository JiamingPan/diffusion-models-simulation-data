#!/usr/bin/env python
"""Check actual DiT imports before submitting the screen's data/GPU jobs."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from simdiff_eval.torch_compat import install_torch_backend_compat

install_torch_backend_compat(entry_point=__name__)

from sample_cosmodiff import _install_sklearn_roc_curve_stub, _reject_known_bad_runtime

_reject_known_bad_runtime()
_install_sklearn_roc_curve_stub()

import torch
import diffusers
import cosmodiff
from cosmodiff import utils, optim, augment, transform
from diffusers import DiTTransformer2DModel, DDPMScheduler, DPMSolverMultistepScheduler

print("Torch:", torch.__version__, torch.__file__)
print("Diffusers:", diffusers.__version__, diffusers.__file__)
print("CosmoDiff:", cosmodiff.__version__, cosmodiff.__file__)
print("DIT RUNTIME IMPORTS PASSED; NO MODEL, DATA OR GPU WORK")
