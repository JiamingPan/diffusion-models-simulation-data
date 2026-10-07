#!/usr/bin/env python
"""Run cosmodiff training after explicitly seeding fresh construction."""
from __future__ import annotations

import argparse
import os
import random
import runpy
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from simdiff_eval.torch_compat import install_torch_backend_compat


TORCH_COMPAT_REPORT = install_torch_backend_compat(entry_point=__name__)

from sample_cosmodiff import _install_sklearn_roc_curve_stub

_install_sklearn_roc_curve_stub()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--cosmodiff-train", type=Path, required=True)
    args = parser.parse_args()
    if not args.config.is_file() or not args.cosmodiff_train.is_file():
        raise FileNotFoundError("config or cosmodiff training entry point is missing")
    if os.environ.get("PYTHONHASHSEED") != str(args.seed):
        raise RuntimeError("PYTHONHASHSEED must be set before this process starts")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    sys.argv = [str(args.cosmodiff_train), "--config", str(args.config)]
    runpy.run_path(str(args.cosmodiff_train), run_name="__main__")


if __name__ == "__main__":
    main()
