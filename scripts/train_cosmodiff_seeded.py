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
