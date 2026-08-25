#!/usr/bin/env python
"""Start one cosmodiff run from a deterministic fresh initialization."""

from __future__ import annotations

import argparse
import json
import os
import random
import runpy
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from simdiff_eval.torch_compat import install_torch_backend_compat


install_torch_backend_compat(entry_point=__name__)

import numpy as np
import torch
import yaml


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def validate_fresh_checkpoint_dir(checkpoint_dir: Path) -> Path:
    checkpoint_dir = checkpoint_dir.expanduser().resolve()
    existing = sorted(checkpoint_dir.glob("checkpoint-epoch-*"))
    if existing:
        raise ValueError(
            "Refusing to initialize a fresh checkpoint directory that already "
            f"contains checkpoints: {checkpoint_dir}"
        )
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    return checkpoint_dir


def git_value(repo: Path, *args: str) -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(repo), *args],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def write_provenance(
    checkpoint_dir: Path,
    *,
    seed: int,
    config: Path,
    train_script: Path,
) -> Path:
    try:
        import diffusers

        diffusers_version = diffusers.__version__
    except ImportError:
        diffusers_version = "unavailable"
    cosmodiff_repo = train_script.resolve().parents[1]
    payload = {
        "mode": "fresh_initialization",
        "seed": int(seed),
        "config": str(config.resolve()),
        "cosmodiff_train": str(train_script.resolve()),
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "python_version": sys.version,
        "torch_version": torch.__version__,
        "diffusers_version": diffusers_version,
        "cuda_available": torch.cuda.is_available(),
        "deterministic_algorithms_enabled": torch.are_deterministic_algorithms_enabled(),
        "cosmodiff_git_branch": git_value(cosmodiff_repo, "rev-parse", "--abbrev-ref", "HEAD"),
        "cosmodiff_git_commit": git_value(cosmodiff_repo, "rev-parse", "--short", "HEAD"),
    }
    path = checkpoint_dir / "fresh_run_provenance.json"
    path.write_text(json.dumps(payload, indent=2) + "\n")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--cosmodiff-train", required=True, type=Path)
    parser.add_argument("--checkpoint-dir", required=True, type=Path)
    parser.add_argument("--seed", required=True, type=int)
    args, extra_args = parser.parse_known_args()

    config_path = args.config.expanduser().resolve()
    train_script = args.cosmodiff_train.expanduser().resolve()
    if not config_path.is_file():
        raise FileNotFoundError(f"Missing fresh training config: {config_path}")
    if not train_script.is_file():
        raise FileNotFoundError(f"Missing cosmodiff training script: {train_script}")

    checkpoint_dir = validate_fresh_checkpoint_dir(args.checkpoint_dir)
    with config_path.open() as handle:
        config = yaml.safe_load(handle)
    configured_output = Path(config["io"]["output_dir"]).expanduser().resolve()
    if configured_output != checkpoint_dir:
        raise ValueError(
            f"Config output directory {configured_output} does not match fresh "
            f"checkpoint directory {checkpoint_dir}"
        )

    seed_everything(args.seed)
    provenance = write_provenance(
        checkpoint_dir,
        seed=args.seed,
        config=config_path,
        train_script=train_script,
    )
    os.environ["PYTHONHASHSEED"] = str(args.seed)
    print(f"Fresh initialization seed: {args.seed}", flush=True)
    print(f"Fresh provenance: {provenance}", flush=True)
    sys.argv = [str(train_script), "--config", str(config_path), *extra_args]
    runpy.run_path(str(train_script), run_name="__main__")


if __name__ == "__main__":
    main()
