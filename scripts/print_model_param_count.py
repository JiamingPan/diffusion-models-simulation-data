#!/usr/bin/env python
"""Print model parameter counts for one or more cosmodiff YAML configs."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
COSMODIFF_ROOT = PROJECT_ROOT / "cosmo_diffusion"
if str(COSMODIFF_ROOT) not in sys.path:
    sys.path.insert(0, str(COSMODIFF_ROOT))

from cosmodiff.utils import parse_config_model  # noqa: E402


def count_params(config_path: Path) -> tuple[str, int, int]:
    with config_path.open() as f:
        config = yaml.safe_load(f)
    config.setdefault("global", {})["device"] = "cpu"
    model, *_ = parse_config_model(config)
    total = sum(param.numel() for param in model.parameters())
    trainable = sum(param.numel() for param in model.parameters() if param.requires_grad)
    model_class = config.get("model", {}).get("class", "unknown")
    return model_class, total, trainable


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("configs", nargs="+", type=Path, help="YAML config paths.")
    parser.add_argument(
        "--target",
        type=Path,
        default=None,
        help="Optional reference config. Percent differences are relative to this model.",
    )
    args = parser.parse_args()

    target_total = None
    if args.target is not None:
        _, target_total, _ = count_params(args.target)
        print(f"target\t{args.target}\t{target_total:,}")

    print("config\tclass\ttotal_params\ttrainable_params\tpct_diff_vs_target")
    for config_path in args.configs:
        model_class, total, trainable = count_params(config_path)
        diff = ""
        if target_total:
            diff = f"{(total - target_total) / target_total * 100:+.2f}%"
        print(f"{config_path}\t{model_class}\t{total:,}\t{trainable:,}\t{diff}")


if __name__ == "__main__":
    main()
