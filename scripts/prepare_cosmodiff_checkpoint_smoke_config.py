#!/usr/bin/env python
"""Create a two-epoch checkpoint round-trip config from a frozen run config."""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--checkpoint-dir", required=True, type=Path)
    args = parser.parse_args()

    with args.source.open() as handle:
        config = yaml.safe_load(handle)
    config["io"]["output_dir"] = str(args.checkpoint_dir.resolve())
    config["train"]["num_epochs"] = 1
    config["train"]["checkpoint_every_n_epochs"] = 1
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as handle:
        yaml.safe_dump(config, handle, sort_keys=False)
    print(f"Wrote checkpoint smoke-test config: {args.output}")


if __name__ == "__main__":
    main()
