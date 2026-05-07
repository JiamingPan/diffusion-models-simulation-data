#!/usr/bin/env python
"""Generate samples for the normalization-fix checkpoint sweep."""

from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from pathlib import Path


RUNS = [
    {
        "run_base": "nf_u64_n500_e100",
        "checkpoint": "nf_u64_n500_e100_checkpoints/checkpoint-epoch-0099",
        "image_size": 128,
    },
    {
        "run_base": "nf_u128_n500_e100",
        "checkpoint": "nf_u128_n500_e100_checkpoints/checkpoint-epoch-0099",
        "image_size": 128,
    },
    {
        "run_base": "nf_u128_n500_e80",
        "checkpoint": "nf_u128_n500_e80_checkpoints/checkpoint-epoch-0079",
        "image_size": 128,
    },
    {
        "run_base": "nf_u256_n500_e100",
        "checkpoint": "nf_u256_n500_e100_checkpoints/checkpoint-epoch-0099",
        "image_size": 128,
    },
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-root", required=True, help="Directory containing *_checkpoints directories.")
    parser.add_argument("--config-dir", default=None, help="Optional directory containing per-run YAML configs.")
    parser.add_argument("--output-dir", default="results/tables/samples", help="Where to write generated .npy files.")
    parser.add_argument("--num-samples", type=int, default=1, help="Total samples to generate per checkpoint.")
    parser.add_argument("--batch-size", type=int, default=1, help="Samples denoised at once on GPU.")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--python-bin", default=sys.executable)
    parser.add_argument("--dry-run", action="store_true", help="Print commands without running them.")
    return parser.parse_args()


def maybe_config(config_dir: Path | None, run_base: str) -> Path | None:
    if config_dir is None:
        return None
    for pattern in (f"{run_base}.yaml", f"{run_base}.yml", f"{run_base}*.yaml", f"{run_base}*.yml"):
        matches = sorted(config_dir.glob(pattern))
        if matches:
            return matches[0]
    return None


def main() -> None:
    args = parse_args()
    checkpoint_root = Path(args.checkpoint_root).expanduser()
    config_dir = Path(args.config_dir).expanduser() if args.config_dir else None
    output_dir = Path(args.output_dir).expanduser()

    for run in RUNS:
        run_base = run["run_base"]
        checkpoint = checkpoint_root / run["checkpoint"]
        output = output_dir / f"{run_base}_seed{args.seed}.npy"
        config = maybe_config(config_dir, run_base)

        cmd = [
            args.python_bin,
            "scripts/sample_cosmodiff.py",
            "--checkpoint",
            str(checkpoint),
        ]
        if config is not None:
            cmd.extend(["--config", str(config)])
        cmd.extend([
            "--output",
            str(output),
            "--num-samples",
            str(args.num_samples),
            "--batch-size",
            str(args.batch_size),
            "--image-size",
            str(run["image_size"]),
            "--seed",
            str(args.seed),
            "--device",
            args.device,
        ])

        print(" ".join(shlex.quote(part) for part in cmd), flush=True)
        if not args.dry_run:
            subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
