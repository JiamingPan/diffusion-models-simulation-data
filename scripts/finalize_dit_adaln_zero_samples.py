#!/usr/bin/env python
"""Validate one paired sample file and write its success receipt."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def scalar(data, key):
    return np.asarray(data[key]).item()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--plan-sha256", required=True)
    parser.add_argument("--index", type=int, required=True)
    args = parser.parse_args()
    if sha256(args.plan) != args.plan_sha256:
        raise ValueError("plan hash changed")
    row = json.loads(args.plan.read_text())["runs"][args.index]
    output = Path(row["sample_path"].format(seed=123, sample_label="dpm50_n512"))
    receipt_path = output.with_suffix(".complete.json")
    if receipt_path.exists():
        raise FileExistsError(f"preserving existing sample receipt: {receipt_path}")
    with np.load(output, allow_pickle=False) as data:
        samples = data["samples"]
        if samples.shape != (512, 1, 128, 128) or not np.isfinite(samples).all():
            raise ValueError(f"invalid generated sample tensor: {samples.shape}")
        if (scalar(data, "requested_checkpoint") != row["checkpoint_dir"]
                or scalar(data, "config_path") != row["config"]
                or scalar(data, "scheduler") != "DPMSolverMultistepScheduler"
                or int(scalar(data, "num_steps")) != 50
                or int(scalar(data, "seed")) != 123):
            raise ValueError("sample provenance differs from the frozen screen contract")
    receipt = {
        "status": "complete", "plan_sha256": args.plan_sha256,
        "run_name": row["run_name"], "config_sha256": row["config_sha256"],
        "samples": str(output), "samples_sha256": sha256(output),
        "shape": [512, 1, 128, 128], "seed": 123,
        "scheduler": "DPMSolverMultistepScheduler", "num_steps": 50,
    }
    pending = receipt_path.with_suffix(".json.pending")
    pending.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    pending.replace(receipt_path)
    print(f"SAMPLE RECEIPT COMPLETE: {receipt_path}")


if __name__ == "__main__":
    main()
