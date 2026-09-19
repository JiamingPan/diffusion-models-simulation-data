#!/usr/bin/env python
"""Audit every selected training subset and publish an immutable receipt."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import torch
import yaml

from dit_zero_screen_data import audit_native_dataset, load_native_training_reference


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--plan-sha256", required=True)
    parser.add_argument("--cosmodiff-manifest-sha256", required=True)
    args = parser.parse_args()
    if sha256(args.plan) != args.plan_sha256:
        raise ValueError("plan hash changed")
    receipt_path = args.plan.parent / "data_preflight.json"
    if receipt_path.exists():
        raise FileExistsError(f"preserving existing data receipt: {receipt_path}")
    if os.environ.get("CUDA_VISIBLE_DEVICES") not in (None, "", "NoDevFiles"):
        raise RuntimeError("data preflight must remain CPU-only")
    from cosmodiff import utils
    plan = json.loads(args.plan.read_text())
    rows = []
    for row in plan["runs"]:
        config = yaml.safe_load(Path(row["config"]).read_text())
        reference, metadata = load_native_training_reference(config)
        parsed = utils.parse_config_data(config)
        audit = audit_native_dataset(parsed, reference, metadata, torch)
        if audit["shape"] != [row["dataset_size"], 1, 128, 128]:
            raise ValueError(f"configured N mismatch for {row['run_name']}: {audit['shape']}")
        audit.update(run_name=row["run_name"], dataset_size=row["dataset_size"],
                     config_sha256=row["config_sha256"])
        rows.append(audit)
        print(f"NATIVE DATA MATCH PASSED: {row['run_name']} {audit}", flush=True)
        del parsed, reference
    receipt = {
        "status": "complete", "plan_sha256": args.plan_sha256,
        "cosmodiff_manifest_sha256": args.cosmodiff_manifest_sha256,
        "runs": rows,
    }
    pending = receipt_path.with_suffix(".json.pending")
    pending.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    pending.replace(receipt_path)
    print(f"DATA PREFLIGHT COMPLETE: {receipt_path}; no model load or GPU work", flush=True)


if __name__ == "__main__":
    main()
