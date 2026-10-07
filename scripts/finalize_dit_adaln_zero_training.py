#!/usr/bin/env python
"""Validate a completed screen checkpoint and write its success receipt."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from dit_checkpoint_path import resolve_checkpoint


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--plan-sha256", required=True)
    parser.add_argument("--index", type=int, required=True)
    args = parser.parse_args()
    if sha256(args.plan) != args.plan_sha256:
        raise ValueError("plan hash changed")
    row = json.loads(args.plan.read_text())["runs"][args.index]
    root = Path(row["checkpoint_dir"])
    final = resolve_checkpoint(root, row['checkpoint_epoch'])
    receipt_path = root / "training_complete.json"
    if receipt_path.exists():
        raise FileExistsError(f"preserving existing training receipt: {receipt_path}")
    config_path = final / "config.json"
    weight_paths = sorted(final.glob("*.safetensors")) + sorted(final.glob("pytorch_model*.bin"))
    if not final.is_dir() or not config_path.is_file() or not weight_paths:
        raise FileNotFoundError(f"final diffusers checkpoint is incomplete: {final}")
    config = json.loads(config_path.read_text())
    if (config.get("_class_name") != "DiTTransformer2DModel"
            or config.get("num_layers") != 16 or config.get("patch_size") != 8):
        raise ValueError("saved checkpoint is not the selected L16 patch-8 model")
    artifacts = {str(path.relative_to(root)): sha256(path)
                 for path in [config_path, *weight_paths]}
    receipt = {
        "status": "complete", "plan_sha256": args.plan_sha256,
        "run_name": row["run_name"], "config_sha256": row["config_sha256"],
        "checkpoint_epoch": row["checkpoint_epoch"], "final_checkpoint": str(final),
        "artifacts_sha256": artifacts,
    }
    pending = receipt_path.with_suffix(".json.pending")
    pending.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    pending.replace(receipt_path)
    print(f"TRAINING RECEIPT COMPLETE: {receipt_path}")


if __name__ == "__main__":
    main()
