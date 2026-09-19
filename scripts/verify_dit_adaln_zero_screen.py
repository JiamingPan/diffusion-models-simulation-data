#!/usr/bin/env python
"""Fail-closed validation for a selected adaLN-Zero DiT screen plan."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess

import yaml


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify_training_receipt(row, plan_sha256):
    root = Path(row["checkpoint_dir"])
    receipt_path = root / "training_complete.json"
    if not receipt_path.is_file():
        raise FileNotFoundError(f"completed training receipt is missing: {receipt_path}")
    receipt = json.loads(receipt_path.read_text())
    expected_final = root / f"checkpoint-epoch-{row['checkpoint_epoch']}"
    if (receipt.get("status") != "complete" or receipt.get("plan_sha256") != plan_sha256
            or receipt.get("run_name") != row["run_name"]
            or receipt.get("config_sha256") != row["config_sha256"]
            or receipt.get("checkpoint_epoch") != row["checkpoint_epoch"]
            or receipt.get("final_checkpoint") != str(expected_final)):
        raise ValueError(f"training receipt changed: {receipt_path}")
    artifacts = receipt.get("artifacts_sha256", {})
    if not artifacts:
        raise ValueError(f"training receipt has no artifacts: {receipt_path}")
    for relative, expected_sha in artifacts.items():
        if sha(root / relative) != expected_sha:
            raise ValueError(f"checkpoint artifact changed: {root / relative}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--plan-sha256", required=True)
    parser.add_argument("--cosmodiff-root", type=Path, required=True)
    parser.add_argument("--cosmodiff-commit", required=True)
    parser.add_argument("--cosmodiff-manifest", type=Path, required=True)
    parser.add_argument("--cosmodiff-manifest-sha256", required=True)
    parser.add_argument("--index", type=int)
    parser.add_argument("--allow-checkpoint", action="store_true")
    parser.add_argument("--require-data-receipt", action="store_true")
    args = parser.parse_args()
    if sha(args.plan) != args.plan_sha256:
        raise ValueError("plan hash changed")
    plan = json.loads(args.plan.read_text())
    runs = plan["runs"]
    if plan["status"] != "draft_not_launch_ready" or not runs:
        raise ValueError("unexpected plan status or empty run list")
    if args.index is not None:
        if not 0 <= args.index < len(runs):
            raise ValueError("array index outside plan")
        runs = [runs[args.index]]
    commit = subprocess.check_output(
        ["git", "-C", str(args.cosmodiff_root), "rev-parse", "HEAD"], text=True).strip()
    if commit != args.cosmodiff_commit:
        raise ValueError("cosmodiff commit changed")
    if sha(args.cosmodiff_manifest) != args.cosmodiff_manifest_sha256:
        raise ValueError("cosmodiff runtime manifest changed")
    manifest = json.loads(args.cosmodiff_manifest.read_text())
    if manifest.get("schema_version") != 1 or manifest.get("base_commit") != commit:
        raise ValueError("cosmodiff runtime manifest contract changed")
    expected_files = set(manifest.get("files", {}))
    actual_files = {
        str(path.relative_to(args.cosmodiff_root))
        for path in (args.cosmodiff_root / "cosmodiff").rglob("*.py")
    }
    actual_files.add("scripts/cosmodiff_train.py")
    if actual_files != expected_files:
        raise ValueError("cosmodiff Python runtime file set changed")
    for relative, expected_sha in manifest["files"].items():
        if sha(args.cosmodiff_root / relative) != expected_sha:
            raise ValueError(f"cosmodiff runtime file changed: {relative}")
    if args.require_data_receipt:
        receipt_path = args.plan.parent / "data_preflight.json"
        if not receipt_path.is_file():
            raise FileNotFoundError(f"completed data preflight is missing: {receipt_path}")
        receipt = json.loads(receipt_path.read_text())
        if (receipt.get("status") != "complete"
                or receipt.get("plan_sha256") != args.plan_sha256
                or receipt.get("cosmodiff_manifest_sha256") != args.cosmodiff_manifest_sha256
                or [item.get("run_name") for item in receipt.get("runs", [])]
                    != [item["run_name"] for item in plan["runs"]]
                or [item.get("config_sha256") for item in receipt.get("runs", [])]
                    != [item["config_sha256"] for item in plan["runs"]]):
            raise ValueError("data preflight receipt changed or does not match the plan")
    utils = (args.cosmodiff_root / "cosmodiff/utils.py").read_text()
    optim = (args.cosmodiff_root / "cosmodiff/optim.py").read_text()
    if "initialize_dit_adaln_zero" not in utils:
        raise ValueError("zero-init factory is missing")
    if "constant_label = data_cfg.get" not in utils:
        raise ValueError("constant-label support is missing")
    class_label_markers = (
        "class_labels=batch_labels", "class_labels=labels",
        "'class_labels': labels", '"class_labels": labels',
    )
    if not any(marker in optim for marker in class_label_markers):
        raise ValueError("DiT class-label forward support is missing")
    for row in runs:
        cfg_path = Path(row["config"])
        if sha(cfg_path) != row["config_sha256"]:
            raise ValueError(f"config hash changed: {cfg_path}")
        cfg = yaml.safe_load(cfg_path.read_text())
        if (row["num_layers"] != 16 or row["target_updates"] != 300000
                or row["patch_size"] != 8 or row["training_seed"] != 123):
            raise ValueError("screen contract changed")
        if (cfg["model"].get("initialization") != "adaln_zero"
                or cfg["model"]["kwargs"].get("norm_type") != "ada_norm_zero"
                or cfg["model"]["kwargs"].get("patch_size") != 8):
            raise ValueError("zero-init model contract changed")
        if not args.allow_checkpoint and Path(row["checkpoint_dir"]).exists():
            raise FileExistsError(f"fresh checkpoint destination already exists: {row['checkpoint_dir']}")
        if args.allow_checkpoint and not Path(row["checkpoint_dir"]).is_dir():
            raise FileNotFoundError(f"checkpoint destination is missing: {row['checkpoint_dir']}")
        if args.allow_checkpoint:
            verify_training_receipt(row, args.plan_sha256)
    print(f"SCREEN PREFLIGHT PASSED: {len(runs)} fresh L16 runs; no training performed")


if __name__ == "__main__":
    main()
