#!/usr/bin/env python
"""Recover EMA evidence erased by the seed-restart audit overwrite bug."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
from typing import Any

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from simdiff_eval.terminal_reports import merge_json_object


RECOVERY_REASON = "recovered_after_seed_restart_audit_last_writer_wins_bug"


def _read_object(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Invalid seed-restart audit {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"Seed-restart audit must contain a JSON object: {path}")
    return payload


def _required(payload: dict[str, Any], key: str) -> Any:
    if key not in payload:
        raise ValueError(f"Seed-restart audit lacks required {key!r} evidence")
    return payload[key]


def _build_ema_restore(payload: dict[str, Any]) -> dict[str, Any]:
    first_loss = _required(payload, "first_resumed_loss")
    if not isinstance(first_loss, (int, float)) or not math.isfinite(float(first_loss)):
        raise ValueError("first_resumed_loss must be one finite number")
    target_state = _required(payload, "target_checkpoint_state")
    if not isinstance(target_state, dict) or not target_state:
        raise ValueError("target_checkpoint_state must be one nonempty object")

    target_checkpoint = Path(_required(payload, "target_checkpoint"))
    if not target_checkpoint.is_dir():
        raise FileNotFoundError(
            f"Validated target checkpoint directory is missing: {target_checkpoint}"
        )

    checkpoint = Path(_required(payload, "checkpoint"))
    config_path = checkpoint / "checkpoint_config.yaml"
    if not config_path.is_file():
        raise FileNotFoundError(f"Checkpoint EMA metadata is missing: {config_path}")
    metadata = yaml.safe_load(config_path.read_text())
    if not isinstance(metadata, dict):
        raise ValueError(f"Checkpoint EMA metadata is not a mapping: {config_path}")

    expected_step = int(_required(payload, "expected_ema_step"))
    expected_sigma_rels = [
        float(value) for value in _required(payload, "ema_sigma_rels")
    ]
    expected_burn_in = int(_required(payload, "original_ema_burn_in"))
    actual_sigma_rels = [
        float(value) for value in metadata.get("ema_sigma_rels", [])
    ]
    actual_burn_in = int(metadata.get("ema_burn_in", -1))
    if actual_sigma_rels != expected_sigma_rels:
        raise ValueError(
            "Checkpoint EMA sigma profiles disagree with the audit: "
            f"checkpoint={actual_sigma_rels}, audit={expected_sigma_rels}"
        )
    if actual_burn_in != expected_burn_in:
        raise ValueError(
            "Checkpoint EMA burn-in disagrees with the audit: "
            f"checkpoint={actual_burn_in}, audit={expected_burn_in}"
        )

    snapshots = [
        checkpoint / "ema" / f"{profile_index}.{expected_step}.pt"
        for profile_index in range(len(expected_sigma_rels))
    ]
    missing = [str(path) for path in snapshots if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing exact EMA snapshot(s): " + ", ".join(missing))

    return {
        "step": expected_step,
        "profiles": len(snapshots),
        "sigma_rels": actual_sigma_rels,
        "burn_in": actual_burn_in,
        "snapshots": [str(path) for path in snapshots],
    }


def backfill_one(
    audit_path: Path,
    *,
    code_revision: str,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Verify and backfill one legacy audit without changing checkpoints."""
    audit_path = Path(audit_path)
    payload = _read_object(audit_path)
    existing_restore = payload.get("ema_restore")
    existing_recovery = payload.get("audit_recovery")
    if existing_restore is not None:
        status = "already_backfilled" if existing_recovery else "already_complete"
        return {"audit": str(audit_path), "status": status}
    if existing_recovery is not None:
        raise ValueError("audit_recovery exists without ema_restore")

    ema_restore = _build_ema_restore(payload)
    recovery = {
        "kind": "ema_restore_backfill",
        "reason": RECOVERY_REASON,
        "source": "immutable_source_checkpoint_ema_inventory",
        "tool_code_revision": str(code_revision),
    }
    if not dry_run:
        merge_json_object(
            audit_path,
            {"ema_restore": ema_restore, "audit_recovery": recovery},
        )
    return {
        "audit": str(audit_path),
        "status": "verified_dry_run" if dry_run else "backfilled",
        "ema_restore": ema_restore,
        "audit_recovery": recovery,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--audit",
        action="append",
        required=True,
        type=Path,
        help="Legacy seed-restart audit JSON; repeat for multiple reports.",
    )
    parser.add_argument("--code-revision", required=True)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    for audit_path in args.audit:
        result = backfill_one(
            audit_path,
            code_revision=args.code_revision,
            dry_run=args.dry_run,
        )
        print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
