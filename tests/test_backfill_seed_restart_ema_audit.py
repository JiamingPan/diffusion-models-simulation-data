from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "scripts" / "backfill_seed_restart_ema_audit.py"


def load_module():
    spec = importlib.util.spec_from_file_location(
        "backfill_seed_restart_ema_audit_for_test", SCRIPT_PATH
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def make_incomplete_audit(tmp_path: Path) -> Path:
    checkpoint = tmp_path / "checkpoint-epoch-9374"
    checkpoint.mkdir()
    (checkpoint / "checkpoint_config.yaml").write_text(
        "ema_sigma_rels: [0.02, 0.1]\nema_burn_in: 1000\n"
    )
    ema_dir = checkpoint / "ema"
    ema_dir.mkdir()
    for index in range(2):
        (ema_dir / f"{index}.1199000.pt").write_bytes(f"profile-{index}".encode())

    target = tmp_path / "checkpoint-epoch-10624"
    target.mkdir()
    audit_path = tmp_path / "resume_audit.json"
    audit_path.write_text(
        json.dumps(
            {
                "checkpoint": str(checkpoint),
                "target_checkpoint": str(target),
                "expected_ema_step": 1_199_000,
                "ema_sigma_rels": [0.02, 0.1],
                "original_ema_burn_in": 1_000,
                "first_resumed_loss": 0.125,
                "target_checkpoint_state": {"validated": True},
            }
        )
        + "\n"
    )
    return audit_path


def test_backfill_recovers_ema_restore_from_immutable_checkpoint_inventory(tmp_path):
    module = load_module()
    audit_path = make_incomplete_audit(tmp_path)

    result = module.backfill_one(audit_path, code_revision="fix123")
    stored = json.loads(audit_path.read_text())

    assert result["status"] == "backfilled"
    assert stored["ema_restore"] == {
        "step": 1_199_000,
        "profiles": 2,
        "sigma_rels": [0.02, 0.1],
        "burn_in": 1_000,
        "snapshots": [
            str(tmp_path / "checkpoint-epoch-9374" / "ema" / "0.1199000.pt"),
            str(tmp_path / "checkpoint-epoch-9374" / "ema" / "1.1199000.pt"),
        ],
    }
    assert stored["audit_recovery"] == {
        "kind": "ema_restore_backfill",
        "reason": "recovered_after_seed_restart_audit_last_writer_wins_bug",
        "source": "immutable_source_checkpoint_ema_inventory",
        "tool_code_revision": "fix123",
    }
    assert stored["first_resumed_loss"] == 0.125
    assert stored["target_checkpoint_state"] == {"validated": True}


def test_backfill_is_idempotent_after_success(tmp_path):
    module = load_module()
    audit_path = make_incomplete_audit(tmp_path)
    module.backfill_one(audit_path, code_revision="fix123")
    before = audit_path.read_bytes()

    result = module.backfill_one(audit_path, code_revision="fix123")

    assert result["status"] == "already_backfilled"
    assert audit_path.read_bytes() == before


def test_backfill_refuses_missing_snapshot_without_modifying_audit(tmp_path):
    module = load_module()
    audit_path = make_incomplete_audit(tmp_path)
    missing = tmp_path / "checkpoint-epoch-9374" / "ema" / "1.1199000.pt"
    missing.unlink()
    before = audit_path.read_bytes()

    with pytest.raises(FileNotFoundError, match="EMA snapshot"):
        module.backfill_one(audit_path, code_revision="fix123")

    assert audit_path.read_bytes() == before


@pytest.mark.parametrize("missing_key", ["first_resumed_loss", "target_checkpoint_state"])
def test_backfill_refuses_audit_without_success_evidence(tmp_path, missing_key):
    module = load_module()
    audit_path = make_incomplete_audit(tmp_path)
    payload = json.loads(audit_path.read_text())
    payload.pop(missing_key)
    audit_path.write_text(json.dumps(payload) + "\n")
    before = audit_path.read_bytes()

    with pytest.raises(ValueError, match=missing_key):
        module.backfill_one(audit_path, code_revision="fix123")

    assert audit_path.read_bytes() == before
