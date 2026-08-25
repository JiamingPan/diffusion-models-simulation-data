from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
from unittest import mock

import pytest

from simdiff_eval import terminal_reports


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_terminal_report_pass_lifecycle_is_atomic_and_validated(tmp_path):
    path = tmp_path / "report.json"
    started = terminal_reports.start_report(
        path,
        payload={"rows": [{"dataset_tag": "d2p08"}]},
        producer_job_id="12345",
    )

    assert json.loads(path.read_text()) == started
    assert started["status"] == "INCOMPLETE"
    assert started["producer_job_id"] == "12345"
    assert started["producer_exit_code"] is None
    assert started["finalized_at_utc"] is None
    assert not list(tmp_path.glob(".report.json.*.tmp"))

    passed = terminal_reports.finalize_report(
        path,
        status="PASS",
        producer_job_id="12345",
        producer_exit_code=0,
    )

    assert passed["rows"] == [{"dataset_tag": "d2p08"}]
    assert passed["status"] == "PASS"
    assert passed["producer_exit_code"] == 0
    assert passed["finalized_at_utc"]
    assert terminal_reports.require_passed_report(
        path, expected_producer_job_id="12345"
    ) == passed


@pytest.mark.parametrize(
    ("status", "exit_code", "message"),
    [
        ("PASS", 7, "PASS requires producer_exit_code 0"),
        ("FAILED", 0, "FAILED requires a nonzero producer_exit_code"),
        ("INCOMPLETE", 0, "terminal status"),
    ],
)
def test_terminal_report_rejects_inconsistent_final_state(
    tmp_path, status, exit_code, message
):
    path = tmp_path / "report.json"
    terminal_reports.start_report(path, payload={}, producer_job_id="12")

    with pytest.raises(ValueError, match=message):
        terminal_reports.finalize_report(
            path,
            status=status,
            producer_job_id="12",
            producer_exit_code=exit_code,
        )

    assert json.loads(path.read_text())["status"] == "INCOMPLETE"


def test_terminal_report_rejects_wrong_producer_and_second_finalization(tmp_path):
    path = tmp_path / "report.json"
    terminal_reports.start_report(path, payload={}, producer_job_id="12")

    with pytest.raises(ValueError, match="producer job ID"):
        terminal_reports.finalize_report(
            path,
            status="PASS",
            producer_job_id="13",
            producer_exit_code=0,
        )

    terminal_reports.finalize_report(
        path, status="FAILED", producer_job_id="12", producer_exit_code=9
    )
    with pytest.raises(ValueError, match="already finalized"):
        terminal_reports.finalize_report(
            path, status="PASS", producer_job_id="12", producer_exit_code=0
        )


def test_pass_consumer_rejects_legacy_incomplete_failed_and_wrong_job(tmp_path):
    path = tmp_path / "report.json"
    path.write_text(json.dumps({"status": "PASS"}))
    with pytest.raises(ValueError, match="schema"):
        terminal_reports.require_passed_report(
            path, expected_producer_job_id="12345"
        )

    terminal_reports.start_report(
        path, payload={}, producer_job_id="12345", overwrite=True
    )
    with pytest.raises(ValueError, match="INCOMPLETE"):
        terminal_reports.require_passed_report(
            path, expected_producer_job_id="12345"
        )

    terminal_reports.finalize_report(
        path, status="FAILED", producer_job_id="12345", producer_exit_code=4
    )
    with pytest.raises(ValueError, match="FAILED"):
        terminal_reports.require_passed_report(
            path, expected_producer_job_id="12345"
        )

    terminal_reports.start_report(
        path, payload={}, producer_job_id="12345", overwrite=True
    )
    terminal_reports.finalize_report(
        path, status="PASS", producer_job_id="12345", producer_exit_code=0
    )
    with pytest.raises(ValueError, match="producer job ID"):
        terminal_reports.require_passed_report(
            path, expected_producer_job_id="54321"
        )


def test_mark_stale_preserves_payload_and_records_reason(tmp_path):
    path = tmp_path / "legacy.json"
    path.write_text(json.dumps({"status": "PASS", "rows": [1, 2]}))

    stale = terminal_reports.mark_stale(path, reason="superseded unsafe report")

    assert stale["status"] == "STALE"
    assert stale["stale_reason"] == "superseded unsafe report"
    assert stale["rows"] == [1, 2]
    assert stale["report_schema_version"] == terminal_reports.REPORT_SCHEMA_VERSION


def test_start_report_refuses_overwrite_by_default(tmp_path):
    path = tmp_path / "report.json"
    path.write_text("do not replace")

    with pytest.raises(FileExistsError):
        terminal_reports.start_report(path, payload={}, producer_job_id=None)

    assert path.read_text() == "do not replace"


def test_incomplete_report_payload_can_be_enriched_before_finalization(tmp_path):
    path = tmp_path / "report.json"
    terminal_reports.start_report(
        path,
        payload={"analysis": "c4-v3"},
        producer_job_id="98",
    )

    enriched = terminal_reports.update_incomplete_report(
        path,
        payload={"analysis": "c4-v3", "row_count": 14336},
        producer_job_id="98",
    )

    assert enriched["status"] == "INCOMPLETE"
    assert enriched["row_count"] == 14336
    assert enriched["started_at_utc"]
    with pytest.raises(ValueError, match="producer job ID"):
        terminal_reports.update_incomplete_report(
            path, payload={}, producer_job_id="wrong"
        )


def test_merge_json_object_is_shallow_and_preserves_independent_fields(tmp_path):
    path = tmp_path / "resume_audit.json"
    terminal_reports.atomic_write_json(
        path,
        {
            "ema_restore": {"step": 1_199_000, "profiles": 2},
            "checkpoint": "/runs/checkpoint-epoch-9374",
        },
    )

    merged = terminal_reports.merge_json_object(
        path,
        {
            "first_resumed_loss": 0.125,
            "checkpoint": "/runs/checkpoint-epoch-9374",
        },
    )

    assert merged == json.loads(path.read_text())
    assert merged["ema_restore"] == {"step": 1_199_000, "profiles": 2}
    assert merged["first_resumed_loss"] == 0.125


def test_merge_json_object_rejects_conflicting_top_level_value(tmp_path):
    path = tmp_path / "resume_audit.json"
    terminal_reports.atomic_write_json(path, {"resume_seed": 456, "kept": True})

    with pytest.raises(ValueError, match="resume_seed"):
        terminal_reports.merge_json_object(path, {"resume_seed": 123})

    assert json.loads(path.read_text()) == {"resume_seed": 456, "kept": True}


def test_merge_json_object_never_deep_merges_nested_values(tmp_path):
    path = tmp_path / "resume_audit.json"
    terminal_reports.atomic_write_json(path, {"ema_restore": {"step": 10}})

    with pytest.raises(ValueError, match="ema_restore"):
        terminal_reports.merge_json_object(
            path,
            {"ema_restore": {"step": 10, "profiles": 2}},
        )

    assert json.loads(path.read_text()) == {"ema_restore": {"step": 10}}


@pytest.mark.parametrize("rank_variable", ["RANK", "SLURM_PROCID", "LOCAL_RANK"])
def test_merge_json_object_refuses_nonzero_rank_writer(
    tmp_path, monkeypatch, rank_variable
):
    path = tmp_path / "resume_audit.json"
    for name in ("RANK", "SLURM_PROCID", "LOCAL_RANK"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv(rank_variable, "1")

    with pytest.raises(RuntimeError, match="rank 0"):
        terminal_reports.merge_json_object(path, {"rank": 1})

    assert not path.exists()


def test_interrupted_merge_keeps_valid_json_and_next_merge_recovers(tmp_path):
    path = tmp_path / "resume_audit.json"
    terminal_reports.atomic_write_json(path, {"ema_restore": {"step": 10}})

    with mock.patch.object(
        terminal_reports.os,
        "replace",
        side_effect=OSError("simulated interruption before publication"),
    ):
        with pytest.raises(OSError, match="simulated interruption"):
            terminal_reports.merge_json_object(path, {"first_resumed_loss": 0.5})

    assert json.loads(path.read_text()) == {"ema_restore": {"step": 10}}
    assert not list(tmp_path.glob(".resume_audit.json.*.tmp"))

    recovered = terminal_reports.merge_json_object(
        path, {"first_resumed_loss": 0.5}
    )
    assert recovered == {
        "ema_restore": {"step": 10},
        "first_resumed_loss": 0.5,
    }


def test_concurrent_process_merges_preserve_every_distinct_key(tmp_path):
    path = tmp_path / "resume_audit.json"
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(REPO_ROOT)
    for name in ("RANK", "SLURM_PROCID", "LOCAL_RANK"):
        environment.pop(name, None)
    program = """
import json
import sys
from pathlib import Path
from simdiff_eval.terminal_reports import merge_json_object

merge_json_object(Path(sys.argv[1]), {sys.argv[2]: json.loads(sys.argv[3])})
"""
    processes = [
        subprocess.Popen(
            [sys.executable, "-c", program, str(path), f"writer_{index}", str(index)],
            cwd=REPO_ROOT,
            env=environment,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        for index in range(8)
    ]

    for process in processes:
        stdout, stderr = process.communicate(timeout=30)
        assert process.returncode == 0, (stdout, stderr)

    assert json.loads(path.read_text()) == {
        f"writer_{index}": index for index in range(8)
    }
