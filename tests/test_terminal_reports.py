from __future__ import annotations

import json
from pathlib import Path

import pytest

from simdiff_eval import terminal_reports


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
