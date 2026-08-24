"""Atomic, fail-closed lifecycle for job terminal-status reports."""

from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import tempfile
from typing import Any


REPORT_SCHEMA_VERSION = 1
TERMINAL_STATUSES = frozenset({"PASS", "FAILED", "STALE"})
LIFECYCLE_FIELDS = frozenset(
    {
        "status",
        "producer_job_id",
        "producer_exit_code",
        "started_at_utc",
        "finalized_at_utc",
        "report_schema_version",
    }
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _normalized_job_id(value: str | int | None) -> str | None:
    return None if value is None else str(value)


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid terminal report {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"terminal report {path} must contain a JSON object")
    return value


def atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    """Durably replace *path* with one complete JSON object."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(
        dict(payload), indent=2, sort_keys=True, allow_nan=False
    ) + "\n"
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        text=True,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        try:
            directory_fd = os.open(path.parent, os.O_RDONLY)
        except OSError:
            directory_fd = None
        if directory_fd is not None:
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
    finally:
        if temporary.exists():
            temporary.unlink()


def start_report(
    path: Path,
    *,
    payload: Mapping[str, Any],
    producer_job_id: str | int | None,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Write the required initial INCOMPLETE state."""
    path = Path(path)
    if path.exists() and not overwrite:
        raise FileExistsError(f"refusing to overwrite terminal report: {path}")
    collision = LIFECYCLE_FIELDS.intersection(payload)
    if collision:
        raise ValueError(
            "payload may not set lifecycle fields: " + ", ".join(sorted(collision))
        )
    report = {
        **dict(payload),
        "status": "INCOMPLETE",
        "producer_job_id": _normalized_job_id(producer_job_id),
        "producer_exit_code": None,
        "started_at_utc": _utc_now(),
        "finalized_at_utc": None,
        "report_schema_version": REPORT_SCHEMA_VERSION,
    }
    atomic_write_json(path, report)
    return report


def update_incomplete_report(
    path: Path,
    *,
    payload: Mapping[str, Any],
    producer_job_id: str | int | None,
) -> dict[str, Any]:
    """Replace producer payload while preserving an INCOMPLETE lifecycle."""
    path = Path(path)
    collision = LIFECYCLE_FIELDS.intersection(payload)
    if collision:
        raise ValueError(
            "payload may not set lifecycle fields: " + ", ".join(sorted(collision))
        )
    current = _read_json(path)
    if current.get("report_schema_version") != REPORT_SCHEMA_VERSION:
        raise ValueError("terminal report has an unsupported or missing schema")
    if current.get("status") != "INCOMPLETE":
        raise ValueError(f"terminal report is already finalized as {current.get('status')}")
    expected_job_id = _normalized_job_id(producer_job_id)
    if current.get("producer_job_id") != expected_job_id:
        raise ValueError(
            "producer job ID mismatch: "
            f"report={current.get('producer_job_id')!r}, caller={expected_job_id!r}"
        )
    updated = {
        **dict(payload),
        "status": "INCOMPLETE",
        "producer_job_id": expected_job_id,
        "producer_exit_code": None,
        "started_at_utc": current.get("started_at_utc"),
        "finalized_at_utc": None,
        "report_schema_version": REPORT_SCHEMA_VERSION,
    }
    if not updated["started_at_utc"]:
        raise ValueError("terminal report lacks its start timestamp")
    atomic_write_json(path, updated)
    return updated


def finalize_report(
    path: Path,
    *,
    status: str,
    producer_job_id: str | int | None,
    producer_exit_code: int,
) -> dict[str, Any]:
    """Atomically transition an INCOMPLETE report to PASS or FAILED."""
    path = Path(path)
    requested_status = str(status)
    exit_code = int(producer_exit_code)
    if requested_status not in {"PASS", "FAILED"}:
        raise ValueError("finalization requires a terminal status of PASS or FAILED")
    if requested_status == "PASS" and exit_code != 0:
        raise ValueError("PASS requires producer_exit_code 0")
    if requested_status == "FAILED" and exit_code == 0:
        raise ValueError("FAILED requires a nonzero producer_exit_code")

    report = _read_json(path)
    if report.get("report_schema_version") != REPORT_SCHEMA_VERSION:
        raise ValueError("terminal report has an unsupported or missing schema")
    if report.get("status") != "INCOMPLETE":
        raise ValueError(f"terminal report is already finalized as {report.get('status')}")
    expected_job_id = _normalized_job_id(producer_job_id)
    if report.get("producer_job_id") != expected_job_id:
        raise ValueError(
            "producer job ID mismatch: "
            f"report={report.get('producer_job_id')!r}, caller={expected_job_id!r}"
        )
    if not report.get("started_at_utc") or report.get("finalized_at_utc") is not None:
        raise ValueError("terminal report has invalid lifecycle timestamps")

    report.update(
        status=requested_status,
        producer_exit_code=exit_code,
        finalized_at_utc=_utc_now(),
    )
    atomic_write_json(path, report)
    return report


def mark_stale(path: Path, *, reason: str) -> dict[str, Any]:
    """Explicitly invalidate a current or legacy report without deleting it."""
    path = Path(path)
    reason = str(reason).strip()
    if not reason:
        raise ValueError("a nonempty stale reason is required")
    report = _read_json(path)
    now = _utc_now()
    report.update(
        status="STALE",
        producer_job_id=_normalized_job_id(report.get("producer_job_id")),
        producer_exit_code=report.get("producer_exit_code"),
        started_at_utc=report.get("started_at_utc") or now,
        finalized_at_utc=now,
        report_schema_version=REPORT_SCHEMA_VERSION,
        stale_reason=reason,
    )
    atomic_write_json(path, report)
    return report


def require_passed_report(
    path: Path, *, expected_producer_job_id: str | int | None
) -> dict[str, Any]:
    """Load a report only when it is an exact finalized success."""
    path = Path(path)
    report = _read_json(path)
    if report.get("report_schema_version") != REPORT_SCHEMA_VERSION:
        raise ValueError("terminal report has an unsupported or missing schema")
    if report.get("status") != "PASS":
        raise ValueError(
            f"terminal report status is {report.get('status')!r}, not PASS"
        )
    if report.get("producer_exit_code") != 0:
        raise ValueError("PASS terminal report does not have producer exit code 0")
    expected = _normalized_job_id(expected_producer_job_id)
    if report.get("producer_job_id") != expected:
        raise ValueError(
            "producer job ID mismatch: "
            f"report={report.get('producer_job_id')!r}, expected={expected!r}"
        )
    if not report.get("started_at_utc") or not report.get("finalized_at_utc"):
        raise ValueError("PASS terminal report lacks finalized lifecycle timestamps")
    return report
