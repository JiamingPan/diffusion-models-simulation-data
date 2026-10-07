#!/usr/bin/env python
"""Append-only experiment ledger.

One JSON object per line in experiments/ledger.jsonl. Entries are never edited;
a later line with the same run_name supersedes the earlier one. Status moves
planned -> submitted -> done | failed.

Usage:
  python scripts/ledger.py plan   --run R --question Q --prediction P --metric M --config C [--gpu-hours H]
  python scripts/ledger.py submit --run R --slurm-id ID
  python scripts/ledger.py result --run R --metrics-json PATH [--verdict supported|unsupported|mixed] [--note TEXT]
  python scripts/ledger.py fail   --run R --note TEXT
  python scripts/ledger.py show   [--run R]
  python scripts/ledger.py check-planned --run R      # exit 0 if R has a planned entry (used by the sbatch hook)
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LEDGER = ROOT / "experiments" / "ledger.jsonl"


def _now() -> str:
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat()


def _git_rev() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, text=True).strip()
    except Exception:
        return "unknown"


def _sha256(path: str | None) -> str | None:
    if not path:
        return None
    p = Path(path)
    if not p.is_absolute():
        p = ROOT / p
    if not p.is_file():
        return None
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _append(entry: dict) -> None:
    LEDGER.parent.mkdir(parents=True, exist_ok=True)
    entry = {"ts": _now(), "git": _git_rev(), **entry}
    with open(LEDGER, "a") as f:
        f.write(json.dumps(entry, sort_keys=True) + "\n")
    print(json.dumps(entry, sort_keys=True))


def _load() -> list[dict]:
    if not LEDGER.is_file():
        return []
    out = []
    with open(LEDGER) as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def _latest(run: str) -> dict | None:
    rows = [r for r in _load() if r.get("run") == run]
    return rows[-1] if rows else None


def cmd_plan(a):
    prev = _latest(a.run)
    if prev and prev.get("status") in ("planned", "submitted", "done"):
        sys.exit(f"refusing: {a.run} already has status {prev['status']} (use a new run name or 'fail' it first)")
    _append({
        "run": a.run, "status": "planned", "question": a.question, "prediction": a.prediction,
        "metric": a.metric, "config": a.config, "config_sha256": _sha256(a.config),
        "gpu_hours_est": a.gpu_hours, "author": os.environ.get("USER", "unknown"),
    })


def cmd_submit(a):
    prev = _latest(a.run)
    if not prev or prev.get("status") != "planned":
        sys.exit(f"refusing: {a.run} is not in status planned (found: {prev and prev.get('status')})")
    _append({"run": a.run, "status": "submitted", "slurm_id": a.slurm_id})


def cmd_result(a):
    prev = _latest(a.run)
    if not prev or prev.get("status") not in ("submitted", "planned"):
        sys.exit(f"refusing: {a.run} has status {prev and prev.get('status')}")
    p = Path(a.metrics_json)
    metrics = json.load(open(p)) if p.is_file() else None
    if metrics is None:
        sys.exit(f"metrics file not found: {a.metrics_json}")
    _append({"run": a.run, "status": "done", "metrics_path": str(p), "metrics": metrics,
             "verdict": a.verdict, "note": a.note})


def cmd_fail(a):
    _append({"run": a.run, "status": "failed", "note": a.note})


def cmd_show(a):
    rows = _load()
    if a.run:
        rows = [r for r in rows if r.get("run") == a.run]
    latest: dict[str, dict] = {}
    for r in rows:
        latest[r["run"]] = r
    for run, r in latest.items():
        print(f"{r['status']:9s} {run:50s} {r.get('metric','')!s:20s} {r.get('prediction','')!s:.60s}")


def cmd_check_planned(a):
    prev = _latest(a.run)
    ok = bool(prev and prev.get("status") == "planned")
    print("planned" if ok else f"not planned ({prev and prev.get('status')})")
    sys.exit(0 if ok else 1)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("plan"); p.add_argument("--run", required=True); p.add_argument("--question", required=True)
    p.add_argument("--prediction", required=True); p.add_argument("--metric", required=True)
    p.add_argument("--config", required=True); p.add_argument("--gpu-hours", type=float, default=0.0)
    p.set_defaults(fn=cmd_plan)
    p = sub.add_parser("submit"); p.add_argument("--run", required=True); p.add_argument("--slurm-id", required=True)
    p.set_defaults(fn=cmd_submit)
    p = sub.add_parser("result"); p.add_argument("--run", required=True); p.add_argument("--metrics-json", required=True)
    p.add_argument("--verdict", choices=["supported", "unsupported", "mixed"], default=None)
    p.add_argument("--note", default=""); p.set_defaults(fn=cmd_result)
    p = sub.add_parser("fail"); p.add_argument("--run", required=True); p.add_argument("--note", required=True)
    p.set_defaults(fn=cmd_fail)
    p = sub.add_parser("show"); p.add_argument("--run", default=None); p.set_defaults(fn=cmd_show)
    p = sub.add_parser("check-planned"); p.add_argument("--run", required=True); p.set_defaults(fn=cmd_check_planned)
    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
