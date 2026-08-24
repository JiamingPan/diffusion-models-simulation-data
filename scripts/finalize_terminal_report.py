#!/usr/bin/env python
"""Validate and finalize shared terminal-status reports."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from simdiff_eval.terminal_reports import (
    finalize_report,
    mark_stale,
    require_passed_report,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    finalize = subparsers.add_parser("finalize")
    finalize.add_argument("path", type=Path)
    finalize.add_argument("--status", choices=("PASS", "FAILED"), required=True)
    finalize.add_argument("--job-id", required=True)
    finalize.add_argument("--exit-code", required=True, type=int)

    require = subparsers.add_parser("require-pass")
    require.add_argument("path", type=Path)
    require.add_argument("--expected-job-id", required=True)

    stale = subparsers.add_parser("mark-stale")
    stale.add_argument("path", type=Path)
    stale.add_argument("--reason", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "finalize":
        report = finalize_report(
            args.path,
            status=args.status,
            producer_job_id=args.job_id,
            producer_exit_code=args.exit_code,
        )
    elif args.command == "require-pass":
        report = require_passed_report(
            args.path, expected_producer_job_id=args.expected_job_id
        )
    else:
        report = mark_stale(args.path, reason=args.reason)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
