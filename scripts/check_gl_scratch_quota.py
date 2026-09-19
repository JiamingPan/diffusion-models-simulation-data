#!/usr/bin/env python
"""Fail closed unless Great Lakes scratch has explicit experiment headroom."""
from __future__ import annotations

import argparse
import re
import sys


def bytes_from_size(value: str) -> float:
    match = re.fullmatch(r"([0-9]+(?:\.[0-9]+)?)([KMGTPE])(?:i?B)?", value)
    if not match:
        raise ValueError(f"unrecognized quota size: {value}")
    return float(match.group(1)) * 1024 ** ("KMGTPE".index(match.group(2)) + 1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--min-gib", type=float, required=True)
    parser.add_argument("--min-files", type=int, required=True)
    args = parser.parse_args()
    rows = [line.split() for line in sys.stdin.read().replace("|", "").splitlines()
            if line.strip().startswith("gl-scratch FILESET")]
    if len(rows) != 1 or len(rows[0]) < 12:
        raise ValueError("could not identify exactly one gl-scratch FILESET quota row")
    row = rows[0]
    used, soft, hard = map(bytes_from_size, row[2:5])
    if float(row[5]) != 0:
        raise ValueError("scratch block accounting has nonzero in_doubt")
    block_headroom = min(soft, hard) - used
    file_used, file_soft, file_hard, file_doubt = map(int, row[7:11])
    file_headroom = min(file_soft, file_hard) - file_used
    if block_headroom < args.min_gib * 2**30:
        raise ValueError(f"scratch headroom is only {block_headroom / 2**30:.1f} GiB")
    if file_doubt != 0 or file_headroom < args.min_files:
        raise ValueError(f"scratch file headroom is ambiguous or too small: {file_headroom}")
    print(f"SCRATCH QUOTA PASSED: {block_headroom / 2**40:.2f} TiB and {file_headroom} files free")


if __name__ == "__main__":
    main()
