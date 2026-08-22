#!/usr/bin/env python
"""Write minimal distribution metadata for an uninstalled source checkout."""

from __future__ import annotations

import argparse
import re
from pathlib import Path


FIELD_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._+!-]*$")


def write_distribution_metadata(
    root: str | Path, *, distribution: str, version: str
) -> Path:
    """Make ``importlib.metadata.version`` work without copying package code."""
    if not FIELD_RE.fullmatch(distribution):
        raise ValueError(f"invalid distribution name: {distribution!r}")
    if not FIELD_RE.fullmatch(version):
        raise ValueError(f"invalid distribution version: {version!r}")

    root = Path(root)
    dist_info = root / f"{distribution.replace('-', '_')}-{version}.dist-info"
    metadata = (
        "Metadata-Version: 2.1\n"
        f"Name: {distribution}\n"
        f"Version: {version}\n"
    )
    metadata_path = dist_info / "METADATA"
    if metadata_path.exists() and metadata_path.read_text() != metadata:
        raise RuntimeError(f"conflicting source metadata: {metadata_path}")
    dist_info.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(metadata)
    print(f"wrote source distribution metadata: {metadata_path}")
    return dist_info


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("distribution")
    parser.add_argument("version")
    args = parser.parse_args()
    write_distribution_metadata(
        args.root,
        distribution=args.distribution,
        version=args.version,
    )


if __name__ == "__main__":
    main()
