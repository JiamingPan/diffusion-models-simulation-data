#!/usr/bin/env python
"""Freeze or verify the Python source tree used by the DiT screen."""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def runtime_files(root: Path) -> list[Path]:
    files = sorted((root / "cosmodiff").rglob("*.py"))
    files.append(root / "scripts/cosmodiff_train.py")
    missing = [path for path in files if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing runtime files: {missing}")
    return files


def build_manifest(root: Path, expected_commit: str) -> dict:
    root = root.resolve()
    actual_commit = subprocess.check_output(
        ["git", "-C", str(root), "rev-parse", "HEAD"], text=True
    ).strip()
    if actual_commit != expected_commit:
        raise ValueError(
            f"cosmodiff base commit changed: expected {expected_commit}, got {actual_commit}"
        )
    return {
        "schema_version": 1,
        "base_commit": actual_commit,
        "root_at_creation": str(root),
        "files": {
            str(path.relative_to(root)): sha256(path) for path in runtime_files(root)
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--commit", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(f"refusing to overwrite runtime manifest: {args.out}")
    manifest = build_manifest(args.root, args.commit)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(
        f"COSMODIFF RUNTIME MANIFEST WRITTEN: {args.out}; "
        f"files={len(manifest['files'])}; sha256={sha256(args.out)}"
    )


if __name__ == "__main__":
    main()
