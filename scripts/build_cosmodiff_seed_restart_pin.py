#!/usr/bin/env python
"""Build and atomically publish an audited cosmodiff seed-restart pin."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import tarfile
import tempfile
from typing import Any


PIN_MANIFEST_NAME = "seed_restart_pin_manifest.json"
EXPECTED_PATCH_NAMES = (
    "patch_cosmodiff_package_metadata.py",
    "patch_cosmodiff_constant_label.py",
    "patch_cosmodiff_dit_class_labels.py",
    "patch_cosmodiff_checkpoint_state.py",
)
PATCH_TARGETS = {
    "patch_cosmodiff_package_metadata.py": ("cosmodiff/__init__.py",),
    "patch_cosmodiff_constant_label.py": ("cosmodiff/utils.py",),
    "patch_cosmodiff_dit_class_labels.py": (
        "cosmodiff/optim.py",
        "cosmodiff/utils.py",
    ),
    "patch_cosmodiff_checkpoint_state.py": ("cosmodiff/optim.py",),
}
REQUIRED_IMPORTS = (
    "cosmodiff",
    "cosmodiff.optim",
    "cosmodiff.utils",
    "cosmodiff.augment",
    "cosmodiff.transform",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def file_inventory(root: Path) -> dict[str, dict[str, Any]]:
    root = Path(root).resolve()
    records: dict[str, dict[str, Any]] = {}
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(root).as_posix()
        if relative == PIN_MANIFEST_NAME:
            continue
        records[relative] = {
            "sha256": sha256_file(path),
            "bytes": path.stat().st_size,
        }
    return records


def _resolved_revision(source_repo: Path, revision: str) -> str:
    return subprocess.run(
        ["git", "rev-parse", f"{revision}^{{commit}}"],
        cwd=source_repo,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _extract_revision(source_repo: Path, revision: str, destination: Path) -> None:
    archive = subprocess.run(
        ["git", "archive", "--format=tar", revision],
        cwd=source_repo,
        check=True,
        capture_output=True,
    ).stdout
    with tarfile.open(fileobj=io.BytesIO(archive), mode="r:") as payload:
        for member in payload.getmembers():
            target = (destination / member.name).resolve()
            if destination.resolve() not in target.parents and target != destination.resolve():
                raise RuntimeError(f"unsafe path in source archive: {member.name}")
            if member.isdir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            if not member.isfile():
                raise RuntimeError(
                    f"unsupported non-file entry in source archive: {member.name}"
                )
            target.parent.mkdir(parents=True, exist_ok=True)
            source = payload.extractfile(member)
            if source is None:
                raise RuntimeError(f"could not read source archive entry: {member.name}")
            with source, target.open("wb") as handle:
                shutil.copyfileobj(source, handle)
            target.chmod(member.mode)


def _remove_patch_backups(root: Path) -> None:
    for path in root.rglob("*.codex_*.bak"):
        path.unlink()


def _target_hashes(root: Path, targets: tuple[str, ...]) -> dict[str, str]:
    hashes: dict[str, str] = {}
    for relative in targets:
        path = root / relative
        if not path.is_file():
            raise FileNotFoundError(f"declared patch target is missing: {path}")
        hashes[relative] = sha256_file(path)
    return hashes


def imported_modules(python_bin: Path, root: Path) -> dict[str, Any]:
    program = (
        "import importlib,json,pathlib\n"
        f"names={list(REQUIRED_IMPORTS)!r}\n"
        "result={}\n"
        "for name in names:\n"
        "    module=importlib.import_module(name)\n"
        "    path=pathlib.Path(module.__file__).resolve()\n"
        "    result[name]=str(path)\n"
        "import cosmodiff\n"
        "print(json.dumps({'paths':result,'version':cosmodiff.__version__},sort_keys=True))\n"
    )
    env = dict(os.environ)
    env["PYTHONNOUSERSITE"] = "1"
    env["PYTHONPATH"] = str(root)
    try:
        completed = subprocess.run(
            [str(python_bin), "-c", program],
            cwd=root,
            env=env,
            check=True,
            capture_output=True,
            text=True,
        )
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(
            "cosmodiff pin import validation failed:\n" + (exc.stderr or exc.stdout)
        ) from exc
    result = json.loads(completed.stdout.strip().splitlines()[-1])
    root = root.resolve()
    relative_paths: dict[str, str] = {}
    for name, raw_path in result["paths"].items():
        path = Path(raw_path).resolve()
        if root not in path.parents:
            raise RuntimeError(f"{name} imported outside pin staging tree: {path}")
        relative_paths[name] = path.relative_to(root).as_posix()
    return {"paths": relative_paths, "version": str(result["version"])}


def _validate_patch_scripts(patch_scripts: list[Path]) -> list[Path]:
    resolved = [Path(path).resolve() for path in patch_scripts]
    names = tuple(path.name for path in resolved)
    if names != EXPECTED_PATCH_NAMES:
        raise ValueError(
            f"patch order must be exactly {EXPECTED_PATCH_NAMES}; found {names}"
        )
    missing = [str(path) for path in resolved if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing patch scripts: {missing}")
    return resolved


def build_pin(
    *,
    source_repo: Path,
    base_revision: str,
    destination: Path,
    python_bin: Path,
    patch_scripts: list[Path],
) -> dict[str, Any]:
    """Build a verified pin in staging and publish it with one rename."""
    source_repo = Path(source_repo).resolve()
    destination = Path(destination).resolve()
    # Keep the venv entry-point path intact.  Resolving this symlink launches
    # Great Lakes' bare base interpreter and silently drops venv packages.
    python_bin = Path(os.path.abspath(os.path.expanduser(str(python_bin))))
    patches = _validate_patch_scripts(patch_scripts)
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite existing pin: {destination}")
    if not python_bin.is_file():
        raise FileNotFoundError(f"Python interpreter is missing: {python_bin}")
    revision = _resolved_revision(source_repo, base_revision)
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{destination.name}.tmp.", dir=destination.parent)
    ).resolve()
    published = False
    try:
        _extract_revision(source_repo, revision, staging)
        patch_records = []
        for script in patches:
            targets = PATCH_TARGETS[script.name]
            before = _target_hashes(staging, targets)
            subprocess.run(
                [str(python_bin), str(script), str(staging)],
                cwd=staging,
                check=True,
            )
            after = _target_hashes(staging, targets)
            _remove_patch_backups(staging)
            patch_records.append(
                {
                    "name": script.name,
                    "script_sha256": sha256_file(script),
                    "status": "applied" if before != after else "already_supported",
                    "targets": {
                        name: {"before_sha256": before[name], "after_sha256": after[name]}
                        for name in targets
                    },
                }
            )

        imports = imported_modules(python_bin, staging)
        manifest = {
            "pin_schema_version": 1,
            "base_revision": revision,
            "python_executable": str(python_bin),
            "patches": patch_records,
            "imports": imports["paths"],
            "cosmodiff_version": imports["version"],
            "inventory": file_inventory(staging),
        }
        (staging / PIN_MANIFEST_NAME).write_text(
            json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n"
        )
        if file_inventory(staging) != manifest["inventory"]:
            raise RuntimeError("pin inventory changed while writing its manifest")
        os.replace(staging, destination)
        published = True
        return json.loads((destination / PIN_MANIFEST_NAME).read_text())
    finally:
        if not published and staging.exists():
            shutil.rmtree(staging)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-repo", type=Path, required=True)
    parser.add_argument("--base-revision", required=True)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--python-bin", type=Path, required=True)
    parser.add_argument("--patch-script", type=Path, action="append", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = build_pin(
        source_repo=args.source_repo,
        base_revision=args.base_revision,
        destination=args.destination,
        python_bin=args.python_bin,
        patch_scripts=args.patch_script,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
