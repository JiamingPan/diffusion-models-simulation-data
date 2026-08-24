#!/usr/bin/env python
"""Verify an immutable, audited cosmodiff seed-restart pin."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
from typing import Any


PIN_MANIFEST_NAME = "seed_restart_pin_manifest.json"
EXPECTED_PATCH_NAMES = (
    "patch_cosmodiff_package_metadata.py",
    "patch_cosmodiff_constant_label.py",
    "patch_cosmodiff_dit_class_labels.py",
    "patch_cosmodiff_checkpoint_state.py",
)
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
    records = {}
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


def require_fragments(path: Path, fragments: tuple[str, ...]) -> None:
    source = path.read_text()
    missing = [fragment for fragment in fragments if fragment not in source]
    if missing:
        raise RuntimeError(f"{path} lacks required runtime contracts: {missing}")


def _runtime_imports(python_bin: Path, pin_root: Path) -> dict[str, Any]:
    program = (
        "import importlib,json,pathlib\n"
        f"names={list(REQUIRED_IMPORTS)!r}\n"
        "paths={}\n"
        "for name in names:\n"
        "    module=importlib.import_module(name)\n"
        "    paths[name]=str(pathlib.Path(module.__file__).resolve())\n"
        "import cosmodiff\n"
        "print(json.dumps({'paths':paths,'version':cosmodiff.__version__},sort_keys=True))\n"
    )
    env = dict(os.environ)
    env["PYTHONNOUSERSITE"] = "1"
    env["PYTHONPATH"] = str(pin_root)
    try:
        completed = subprocess.run(
            [str(python_bin), "-c", program],
            cwd=pin_root,
            env=env,
            check=True,
            capture_output=True,
            text=True,
        )
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(
            "cosmodiff pin import verification failed:\n" + (exc.stderr or exc.stdout)
        ) from exc
    result = json.loads(completed.stdout.strip().splitlines()[-1])
    relative = {}
    for name, raw_path in result["paths"].items():
        path = Path(raw_path).resolve()
        if pin_root not in path.parents:
            raise RuntimeError(f"{name} imported outside immutable pin: {path}")
        relative[name] = path.relative_to(pin_root).as_posix()
    return {"paths": relative, "version": str(result["version"])}


def verify_pin(
    pin_root: Path,
    manifest_path: Path,
    *,
    expected_base_revision: str,
    python_bin: Path,
    expected_patch_scripts: list[Path],
    check_source_contract: bool = True,
) -> dict[str, Any]:
    """Fail when any pin source, patch, import, or provenance field differs."""
    pin_root = Path(pin_root).resolve()
    manifest_path = Path(manifest_path).resolve()
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("pin_schema_version") != 1:
        raise RuntimeError("unsupported cosmodiff pin schema")
    if manifest.get("base_revision") != str(expected_base_revision):
        raise RuntimeError(
            f"cosmodiff pin base revision mismatch: {manifest.get('base_revision')}"
        )
    patch_rows = manifest.get("patches", [])
    if tuple(row.get("name") for row in patch_rows) != EXPECTED_PATCH_NAMES:
        raise RuntimeError("cosmodiff pin patch order is not the declared exact set")
    expected_patch_scripts = [Path(path).resolve() for path in expected_patch_scripts]
    if tuple(path.name for path in expected_patch_scripts) != EXPECTED_PATCH_NAMES:
        raise RuntimeError("expected patch scripts are not the declared exact set")
    for row, script in zip(patch_rows, expected_patch_scripts, strict=True):
        if row.get("script_sha256") != sha256_file(script):
            raise RuntimeError(f"patch script hash mismatch: {script}")
        if row.get("status") not in {"applied", "already_supported"}:
            raise RuntimeError(f"invalid patch status for {script.name}")
    actual_inventory = file_inventory(pin_root)
    if actual_inventory != manifest.get("inventory"):
        raise RuntimeError("cosmodiff pin inventory differs from its manifest")
    runtime = _runtime_imports(Path(python_bin).resolve(), pin_root)
    if runtime["paths"] != manifest.get("imports"):
        raise RuntimeError("cosmodiff pin import paths differ from its manifest")
    if runtime["version"] != manifest.get("cosmodiff_version"):
        raise RuntimeError("cosmodiff pin version differs from its manifest")
    if check_source_contract:
        verify_source_contract(pin_root)
    return manifest


def verify_source_contract(cosmodiff_dir: Path) -> None:
    optim = cosmodiff_dir / "cosmodiff/optim.py"
    utils = cosmodiff_dir / "cosmodiff/utils.py"
    require_fragments(
        optim,
        (
            "resume_from_checkpoint",
            "accelerator.load_state(resume_from_checkpoint)",
            "from ema_pytorch import PostHocEMA",
            "gradient_accumulation_steps=gradient_accumulation_steps",
            "global_step = 0",
            "global_step += 1",
            "global_step >= ema_burn_in",
            "ema.update()",
            "ema.checkpoint_folder = Path(ckpt_save_path) / 'ema'",
            "ema.checkpoint()",
            "class_labels",
            "noise_scheduler.save_pretrained(ckpt_save_path)",
            '"checkpoint_config.yaml"',
            "accelerator.save_state(ckpt_save_path)",
        ),
    )
    require_fragments(
        utils,
        (
            "def parse_config_data(config: dict)",
            "class ArrayDataset",
            "self.labels = labels",
        ),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cosmodiff_dir", type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--expected-base-revision")
    parser.add_argument("--python-bin", type=Path, default=Path(os.sys.executable))
    parser.add_argument("--patch-script", type=Path, action="append")
    parser.add_argument("--skip-source-contract", action="store_true")
    args = parser.parse_args()
    manifest_path = args.manifest or args.cosmodiff_dir / PIN_MANIFEST_NAME
    manifest = json.loads(manifest_path.read_text())
    expected_revision = args.expected_base_revision or manifest.get("base_revision")
    if not args.patch_script:
        raise ValueError("--patch-script must supply the exact four patch scripts")
    verify_pin(
        args.cosmodiff_dir,
        manifest_path,
        expected_base_revision=expected_revision,
        python_bin=args.python_bin,
        expected_patch_scripts=args.patch_script,
        check_source_contract=not args.skip_source_contract,
    )
    print(f"PASS: immutable cosmodiff seed-restart pin at {args.cosmodiff_dir}")


if __name__ == "__main__":
    main()
