#!/usr/bin/env python
"""Verify an immutable, audited cosmodiff seed-restart pin."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in os.sys.path:
    os.sys.path.insert(0, str(REPO_ROOT))

from simdiff_eval.seed_restart_runtime import (
    RUNTIME_DIR_NAME,
    normalize_runtime_audit,
    run_runtime_audit,
    runtime_file_inventory,
)


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


def verify_pin(
    pin_root: Path,
    manifest_path: Path,
    *,
    expected_base_revision: str,
    python_bin: Path,
    expected_patch_scripts: list[Path],
    code_root: Path,
    expected_torch_prefix: Path,
    incompatible_paths: Sequence[Path] = (),
    approved_residual_paths: Sequence[Path] = (),
    check_source_contract: bool = True,
) -> dict[str, Any]:
    """Fail when any pin source, patch, import, or provenance field differs."""
    pin_root = Path(pin_root).resolve()
    manifest_path = Path(manifest_path).resolve()
    code_root = Path(code_root).resolve()
    expected_torch_prefix = Path(expected_torch_prefix).resolve()
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("pin_schema_version") != 3:
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
    constant_label_patch = next(
        row
        for row in patch_rows
        if row.get("name") == "patch_cosmodiff_constant_label.py"
    )
    constant_label_support = manifest.get("constant_label_support")
    if not isinstance(constant_label_support, dict):
        raise RuntimeError("cosmodiff pin lacks constant-label support provenance")
    expected_support_keys = {
        "native_in_base_revision",
        "effective_in_published_pin",
        "provenance",
        "utils_path",
    }
    if set(constant_label_support) != expected_support_keys:
        raise RuntimeError("cosmodiff pin constant-label support fields differ")
    native_support = constant_label_support.get("native_in_base_revision")
    if not isinstance(native_support, bool):
        raise RuntimeError("cosmodiff pin native constant-label flag is not boolean")
    if constant_label_support.get("effective_in_published_pin") is not True:
        raise RuntimeError("published cosmodiff pin lacks effective constant-label support")
    expected_provenance = (
        "base_revision" if native_support else "patch_cosmodiff_constant_label.py"
    )
    expected_patch_status = "already_supported" if native_support else "applied"
    if constant_label_support.get("provenance") != expected_provenance:
        raise RuntimeError("cosmodiff pin constant-label provenance is inconsistent")
    if constant_label_support.get("utils_path") != "cosmodiff/utils.py":
        raise RuntimeError("cosmodiff pin constant-label source path is inconsistent")
    if constant_label_patch.get("status") != expected_patch_status:
        raise RuntimeError("cosmodiff pin constant-label patch status is inconsistent")
    actual_inventory = file_inventory(pin_root)
    if actual_inventory != manifest.get("inventory"):
        raise RuntimeError("cosmodiff pin inventory differs from its manifest")
    # Preserve the venv entry-point path; resolving its symlink would launch
    # the bare base interpreter without the venv's installed packages.
    runtime_python = Path(os.path.abspath(os.path.expanduser(str(python_bin))))
    if manifest.get("python_executable") != str(runtime_python):
        raise RuntimeError(
            "cosmodiff pin Python executable differs from its manifest: "
            f"{runtime_python} != {manifest.get('python_executable')}"
        )
    compatibility = manifest.get("runtime_compatibility")
    if not isinstance(compatibility, dict) or compatibility.get("schema_version") != 1:
        raise RuntimeError("unsupported seed-restart runtime compatibility schema")
    if compatibility.get("runtime_root") != RUNTIME_DIR_NAME:
        raise RuntimeError("seed-restart runtime root differs from its manifest")
    if compatibility.get("python_executable") != str(runtime_python):
        raise RuntimeError("runtime compatibility Python executable mismatch")
    if compatibility.get("python_executable_resolved") != str(runtime_python.resolve()):
        raise RuntimeError("runtime compatibility resolved Python executable mismatch")
    if compatibility.get("expected_torch_prefix") != str(expected_torch_prefix):
        raise RuntimeError("runtime compatibility Torch prefix mismatch")
    expected_incompatible = [
        str(Path(path).resolve()) for path in incompatible_paths
    ]
    if compatibility.get("incompatible_paths") != expected_incompatible:
        raise RuntimeError("runtime compatibility incompatible-path set mismatch")
    expected_residual = [
        str(Path(path).resolve()) for path in approved_residual_paths
    ]
    if compatibility.get("approved_residual_paths") != expected_residual:
        raise RuntimeError("runtime compatibility residual-path set mismatch")

    canonical = code_root / "simdiff_eval/torch_compat.py"
    canonical_row = compatibility.get("canonical_shim", {})
    if canonical_row.get("path") != "simdiff_eval/torch_compat.py":
        raise RuntimeError("canonical Torch compatibility path mismatch")
    if canonical_row.get("sha256") != sha256_file(canonical):
        raise RuntimeError("canonical Torch compatibility hash mismatch")
    runtime_root = pin_root / RUNTIME_DIR_NAME
    sitecustomize = runtime_root / "sitecustomize.py"
    sitecustomize_row = compatibility.get("sitecustomize", {})
    if sitecustomize_row.get("path") != f"{RUNTIME_DIR_NAME}/sitecustomize.py":
        raise RuntimeError("sitecustomize manifest path mismatch")
    if sitecustomize_row.get("sha256") != sha256_file(sitecustomize):
        raise RuntimeError("sitecustomize manifest hash mismatch")
    actual_sklearn_files = {
        name: row
        for name, row in runtime_file_inventory(runtime_root).items()
        if name.startswith("sklearn/")
    }
    if compatibility.get("sklearn_stub", {}).get("files") != actual_sklearn_files:
        raise RuntimeError("sklearn stub hashes differ from the pin manifest")

    runtime_report = run_runtime_audit(
        runtime_python,
        pin_root=pin_root,
        runtime_root=runtime_root,
        code_root=code_root,
        expected_torch_prefix=expected_torch_prefix,
        incompatible_paths=incompatible_paths,
        approved_residual_paths=approved_residual_paths,
    )
    normalized = normalize_runtime_audit(
        runtime_report,
        pin_root=pin_root,
        code_root=code_root,
    )
    if normalized != compatibility.get("runtime_audit"):
        raise RuntimeError("seed-restart runtime audit differs from its manifest")
    runtime_paths = {
        name: value.removeprefix("<PIN_ROOT>/")
        for name, value in normalized["cosmodiff"]["modules"].items()
    }
    if runtime_paths != manifest.get("imports"):
        raise RuntimeError("cosmodiff pin import paths differ from its manifest")
    if normalized["cosmodiff"]["version"] != manifest.get("cosmodiff_version"):
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
    parser.add_argument("--code-root", type=Path, required=True)
    parser.add_argument("--expected-torch-prefix", type=Path, required=True)
    parser.add_argument(
        "--incompatible-python-path",
        type=Path,
        action="append",
        default=[],
    )
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
        code_root=args.code_root,
        expected_torch_prefix=args.expected_torch_prefix,
        incompatible_paths=args.incompatible_python_path,
        check_source_contract=not args.skip_source_contract,
    )
    print(f"PASS: immutable cosmodiff seed-restart pin at {args.cosmodiff_dir}")


if __name__ == "__main__":
    main()
