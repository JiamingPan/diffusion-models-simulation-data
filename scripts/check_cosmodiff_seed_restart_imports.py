#!/usr/bin/env python
"""Audit every import used by the immutable cosmodiff seed-restart pin."""

from __future__ import annotations

from simdiff_eval.torch_compat import install_torch_backend_compat

TORCH_COMPAT_REPORT = install_torch_backend_compat(
    entry_point="scripts.check_cosmodiff_seed_restart_imports"
)

import argparse
import importlib
import json
import os
from pathlib import Path
import sys
from typing import Any, Sequence


REQUIRED_COSMODIFF_MODULES = (
    "cosmodiff",
    "cosmodiff.optim",
    "cosmodiff.utils",
    "cosmodiff.augment",
    "cosmodiff.transform",
)
OUTPUT_PREFIX = "SEED_RESTART_RUNTIME_JSON="


def _module_path(module: Any, label: str) -> Path:
    raw = getattr(module, "__file__", None)
    if not raw:
        raise RuntimeError(f"{label} has no import file: {module!r}")
    return Path(raw).resolve()


def _require_under(path: Path, root: Path, label: str) -> None:
    root = root.resolve()
    if not path.is_relative_to(root):
        raise RuntimeError(f"{label} resolved outside expected prefix {root}: {path}")


def _require_not_under(
    path: Path,
    incompatible_paths: Sequence[Path],
    label: str,
) -> None:
    for root in incompatible_paths:
        root = root.resolve()
        if path == root or path.is_relative_to(root):
            raise RuntimeError(f"{label} resolved under incompatible prefix {root}: {path}")


def collect_runtime_report(
    *,
    pin_root: Path,
    runtime_root: Path,
    code_root: Path,
    expected_torch_prefix: Path,
    incompatible_paths: Sequence[Path],
) -> dict[str, Any]:
    """Import and validate the exact runtime selected by the child process."""

    pin_root = Path(pin_root).resolve()
    runtime_root = Path(runtime_root).resolve()
    code_root = Path(code_root).resolve()
    expected_torch_prefix = Path(expected_torch_prefix).resolve()
    incompatible_paths = tuple(Path(path).resolve() for path in incompatible_paths)

    import sitecustomize
    import numpy
    import sklearn
    import torch

    import diffusers
    from diffusers import DDPMScheduler, DiTTransformer2DModel

    import huggingface_hub
    from huggingface_hub import hf_hub_download, snapshot_download

    del DDPMScheduler, DiTTransformer2DModel, hf_hub_download, snapshot_download

    sitecustomize_path = _module_path(sitecustomize, "sitecustomize")
    torch_path = _module_path(torch, "torch")
    sklearn_path = _module_path(sklearn, "sklearn")
    numpy_path = _module_path(numpy, "numpy")
    diffusers_path = _module_path(diffusers, "diffusers")
    hub_path = _module_path(huggingface_hub, "huggingface_hub")

    _require_under(
        sitecustomize_path,
        runtime_root,
        "sitecustomize",
    )
    expected_sitecustomize = runtime_root / "sitecustomize.py"
    if sitecustomize_path != expected_sitecustomize:
        raise RuntimeError(
            f"sitecustomize mismatch: expected {expected_sitecustomize}, found {sitecustomize_path}"
        )
    _require_under(torch_path, expected_torch_prefix, "torch")
    _require_under(sklearn_path, runtime_root, "sklearn")
    for label, path in (
        ("torch", torch_path),
        ("sklearn", sklearn_path),
        ("numpy", numpy_path),
    ):
        _require_not_under(path, incompatible_paths, label)

    from simdiff_eval import torch_compat

    canonical_path = _module_path(torch_compat, "simdiff_eval.torch_compat")
    _require_under(canonical_path, code_root, "canonical Torch compatibility module")
    auditor_path = Path(__file__).resolve()
    _require_under(auditor_path, code_root, "seed-restart runtime auditor")

    cosmodiff_paths: dict[str, str] = {}
    cosmodiff_modules: dict[str, Any] = {}
    for name in REQUIRED_COSMODIFF_MODULES:
        module = importlib.import_module(name)
        path = _module_path(module, name)
        _require_under(path, pin_root, name)
        cosmodiff_paths[name] = str(path)
        cosmodiff_modules[name] = module

    sklearn_kind = getattr(sklearn, "RUNTIME_KIND", None)
    if sklearn_kind != "simdiff-seed-restart-stub":
        raise RuntimeError(
            "sklearn is not the audited DiT seed-restart stub: "
            f"kind={sklearn_kind!r} file={sklearn_path}"
        )

    pythonpath = [
        str(Path(entry).resolve())
        for entry in os.environ.get("PYTHONPATH", "").split(os.pathsep)
        if entry
    ]
    return {
        "python": {
            "executable": sys.executable,
            "path": [str(Path(entry).resolve()) if entry else "" for entry in sys.path],
            "pythonpath": pythonpath,
        },
        "sitecustomize": {"file": str(sitecustomize_path)},
        "torch": {
            "file": str(torch_path),
            "version": str(getattr(torch, "__version__", "unknown")),
            "compat": TORCH_COMPAT_REPORT,
        },
        "sklearn": {
            "file": str(sklearn_path),
            "runtime_kind": str(sklearn_kind),
            "version": str(getattr(sklearn, "__version__", "unknown")),
        },
        "numpy": {
            "file": str(numpy_path),
            "version": str(getattr(numpy, "__version__", "unknown")),
        },
        "diffusers": {
            "file": str(diffusers_path),
            "version": str(getattr(diffusers, "__version__", "unknown")),
            "symbols": ["DDPMScheduler", "DiTTransformer2DModel"],
        },
        "huggingface_hub": {
            "file": str(hub_path),
            "version": str(getattr(huggingface_hub, "__version__", "unknown")),
            "symbols": ["hf_hub_download", "snapshot_download"],
        },
        "cosmodiff": {
            "modules": cosmodiff_paths,
            "version": str(getattr(cosmodiff_modules["cosmodiff"], "__version__", "unknown")),
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pin-root", type=Path, required=True)
    parser.add_argument("--runtime-root", type=Path, required=True)
    parser.add_argument("--code-root", type=Path, required=True)
    parser.add_argument("--expected-torch-prefix", type=Path, required=True)
    parser.add_argument(
        "--incompatible-python-path",
        type=Path,
        action="append",
        default=[],
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = collect_runtime_report(
        pin_root=args.pin_root,
        runtime_root=args.runtime_root,
        code_root=args.code_root,
        expected_torch_prefix=args.expected_torch_prefix,
        incompatible_paths=args.incompatible_python_path,
    )
    print(OUTPUT_PREFIX + json.dumps(report, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
