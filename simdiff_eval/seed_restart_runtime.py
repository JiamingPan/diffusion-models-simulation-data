"""Build the deterministic Python runtime used by DiT seed-restart pins."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Any, Mapping, Sequence


RUNTIME_SCHEMA_VERSION = 1
RUNTIME_DIR_NAME = "seed_restart_runtime"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def runtime_file_inventory(runtime_root: Path) -> dict[str, dict[str, Any]]:
    """Return stable hashes for every file in an audited runtime root."""

    root = Path(runtime_root).resolve()
    return {
        path.relative_to(root).as_posix(): {
            "sha256": _sha256_file(path),
            "bytes": path.stat().st_size,
        }
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def _sitecustomize_source(*, code_root: Path, entry_point: str) -> str:
    canonical = (code_root / "simdiff_eval/torch_compat.py").resolve()
    return (
        "from pathlib import Path\n"
        "from simdiff_eval import torch_compat as _torch_compat\n"
        "\n"
        f"_EXPECTED = Path({str(canonical)!r})\n"
        "if Path(_torch_compat.__file__).resolve() != _EXPECTED.resolve():\n"
        "    raise RuntimeError(\n"
        "        'canonical torch compatibility module resolved outside the pinned code root: '\n"
        "        f'{_torch_compat.__file__} != {_EXPECTED}'\n"
        "    )\n"
        "_torch_compat.install_torch_backend_compat(\n"
        f"    entry_point={entry_point!r},\n"
        ")\n"
    )


def _sklearn_init_source() -> str:
    return (
        '"""Narrow sklearn import stub for the DiT seed-restart runtime."""\n'
        "\n"
        "from . import metrics\n"
        "\n"
        "__version__ = '0+simdiff-seed-restart-stub'\n"
        "RUNTIME_KIND = 'simdiff-seed-restart-stub'\n"
        "__all__ = ['metrics']\n"
    )


def _sklearn_metrics_source() -> str:
    return (
        '"""Unsupported sklearn surface required only for import isolation."""\n'
        "\n"
        "def roc_curve(*args, **kwargs):\n"
        "    raise RuntimeError(\n"
        "        'sklearn.metrics.roc_curve is unavailable in the DiT seed-restart '\n"
        "        'stub; this runtime cannot load sklearn estimators'\n"
        "    )\n"
    )


def _write_exact_files(root: Path, sources: Mapping[str, str]) -> None:
    existing = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file()
    } if root.exists() else set()
    unexpected = existing - set(sources)
    if unexpected:
        raise RuntimeError(
            f"unexpected existing runtime assets under {root}: {sorted(unexpected)}"
        )
    for relative, source in sources.items():
        path = root / relative
        if path.exists() and path.read_text() != source:
            raise RuntimeError(f"different existing runtime asset: {path}")
    for relative, source in sources.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            path.write_text(source)


def write_sitecustomize(
    path: Path,
    *,
    code_root: Path,
    entry_point: str,
) -> Path:
    """Write a thin adapter around the canonical compatibility module."""

    target = Path(path)
    code_root = Path(code_root).resolve()
    canonical = code_root / "simdiff_eval/torch_compat.py"
    if not canonical.is_file():
        raise FileNotFoundError(f"canonical Torch compatibility module is missing: {canonical}")
    source = _sitecustomize_source(code_root=code_root, entry_point=entry_point)
    if target.exists() and target.read_text() != source:
        raise RuntimeError(f"different existing runtime asset: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    if not target.exists():
        target.write_text(source)
    return target


def write_runtime_assets(
    runtime_root: Path,
    *,
    code_root: Path,
    entry_point: str,
) -> dict[str, Any]:
    """Create or validate the exact runtime files included in a pin."""

    root = Path(runtime_root).resolve()
    code_root = Path(code_root).resolve()
    canonical = code_root / "simdiff_eval/torch_compat.py"
    if not canonical.is_file():
        raise FileNotFoundError(f"canonical Torch compatibility module is missing: {canonical}")
    sources = {
        "sitecustomize.py": _sitecustomize_source(
            code_root=code_root,
            entry_point=entry_point,
        ),
        "sklearn/__init__.py": _sklearn_init_source(),
        "sklearn/metrics/__init__.py": _sklearn_metrics_source(),
    }
    _write_exact_files(root, sources)
    return {
        "schema_version": RUNTIME_SCHEMA_VERSION,
        "runtime_root": str(root),
        "entry_point": entry_point,
        "files": runtime_file_inventory(root),
    }


def build_child_env(
    base_env: Mapping[str, str],
    *,
    runtime_root: Path,
    code_root: Path,
    pin_root: Path,
    incompatible_paths: Sequence[Path] = (),
    approved_residual_paths: Sequence[Path] = (),
) -> dict[str, str]:
    """Return a child environment with one explicit, audited import order."""

    ordered = [
        Path(runtime_root).resolve(),
        Path(code_root).resolve(),
        Path(pin_root).resolve(),
        *(Path(path).resolve() for path in approved_residual_paths),
    ]
    incompatible = {Path(path).resolve() for path in incompatible_paths}
    conflicts = [path for path in ordered if path in incompatible]
    if conflicts:
        raise RuntimeError(
            f"approved child runtime includes incompatible paths: {conflicts}"
        )
    if len(set(ordered)) != len(ordered):
        raise RuntimeError(f"child runtime path roles must be unique: {ordered}")
    env = dict(base_env)
    env["PYTHONNOUSERSITE"] = "1"
    env["PYTHONPATH"] = os.pathsep.join(str(path) for path in ordered)
    return env
