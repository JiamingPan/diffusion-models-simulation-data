"""Small helpers for filtering an incompatible Python path per process."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable


def sanitize_sys_path(
    paths: Iterable[str],
    incompatible_paths: Iterable[str],
) -> list[str]:
    """Remove only exact configured entries from a Python search path."""
    blocked = {str(path) for path in incompatible_paths if str(path)}
    return [str(path) for path in paths if str(path) not in blocked]


def write_sitecustomize(path: str | Path, incompatible_paths: Iterable[str]) -> Path:
    """Write a startup hook that applies the same exact-entry filtering."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    configured = tuple(str(item) for item in incompatible_paths if str(item))
    destination.write_text(
        """from __future__ import annotations

import os
import sys

blocked = set(CONFIGURED_PATHS)
blocked.update({
    item for item in os.environ.get("PROBE_CONTROLS_INCOMPATIBLE_PATHS", "").split(os.pathsep)
    if item
})
sys.path[:] = [entry for entry in sys.path if entry not in blocked]
""".replace("CONFIGURED_PATHS", repr(configured))
    )
    return destination


def incompatible_paths_from_env() -> tuple[str, ...]:
    return tuple(
        item
        for item in os.environ.get("PROBE_CONTROLS_INCOMPATIBLE_PATHS", "").split(os.pathsep)
        if item
    )
