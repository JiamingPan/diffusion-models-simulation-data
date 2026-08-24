#!/usr/bin/env python
"""Patch cosmo_diffusion import when running from a source checkout.

Some ``cosmodiff/__init__.py`` versions unconditionally call
``importlib.metadata.version("cosmodiff")``.  That works after an editable or
wheel install, but it fails on Great Lakes where we put the source checkout on
``PYTHONPATH``.  This patch keeps the installed-package behavior when metadata
exists and falls back to a source-checkout version string otherwise.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path


MARKER = "PackageNotFoundError as _PackageNotFoundError"


def source_for_patch(path: Path) -> str:
    backup = path.with_suffix(path.suffix + ".codex_package_metadata.bak")
    if not backup.exists():
        backup.write_text(path.read_text())
    return path.read_text()


def patch_init(path: Path) -> bool:
    source = path.read_text()
    if MARKER in source or re.search(
        r"^from\s+\.version\s+import\s+__version__\s*$",
        source,
        flags=re.MULTILINE,
    ):
        print("cosmo_diffusion package-metadata patch: ok")
        return False
    source = source_for_patch(path)

    updated = source
    updated = updated.replace(
        "from importlib.metadata import version as _pkg_version",
        (
            "from importlib.metadata import PackageNotFoundError as _PackageNotFoundError\n"
            "from importlib.metadata import version as _pkg_version"
        ),
        1,
    )

    pattern = re.compile(r'^__version__\s*=\s*_pkg_version\("cosmodiff"\)\s*$', flags=re.MULTILINE)
    replacement = (
        "try:\n"
        "    __version__ = _pkg_version(\"cosmodiff\")\n"
        "except _PackageNotFoundError:\n"
        "    __version__ = \"0+source\""
    )
    updated, n = pattern.subn(replacement, updated, count=1)
    if n != 1:
        raise RuntimeError(f"Could not patch {path}; expected __version__ assignment not found.")

    path.write_text(updated)
    print("cosmo_diffusion package-metadata patch: patched")
    return True


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("cosmodiff_dir", type=Path)
    args = parser.parse_args()

    init_path = args.cosmodiff_dir / "cosmodiff" / "__init__.py"
    if not init_path.exists():
        raise FileNotFoundError(f"Missing {init_path}")
    patch_init(init_path)


if __name__ == "__main__":
    main()
