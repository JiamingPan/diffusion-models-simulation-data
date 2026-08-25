from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
from types import ModuleType, SimpleNamespace

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _write_import_fixture(root: Path) -> None:
    (root / "diffusers").mkdir(parents=True)
    (root / "torch.py").write_text(
        "float16 = 'float16'\n"
        "uint8 = 'uint8'\n"
    )
    (root / "diffusers/__init__.py").write_text(
        "import torch\n"
        "DEVICE_EMPTY_CACHE = {'xpu': torch.xpu.empty_cache}\n"
    )


def _run_child(program: str, *python_paths: Path) -> subprocess.CompletedProcess[str]:
    env = dict(os.environ)
    env["PYTHONNOUSERSITE"] = "1"
    env["PYTHONPATH"] = os.pathsep.join(str(path) for path in python_paths)
    return subprocess.run(
        [sys.executable, "-c", program],
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )


def _fake_torch() -> ModuleType:
    torch = ModuleType("torch")
    torch.float16 = "float16"
    torch.uint8 = "uint8"
    torch.utils = SimpleNamespace(_pytree=SimpleNamespace())
    torch.distributed = SimpleNamespace()
    return torch


def test_unshimmed_child_reproduces_diffusers_xpu_failure(tmp_path):
    _write_import_fixture(tmp_path)

    completed = _run_child("import diffusers", tmp_path)

    assert completed.returncode != 0
    assert "AttributeError" in completed.stderr
    assert "has no attribute 'xpu'" in completed.stderr


def test_shimmed_child_imports_diffusers_when_torch_lacks_xpu(tmp_path):
    _write_import_fixture(tmp_path)
    program = (
        "from simdiff_eval.torch_compat import install_torch_backend_compat\n"
        "report = install_torch_backend_compat(entry_point='tests.shimmed_child')\n"
        "import diffusers, json, torch\n"
        "print(json.dumps({'report': report, 'xpu': torch.xpu.is_available()}))\n"
    )

    completed = _run_child(program, tmp_path, REPO_ROOT)

    assert completed.returncode == 0, completed.stderr
    payload = json.loads(completed.stdout.strip().splitlines()[-1])
    assert payload["xpu"] is False
    assert payload["report"]["schema_version"] == 1
    assert payload["report"]["entry_points"] == ["tests.shimmed_child"]


def test_installer_adds_unavailable_backends_without_selecting_them():
    from simdiff_eval.torch_compat import install_torch_backend_compat

    torch = _fake_torch()

    report = install_torch_backend_compat(
        entry_point="tests.unavailable_backends",
        torch_module=torch,
    )

    for name in ("xpu", "mps", "npu"):
        backend = getattr(torch, name)
        assert backend.is_available() is False
        assert backend.device_count() == 0
        assert backend.empty_cache() is None
    assert report["schema_version"] == 1
    assert {"xpu", "mps", "npu"}.issubset(report["installed_attributes"])


def test_installer_is_idempotent_and_preserves_complete_real_backend():
    from simdiff_eval.torch_compat import install_torch_backend_compat

    torch = _fake_torch()
    real_xpu = SimpleNamespace(
        is_available=lambda: True,
        device_count=lambda: 2,
        empty_cache=lambda: None,
        manual_seed=lambda seed: None,
    )
    torch.xpu = real_xpu

    first = install_torch_backend_compat(
        entry_point="tests.first_call",
        torch_module=torch,
    )
    second = install_torch_backend_compat(
        entry_point="tests.second_call",
        torch_module=torch,
    )

    assert torch.xpu is real_xpu
    assert first["installed_attributes"] == second["installed_attributes"]
    assert second["entry_points"] == ["tests.first_call", "tests.second_call"]


def test_installer_fails_when_diffusers_was_imported_before_the_shim(monkeypatch):
    from simdiff_eval.torch_compat import (
        TorchCompatibilityError,
        install_torch_backend_compat,
    )

    torch = _fake_torch()
    monkeypatch.setitem(sys.modules, "diffusers", ModuleType("diffusers"))

    with pytest.raises(
        TorchCompatibilityError,
        match="tests.late_entry.*already imported",
    ):
        install_torch_backend_compat(
            entry_point="tests.late_entry",
            torch_module=torch,
        )
