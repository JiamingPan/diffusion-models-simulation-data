"""Regression tests for the diagnostic's two-code-root runtime contract."""

from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
SBATCH = ROOT / "scripts/slurm/evaluate_dit_high_noise.sbatch"
ENTRYPOINT = ROOT / "scripts/evaluate_dit_high_noise.py"
RUNTIME_REVISION = "555f350f82c913ff06150c96e66709719856cd83"


def test_preflight_verifies_original_pin_code_root(tmp_path):
    runtime = tmp_path / "runtime-code"
    adapter = tmp_path / "adapter-code"
    runtime.mkdir()
    adapter.mkdir()

    git = tmp_path / "git"
    git.write_text(
        "#!/bin/bash\n"
        "if [[ \"$3\" == rev-parse ]]; then\n"
        "  if [[ \"$2\" == \"$RUNTIME_CODE_ROOT\" ]]; then\n"
        f"    echo {RUNTIME_REVISION}\n"
        "  else\n"
        "    echo adapter\n"
        "  fi\n"
        "fi\n"
    )
    git.chmod(0o755)

    python = tmp_path / "fake-python"
    python.write_text(
        "#!/bin/bash\n"
        "set -eu\n"
        "test \"$PWD\" = \"$RUNTIME_CODE_ROOT\"\n"
        "test \"$PYTHONPATH\" = "
        "\"$COSMODIFF_PIN_ROOT/seed_restart_runtime:$RUNTIME_CODE_ROOT:$COSMODIFF_PIN_ROOT\"\n"
        "[[ \"$*\" == *\"--code-root $RUNTIME_CODE_ROOT\"* ]]\n"
        "[[ \"$*\" == *\"$RUNTIME_CODE_ROOT/scripts/patch_cosmodiff_package_metadata.py\"* ]]\n"
        "[[ \"$*\" != *\"--code-root $CODE_ROOT\"* ]]\n"
        "echo VERIFIED_ORIGINAL_RUNTIME_ROOT\n"
    )
    python.chmod(0o755)

    env = dict(
        os.environ,
        PATH=f"{tmp_path}:{os.environ['PATH']}",
        RUNTIME_CODE_ROOT=str(runtime),
        CODE_ROOT=str(adapter),
        EXPECTED_COMMIT="adapter",
        PROJECT_DIR=str(tmp_path),
        COSMODIFF_PIN_ROOT=str(tmp_path / "pin"),
        COSMODIFF_PIN_MANIFEST="manifest",
        EXPECTED_COSMODIFF_BASE_REVISION="base",
        PYTHON_BIN=str(python),
        PREFLIGHT_ONLY="1",
    )
    result = subprocess.run(
        ["bash", str(SBATCH)], env=env, text=True, capture_output=True
    )
    assert result.returncode == 0, result.stderr
    assert "VERIFIED_ORIGINAL_RUNTIME_ROOT" in result.stdout
    assert "RUNTIME PREFLIGHT PASSED; NO DIAGNOSTIC" in result.stdout


def test_entrypoint_can_add_adapter_module_to_preloaded_runtime_package(tmp_path):
    runtime = tmp_path / "runtime"
    package = runtime / "simdiff_eval"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("")
    (package / "torch_compat.py").write_text(
        "def install_torch_backend_compat(*, entry_point):\n"
        "    return entry_point\n"
    )
    (runtime / "sitecustomize.py").write_text(
        "from simdiff_eval import torch_compat\n"
        "torch_compat.install_torch_backend_compat(entry_point='test.sitecustomize')\n"
    )
    env = dict(
        os.environ,
        PYTHONNOUSERSITE="1",
        PYTHONPATH=str(runtime),
    )
    result = subprocess.run(
        [sys.executable, str(ENTRYPOINT), "--help"],
        cwd=runtime,
        env=env,
        text=True,
        capture_output=True,
    )
    assert result.returncode == 0, result.stderr
    assert "terminal high-noise regime" in result.stdout
