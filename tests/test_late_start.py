from __future__ import annotations

import importlib.util
import os
from pathlib import Path
import subprocess
from types import SimpleNamespace

import numpy as np
import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]
SBATCH = ROOT / "scripts/slurm/sample_dit_late_start_screen.sbatch"
RUNTIME_REVISION = "555f350f82c913ff06150c96e66709719856cd83"


def load_script(name: str):
    path = ROOT / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class FakeScheduler:
    def __init__(self):
        self.timesteps = torch.tensor([], dtype=torch.long)
        self.begin_index = None
        self.calls = []

    def set_timesteps(self, steps, device=None):
        assert steps == 5
        self.timesteps = torch.tensor([499, 449, 399, 349, 0], device=device)
        self.begin_index = None
        self.calls.append("set_timesteps")

    def set_begin_index(self, index):
        self.begin_index = int(index)
        self.calls.append(("set_begin_index", int(index)))

    def add_noise(self, x0, noise, timesteps):
        assert self.begin_index is not None
        assert set(timesteps.tolist()) == {399}
        self.calls.append("add_noise")
        return noise

    def step(self, prediction, timestep, images, generator=None):
        assert self.begin_index == 2
        self.calls.append(("step", int(timestep)))
        return SimpleNamespace(prev_sample=images)


class FakeModel:
    def eval(self):
        return self

    def __call__(self, images, **kwargs):
        return (torch.zeros_like(images),)


def test_late_start_aligns_scheduler_begin_index_before_noise():
    module = load_script("sample_late_start")
    scheduler = FakeScheduler()
    images, actual, steps = module.generate_late_start(
        FakeModel(), scheduler,
        x_bar=torch.zeros(1, 1, 4, 4), t_start=400, init="mean",
        batch_size=2, num_steps=5, device=torch.device("cpu"),
        generator=torch.Generator().manual_seed(3),
        class_labels=torch.zeros(2, dtype=torch.long),
    )
    assert images.shape == (2, 1, 4, 4)
    assert (actual, steps) == (399, 3)
    assert scheduler.calls[:3] == [
        "set_timesteps", ("set_begin_index", 2), "add_noise"
    ]
    assert scheduler.calls[-3:] == [("step", 399), ("step", 349), ("step", 0)]


def test_evaluator_accepts_only_frozen_raw_protocol():
    module = load_script("evaluate_late_start")
    record = {
        "late_start_requested": np.asarray(399),
        "late_start_actual": np.asarray(399),
        "late_start_steps_run": np.asarray(40),
        "seed": np.asarray(123),
        "num_steps": np.asarray(50),
        "ema_sigma_rel": np.asarray(-1.0),
        "resolved_checkpoint": np.asarray("checkpoint"),
        "config_sha256": np.asarray("abc"),
        "training_mean_sha256": np.asarray("mean-hash"),
        "late_start_init": np.asarray("mean"),
        "scheduler_class": np.asarray("DPMSolverMultistepScheduler"),
    }
    result = module.validate_provenance(record, training_mean_sha256="mean-hash")
    assert result["t_start"] == 399
    assert result["steps_run"] == 40
    for key, value in (
        ("training_mean_sha256", "wrong"),
        ("seed", 124),
        ("ema_sigma_rel", 0.05),
    ):
        changed = dict(record)
        changed[key] = np.asarray(value)
        with pytest.raises(ValueError):
            module.validate_provenance(changed, training_mean_sha256="mean-hash")


def test_great_lakes_preflight_uses_frozen_runtime_root(tmp_path):
    runtime = tmp_path / "runtime-code"
    adapter = tmp_path / "adapter-code"
    project = tmp_path / "project"
    pin = tmp_path / "pin"
    for path in (runtime, adapter, project, pin):
        path.mkdir()

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
        "if [[ \"$*\" == *verify_cosmodiff_seed_restart_runtime.py* ]]; then\n"
        "  test \"$PWD\" = \"$RUNTIME_CODE_ROOT\"\n"
        "  test \"$PYTHONPATH\" = "
        "\"$COSMODIFF_PIN_ROOT/seed_restart_runtime:$RUNTIME_CODE_ROOT:$COSMODIFF_PIN_ROOT\"\n"
        "  [[ \"$*\" == *\"--code-root $RUNTIME_CODE_ROOT\"* ]]\n"
        "  [[ \"$*\" != *\"--code-root $CODE_ROOT\"* ]]\n"
        "elif [[ \"$*\" == *resolve_dit_high_noise_models.py* ]]; then\n"
        "  exit 0\n"
        "else\n"
        "  echo \"unexpected python invocation: $*\" >&2\n"
        "  exit 9\n"
        "fi\n"
    )
    python.chmod(0o755)

    env = dict(
        os.environ,
        PATH=f"{tmp_path}:{os.environ['PATH']}",
        RUNTIME_CODE_ROOT=str(runtime),
        CODE_ROOT=str(adapter),
        EXPECTED_COMMIT="adapter",
        PROJECT_DIR=str(project),
        COSMODIFF_PIN_ROOT=str(pin),
        COSMODIFF_PIN_MANIFEST="manifest",
        EXPECTED_COSMODIFF_BASE_REVISION="base",
        PYTHON_BIN=str(python),
        PREFLIGHT_ONLY="1",
    )
    result = subprocess.run(
        ["bash", str(SBATCH)], env=env, text=True, capture_output=True
    )
    assert result.returncode == 0, result.stderr
    assert "LATE-START PREFLIGHT PASSED; NO SAMPLING" in result.stdout
