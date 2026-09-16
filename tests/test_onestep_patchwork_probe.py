from __future__ import annotations

import importlib.util
import os
from pathlib import Path
import subprocess
from types import SimpleNamespace

import pytest
import torch
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "onestep_patchwork_probe.py"
SBATCH = ROOT / "scripts/slurm/evaluate_dit_onestep_patchwork.sbatch"
RUNTIME_REVISION = "555f350f82c913ff06150c96e66709719856cd83"


def load_module():
    spec = importlib.util.spec_from_file_location("onestep_patchwork_probe", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class FakeScheduler:
    def __init__(self):
        self.noise_ids = []
        self.alphas_cumprod = torch.tensor([0.25])
        self.config = SimpleNamespace(prediction_type="v_prediction")

    def add_noise(self, source, noise, timesteps):
        self.noise_ids.append(noise.data_ptr())
        assert timesteps.tolist() == [0, 0]
        return source + 2 * noise


def test_no_trace_and_trace_receive_identical_noise_tensor():
    module = load_module()
    scheduler = FakeScheduler()
    mean = torch.zeros(2, 1, 4, 4)
    traced = torch.ones(2, 1, 4, 4)
    noise = torch.arange(32, dtype=torch.float32).reshape(2, 1, 4, 4)
    pair = module.add_shared_noise_pair(
        scheduler,
        mean=mean,
        traced=traced,
        noise=noise,
        timestep=0,
    )
    assert scheduler.noise_ids == [noise.data_ptr(), noise.data_ptr()]
    torch.testing.assert_close(pair["trace"] - pair["no_trace"], traced - mean)


def test_shared_noise_pair_rejects_shape_mismatch():
    module = load_module()
    with pytest.raises(ValueError, match="shapes must match"):
        module.add_shared_noise_pair(
            FakeScheduler(),
            mean=torch.zeros(2, 1, 4, 4),
            traced=torch.zeros(1, 1, 4, 4),
            noise=torch.zeros(2, 1, 4, 4),
            timestep=0,
        )


class ZeroVelocityModel:
    def __call__(self, x_t, **kwargs):
        return (torch.zeros_like(x_t),)


def test_v_prediction_is_converted_to_x0_estimate():
    module = load_module()
    scheduler = FakeScheduler()
    x_t = torch.full((2, 1, 4, 4), 4.0)
    result = module.predict_x0(
        ZeroVelocityModel(), scheduler, x_t, t=0, class_label=0
    )
    torch.testing.assert_close(result, torch.full_like(x_t, 2.0))


def test_gallery_png_does_not_require_matplotlib(tmp_path, monkeypatch):
    module = load_module()
    gallery = {}
    for timestep in (10, 5):
        for kind in ("no_trace", "trace"):
            gallery[(timestep, kind)] = np.linspace(
                -1, 1, 4 * 1 * 8 * 8, dtype=np.float32
            ).reshape(4, 1, 8, 8)
    output = tmp_path / "gallery.png"
    monkeypatch.setitem(__import__("sys").modules, "matplotlib", None)
    module.write_gallery_png(
        gallery, timesteps=[10, 5], label="fixture", output=output
    )
    assert output.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")


def test_great_lakes_preflight_records_missing_l12_and_uses_frozen_runtime(tmp_path):
    runtime = tmp_path / "runtime"
    code = tmp_path / "code"
    project = tmp_path / "project"
    pin = tmp_path / "pin"
    l12 = tmp_path / "l12-empty-checkpoint"
    for path in (runtime, code, project, pin, l12):
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
        "  [[ \"$*\" == *\"--code-root $RUNTIME_CODE_ROOT\"* ]]\n"
        "elif [[ \"$*\" == *resolve_dit_high_noise_models.py* ]]; then\n"
        "  echo \"{\\\"name\\\": \\\"fixture\\\", \\\"checkpoint\\\": \\\"$L12_FIXTURE\\\", \\\"config\\\": \\\"fixture.yaml\\\"}\"\n"
        "elif [[ \"$1\" == - ]]; then\n"
        "  sed -n '1,99p' >/dev/null\n"
        "  echo \"$L12_FIXTURE\"\n"
        "elif [[ \"$*\" == *onestep_patchwork_probe.py* && \"$*\" == *--help* ]]; then\n"
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
        PROJECT_DIR=str(project),
        CODE_ROOT=str(code),
        EXPECTED_COMMIT="adapter",
        RUNTIME_CODE_ROOT=str(runtime),
        PYTHON_BIN=str(python),
        COSMODIFF_PIN_ROOT=str(pin),
        COSMODIFF_PIN_MANIFEST="manifest",
        EXPECTED_COSMODIFF_BASE_REVISION="base",
        L12_FIXTURE=str(l12),
        PREFLIGHT_ONLY="1",
    )
    result = subprocess.run(
        ["bash", str(SBATCH)], env=env, text=True, capture_output=True
    )
    assert result.returncode == 0, result.stderr
    assert "L12 OMITTED" in result.stdout
    assert "ONESTEP PATCHWORK PREFLIGHT PASSED; NO GPU PROBE" in result.stdout
    assert not (project / "results/onestep_probe_v2").exists()
