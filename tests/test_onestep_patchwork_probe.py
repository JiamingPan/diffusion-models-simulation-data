from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "onestep_patchwork_probe.py"


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
