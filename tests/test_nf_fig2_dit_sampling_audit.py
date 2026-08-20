import importlib.util
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def load_sample_module():
    path = ROOT / "scripts" / "sample_cosmodiff.py"
    spec = importlib.util.spec_from_file_location("sample_cosmodiff_audit_test", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_scheduler_audit_records_executed_schedule_and_terminal_sigma():
    module = load_sample_module()

    class FakeScheduler:
        timesteps = np.asarray([999, 750, 500, 250, 0])
        sigmas = np.asarray([10.0, 3.0, 1.0, 0.1, 0.0])

    audit = module.scheduler_audit_metadata(FakeScheduler(), requested_steps=5)
    assert audit == {
        "scheduler_class": "FakeScheduler",
        "requested_inference_steps": 5,
        "executed_inference_steps": 5,
        "first_timestep": 999.0,
        "final_timestep": 0.0,
        "terminal_sigma": 0.0,
        "terminal_sigma_is_zero": True,
        "terminal_sigma_verifiable": True,
    }


def test_scheduler_audit_does_not_infer_sigma_from_timestep():
    module = load_sample_module()

    class NoSigmaScheduler:
        timesteps = np.asarray([9, 4, 0])

    audit = module.scheduler_audit_metadata(NoSigmaScheduler(), requested_steps=3)
    assert np.isnan(audit["terminal_sigma"])
    assert not audit["terminal_sigma_is_zero"]
    assert not audit["terminal_sigma_verifiable"]


def test_npz_output_records_scheduler_audit(tmp_path):
    module = load_sample_module()
    output = tmp_path / "samples.npz"
    samples = np.arange(12, dtype=np.float32).reshape(3, 1, 2, 2)
    audit = {
        "scheduler_class": "DPMSolverMultistepScheduler",
        "requested_inference_steps": 50,
        "executed_inference_steps": 50,
        "first_timestep": 999.0,
        "final_timestep": 0.0,
        "terminal_sigma": 0.0,
        "terminal_sigma_is_zero": True,
        "terminal_sigma_verifiable": True,
    }

    module.save_sample_output(
        output,
        samples,
        requested_checkpoint=Path("/runs/example"),
        resolved_checkpoint=Path("/runs/example/checkpoint-epoch-1"),
        config_path=Path("/configs/example.yaml"),
        scheduler_name="DPMSolverMultistepScheduler",
        num_steps=50,
        seed=123,
        scheduler_audit=audit,
    )

    with np.load(output) as data:
        np.testing.assert_array_equal(data["samples"], samples)
        assert int(data["executed_inference_steps"].item()) == 50
        assert float(data["terminal_sigma"].item()) == 0.0
        assert bool(data["terminal_sigma_is_zero"].item())
        assert bool(data["terminal_sigma_verifiable"].item())
