from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace
import sys
import subprocess

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]


def load_module():
    spec = importlib.util.spec_from_file_location(
        "trace_trajectories", ROOT / "scripts/trace_trajectories.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class Scheduler:
    def __init__(self):
        self.config = SimpleNamespace(prediction_type="v_prediction")
        self.alphas_cumprod = torch.tensor([0.8, 0.5, 0.2])
        self.calls = 0

    def set_timesteps(self, n):
        assert n == 3
        self.timesteps = torch.tensor([2, 1, 0])

    def step(self, v, t, x):
        self.calls += 1
        return SimpleNamespace(prev_sample=x - v * 0.1)


class Model:
    def __init__(self):
        self.batch_sizes = []

    def __call__(self, x, **kwargs):
        self.batch_sizes.append(len(x))
        return (x * 0.1,)


def test_noise_is_identical_and_hashed_sample_by_sample():
    m = load_module()
    first = m.make_initial_noise(3, (1, 16, 16), 123)
    second = m.make_initial_noise(3, (1, 16, 16), 123)
    assert torch.equal(first, second)
    assert m.noise_hashes(first) == m.noise_hashes(second)
    assert len(set(m.noise_hashes(first))) == 3
    assert m.noise_hashes(first) != m.noise_hashes(m.make_initial_noise(3, (1, 16, 16), 124))


def test_top2_returns_sorted_values_without_mutating_samples():
    m = load_module()
    samples = np.array([[[[2.0, 0.0]]]], dtype=np.float32)
    ref = np.array([[[[0.0, 1.0]]], [[[1.0, 0.0]]], [[[0.6, 0.8]]]], dtype=np.float32)
    before = samples.copy()
    top, second = m.top2_cos(samples, ref)
    np.testing.assert_allclose(top, [1])
    np.testing.assert_allclose(second, [.6])
    np.testing.assert_array_equal(samples, before)


def test_trajectory_batches_model_but_steps_scheduler_once_per_timestep():
    m = load_module()
    rng = np.random.default_rng(7)
    ref = rng.normal(size=(4, 1, 16, 16)).astype(np.float32)
    initial = m.make_initial_noise(3, ref.shape[1:], 123)
    before = initial.clone()
    model, scheduler = Model(), Scheduler()
    metrics, timesteps, kept, final, assignments = m.run(
        model, scheduler, ref, initial_noise=initial,
        num_steps=3, batch_size=2, class_label=0, device=torch.device("cpu"),
        keep_steps={0, 2},
    )
    assert scheduler.calls == 3
    assert model.batch_sizes == [2, 1] * 3
    assert timesteps.tolist() == [2, 1, 0]
    assert np.isnan(metrics["persistence"][0]).all()
    np.testing.assert_allclose(metrics["persistence"][1], (assignments[1] == assignments[0]).mean(axis=1))
    assert final.shape == (3, 1, 16, 16)
    assert set(kept) == {0, 2}
    assert "x" in kept[0] and "x0" in kept[0]
    assert torch.equal(initial, before)


def test_v_prediction_conversion_is_correct_at_each_timestep():
    m = load_module()
    ref = np.random.default_rng(9).normal(size=(3, 1, 16, 16)).astype(np.float32)
    initial = torch.ones(2, 1, 16, 16)
    _, _, kept, _, _ = m.run(
        Model(), Scheduler(), ref, initial_noise=initial, num_steps=3,
        batch_size=2, class_label=0, device=torch.device("cpu"), keep_steps={0},
    )
    np.testing.assert_allclose(kept[0]["x0"], np.sqrt(.2) - np.sqrt(.8) * .1, rtol=1e-6)


def test_persistence_controls_include_frozen_and_independent_noise():
    m = load_module()
    ref = np.random.default_rng(2).normal(size=(8, 1, 16, 16)).astype(np.float32)
    rows = m.persistence_controls(ref, n=8, seed=123)
    by_name = {row["control"]: row for row in rows}
    assert by_name["frozen_gaussian"]["persistence_median"] == 1
    assert by_name["frozen_gaussian"]["majority_median"] < .8
    assert by_name["independent_gaussian"]["persistence_median"] < .5
    assert by_name["frozen_synthetic_patchwork"]["persistence_median"] == 1
    assert by_name["frozen_corrupted_single_map"]["persistence_median"] == 1


def fixture_npz(path, m, noise, ref_hash="reference", timestep_shift=0):
    np.savez(path, initial_noise=noise.numpy(), initial_noise_sha256=np.array(m.noise_hashes(noise)),
             reference_sha256=ref_hash, timesteps=np.array([2, 1, 0]) + timestep_shift,
             final_cos=np.array([.999, .3, .85]), outcome=np.array(["copy", "junk", "other"]),
             label=path.name.split("_trajectory")[0], seed=123)


def test_pairing_fails_closed_on_changed_noise_reference_or_schedule(tmp_path):
    m = load_module()
    noise = m.make_initial_noise(3, (1, 16, 16), 123)
    paths = [tmp_path / f"model{i}_trajectory.npz" for i in range(3)]
    for path in paths:
        fixture_npz(path, m, noise)
    rows = m.paired_outcomes(paths)
    assert len(rows) == 3 and rows[1]["model0_outcome"] == "junk"
    fixture_npz(paths[2], m, noise, ref_hash="different")
    with pytest.raises(ValueError, match="reference"):
        m.paired_outcomes(paths)
    fixture_npz(paths[2], m, m.make_initial_noise(3, (1, 16, 16), 124))
    with pytest.raises(ValueError, match="noise"):
        m.paired_outcomes(paths)
    fixture_npz(paths[2], m, noise, timestep_shift=1)
    with pytest.raises(ValueError, match="timesteps"):
        m.paired_outcomes(paths)


def test_pillow_curves_and_gallery_work_without_matplotlib(tmp_path, monkeypatch):
    m = load_module()
    monkeypatch.setitem(sys.modules, "matplotlib", None)
    metrics = {name: np.ones((3, 3), dtype=np.float32) * .5 for name in m.METRICS}
    metrics["persistence"][0] = np.nan
    fields = np.zeros((3, 1, 16, 16), dtype=np.float32)
    kept = {0: {"x": fields, "x0": fields}, 2: {"x": fields, "x0": fields}}
    m.write_figures(tmp_path, label="fixture", metrics=metrics, timesteps=np.array([2, 1, 0]),
                    snr=np.array([0., .2, 1.]), kept=kept, final=fields,
                    final_cos=np.array([1., .3, .85]), outcome=np.array(["copy", "junk", "other"]))
    for name in ("fixture_trajectory.png", "fixture_gallery.png"):
        assert (tmp_path / name).read_bytes().startswith(b"\x89PNG")


def test_atomic_output_has_terminal_manifest_only_after_all_files(tmp_path):
    m = load_module()
    output = tmp_path / "published"
    with pytest.raises(RuntimeError):
        with m.atomic_output(output) as pending:
            (pending / "partial.csv").write_text("header\n")
            raise RuntimeError("interrupted")
    assert not output.exists()
    with m.atomic_output(output) as pending:
        (pending / "complete.json").write_text('{"status":"complete"}')
    assert (output / "complete.json").is_file()
    with pytest.raises(FileExistsError):
        with m.atomic_output(output):
            pass


def test_paired_curves_use_the_same_indices_on_l8_and_l16(tmp_path):
    m = load_module()
    initial = m.make_initial_noise(3, (1, 16, 16), 123)
    paths = []
    for label in ("dit_l8_200k", "dit_l16_fresh300k"):
        path = tmp_path / f"{label}_trajectory.npz"
        metric = np.array([[.1, .9, .5], [.2, .8, .6]], dtype=np.float32)
        np.savez(path, label=label, seed=123, initial_noise=initial.numpy(),
                 initial_noise_sha256=np.asarray(m.noise_hashes(initial)), reference_sha256="same",
                 timesteps=np.array([399, 299]), snr=np.array([.1, .5]),
                 final_cos=np.array([1., .4, .85]), outcome=np.array(["copy", "junk", "other"]),
                 **{name: metric for name in m.METRICS})
        paths.append(path)
    rows = m.paired_curve_rows(paths)
    l8_junk = [row for row in rows if row["cohort_outcome"] == "junk" and row["label"] == "dit_l8_200k"]
    assert len(l8_junk) == 2
    assert l8_junk[0]["n"] == 1
    assert l8_junk[0]["majority"] == pytest.approx(.9)


def test_command_line_help_in_a_real_subprocess_does_not_load_matplotlib():
    result = subprocess.run([sys.executable, str(ROOT / "scripts/trace_trajectories.py"), "--help"],
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert "--pair-files" in result.stdout


def test_original_sampler_loop_and_instrumented_loop_have_identical_final_fields():
    m = load_module()
    reference = np.random.default_rng(2).normal(size=(4, 1, 16, 16)).astype(np.float32)
    initial = m.make_initial_noise(3, reference.shape[1:], 123)
    scheduler, model = Scheduler(), Model()
    scheduler.set_timesteps(3)
    x = initial.clone()
    for t in scheduler.timesteps:
        ts = torch.full((len(x),), int(t), dtype=torch.long)
        v = model(x, timestep=ts, class_labels=torch.zeros_like(ts), return_dict=False)[0]
        x = scheduler.step(v, t, x).prev_sample
    _, _, _, actual, _ = m.run(Model(), Scheduler(), reference, initial_noise=initial, num_steps=3,
                               batch_size=2, class_label=0, device=torch.device("cpu"), keep_steps=set())
    np.testing.assert_array_equal(actual, x.numpy())


def test_completed_report_and_artifact_hashes_are_required_before_pairing(tmp_path):
    import hashlib
    import json
    m = load_module()
    path = tmp_path / "fixture_trajectory.npz"
    path.write_bytes(b"fixture")
    table = path.with_suffix(".csv")
    table.write_text("value\n1\n")
    report = {"status": "complete", "artifacts_sha256": {
        file.name: hashlib.sha256(file.read_bytes()).hexdigest() for file in (path, table)}}
    path.with_suffix(".json").write_text(json.dumps(report))
    m.validate_completed_trajectory(path)
    table.write_text("value\n2\n")
    with pytest.raises(ValueError, match="hash mismatch"):
        m.validate_completed_trajectory(path)
    report["status"] = "running"
    path.with_suffix(".json").write_text(json.dumps(report))
    with pytest.raises(ValueError, match="not complete"):
        m.validate_completed_trajectory(path)


def test_great_lakes_wrapper_preflight_has_no_result_writes(tmp_path):
    import os
    for name in ("runtime", "code", "project", "pin", "l12"):
        (tmp_path / name).mkdir()
    git = tmp_path / "git"
    git.write_text('#!/bin/bash\nif [[ "$3" == rev-parse ]]; then\n'
                   'if [[ "$2" == "$RUNTIME_CODE_ROOT" ]]; then\n'
                   'echo 555f350f82c913ff06150c96e66709719856cd83\nelse\necho adapter\nfi\nfi\n')
    git.chmod(0o755)
    python = tmp_path / "fake-python"
    python.write_text('#!/bin/bash\nset -eu\n'
                      'if [[ "$*" == *verify_cosmodiff_seed_restart_runtime.py* ]]; then\n'
                      'test "$PWD" = "$RUNTIME_CODE_ROOT"\n'
                      'elif [[ "$*" == *resolve_dit_high_noise_models.py* ]]; then\n'
                      'echo "{\\"name\\":\\"fixture\\",\\"checkpoint\\":\\"$L12_FIXTURE\\",\\"config\\":\\"fixture.yaml\\"}"\n'
                      'elif [[ "$1" == - ]]; then\nsed -n "1,99p" >/dev/null\necho "$L12_FIXTURE"\n'
                      'elif [[ "$*" == *trace_trajectories.py* && "$*" == *--help* ]]; then\nexit 0\n'
                      'else\necho "unexpected compute invocation: $*" >&2\nexit 9\nfi\n')
    python.chmod(0o755)
    env = dict(os.environ, PATH=f"{tmp_path}:{os.environ['PATH']}",
               PROJECT_DIR=str(tmp_path / "project"), CODE_ROOT=str(tmp_path / "code"),
               EXPECTED_COMMIT="adapter", RUNTIME_CODE_ROOT=str(tmp_path / "runtime"),
               PYTHON_BIN=str(python), COSMODIFF_PIN_ROOT=str(tmp_path / "pin"),
               COSMODIFF_PIN_MANIFEST="manifest", EXPECTED_COSMODIFF_BASE_REVISION="base",
               L12_FIXTURE=str(tmp_path / "l12"), PREFLIGHT_ONLY="1")
    result = subprocess.run(["bash", str(ROOT / "scripts/slurm/evaluate_dit_trajectories.sbatch")],
                            env=env, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert "L12 OMITTED" in result.stdout
    assert "TRAJECTORY PREFLIGHT PASSED; NO GPU PROBE" in result.stdout
    assert not (tmp_path / "project/results").exists()
