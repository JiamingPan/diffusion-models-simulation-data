from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import sys
import subprocess
import os

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def isolate_diffusers_modules():
    """Native scheduler imports must not leak into later fake-torch guard tests."""
    before = {key: value for key, value in sys.modules.items()
              if key == "diffusers" or key.startswith("diffusers.")}
    yield
    for key in list(sys.modules):
        if key == "diffusers" or key.startswith("diffusers."):
            if key in before:
                sys.modules[key] = before[key]
            else:
                del sys.modules[key]


def load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_scheduler_kwargs_are_objects_and_applied_not_ignored(monkeypatch):
    sc = load("sample_cosmodiff")
    assert sc.parse_scheduler_kwargs('{"algorithm_type":"sde-dpmsolver++"}') == {
        "algorithm_type": "sde-dpmsolver++"}
    for bad in ("[]", "null", "broken", '{"x":NaN}'):
        with pytest.raises(Exception):
            sc.parse_scheduler_kwargs(bad)

    class Scheduler:
        def __init__(self, algorithm_type="dpmsolver++"):
            self.config = {"algorithm_type": algorithm_type}

        @classmethod
        def from_config(cls, config, **kwargs):
            return cls(**kwargs)

    monkeypatch.setitem(sys.modules, "diffusers", SimpleNamespace(Scheduler=Scheduler))
    result = sc.build_inference_scheduler(SimpleNamespace(config={}), "Scheduler",
                                         {"algorithm_type": "sde-dpmsolver++"})
    assert result.config["algorithm_type"] == "sde-dpmsolver++"
    with pytest.raises(ValueError, match="unknown"):
        sc.build_inference_scheduler(SimpleNamespace(config={}), "Scheduler", {"typo": 1})
    with pytest.raises(ValueError):
        sc.build_inference_scheduler(SimpleNamespace(config={}), None, {"solver_order": 2})


class Scheduler:
    def __init__(self, stochastic):
        self.config = SimpleNamespace(num_train_timesteps=4)
        self.stochastic = stochastic
        self.calls = []
        self.init_noise_sigma = 1.0

    def set_timesteps(self, n):
        self.timesteps = torch.arange(n - 1, -1, -1)

    def step(self, v, t, x, generator=None):
        self.calls.append(len(x))
        noise = torch.randn(x.shape, generator=generator) if self.stochastic else 0
        return SimpleNamespace(prev_sample=x - .1 * v + .01 * noise)


class Model:
    def __init__(self):
        self.chunks = []

    def eval(self):
        return self

    def to(self, device):
        return self

    def __call__(self, x, **kwargs):
        self.chunks.append(len(x))
        return (x * .1,)


def test_supplied_noise_is_preserved_and_step_rng_does_not_change_it():
    sc = load("sample_cosmodiff")
    initial = torch.randn(5, 1, 4, 4, generator=torch.Generator().manual_seed(123))
    before = initial.clone()
    results = []
    for chunk in (2, 5):
        scheduler, model = Scheduler(True), Model()
        result = sc.generate_samples(
            model, scheduler, batch_size=5, image_shape=(1, 4, 4), num_steps=4,
            device=torch.device("cpu"), generator=torch.Generator().manual_seed(123),
            initial_noise=initial, model_batch_size=chunk,
            class_labels=torch.zeros(5, dtype=torch.long))
        results.append(result)
        assert scheduler.calls == [5] * 4
        assert max(model.chunks) <= chunk
    assert torch.equal(initial, before)
    assert torch.equal(*results)
    with pytest.raises(ValueError):
        sc.generate_samples(Model(), Scheduler(False), batch_size=5, image_shape=(1, 4, 4),
                            num_steps=4, device=torch.device("cpu"), generator=None,
                            initial_noise=torch.zeros(4, 1, 4, 4))


def test_matrix_noise_extends_original_32_and_is_fail_closed():
    m = load("dit_sampler_matrix")
    original = m.make_initial_noise(32, (1, 128, 128), 123).numpy()
    extended = m.extend_noise(original, 128, 123)
    np.testing.assert_array_equal(extended[:32], original)
    assert extended.shape == (128, 1, 128, 128)
    assert m.STEP_SEED != 123
    first_step_noise = m.make_initial_noise(128, (1, 128, 128), m.STEP_SEED).numpy()
    assert not np.array_equal(first_step_noise, extended)
    with pytest.raises(ValueError, match="original"):
        m.extend_noise(original + .01, 128, 123)


def matrix_record(condition="sde_dpmpp50"):
    m = load("dit_sampler_matrix")
    c = next(c for c in m.CONDITIONS if c["name"] == condition)
    initial = m.make_initial_noise(128, (1, 4, 4), 123).numpy()
    config = {"algorithm_type": c["kwargs"].get("algorithm_type", ""), "solver_order": 2,
              "prediction_type": "v_prediction"}
    return {
        "protocol": np.asarray("dit_sampler_matrix_v1"),
        "matrix_condition": np.asarray(condition),
        "scheduler_class": np.asarray(c["scheduler"]),
        "scheduler_config_json": np.asarray(json.dumps(config)),
        "num_steps": np.asarray(c["steps"]), "executed_inference_steps": np.asarray(c["steps"]),
        "seed": np.asarray(123), "step_seed": np.asarray(m.STEP_SEED),
        "ema_sigma_rel": np.asarray(-1.), "resolved_checkpoint": np.asarray("checkpoint"),
        "config_sha256": np.asarray("config"), "reference_sha256": np.asarray("reference"),
        "initial_noise": initial,
        "initial_noise_sha256": np.asarray([m.array_hash(x) for x in initial]),
        "initial_noise_batch_sha256": np.asarray(m.array_hash(initial)),
    }


def test_matrix_evaluation_requires_explicit_mode_and_real_noise_provenance():
    evaluator = load("evaluate_late_start")
    record = matrix_record()
    result = evaluator.validate_matrix_provenance(record, reference_sha256="reference",
                                                config_sha256="config")
    assert result["matrix_condition"] == "sde_dpmpp50"
    with pytest.raises(ValueError):
        evaluator.validate_provenance(record, training_mean_sha256="mean")
    for key, value in (("seed", 124), ("step_seed", 123), ("ema_sigma_rel", .1),
                       ("num_steps", 200), ("reference_sha256", "wrong"),
                       ("config_sha256", "wrong"), ("initial_noise_batch_sha256", "wrong")):
        changed = dict(record, **{key: np.asarray(value)})
        with pytest.raises(ValueError):
            evaluator.validate_matrix_provenance(changed, reference_sha256="reference",
                                                 config_sha256="config")
    changed = dict(record)
    changed["initial_noise"] = record["initial_noise"].copy()
    changed["initial_noise"][0, 0, 0, 0] += 1
    with pytest.raises(ValueError):
        evaluator.validate_matrix_provenance(changed, reference_sha256="reference",
                                             config_sha256="config")


def test_conditions_cover_steps_and_stochastic_equal_step_control():
    m = load("dit_sampler_matrix")
    assert [(c["name"], c["steps"]) for c in m.CONDITIONS] == [
        ("dpm50", 50), ("dpm200", 200), ("ddpm500", 500), ("sde_dpmpp50", 50)]
    assert m.CONDITIONS[-1]["kwargs"] == {"algorithm_type": "sde-dpmsolver++", "solver_order": 2}


def test_optional_l12_is_omitted_but_mandatory_missing_weights_raise(tmp_path):
    m = load("dit_sampler_matrix")
    rows = []
    for name in ("dit_l8_200k", "dit_l12_200k", "dit_l16_fresh300k", "dit_l16_seed456_500k"):
        checkpoint = tmp_path / name
        checkpoint.mkdir()
        (checkpoint / "config.json").write_text('{}')
        if "l12" not in name:
            (checkpoint / "diffusion_pytorch_model.safetensors").write_bytes(b"weights")
        config = tmp_path / f"{name}.yaml"
        config.write_text("data: {}\nnoise_scheduler: {}\n")
        rows.append({"name": name, "checkpoint": str(checkpoint), "config": str(config)})
    selected, omitted = m.select_models(rows)
    assert len(selected) == 3 and omitted[0]["name"] == "dit_l12_200k"
    (tmp_path / "dit_l12_200k/diffusion_pytorch_model.safetensors").write_bytes(b"weights")
    assert len(m.select_models(rows)[0]) == 4
    (tmp_path / "dit_l8_200k/diffusion_pytorch_model.safetensors").unlink()
    with pytest.raises(FileNotFoundError):
        m.select_models(rows)


def test_matrix_wrapper_runs_preflight_before_jobs_and_keeps_frozen_runtime():
    text = (ROOT / "scripts/slurm/sample_dit_sampler_matrix.sbatch").read_text()
    assert "#SBATCH --nodes=1" in text and "#SBATCH --gres=gpu:1" in text
    assert "seed_restart_runtime:$RUNTIME_CODE_ROOT:$COSMODIFF_PIN_ROOT" in text
    assert "555f350f82c913ff06150c96e66709719856cd83" in text
    assert "--runtime-preflight" in text and "PREFLIGHT_ONLY" in text
    assert "sbatch " not in text and "srun " not in text


def make_project(tmp_path, monkeypatch):
    m = load("dit_sampler_matrix")
    evaluator = load("evaluate_late_start")
    project = tmp_path / "project"
    project.mkdir()
    raw = np.random.default_rng(7).normal(size=(256, 128, 128)).astype(np.float32)
    np.save(project / "real.npy", raw)
    rows = []
    for name in ("dit_l8_200k", "dit_l16_fresh300k", "dit_l16_seed456_500k"):
        checkpoint = project / name
        checkpoint.mkdir()
        (checkpoint / "config.json").write_text('{}')
        (checkpoint / "diffusion_pytorch_model.safetensors").write_bytes(b"weights")
        config = project / f"{name}.yaml"
        config.write_text(f"data:\n  img_path: {project / 'real.npy'}\n"
                          "  img_read_fn: npy_read_fn\n  two_dim: true\n  normalization: none\n"
                          "noise_scheduler:\n  class: DDPMScheduler\n  kwargs:\n"
                          "    num_train_timesteps: 500\n    prediction_type: v_prediction\n"
                          "    beta_schedule: squaredcos_cap_v2\n    rescale_betas_zero_snr: true\n"
                          "    clip_sample: false\n")
        rows.append({"name": name, "checkpoint": str(checkpoint), "config": str(config)})
        reference = evaluator.load_reference(config, ROOT)
        directory = project / "results/dit_l16_trajectory_screen" / name
        directory.mkdir(parents=True)
        path = directory / f"{name}_trajectory.npz"
        np.savez_compressed(path, initial_noise=m.make_initial_noise(32, (1, 128, 128), 123).numpy(),
                            reference_sha256=m.array_hash(reference), seed=123,
                            final=np.zeros((32, 1, 128, 128), dtype=np.float32),
                            outcome=np.full(32, "junk"))
        path.with_suffix(".csv").write_text("step,n\n0,32\n")
        path.with_suffix(".json").write_text(json.dumps({"status": "complete", "raw_weights": True,
            "num_steps": 50, "scheduler_config": {"algorithm_type": "dpmsolver++", "solver_order": 2},
            "config_sha256": m.file_hash(config), "checkpoint": str(checkpoint), "artifacts_sha256": {
            path.name: m.file_hash(path), path.with_suffix(".csv").name: m.file_hash(path.with_suffix(".csv"))}}))
    monkeypatch.setattr(m, "resolve", lambda project, **kwargs: rows)
    monkeypatch.setenv("EXPECTED_COMMIT", "adapter")
    output = tmp_path / "matrix"
    m.prepare(project, output)
    monkeypatch.setenv("MATRIX_PLAN_SHA256", m.file_hash(output / "plan/plan.json"))
    return m, project, output


def test_plan_publication_no_overwrite_and_tampering_fails(tmp_path, monkeypatch):
    m, project, output = make_project(tmp_path, monkeypatch)
    plan, noise = m.load_plan(output)
    assert len(plan["tasks"]) == 12 and noise.shape == (128, 1, 128, 128)
    assert m.array_hash(noise[:32]) == plan["original32_sha256"]
    with pytest.raises(FileExistsError):
        m.prepare(project, output)
    checkpoint = Path(plan["models"][0]["checkpoint"]) / "diffusion_pytorch_model.safetensors"
    checkpoint.write_bytes(b"changed")
    with pytest.raises(ValueError, match="checkpoint changed"):
        m.validate_inputs(plan["models"][0])
    with (output / "plan/initial_noise.npz").open("ab") as f:
        f.write(b"changed")
    with pytest.raises(ValueError, match="noise file hash"):
        m.load_plan(output)


def test_actual_evaluator_child_accepts_matrix_but_default_rejects(tmp_path):
    m = load("dit_sampler_matrix")
    raw = np.random.default_rng(8).normal(size=(256, 128, 128)).astype(np.float32)
    np.save(tmp_path / "real.npy", raw)
    config = tmp_path / "config.yaml"
    config.write_text(f"data:\n  img_path: {tmp_path / 'real.npy'}\n"
                      "  img_read_fn: npy_read_fn\n  two_dim: true\n  normalization: none\n")
    evaluator = load("evaluate_late_start")
    reference = evaluator.load_reference(config, ROOT)
    record = matrix_record()
    initial = m.make_initial_noise(128, (1, 128, 128), 123).numpy()
    record.update(initial_noise=initial,
                  initial_noise_sha256=np.asarray([m.array_hash(x) for x in initial]),
                  initial_noise_batch_sha256=np.asarray(m.array_hash(initial)))
    record.update(reference_sha256=np.asarray(m.array_hash(reference)),
                  config_sha256=np.asarray(m.file_hash(config)), samples=record["initial_noise"].copy())
    samples = tmp_path / "samples.npz"
    np.savez_compressed(samples, **record)
    command = [sys.executable, str(ROOT / "scripts/evaluate_late_start.py"), "--config", str(config),
               "--samples", str(samples), "--eval-root", str(ROOT)]
    default = subprocess.run(command, capture_output=True, text=True)
    assert default.returncode != 0 and "late_start_requested" in default.stderr
    accepted = subprocess.run([*command, "--protocol", "sampler-matrix", "--out", str(tmp_path / "summary.csv")],
                              capture_output=True, text=True)
    assert accepted.returncode == 0, accepted.stderr
    assert "matrix_condition=sde_dpmpp50" in accepted.stdout
    again = subprocess.run([*command, "--protocol", "sampler-matrix", "--out", str(tmp_path / "summary.csv")],
                           capture_output=True, text=True)
    assert again.returncode != 0 and "refusing to overwrite" in again.stderr


@pytest.mark.parametrize("condition", ["dpm50", "dpm200", "ddpm500", "sde_dpmpp50"])
def test_real_diffusers_cpu_steps_are_finite_and_step_rng_reproducible(tmp_path, condition):
    pytest.importorskip("diffusers")
    m = load("dit_sampler_matrix")
    config = tmp_path / "scheduler.yaml"
    config.write_text("noise_scheduler:\n  class: DDPMScheduler\n  kwargs:\n"
                      "    num_train_timesteps: 500\n    prediction_type: v_prediction\n"
                      "    beta_schedule: squaredcos_cap_v2\n    rescale_betas_zero_snr: true\n"
                      "    clip_sample: false\n")
    c = next(c for c in m.CONDITIONS if c["name"] == condition)
    initial = m.make_initial_noise(3, (1, 4, 4), 123)
    outputs = []
    for chunk in (1, 3):
        scheduler = m.scheduler_for(config, c)
        with torch.no_grad():
            result = m.sc.generate_samples(Model(), scheduler, batch_size=3, image_shape=(1, 4, 4),
                num_steps=c["steps"], device=torch.device("cpu"), initial_noise=initial,
                model_batch_size=chunk, class_labels=torch.zeros(3, dtype=torch.long),
                generator=torch.Generator(device="cpu").manual_seed(m.STEP_SEED))
        assert torch.isfinite(result).all()
        assert len(scheduler.timesteps) == c["steps"] and int(scheduler.timesteps[0]) == 499
        outputs.append(result)
    torch.testing.assert_close(*outputs, rtol=0, atol=0)


def test_real_scheduler_task_and_child_score_publish_only_after_success(tmp_path, monkeypatch):
    pytest.importorskip("diffusers")
    m, _, output = make_project(tmp_path, monkeypatch)
    monkeypatch.setattr(m.sc, "_load_for_sampling", lambda *args: (Model(), None))
    monkeypatch.setattr(m.sc, "_install_sklearn_roc_curve_stub", lambda: None)
    torch.set_num_threads(2)
    m.runtime_preflight(output)
    m.run_task(output, 0, ROOT, torch.device("cpu"))
    directory = output / "tasks/dit_l8_200k__dpm50"
    report = json.loads((directory / "complete.json").read_text())
    assert report["status"] == "complete" and report["n"] == 128
    assert report["qualification"]  # Deliberately different fake baseline must not be concealed.
    assert set(report["artifacts_sha256"]) == {"samples.npz", "summary.csv", "outcomes.csv"}
    for name, digest in report["artifacts_sha256"].items():
        assert m.file_hash(directory / name) == digest
    with pytest.raises(FileExistsError):
        m.run_task(output, 0, ROOT, torch.device("cpu"))
    with pytest.raises(FileNotFoundError):
        m.summarize(output)
    assert not (output / "comparison").exists()


def test_complete_aggregate_rejects_metadata_counts_and_order_then_publishes(tmp_path, monkeypatch):
    """Synthetic completed arms test report integrity, not trained-model quality."""
    m, _, output = make_project(tmp_path, monkeypatch)
    evaluator = load("evaluate_late_start")
    plan, noise = m.load_plan(output)
    reference = evaluator.load_reference(Path(plan["models"][0]["config"]), ROOT)
    power, k = evaluator.radial_power(reference)
    scores = evaluator.score(noise, reference, power.mean(axis=0), k)
    cos = evaluator.max_cosine(noise, reference)
    for task in plan["tasks"]:
        model = next(r for r in plan["models"] if r["name"] == task["model"])
        condition = next(c for c in m.CONDITIONS if c["name"] == task["condition"])
        directory = output / "tasks" / f"{task['model']}__{task['condition']}"
        directory.mkdir(parents=True)
        record = matrix_record(task["condition"])
        record.update(samples=noise, initial_noise=noise,
            initial_noise_sha256=np.asarray([m.array_hash(x) for x in noise]),
            initial_noise_batch_sha256=np.asarray(m.array_hash(noise)),
            config_sha256=np.asarray(model["config_sha256"]),
            reference_sha256=np.asarray(plan["reference_sha256"]),
            resolved_checkpoint=np.asarray(model["checkpoint"]))
        np.savez(directory / "samples.npz", **record)
        provenance = evaluator.validate_matrix_provenance(record,
            reference_sha256=plan["reference_sha256"], config_sha256=model["config_sha256"])
        m.write_csv(directory / "summary.csv", [{"file": "samples.npz", **provenance, **scores}])
        m.write_csv(directory / "outcomes.csv", [{"sample": i,
            "initial_noise_sha256": m.array_hash(noise[i]), "outcome": "junk", "max_cos": float(cos[i])}
            for i in range(128)])
        report = {"status": "complete", "protocol": m.PROTOCOL, "task": task,
            "condition": condition, "input": model, "n": 128, "seed": 123,
            "step_seed": m.STEP_SEED, "raw_weights": True, "runtime_versions": plan["runtime_versions"],
            "plan_sha256": m.file_hash(output / "plan/plan.json"),
            "noise_batch_sha256": m.array_hash(noise), "reference_sha256": plan["reference_sha256"],
            "qualification": ["Synthetic test fixture"],
            "artifacts_sha256": {p.name: m.file_hash(p) for p in directory.iterdir()}}
        (directory / "complete.json").write_text(json.dumps(report))
    directory = output / "tasks/dit_l8_200k__dpm50"
    original_summary = (directory / "summary.csv").read_text()
    original_outcomes = (directory / "outcomes.csv").read_text()

    def replace_artifact(name, content):
        (directory / name).write_text(content)
        report = json.loads((directory / "complete.json").read_text())
        report["artifacts_sha256"][name] = m.file_hash(directory / name)
        (directory / "complete.json").write_text(json.dumps(report))

    replace_artifact("summary.csv", original_summary.replace("dpm50,", "ddpm500,"))
    with pytest.raises(ValueError, match="summary provenance"):
        m.summarize(output)
    replace_artifact("summary.csv", original_summary)
    import csv
    with (directory / "summary.csv").open() as f:
        changed = next(csv.DictReader(f))
    changed["junk_count"] = 127
    m.write_csv(directory / "summary.csv", [changed])
    replace_artifact("summary.csv", (directory / "summary.csv").read_text())
    with pytest.raises(ValueError, match="counts do not reconcile"):
        m.summarize(output)
    replace_artifact("summary.csv", original_summary)
    lines = original_outcomes.splitlines(keepends=True)
    replace_artifact("outcomes.csv", "".join([lines[0], lines[2], lines[1], *lines[3:]]))
    with pytest.raises(ValueError, match="pairing/order"):
        m.summarize(output)
    replace_artifact("outcomes.csv", original_outcomes)
    assert not (output / "comparison").exists()
    m.summarize(output)
    report = json.loads((output / "comparison/complete.json").read_text())
    assert report["status"] == "complete" and report["tasks"] == 12 and report["n"] == 128
    with (output / "comparison/summary_original32.csv").open() as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 12 and all(int(r["junk_count"]) == 32 for r in rows)
    for name, digest in report["artifacts_sha256"].items():
        assert m.file_hash(output / "comparison" / name) == digest
    with pytest.raises(FileExistsError):
        m.summarize(output)


def test_wrapper_preflight_uses_real_pin_root_and_no_gpu_task(tmp_path):
    for name in ("runtime", "adapter", "project", "pin"):
        (tmp_path / name).mkdir()
    git = tmp_path / "git"
    git.write_text('#!/bin/bash\nif [[ "$3" == rev-parse ]]; then\n'
                   ' if [[ "$2" == "$RUNTIME_CODE_ROOT" ]]; then\n'
                   '  echo 555f350f82c913ff06150c96e66709719856cd83\n'
                   ' else echo adapter; fi\nfi\n')
    git.chmod(0o755)
    python = tmp_path / "python"
    python.write_text('#!/bin/bash\nset -eu\n'
                      'test "$PYTHONPATH" = "$COSMODIFF_PIN_ROOT/seed_restart_runtime:$RUNTIME_CODE_ROOT:$COSMODIFF_PIN_ROOT"\n'
                      'test "$PYTHONNOUSERSITE" = 1\n'
                      'if [[ "$*" == *verify_cosmodiff_seed_restart_runtime.py* ]]; then\n'
                      ' [[ "$*" == *"--code-root $RUNTIME_CODE_ROOT"* ]]\n'
                      'elif [[ "$*" == *dit_sampler_matrix.py* ]]; then\n'
                      ' [[ "$*" == *--runtime-preflight* ]]\n'
                      ' [[ "$*" != *--task* ]]\n'
                      'else exit 9; fi\n')
    python.chmod(0o755)
    env = {**os.environ, "PATH": f"{tmp_path}:{os.environ['PATH']}",
           "PROJECT_DIR": str(tmp_path / "project"), "CODE_ROOT": str(tmp_path / "adapter"),
           "EXPECTED_COMMIT": "adapter", "RUNTIME_CODE_ROOT": str(tmp_path / "runtime"),
           "COSMODIFF_PIN_ROOT": str(tmp_path / "pin"), "COSMODIFF_PIN_MANIFEST": "manifest",
           "EXPECTED_COSMODIFF_BASE_REVISION": "base", "MATRIX_PLAN_SHA256": "plan",
           "PYTHON_BIN": str(python), "PREFLIGHT_ONLY": "1"}
    result = subprocess.run(["bash", str(ROOT / "scripts/slurm/sample_dit_sampler_matrix.sbatch")],
                            env=env, text=True, capture_output=True)
    assert result.returncode == 0, result.stderr
    assert "NO MODEL LOAD OR GPU SAMPLING OR JOB SUBMISSION" in result.stdout
