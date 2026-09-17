from copy import deepcopy
from pathlib import Path
import sys

import numpy as np
import pytest
import torch

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
from dit_a40_ablation import ARMS, arm_configs, single_a40, validate_runtime_versions
from dit_zero_init import zero_modulation_and_output
from prepare_nf_generalize_fig2_dit_configs import build_config
from patch_memorization_test import patch_cos_stats


def baseline():
    config = build_config("baseline", "dit_l16", [{"path": "/example.npy", "n_samples": 16}], 256)
    config["train"]["num_epochs"] = 9375
    return config


def matrix_versions():
    return {"python": "3.10.9", "torch": "2.1.2+cu118", "numpy": "1.26.4",
            "diffusers": "0.38.0", "huggingface-hub": "0.36.2"}


def test_actual_great_lakes_versions_match_without_a_downgrade():
    assert validate_runtime_versions(matrix_versions(), matrix_versions()) is None


@pytest.mark.parametrize("key", list(matrix_versions()))
def test_runtime_drift_fails_with_expected_and_actual_values(key):
    actual = matrix_versions()
    actual[key] = "unexpected"
    with pytest.raises(RuntimeError, match="differs from frozen working matrix") as error:
        validate_runtime_versions(matrix_versions(), actual)
    assert key in str(error.value)
    assert "unexpected" in str(error.value)
    assert matrix_versions()[key] in str(error.value)


@pytest.mark.parametrize("version", ["0.35.1", "0.39.0"])
def test_untested_matrix_version_must_not_silently_change_the_recipe(version):
    expected = matrix_versions()
    expected["diffusers"] = version
    with pytest.raises(RuntimeError, match="has not been tested"):
        validate_runtime_versions(expected)


@pytest.mark.parametrize("expected", [None, {}, {"diffusers": "0.38.0"},
    {**matrix_versions(), "torch": None}])
def test_missing_runtime_contract_fails_closed(expected):
    with pytest.raises(ValueError, match="missing/incomplete"):
        validate_runtime_versions(expected)


def test_three_arms_only_requested_training_differences():
    original = baseline()
    snapshot = deepcopy(original)
    rows = arm_configs(original, Path("/scratch/test"))
    assert [(r["name"], r["patch_size"], r["initialization"]) for r in rows] == ARMS
    assert original == snapshot
    assert len({r["checkpoint_dir"] for r in rows}) == 3
    for row in rows:
        config = deepcopy(row["config_data"])
        config["io"]["output_dir"] = original["io"]["output_dir"]
        config["model"]["kwargs"]["patch_size"] = 8
        config["train"]["checkpoint_every_n_epochs"] = original["train"]["checkpoint_every_n_epochs"]
        assert config == original
        assert row["target_updates"] == 9375 * 32 == 300000


@pytest.mark.parametrize("section,key,value", [("train", "batch_size", 4), ("data", "constant_label", 1)])
def test_wrong_baseline_rejected(section, key, value):
    config = baseline()
    config[section][key] = value
    with pytest.raises(ValueError):
        arm_configs(config, Path("/scratch/test"))


def tiny_dit(patch):
    from simdiff_eval.torch_compat import install_torch_backend_compat
    install_torch_backend_compat(entry_point="test_dit_a40_ablation")
    diffusers = pytest.importorskip("diffusers")
    torch.manual_seed(123)
    return diffusers.DiTTransformer2DModel(sample_size=16, patch_size=patch,
        in_channels=1, out_channels=1, num_layers=16, num_attention_heads=2,
        attention_head_dim=8, num_embeds_ada_norm=1, norm_num_groups=4)


@pytest.mark.parametrize("patch", [4, 8])
def test_native_dit_zero_only_targets_and_first_optimizer_step(patch):
    model = tiny_dit(patch)
    before = {k: v.clone() for k, v in model.state_dict().items()}
    report = zero_modulation_and_output(model, fresh=True)
    assert report["zeroed_tensors"] == 36
    targets = {f"{name}.{field}" for name in report["linear_modules"] for field in ("weight", "bias")}
    for name, value in model.state_dict().items():
        if name in targets:
            assert torch.count_nonzero(value) == 0
        else:
            assert torch.equal(value, before[name])
    x = torch.randn(2, 1, 16, 16)
    pred = model(x, timestep=torch.tensor([249, 399]), class_labels=torch.zeros(2, dtype=torch.long)).sample
    assert torch.count_nonzero(pred) == 0
    target = torch.randn_like(pred)
    # Same v/min-SNR functional form, nonzero SNR away from terminal endpoint.
    snr = torch.tensor([.976, .104])
    weights = snr.clamp(max=5) / (snr + 1)
    loss = (weights * ((pred-target)**2).mean(dim=(1, 2, 3))).mean()
    loss.backward()
    assert torch.isfinite(loss)
    assert torch.count_nonzero(model.proj_out_2.weight.grad) > 0
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=.01)
    optimizer.step()
    assert torch.isfinite(model.proj_out_2.weight).all()
    assert torch.count_nonzero(model.proj_out_2.weight) > 0


def test_zero_init_rejects_resume_and_duplicate():
    model = tiny_dit(8)
    before = model.proj_out_2.weight.clone()
    with pytest.raises(RuntimeError):
        zero_modulation_and_output(model, fresh=False)
    assert torch.equal(model.proj_out_2.weight, before)
    zero_modulation_and_output(model, fresh=True)
    with pytest.raises(RuntimeError):
        zero_modulation_and_output(model, fresh=True)


def test_bad_api_validation_is_all_or_nothing():
    model = tiny_dit(8)
    before = model.transformer_blocks[0].norm1.linear.weight.clone()
    model.proj_out_2 = torch.nn.Linear(16, 63)
    with pytest.raises(ValueError):
        zero_modulation_and_output(model, fresh=True)
    assert torch.equal(model.transformer_blocks[0].norm1.linear.weight, before)


def test_patch_search_matches_dense_and_does_not_modify_inputs():
    from patchwork_check import to_patches
    rng = np.random.default_rng(123)
    real = rng.normal(size=(4, 1, 16, 16)).astype(np.float32)
    samples = real[:2].copy()
    real_before, samples_before = real.copy(), samples.copy()
    same, anywhere, residual = patch_cos_stats(samples, real, query_chunk=3, bank_chunk=5)
    expected = (to_patches(samples).reshape(-1, 64) @ to_patches(real).reshape(-1, 64).T).max(axis=1)
    assert np.allclose(anywhere.ravel(), expected, atol=1e-6)
    assert np.allclose(same, 1, atol=1e-6)
    assert np.allclose(residual, 0, atol=1e-6)
    assert np.array_equal(real, real_before)
    assert np.array_equal(samples, samples_before)


def test_a40_no_ddp_or_other_gpu(monkeypatch):
    monkeypatch.setenv("WORLD_SIZE", "2")
    with pytest.raises(RuntimeError, match="single-process"):
        single_a40(torch)
    monkeypatch.setenv("WORLD_SIZE", "1")
    monkeypatch.setenv("RANK", "1")
    with pytest.raises(RuntimeError, match="nonzero rank"):
        single_a40(torch)


def test_job_has_one_gpu_and_no_pruning():
    script = (SCRIPTS / "slurm/dit_a40_ablation.sbatch").read_text()
    assert "#SBATCH --nodes=1" in script
    assert "#SBATCH --gres=gpu:1" in script
    assert "prune" not in script and "sbatch " not in script and "srun " not in script


def test_single_owner_report_publication_and_crash_recovery(tmp_path):
    from dit_a40_ablation import publish_json
    from trace_trajectories import atomic_output
    import json
    target = tmp_path / "complete_record"
    with pytest.raises(RuntimeError, match="interrupted"):
        with atomic_output(target) as pending:
            (pending / "complete.json").write_text('{"status":"partial"}')
            raise RuntimeError("interrupted")
    assert not target.exists()
    assert not (tmp_path / ".complete_record.lock").exists()
    retained = list(tmp_path.glob(".complete_record.pending.*"))
    assert len(retained) == 1
    assert json.loads((retained[0] / "complete.json").read_text())["status"] == "partial"
    publish_json(tmp_path / "complete.json", {"status": "complete"})
    assert json.loads((target / "complete.json").read_text())["status"] == "complete"
    with pytest.raises(FileExistsError):
        publish_json(tmp_path / "complete.json", {"status": "changed"})


def test_live_publication_lock_refuses_second_owner(tmp_path):
    from trace_trajectories import atomic_output
    target = tmp_path / "result"
    with atomic_output(target) as pending:
        with pytest.raises(FileExistsError):
            with atomic_output(target):
                pytest.fail("second writer must not enter")
        (pending / "ok.json").write_text('{}')
    assert (target / "ok.json").is_file()


def test_frozen_plan_rejects_modified_yaml_and_noise(tmp_path, monkeypatch):
    import json
    import dit_a40_ablation as ab
    source = tmp_path / "baseline.yaml"
    import yaml
    source.write_text(yaml.safe_dump(baseline()))
    output = tmp_path / "experiment"
    directory = output / "plan"
    directory.mkdir(parents=True)
    matrix_path = tmp_path / "matrix_plan.json"
    matrix_path.write_text(json.dumps({"runtime_versions": matrix_versions()}))
    monkeypatch.setattr(ab, "MATRIX_SHA", ab.file_hash(matrix_path))
    rows = ab.arm_configs(baseline(), output / "checkpoints")
    for row in rows:
        config = row.pop("config_data")
        path = directory / f"{row['name']}.yaml"
        path.write_text(yaml.safe_dump(config))
        row.update(config=str(path), config_sha256=ab.file_hash(path))
    noise = np.random.default_rng(123).normal(size=(128, 1, 128, 128)).astype(np.float32)
    np.savez_compressed(directory / "initial_noise.npz", initial_noise=noise)
    plan = {"status": "prepared", "protocol": ab.NAME, "code_revision": "testcode",
        "arms": rows, "baseline": {"config": str(source), "config_sha256": ab.file_hash(source)},
        "matrix_plan_path": str(matrix_path), "matrix_plan_sha256": ab.MATRIX_SHA,
        "runtime_versions": matrix_versions(),
        "noise_file_sha256": ab.file_hash(directory / "initial_noise.npz"), "noise_batch_sha256": ab.array_hash(noise)}
    path = directory / "plan.json"
    path.write_text(json.dumps(plan))
    monkeypatch.setenv("EXPECTED_COMMIT", "testcode")
    monkeypatch.setenv("ABLATION_PLAN_SHA256", ab.file_hash(path))
    _, actual = ab.load_plan(output)
    assert np.array_equal(actual, noise)
    for versions in (None, {**matrix_versions(), "torch": "different"}):
        edited = {**plan, "runtime_versions": versions}
        path.write_text(json.dumps(edited))
        monkeypatch.setenv("ABLATION_PLAN_SHA256", ab.file_hash(path))
        with pytest.raises(ValueError, match="runtime"):
            ab.load_plan(output)
    path.write_text(json.dumps(plan))
    monkeypatch.setenv("ABLATION_PLAN_SHA256", ab.file_hash(path))
    matrix_text = matrix_path.read_text()
    matrix_path.write_text(matrix_text + "\n")
    with pytest.raises(ValueError, match="original reviewed sampler matrix plan changed"):
        ab.load_plan(output)
    matrix_path.write_text(matrix_text)
    config_path = Path(rows[0]["config"])
    text = config_path.read_text()
    config_path.write_text(text + "# edited\n")
    with pytest.raises(ValueError, match="config changed"):
        ab.load_plan(output)
    config_path.write_text(text)
    np.savez_compressed(directory / "initial_noise.npz", initial_noise=noise + .1)
    with pytest.raises(ValueError, match="noise changed"):
        ab.load_plan(output)


def test_failed_first_cpu_update_cannot_emit_preflight_pass():
    source = (SCRIPTS / "dit_a40_ablation.py").read_text()
    assert source.index('raise RuntimeError("native first CPU update is nonfinite")') < source.index('print("ABLATION PREFLIGHT PASSED;')
    assert 'checkpoint_dir.mkdir(parents=True, exist_ok=False)' in source
    assert 'zero_modulation_and_output(model, fresh=True)' in source


def test_all_entrypoints_check_the_frozen_runtime_and_prepare_records_it():
    source = (SCRIPTS / "dit_a40_ablation.py").read_text()
    assert source.count('runtime(plan["runtime_versions"])') == 3
    assert '"runtime_versions": deepcopy(matrix["runtime_versions"])' in source
    assert "this init ablation is verified against diffusers 0.35.1" not in source
