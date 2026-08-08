from __future__ import annotations

import json
import sys
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest
import yaml


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import prepare_nf_generalize_fig2_dit_l16_continue500k_v2_configs as prep
import check_nf_generalize_fig2_dit_l16_continue500k_v2 as precheck
import validate_nf_generalize_fig2_dit_sample as sample_guard


EXPECTED_TAGS = tuple(f"d2p{power:02d}" for power in range(6, 16))


def _source_config(output_dir: Path, *, num_epochs: int) -> dict:
    return {
        "io": {"output_dir": str(output_dir), "logging": "tensorboard"},
        "data": {
            "data_path": "/scratch/camels",
            "sample_size": 128,
            "n_samples": 64,
            "augment": False,
        },
        "model": {
            "sample_size": 128,
            "patch_size": 8,
            "num_layers": 16,
            "num_attention_heads": 12,
            "attention_head_dim": 64,
            "norm_num_groups": 32,
            "num_classes": 1,
            "class_label": 0,
        },
        "noise_scheduler": {"prediction_type": "v_prediction"},
        "optimizer": {"learning_rate": 1.0e-4},
        "lr_scheduler": {"T_0": 4000},
        "train": {
            "num_epochs": num_epochs,
            "checkpoint_every_n_epochs": 625,
            "batch_size": 8,
            "seed": 123,
        },
    }


def _write_source_sweep(project_dir: Path) -> list[dict]:
    source_root = project_dir / "source_checkpoints" / prep.SOURCE_SWEEP_NAME
    rows = []
    for power, tag in zip(range(6, 16), EXPECTED_TAGS):
        dataset_size = 2**power
        steps_per_epoch = max(1, dataset_size // 8)
        final_num_epochs = prep.ceil_div(300_000, steps_per_epoch)
        final_epoch = final_num_epochs - 1
        run_name = f"nf_fig2_dit_l16_{tag}_noaug_fresh300k_v2_seed123"
        checkpoint_dir = source_root / f"{run_name}_checkpoints"
        checkpoint = checkpoint_dir / f"checkpoint-epoch-{final_epoch:04d}"
        config_path = (
            project_dir
            / "local"
            / prep.SOURCE_SWEEP_NAME
            / "configs"
            / f"{run_name}.yaml"
        )
        config_path.parent.mkdir(parents=True, exist_ok=True)
        config = _source_config(checkpoint_dir, num_epochs=final_num_epochs)
        config["data"]["n_samples"] = dataset_size
        config_path.write_text(yaml.safe_dump(config, sort_keys=False))
        rows.append(
            {
                "manifest_version": 1,
                "sweep_name": prep.SOURCE_SWEEP_NAME,
                "run_name": run_name,
                "arch": "dit_l16",
                "dataset_tag": tag,
                "dataset_size": dataset_size,
                "optimizer_steps_per_epoch": steps_per_epoch,
                "target_total_updates": 300_000,
                "expected_final_epoch": final_epoch,
                "expected_checkpoint": str(checkpoint),
                "checkpoint_dir": str(checkpoint_dir),
                "config": str(config_path.relative_to(project_dir)),
                "sample_label": "dpm50_fresh300k_v2",
                "sample_path": (
                    f"results/{prep.SOURCE_SWEEP_NAME}/samples/"
                    f"{run_name}_seed123_dpm50_fresh300k_v2.npz"
                ),
            }
        )
    manifest = project_dir / "local" / prep.SOURCE_SWEEP_NAME / "manifest.json"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(json.dumps(rows, indent=2) + "\n")
    return rows


def test_build_continuation_rows_covers_all_ten_runs_and_five_stages(tmp_path):
    source_rows = _write_source_sweep(tmp_path)
    rows = prep.build_continuation_rows(
        tmp_path,
        source_rows,
        checkpoint_root=tmp_path / "continuation_checkpoints" / prep.CONTINUE_SWEEP_NAME,
    )

    assert len(rows) == 50
    assert {row["dataset_tag"] for row in rows} == set(EXPECTED_TAGS)
    for target in prep.TARGET_UPDATES:
        stage_rows = [row for row in rows if row["target_total_updates"] == target]
        assert [row["dataset_tag"] for row in stage_rows] == list(EXPECTED_TAGS)
        assert {row["sample_label"] for row in stage_rows} == {
            f"dpm50_cont_{target // 1000}k"
        }

    for row in rows:
        steps = row["optimizer_steps_per_epoch"]
        target = row["target_total_updates"]
        expected_epoch = prep.ceil_div(target, steps) - 1
        assert row["expected_final_epoch"] == expected_epoch
        assert row["expected_checkpoint"].endswith(
            f"checkpoint-epoch-{expected_epoch:04d}"
        )
        assert prep.CONTINUE_SWEEP_NAME in row["expected_checkpoint"]
        assert prep.SOURCE_SWEEP_NAME in row["source_checkpoint"]
        assert row["source_checkpoint"] != row["expected_checkpoint"]
        assert row["source_config_sha256"] == prep.sha256_file(
            tmp_path / row["source_config"]
        )

        stage_index = prep.TARGET_UPDATES.index(target)
        if stage_index == 0:
            assert row["previous_expected_checkpoint"] == row["seed_checkpoint"]
        else:
            previous_target = prep.TARGET_UPDATES[stage_index - 1]
            previous_epoch = prep.ceil_div(previous_target, steps) - 1
            assert row["previous_expected_checkpoint"].endswith(
                f"checkpoint-epoch-{previous_epoch:04d}"
            )


def test_stage_spacing_matches_cosine_restart_period():
    checkpoints = (prep.SOURCE_TARGET_UPDATES,) + prep.TARGET_UPDATES
    assert all(
        (right - left) == prep.STAGE_UPDATES
        for left, right in zip(checkpoints, checkpoints[1:])
    )
    assert prep.STAGE_UPDATES % prep.RESTART_PERIOD_UPDATES == 0


@pytest.mark.parametrize("mode", ["missing", "duplicate"])
def test_source_rows_must_have_each_dataset_tag_once(tmp_path, mode):
    rows = _write_source_sweep(tmp_path)
    if mode == "missing":
        rows.pop()
    else:
        rows[-1] = deepcopy(rows[-2])

    with pytest.raises(ValueError, match="dataset tags"):
        prep.build_continuation_rows(tmp_path, rows)


def test_clone_changes_only_three_allowed_config_keys(tmp_path):
    source_rows = _write_source_sweep(tmp_path)
    row = prep.build_continuation_rows(tmp_path, source_rows)[0]
    source_path = tmp_path / row["source_config"]
    continuation_path = tmp_path / row["config"]

    continuation = prep.build_continuation_config(source_path, row)
    continuation_path.parent.mkdir(parents=True, exist_ok=True)
    continuation_path.write_text(yaml.safe_dump(continuation, sort_keys=False))
    prep.assert_continuation_config(source_path, continuation_path, row)

    broken = yaml.safe_load(continuation_path.read_text())
    broken["model"]["patch_size"] = 4
    continuation_path.write_text(yaml.safe_dump(broken, sort_keys=False))
    with pytest.raises(ValueError, match=r"model\.patch_size"):
        prep.assert_continuation_config(source_path, continuation_path, row)


def test_seed_checkpoint_copy_is_complete_and_idempotent(tmp_path):
    source_rows = _write_source_sweep(tmp_path)
    rows = prep.build_continuation_rows(
        tmp_path,
        source_rows,
        checkpoint_root=tmp_path / "continuation_checkpoints" / prep.CONTINUE_SWEEP_NAME,
    )
    first_stage = [row for row in rows if row["continue_stage"] == 1]

    for row in first_stage:
        source = Path(row["source_checkpoint"])
        source.mkdir(parents=True)
        (source / "diffusion_pytorch_model.safetensors").write_bytes(b"model")
        (source / "optimizer.bin").write_bytes(b"optimizer")
        (source / "scheduler.bin").write_bytes(b"scheduler")
        (source / "scaler.pt").write_bytes(b"scaler")
        (source / "random_states_0.pkl").write_bytes(b"rng")
        (source / "ema_state.pt").write_bytes(b"ema")

    prep.seed_continuation_directories(rows)
    prep.seed_continuation_directories(rows)

    for row in first_stage:
        assert prep.checkpoint_inventory(Path(row["source_checkpoint"])) == (
            prep.checkpoint_inventory(Path(row["seed_checkpoint"]))
        )

    corrupt = Path(first_stage[0]["seed_checkpoint"]) / "optimizer.bin"
    corrupt.write_bytes(b"wrong")
    with pytest.raises(ValueError, match="not byte-identical"):
        prep.seed_continuation_directories(rows)


def test_existing_manifest_must_be_identical(tmp_path):
    source_rows = _write_source_sweep(tmp_path)
    rows = prep.build_continuation_rows(tmp_path, source_rows)
    path = prep.continuation_manifest_path(tmp_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rows, indent=2) + "\n")
    assert prep.write_frozen_json(path, rows, use_existing=True) is False

    changed = deepcopy(rows)
    changed[0]["sample_label"] = "wrong"
    with pytest.raises(ValueError, match="differs from the frozen content"):
        prep.write_frozen_json(path, changed, use_existing=True)


def test_analysis_manifest_includes_frozen_300k_baseline(tmp_path):
    source_rows = _write_source_sweep(tmp_path)
    rows = prep.build_continuation_rows(tmp_path, source_rows)
    analysis = prep.build_analysis_manifest(source_rows, rows)

    assert len(analysis) == 60
    baseline = [row for row in analysis if row["target_total_updates"] == 300_000]
    assert len(baseline) == 10
    assert {row["sample_label"] for row in baseline} == {
        "dpm50_fresh300k_v2"
    }
    assert all(row["source_sweep_name"] == prep.SOURCE_SWEEP_NAME for row in baseline)


def _write_complete_source_checkpoint(row: dict) -> Path:
    checkpoint = Path(row["source_checkpoint"])
    checkpoint.mkdir(parents=True, exist_ok=True)
    (checkpoint / "config.json").write_text(
        json.dumps(
            {
                "_class_name": "DiTTransformer2DModel",
                "sample_size": 128,
                "patch_size": 8,
                "num_layers": 16,
                "num_attention_heads": 12,
                "attention_head_dim": 64,
                "norm_num_groups": 32,
            }
        )
    )
    (checkpoint / "checkpoint_config.yaml").write_text(
        yaml.safe_dump(
            {
                "noise_scheduler": {"class": "diffusers.DDPMScheduler"},
                "optimizer": {"class": "torch.optim.AdamW"},
                "lr_scheduler": {
                    "class": "torch.optim.lr_scheduler.CosineAnnealingWarmRestarts"
                },
            }
        )
    )
    for group, alternatives in precheck.REQUIRED_STATE_GROUPS.items():
        name = alternatives[0]
        path = checkpoint / name
        if name.endswith("/"):
            path.mkdir(parents=True)
            (path / "state.pt").write_bytes(group.encode())
        else:
            if path.exists():
                continue
            path.write_bytes(group.encode())
    return checkpoint


def test_precheck_validates_complete_source_checkpoint(tmp_path):
    source_rows = _write_source_sweep(tmp_path)
    row = prep.build_continuation_rows(
        tmp_path,
        source_rows,
        checkpoint_root=tmp_path / "continuation",
    )[0]
    _write_complete_source_checkpoint(row)

    report = precheck.validate_source_row(row, tmp_path)
    assert report["dataset_tag"] == "d2p06"
    assert report["architecture"]["patch_size"] == 8
    assert set(report["resume_state"]) == set(precheck.REQUIRED_STATE_GROUPS)


@pytest.mark.parametrize("missing_group", sorted(precheck.REQUIRED_STATE_GROUPS))
def test_precheck_rejects_each_missing_resume_state(tmp_path, missing_group):
    source_rows = _write_source_sweep(tmp_path)
    row = prep.build_continuation_rows(
        tmp_path,
        source_rows,
        checkpoint_root=tmp_path / "continuation",
    )[0]
    checkpoint = _write_complete_source_checkpoint(row)
    for name in precheck.REQUIRED_STATE_GROUPS[missing_group]:
        path = checkpoint / name.rstrip("/")
        if path.is_dir():
            import shutil

            shutil.rmtree(path)
        elif path.exists():
            path.unlink()

    with pytest.raises((FileNotFoundError, ValueError), match=missing_group):
        precheck.validate_source_row(row, tmp_path)


def test_precheck_rejects_source_config_digest_change(tmp_path):
    source_rows = _write_source_sweep(tmp_path)
    row = prep.build_continuation_rows(
        tmp_path,
        source_rows,
        checkpoint_root=tmp_path / "continuation",
    )[0]
    _write_complete_source_checkpoint(row)
    source_path = tmp_path / row["source_config"]
    source_path.write_text(source_path.read_text() + "\n# changed\n")

    with pytest.raises(ValueError, match="digest"):
        precheck.validate_source_row(row, tmp_path)


@pytest.mark.parametrize(
    ("dotted_path", "value", "message"),
    [
        ("model.patch_size", 4, "patch_size"),
        ("data.augment", True, "augmentation"),
        ("data.constant_label", 1, "constant_label"),
    ],
)
def test_precheck_rejects_architecture_or_data_contract(
    tmp_path, dotted_path, value, message
):
    source_rows = _write_source_sweep(tmp_path)
    row = prep.build_continuation_rows(
        tmp_path,
        source_rows,
        checkpoint_root=tmp_path / "continuation",
    )[0]
    _write_complete_source_checkpoint(row)
    source_path = tmp_path / row["source_config"]
    config = yaml.safe_load(source_path.read_text())
    section, key = dotted_path.split(".")
    config[section][key] = value
    source_path.write_text(yaml.safe_dump(config, sort_keys=False))
    row["source_config_sha256"] = prep.sha256_file(source_path)

    with pytest.raises(ValueError, match=message):
        precheck.validate_source_row(row, tmp_path)


def _architecture_config(path: Path, depth: int, *, patch_size: int = 8) -> Path:
    config = _source_config(path.parent / f"l{depth}", num_epochs=10)
    config["model"]["num_layers"] = depth
    config["model"]["patch_size"] = patch_size
    path.write_text(yaml.safe_dump(config, sort_keys=False))
    return path


def test_architecture_comparison_allows_only_depth_and_output(tmp_path):
    configs = {
        "dit_l8": _architecture_config(tmp_path / "l8.yaml", 8),
        "dit_l12": _architecture_config(tmp_path / "l12.yaml", 12),
        "dit_l16": _architecture_config(tmp_path / "l16.yaml", 16),
    }
    report = precheck.compare_architecture_configs(configs)
    assert [report[name]["num_layers"] for name in configs] == [8, 12, 16]
    assert {report[name]["patch_size"] for name in configs} == {8}

    _architecture_config(configs["dit_l16"], 16, patch_size=4)
    with pytest.raises(ValueError, match="patch_size"):
        precheck.compare_architecture_configs(configs)


def test_slurm_precheck_requires_report_and_endpoint_model_loads():
    script = (
        ROOT
        / "scripts"
        / "slurm"
        / "precheck_nf_generalize_fig2_dit_l16_continue500k_v2.sbatch"
    )
    text = script.read_text()

    assert "#SBATCH --gres=gpu:1" in text
    assert "check_nf_generalize_fig2_dit_l16_continue500k_v2.py" in text
    assert "--runtime-check" in text
    assert "precheck_report.json" in text
    assert 'for DATASET_TAG in d2p06 d2p15' in text
    assert text.count("--preflight-only") == 1
    assert "test -s" in text


def test_training_array_has_exact_target_and_recovery_contract():
    script = (
        ROOT
        / "scripts"
        / "slurm"
        / "train_nf_generalize_fig2_dit_l16_continue500k_v2_array.sbatch"
    )
    text = script.read_text()

    assert "#SBATCH --time=48:00:00" in text
    assert "#SBATCH --array=0-9%2" in text
    assert "run_cosmodiff_train_with_dit_resume.py" in text
    assert "--minimum-checkpoint" in text
    assert "--target-checkpoint" in text
    assert "precheck_report.json" in text
    assert "completion" in text
    assert "nf_generalize_fig2_dit_l16_continue500k_v2" in text
    assert "nf_generalize_fig2_dit_l16_continue/" not in text


def test_submission_chain_is_five_stage_afterok_only():
    script = (
        ROOT
        / "scripts"
        / "slurm"
        / "submit_nf_generalize_fig2_dit_l16_continue500k_v2.sh"
    )
    text = script.read_text()

    assert "START_STAGE" in text
    assert "START_STAGE < 1 || START_STAGE > 5" in text
    assert "REUSE_EXISTING_MANIFEST" in text
    assert "--array=0-9%2" in text
    assert "for STAGE in $(seq \"${START_STAGE}\" 5)" in text
    assert "afterok" in text
    assert "afterany" not in text
    assert "previous_expected_checkpoint" in text
    assert "nf_generalize_fig2_dit_l16_continue/" not in text


def test_dpm_sampling_array_uses_fixed_auditable_protocol():
    script = (
        ROOT
        / "scripts"
        / "slurm"
        / "sample_nf_generalize_fig2_dit_l16_continue500k_v2_array.sbatch"
    )
    text = script.read_text()

    assert "#SBATCH --array=0-9%2" in text
    assert "DPMSolverMultistepScheduler" in text
    assert "SAMPLER_STEPS=50" in text
    assert "NUM_SAMPLES=512" in text
    assert "BATCH_SIZE=8" in text
    assert "SEED=123" in text
    assert "validate_nf_generalize_fig2_dit_sample.py" in text


def test_ddpm_controls_are_only_transition_and_high_data_endpoints():
    script = (
        ROOT
        / "scripts"
        / "slurm"
        / "sample_nf_generalize_fig2_dit_l16_continue500k_v2_ddpm_controls.sbatch"
    )
    text = script.read_text()

    assert "#SBATCH --array=0-3%2" in text
    assert "d2p08:0" in text
    assert "d2p08:5" in text
    assert "d2p11:0" in text
    assert "d2p11:5" in text
    assert "DDPMScheduler" in text
    assert "SAMPLER_STEPS=500" in text
    assert "NUM_SAMPLES=512" in text
    assert "BATCH_SIZE=8" in text
    assert "SEED=123" in text


def test_physics_analysis_is_exact_subset_and_fails_on_missing_samples():
    script = (
        ROOT
        / "scripts"
        / "analyze_nf_generalize_fig2_dit_l16_continue500k_v2_physics.py"
    )
    slurm = (
        ROOT
        / "scripts"
        / "slurm"
        / "analyze_nf_generalize_fig2_dit_l16_continue500k_v2_physics.sbatch"
    )

    source = script.read_text()
    wrapper = slurm.read_text()
    assert "iter_real_reference_batches_from_config" in source
    assert "batch_power_spectra" in source
    assert "selected_power_bin_statistics" in source
    assert "patch_boundary_statistics" in source
    assert "--baseline-manifest" in source
    assert "skip-missing" not in source
    assert "nf_generalize_fig2_dit_l16_continue500k_v2_physics_summary.csv" in source
    assert "nf_generalize_fig2_dit_l16_continue500k_v2_pk_selected_bins.csv" in source
    assert "nf_generalize_fig2_dit_l16_continue500k_v2_patch_boundaries.csv" in source
    assert "nf_generalize_fig2_dit_l16_continue500k_v2_curves.npz" in source

    assert "#SBATCH --partition=standard" in wrapper
    assert "#SBATCH --cpus-per-task=8" in wrapper
    assert "#SBATCH --mem=80gb" in wrapper
    assert "SAMPLE_LABEL" in wrapper
    assert "OUT_PREFIX" in wrapper


def _write_sample_file(path: Path, checkpoint: Path, *, value: float = 0.0) -> None:
    np.savez(
        path,
        samples=np.full((512, 1, 128, 128), value, dtype=np.float32),
        requested_checkpoint=np.asarray(str(checkpoint)),
        resolved_checkpoint=np.asarray(str(checkpoint)),
        scheduler=np.asarray("DPMSolverMultistepScheduler"),
        num_steps=np.asarray(50),
        seed=np.asarray(123),
        scheduler_class=np.asarray("DPMSolverMultistepScheduler"),
        requested_inference_steps=np.asarray(50),
        executed_inference_steps=np.asarray(50),
        first_timestep=np.asarray(999.0),
        final_timestep=np.asarray(0.0),
        terminal_sigma=np.asarray(0.0),
        terminal_sigma_is_zero=np.asarray(True),
        terminal_sigma_verifiable=np.asarray(True),
    )


def test_sample_guard_rejects_identical_outputs_from_different_checkpoints(tmp_path):
    first_checkpoint = tmp_path / "checkpoint-epoch-100"
    second_checkpoint = tmp_path / "checkpoint-epoch-200"
    first_checkpoint.mkdir()
    second_checkpoint.mkdir()
    first = tmp_path / "a_dpm50_cont_340k.npz"
    second = tmp_path / "b_dpm50_cont_340k.npz"
    _write_sample_file(first, first_checkpoint)
    _write_sample_file(second, second_checkpoint)

    sample_guard.validate_sample_file(
        first,
        requested_checkpoint=first_checkpoint,
        scheduler="DPMSolverMultistepScheduler",
        requested_steps=50,
    )
    with pytest.raises(ValueError, match="byte-identical"):
        sample_guard.reject_cross_checkpoint_duplicates(
            tmp_path, sample_label="dpm50_cont_340k"
        )
