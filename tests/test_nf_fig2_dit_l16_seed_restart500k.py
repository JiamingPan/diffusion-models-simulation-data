import importlib.util
import json
from pathlib import Path

import pytest
import torch
import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts/prepare_nf_generalize_fig2_dit_l16_seed_restart500k_configs.py"
CHECK_SCRIPT = REPO_ROOT / "scripts/check_nf_generalize_fig2_dit_l16_seed_restart500k.py"
SUBMIT_SCRIPT = REPO_ROOT / "scripts/slurm/submit_nf_generalize_fig2_dit_l16_seed_restart500k.sh"
PRECHECK_SCRIPT = REPO_ROOT / "scripts/slurm/precheck_nf_generalize_fig2_dit_l16_seed_restart500k.sbatch"
TRAIN_SCRIPT = REPO_ROOT / "scripts/slurm/train_nf_generalize_fig2_dit_l16_seed_restart500k_array.sbatch"


def load_module():
    spec = importlib.util.spec_from_file_location("seed_restart_prep", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def load_check_module():
    spec = importlib.util.spec_from_file_location("seed_restart_check", CHECK_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def source_rows(project_dir: Path):
    rows = []
    for tag, size, steps, epoch, actual in (
        ("d2p08", 256, 32, 9_374, 300_000),
        ("d2p10", 1_024, 128, 2_343, 300_032),
    ):
        run_name = f"source_{tag}"
        config = {
            "io": {"output_dir": f"/source/{tag}"},
            "data": {
                "img_path": ["a.npy"],
                "n_samples": [size // 16],
                "seed": None,
                "reshape": "2d",
                "zthin": 8,
            },
            "model": {"class": "DiTTransformer2DModel"},
            "train": {
                "num_epochs": epoch + 1,
                "checkpoint_every_n_epochs": 10,
                "ema_sigma_rels": [0.02, 0.10],
                "ema_update_every": 1,
                "ema_burn_in": 1_000,
            },
        }
        config_path = project_dir / f"{run_name}.yaml"
        config_path.write_text(yaml.safe_dump(config, sort_keys=False))
        checkpoint = project_dir / f"{run_name}_checkpoint-epoch-{epoch:04d}"
        checkpoint.mkdir()
        (checkpoint / "weights.bin").write_bytes(f"weights-{tag}".encode())
        for name in (
            "diffusion_pytorch_model.safetensors",
            "config.json",
            "checkpoint_config.yaml",
            "optimizer.pkl",
            "lr_scheduler.pkl",
            "noise_scheduler.pkl",
            "scaler.pt",
            "random_states_0.pkl",
        ):
            (checkpoint / name).write_bytes(f"{name}-{tag}".encode())
        (checkpoint / "config.json").write_text(
            json.dumps({"_class_name": "DiTTransformer2DModel"})
        )
        (checkpoint / "checkpoint_config.yaml").write_text(
            yaml.safe_dump(
                {
                    "ema_sigma_rels": [0.02, 0.10],
                    "ema_burn_in": 1_000,
                    "optimizer": {"class": "torch.optim.AdamW"},
                    "lr_scheduler": {
                        "class": "torch.optim.lr_scheduler.CosineAnnealingWarmRestarts"
                    },
                    "noise_scheduler": {"class": "diffusers.DDPMScheduler"},
                }
            )
        )
        ema_dir = checkpoint / "ema"
        ema_dir.mkdir()
        ema_step = actual - 1_000
        for profile_index in (0, 1):
            torch.save(
                {
                    "step": torch.tensor(ema_step),
                    "initted": torch.tensor(True),
                    "ema_model.weight": torch.tensor([float(profile_index)]),
                },
                ema_dir / f"{profile_index}.{ema_step}.pt",
            )
        rows.append(
            {
                "sweep_name": "nf_generalize_fig2_dit_l16_fresh300k_v2",
                "arch": "dit_l16",
                "dataset_tag": tag,
                "dataset_size": size,
                "run_name": run_name,
                "config": str(config_path),
                "expected_checkpoint": str(checkpoint),
                "expected_final_epoch": epoch,
                "optimizer_steps_per_epoch": steps,
                "actual_total_updates": actual,
                "target_total_updates": 300_000,
            }
        )
    return rows


def test_seed_restart_rows_cover_only_two_runs_and_reach_500k(tmp_path):
    module = load_module()
    rows = module.build_seed_restart_rows(
        tmp_path,
        source_rows(tmp_path),
        checkpoint_root=tmp_path / "new_checkpoints",
    )

    assert len(rows) == 10
    assert {row["dataset_tag"] for row in rows} == {"d2p08", "d2p10"}
    assert {row["resume_seed"] for row in rows} == {456}
    assert {row["target_total_updates"] for row in rows} == {
        340_000,
        380_000,
        420_000,
        460_000,
        500_000,
    }
    assert all(
        row["apply_resume_seed"] is (row["continue_stage"] == 1)
        for row in rows
    )
    final = {row["dataset_tag"]: row for row in rows if row["continue_stage"] == 5}
    assert final["d2p08"]["expected_final_epoch"] == 15_624
    assert final["d2p08"]["actual_total_updates"] == 500_000
    assert final["d2p10"]["expected_final_epoch"] == 3_906
    assert final["d2p10"]["actual_total_updates"] == 500_096
    assert "resume456" in final["d2p08"]["run_name"]
    assert "resume456" in final["d2p10"]["run_name"]


def test_seed_restart_config_changes_only_runtime_length_and_output(tmp_path):
    module = load_module()
    sources = source_rows(tmp_path)
    row = module.build_seed_restart_rows(
        tmp_path,
        sources,
        checkpoint_root=tmp_path / "new_checkpoints",
    )[0]
    source_config = Path(sources[0]["config"])
    destination = tmp_path / "continuation.yaml"
    destination.write_text(
        yaml.safe_dump(
            module.build_seed_restart_config(source_config, row),
            sort_keys=False,
        )
    )

    module.assert_seed_restart_config(source_config, destination, row)
    source = yaml.safe_load(source_config.read_text())
    continuation = yaml.safe_load(destination.read_text())
    assert continuation["data"] == source["data"]
    assert continuation["model"] == source["model"]
    assert continuation["train"]["ema_sigma_rels"] == [0.02, 0.10]
    assert continuation["train"]["ema_burn_in"] == 1_000
    assert continuation["io"]["output_dir"] == row["checkpoint_dir"]


def test_seed_restart_checkpoint_copy_is_byte_identical_and_refuses_conflict(tmp_path):
    module = load_module()
    rows = module.build_seed_restart_rows(
        tmp_path,
        source_rows(tmp_path),
        checkpoint_root=tmp_path / "new_checkpoints",
    )

    module.seed_restart_directories(rows)
    first = next(row for row in rows if row["continue_stage"] == 1)
    assert module.checkpoint_inventory(Path(first["source_checkpoint"])) == module.checkpoint_inventory(
        Path(first["seed_checkpoint"])
    )

    (Path(first["seed_checkpoint"]) / "weights.bin").write_bytes(b"changed")
    with pytest.raises(ValueError, match="not byte-identical"):
        module.seed_restart_directories(rows)


def test_seed_restart_preflight_requires_complete_state_and_exact_ema_step(tmp_path):
    prep = load_module()
    check = load_check_module()
    rows = prep.build_seed_restart_rows(
        tmp_path,
        source_rows(tmp_path),
        checkpoint_root=tmp_path / "new_checkpoints",
    )
    for row in rows:
        config_path = prep._project_path(tmp_path, row["config"])
        config_path.parent.mkdir(parents=True, exist_ok=True)
        config_path.write_text(
            yaml.safe_dump(
                prep.build_seed_restart_config(
                    prep._project_path(tmp_path, row["source_config"]),
                    row,
                ),
                sort_keys=False,
            )
        )
    prep.seed_restart_directories(rows)

    report = check.validate_seed_restart_row(rows[0], tmp_path)
    assert report["dataset_tag"] == "d2p08"
    assert report["source_updates"] == 300_000
    assert report["expected_ema_step"] == 299_000
    assert len(report["ema_snapshots"]) == 2
    assert report["seed_checkpoint_byte_identical"] is True

    seed_checkpoint = Path(rows[0]["seed_checkpoint"])
    checkpoint_config = yaml.safe_load(
        (seed_checkpoint / "checkpoint_config.yaml").read_text()
    )
    checkpoint_config["ema_sigma_rels"] = [0.03, 0.10]
    (seed_checkpoint / "checkpoint_config.yaml").write_text(
        yaml.safe_dump(checkpoint_config)
    )
    with pytest.raises(ValueError, match="EMA sigma profiles"):
        check.validate_seed_restart_row(rows[0], tmp_path)

    checkpoint_config["ema_sigma_rels"] = [0.02, 0.10]
    (seed_checkpoint / "checkpoint_config.yaml").write_text(
        yaml.safe_dump(checkpoint_config)
    )
    (seed_checkpoint / "scaler.pt").unlink()
    with pytest.raises(FileNotFoundError, match="scaler"):
        check.validate_seed_restart_row(rows[0], tmp_path)


def test_slurm_jobs_pin_the_exact_external_cosmodiff_revision():
    submit = SUBMIT_SCRIPT.read_text()
    for path in (PRECHECK_SCRIPT, TRAIN_SCRIPT):
        source = path.read_text()
        assert "EXPECTED_COSMODIFF_COMMIT" in source
        assert 'git -C "${COSMODIFF_DIR}" rev-parse HEAD' in source
    assert "EXPECTED_COSMODIFF_COMMIT" in submit
    assert "EXPECTED_COSMODIFF_COMMIT=${EXPECTED_COSMODIFF_COMMIT}" in submit
    train = TRAIN_SCRIPT.read_text()
    assert "rev-parse --abbrev-ref" not in train
    assert "patch_cosmodiff_constant_label.py" not in train
    assert "patch_cosmodiff_dit_class_labels.py" not in train
    assert "verify_cosmodiff_seed_restart_runtime.py" in PRECHECK_SCRIPT.read_text()
