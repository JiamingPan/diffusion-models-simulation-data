import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import warnings

import pytest
import torch
import yaml

from simdiff_eval.terminal_reports import start_report


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts/prepare_nf_generalize_fig2_dit_l16_seed_restart500k_configs.py"
CHECK_SCRIPT = REPO_ROOT / "scripts/check_nf_generalize_fig2_dit_l16_seed_restart500k.py"
SUBMIT_SCRIPT = REPO_ROOT / "scripts/slurm/submit_nf_generalize_fig2_dit_l16_seed_restart500k.sh"
PRECHECK_SCRIPT = REPO_ROOT / "scripts/slurm/precheck_nf_generalize_fig2_dit_l16_seed_restart500k.sbatch"
TRAIN_SCRIPT = REPO_ROOT / "scripts/slurm/train_nf_generalize_fig2_dit_l16_seed_restart500k_array.sbatch"
SOURCE_METADATA_SCRIPT = REPO_ROOT / "scripts/write_source_checkout_metadata.py"
PIN_BUILDER_SCRIPT = REPO_ROOT / "scripts/build_cosmodiff_seed_restart_pin.py"
PIN_VERIFY_SCRIPT = REPO_ROOT / "scripts/verify_cosmodiff_seed_restart_runtime.py"
PACKAGE_METADATA_PATCH = REPO_ROOT / "scripts/patch_cosmodiff_package_metadata.py"


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


def load_source_metadata_module():
    spec = importlib.util.spec_from_file_location(
        "source_checkout_metadata", SOURCE_METADATA_SCRIPT
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def load_path_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def make_pin_source_repo(
    tmp_path: Path, *, required_runtime_module: str | None = None
) -> tuple[Path, str]:
    source = tmp_path / "cosmodiff_source"
    package = source / "cosmodiff"
    package.mkdir(parents=True)
    runtime_import = (
        f"import {required_runtime_module}\n" if required_runtime_module else ""
    )
    (package / "__init__.py").write_text(
        runtime_import + '__version__ = "0+fixture"\n'
    )
    for name in ("optim", "utils", "augment", "transform"):
        (package / f"{name}.py").write_text(f'VALUE = "{name}"\n')
    (source / "scripts").mkdir()
    (source / "scripts/cosmodiff_train.py").write_text("print('fixture')\n")
    subprocess.run(["git", "init", "-q"], cwd=source, check=True)
    subprocess.run(["git", "config", "user.email", "fixture@example.invalid"], cwd=source, check=True)
    subprocess.run(["git", "config", "user.name", "Fixture"], cwd=source, check=True)
    subprocess.run(["git", "add", "."], cwd=source, check=True)
    subprocess.run(["git", "commit", "-qm", "fixture"], cwd=source, check=True)
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=source,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    return source, revision


def make_pin_patch_scripts(tmp_path: Path) -> list[Path]:
    patch_root = tmp_path / "patches"
    patch_root.mkdir()
    targets = {
        "patch_cosmodiff_package_metadata.py": "cosmodiff/__init__.py",
        "patch_cosmodiff_constant_label.py": "cosmodiff/utils.py",
        "patch_cosmodiff_dit_class_labels.py": "cosmodiff/optim.py",
        "patch_cosmodiff_checkpoint_state.py": "cosmodiff/optim.py",
    }
    paths = []
    for index, (name, target) in enumerate(targets.items(), start=1):
        path = patch_root / name
        path.write_text(
            "from pathlib import Path\n"
            "import sys\n"
            f"path = Path(sys.argv[1]) / {target!r}\n"
            f"path.write_text(path.read_text() + '# patch {index}\\n')\n"
        )
        paths.append(path)
    return paths


def make_pin_third_party_runtime(tmp_path: Path) -> Path:
    root = tmp_path / "pin_third_party"
    diffusers = root / "diffusers"
    hub = root / "huggingface_hub"
    diffusers.mkdir(parents=True)
    hub.mkdir()
    (diffusers / "__init__.py").write_text(
        "__version__ = '0.35.0.fixture'\n"
        "class DDPMScheduler: pass\n"
        "class DiTTransformer2DModel: pass\n"
    )
    (hub / "__init__.py").write_text(
        "__version__ = '0.25.0.fixture'\n"
        "def hf_hub_download(*args, **kwargs): return None\n"
        "def snapshot_download(*args, **kwargs): return None\n"
    )
    return root


def make_pin_code_root(tmp_path: Path) -> Path:
    code_root = tmp_path / "frozen_code"
    shutil.copytree(REPO_ROOT / "simdiff_eval", code_root / "simdiff_eval")
    scripts = code_root / "scripts"
    scripts.mkdir()
    shutil.copy2(
        REPO_ROOT / "scripts/check_cosmodiff_seed_restart_imports.py",
        scripts / "check_cosmodiff_seed_restart_imports.py",
    )
    return code_root


def test_immutable_pin_builder_records_ordered_patches_imports_and_inventory(tmp_path):
    builder = load_path_module(PIN_BUILDER_SCRIPT, "pin_builder")
    verifier = load_path_module(PIN_VERIFY_SCRIPT, "pin_verifier")
    source, revision = make_pin_source_repo(tmp_path)
    patches = make_pin_patch_scripts(tmp_path)
    third_party = make_pin_third_party_runtime(tmp_path)
    destination = tmp_path / "published_pin"

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        manifest = builder.build_pin(
            source_repo=source,
            base_revision=revision,
            destination=destination,
            python_bin=Path(sys.executable),
            patch_scripts=patches,
            code_root=REPO_ROOT,
            expected_torch_prefix=Path(sys.prefix),
            approved_residual_paths=(third_party,),
        )
    assert not caught

    assert destination.is_dir()
    assert manifest["base_revision"] == revision
    assert [row["name"] for row in manifest["patches"]] == [
        "patch_cosmodiff_package_metadata.py",
        "patch_cosmodiff_constant_label.py",
        "patch_cosmodiff_dit_class_labels.py",
        "patch_cosmodiff_checkpoint_state.py",
    ]
    assert [row["status"] for row in manifest["patches"]] == [
        "applied",
        "applied",
        "applied",
        "applied",
    ]
    assert all(row["script_sha256"] for row in manifest["patches"])
    assert all(row["targets"] for row in manifest["patches"])
    assert set(manifest["imports"]) == {
        "cosmodiff",
        "cosmodiff.optim",
        "cosmodiff.utils",
        "cosmodiff.augment",
        "cosmodiff.transform",
    }
    assert manifest["cosmodiff_version"] == "0+fixture"
    assert manifest["pin_schema_version"] == 2
    runtime = manifest["runtime_compatibility"]
    assert runtime["schema_version"] == 1
    assert runtime["runtime_root"] == "seed_restart_runtime"
    assert runtime["canonical_shim"]["path"] == "simdiff_eval/torch_compat.py"
    assert len(runtime["canonical_shim"]["sha256"]) == 64
    assert runtime["sitecustomize"]["path"] == (
        "seed_restart_runtime/sitecustomize.py"
    )
    assert len(runtime["sitecustomize"]["sha256"]) == 64
    assert runtime["sklearn_stub"]["files"]
    assert runtime["python_executable"] == str(Path(sys.executable).absolute())
    assert runtime["runtime_audit"]["sklearn"]["runtime_kind"] == (
        "simdiff-seed-restart-stub"
    )
    assert runtime["runtime_audit"]["numpy"]["file"]
    assert runtime["runtime_audit"]["diffusers"]["version"] == (
        "0.35.0.fixture"
    )
    assert runtime["numpy_compatibility"]["status"] == (
        "not_required_after_import_audit"
    )
    assert not list(destination.rglob("*.bak"))
    assert verifier.verify_pin(
        destination,
        destination / builder.PIN_MANIFEST_NAME,
        expected_base_revision=revision,
        python_bin=Path(sys.executable),
        expected_patch_scripts=patches,
        code_root=REPO_ROOT,
        expected_torch_prefix=Path(sys.prefix),
        approved_residual_paths=(third_party,),
        check_source_contract=False,
    )["base_revision"] == revision


def test_pin_builder_and_verifier_preserve_virtualenv_python_symlink(tmp_path):
    builder = load_path_module(PIN_BUILDER_SCRIPT, "pin_builder_venv")
    verifier = load_path_module(PIN_VERIFY_SCRIPT, "pin_verifier_venv")
    venv = tmp_path / "fixture_venv"
    subprocess.run(
        [sys.executable, "-m", "venv", "--system-site-packages", str(venv)],
        check=True,
    )
    python_bin = venv / "bin/python"
    site_packages = subprocess.run(
        [
            str(python_bin),
            "-c",
            "import site; print(site.getsitepackages()[0])",
        ],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    Path(site_packages, "pin_runtime_marker.py").write_text("VALUE = 1\n")

    source, revision = make_pin_source_repo(
        tmp_path, required_runtime_module="pin_runtime_marker"
    )
    patches = make_pin_patch_scripts(tmp_path)
    third_party = make_pin_third_party_runtime(tmp_path)
    destination = tmp_path / "published_pin"
    manifest = builder.build_pin(
        source_repo=source,
        base_revision=revision,
        destination=destination,
        python_bin=python_bin,
        patch_scripts=patches,
        code_root=REPO_ROOT,
        expected_torch_prefix=Path(sys.prefix),
        approved_residual_paths=(third_party,),
    )

    assert manifest["python_executable"] == str(python_bin.absolute())
    assert verifier.verify_pin(
        destination,
        destination / builder.PIN_MANIFEST_NAME,
        expected_base_revision=revision,
        python_bin=python_bin,
        expected_patch_scripts=patches,
        code_root=REPO_ROOT,
        expected_torch_prefix=Path(sys.prefix),
        approved_residual_paths=(third_party,),
        check_source_contract=False,
    )["base_revision"] == revision


def test_immutable_pin_verifier_rejects_modified_or_extra_files(tmp_path):
    builder = load_path_module(PIN_BUILDER_SCRIPT, "pin_builder_tamper")
    verifier = load_path_module(PIN_VERIFY_SCRIPT, "pin_verifier_tamper")
    source, revision = make_pin_source_repo(tmp_path)
    patches = make_pin_patch_scripts(tmp_path)
    third_party = make_pin_third_party_runtime(tmp_path)
    destination = tmp_path / "published_pin"
    builder.build_pin(
        source_repo=source,
        base_revision=revision,
        destination=destination,
        python_bin=Path(sys.executable),
        patch_scripts=patches,
        code_root=REPO_ROOT,
        expected_torch_prefix=Path(sys.prefix),
        approved_residual_paths=(third_party,),
    )
    manifest_path = destination / builder.PIN_MANIFEST_NAME

    (destination / "cosmodiff/optim.py").write_text("tampered\n")
    with pytest.raises(RuntimeError, match="inventory"):
        verifier.verify_pin(
            destination,
            manifest_path,
            expected_base_revision=revision,
            python_bin=Path(sys.executable),
            expected_patch_scripts=patches,
            code_root=REPO_ROOT,
            expected_torch_prefix=Path(sys.prefix),
            approved_residual_paths=(third_party,),
            check_source_contract=False,
        )

    (destination / "cosmodiff/optim.py").write_text(
        'VALUE = "optim"\n# patch 3\n# patch 4\n'
    )
    (destination / "unexpected.txt").write_text("not declared\n")
    with pytest.raises(RuntimeError, match="inventory"):
        verifier.verify_pin(
            destination,
            manifest_path,
            expected_base_revision=revision,
            python_bin=Path(sys.executable),
            expected_patch_scripts=patches,
            code_root=REPO_ROOT,
            expected_torch_prefix=Path(sys.prefix),
            approved_residual_paths=(third_party,),
            check_source_contract=False,
        )


def test_immutable_pin_verifier_rejects_changed_canonical_shim(tmp_path):
    builder = load_path_module(PIN_BUILDER_SCRIPT, "pin_builder_shim_tamper")
    verifier = load_path_module(PIN_VERIFY_SCRIPT, "pin_verifier_shim_tamper")
    source, revision = make_pin_source_repo(tmp_path)
    patches = make_pin_patch_scripts(tmp_path)
    third_party = make_pin_third_party_runtime(tmp_path)
    code_root = make_pin_code_root(tmp_path)
    destination = tmp_path / "published_pin"
    builder.build_pin(
        source_repo=source,
        base_revision=revision,
        destination=destination,
        python_bin=Path(sys.executable),
        patch_scripts=patches,
        code_root=code_root,
        expected_torch_prefix=Path(sys.prefix),
        approved_residual_paths=(third_party,),
    )
    canonical = code_root / "simdiff_eval/torch_compat.py"
    canonical.write_text(canonical.read_text() + "\n# tampered after publication\n")

    with pytest.raises(RuntimeError, match="canonical Torch compatibility hash"):
        verifier.verify_pin(
            destination,
            destination / builder.PIN_MANIFEST_NAME,
            expected_base_revision=revision,
            python_bin=Path(sys.executable),
            expected_patch_scripts=patches,
            code_root=code_root,
            expected_torch_prefix=Path(sys.prefix),
            approved_residual_paths=(third_party,),
            check_source_contract=False,
        )

def test_package_metadata_patch_accepts_source_version_module_layout(tmp_path):
    module = load_path_module(PACKAGE_METADATA_PATCH, "package_metadata_modern")
    init_path = tmp_path / "__init__.py"
    original = (
        "from . import utils\n"
        "from . import augment\n"
        "from . import optim\n"
        "from .version import __version__\n"
    )
    init_path.write_text(original)

    changed = module.patch_init(init_path)

    assert changed is False
    assert init_path.read_text() == original


def test_pin_import_failure_reports_the_missing_runtime_dependency(tmp_path):
    builder = load_path_module(PIN_BUILDER_SCRIPT, "pin_builder_import_error")
    package = tmp_path / "cosmodiff"
    package.mkdir()
    (package / "__init__.py").write_text("import missing_pin_dependency\n")
    runtime_root = tmp_path / "seed_restart_runtime"
    third_party = make_pin_third_party_runtime(tmp_path)
    from simdiff_eval.seed_restart_runtime import write_runtime_assets

    write_runtime_assets(
        runtime_root,
        code_root=REPO_ROOT,
        entry_point="tests.pin_import_failure",
    )

    with pytest.raises(RuntimeError, match="missing_pin_dependency"):
        builder.imported_modules(
            Path(sys.executable),
            tmp_path,
            runtime_root=runtime_root,
            code_root=REPO_ROOT,
            expected_torch_prefix=Path(sys.prefix),
            approved_residual_paths=(third_party,),
        )


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
                "gradient_accumulation_steps": 4,
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
        ema_step = actual * 4 - 1_000
        for profile_index in (0, 1):
            torch.save(
                {
                    "step": torch.tensor(ema_step, dtype=torch.float16),
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
    assert {row["microbatches_per_optimizer_step"] for row in rows} == {4}
    first = {row["dataset_tag"]: row for row in rows if row["continue_stage"] == 1}
    assert first["d2p08"]["source_ema_step"] == 1_199_000
    assert first["d2p08"]["previous_expected_ema_step"] == 1_199_000
    assert first["d2p08"]["expected_ema_step"] == 1_359_000
    assert first["d2p08"]["first_resumed_optimizer_step"] == 300_001
    assert first["d2p08"]["first_resumed_microbatch_step"] == 1_200_001
    assert first["d2p10"]["source_ema_step"] == 1_199_128
    assert first["d2p10"]["expected_ema_step"] == 1_359_384
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
    assert report["source_microbatches"] == 1_200_000
    assert report["expected_ema_step"] == 1_199_000
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


def test_seed_restart_checker_creates_only_an_incomplete_terminal_report(tmp_path):
    prep = load_module()
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
                    prep._project_path(tmp_path, row["source_config"]), row
                ),
                sort_keys=False,
            )
        )
    prep.seed_restart_directories(rows)
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps(rows, indent=2) + "\n")
    report = tmp_path / "precheck.json"
    env = dict(os.environ)
    env["PYTHONPATH"] = str(REPO_ROOT)
    env["SLURM_JOB_ID"] = "777"

    subprocess.run(
        [
            sys.executable,
            str(CHECK_SCRIPT),
            "--project-dir",
            str(tmp_path),
            "--manifest",
            str(manifest),
            "--stage",
            "1",
            "--report",
            str(report),
        ],
        check=True,
        env=env,
        capture_output=True,
        text=True,
    )

    payload = json.loads(report.read_text())
    assert payload["status"] == "INCOMPLETE"
    assert payload["producer_job_id"] == "777"
    assert payload["producer_exit_code"] is None
    assert payload["finalized_at_utc"] is None
    assert payload["report_schema_version"] == 1


def test_seed_restart_checker_enriches_the_prestarted_job_report(tmp_path):
    prep = load_module()
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
                    prep._project_path(tmp_path, row["source_config"]), row
                ),
                sort_keys=False,
            )
        )
    prep.seed_restart_directories(rows)
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps(rows, indent=2) + "\n")
    report = tmp_path / "precheck.json"
    initial = start_report(report, payload={"stage": 1}, producer_job_id="778")
    env = dict(os.environ)
    env["PYTHONPATH"] = str(REPO_ROOT)
    env["SLURM_JOB_ID"] = "778"

    subprocess.run(
        [
            sys.executable,
            str(CHECK_SCRIPT),
            "--project-dir",
            str(tmp_path),
            "--manifest",
            str(manifest),
            "--stage",
            "1",
            "--report",
            str(report),
        ],
        check=True,
        env=env,
        capture_output=True,
        text=True,
    )

    enriched = json.loads(report.read_text())
    assert enriched["started_at_utc"] == initial["started_at_utc"]
    assert enriched["status"] == "INCOMPLETE"
    assert len(enriched["rows"]) == 2


def test_seed_restart_precheck_starts_report_immediately_after_pin_verification():
    source = PRECHECK_SCRIPT.read_text()
    verify_index = source.index("verify_cosmodiff_seed_restart_runtime.py")
    start_index = source.index('"${FINALIZER}" start')
    prepare_index = source.index('"${PYTHON_BIN}" "${PREPARE}"')
    check_index = source.index('"${PYTHON_BIN}" "${CHECK}"')
    assert verify_index < start_index < prepare_index < check_index


def test_seed_restart_wrappers_consume_verified_pin_and_exact_precheck_job():
    precheck = PRECHECK_SCRIPT.read_text()
    train = TRAIN_SCRIPT.read_text()
    submit = SUBMIT_SCRIPT.read_text()

    for source in (precheck, train, submit):
        assert "COSMODIFF_PIN_ROOT" in source
        assert "COSMODIFF_PIN_MANIFEST" in source
    assert "finalize_terminal_report.py" in precheck
    assert "--status FAILED" in precheck
    assert "--status PASS" in precheck
    assert "EXPECTED_PRECHECK_JOB_ID" in train
    assert "require-pass" in train
    assert "--expected-job-id" in train
    assert "COSMODIFF_DIR_OVERRIDE" not in submit


def test_seed_restart_gpu_jobs_stub_optional_sklearn_before_diffusers_imports():
    """Great Lakes' system sklearn must not enter the DiT resume import path."""
    for path in (PRECHECK_SCRIPT, TRAIN_SCRIPT):
        source = path.read_text()
        assert "STUB_ROOT=" in source
        assert 'mkdir -p "${STUB_ROOT}/sklearn/metrics"' in source
        assert 'write_diffusers_runtime_sitecustomize.py' in source
        assert "sklearn.metrics.roc_curve is stubbed for DiT seed restart" in source
        assert 'export PYTHONPATH="${STUB_ROOT}:${COSMODIFF_PIN_ROOT}:${CODE_ROOT}' in source

        stub_index = source.index('mkdir -p "${STUB_ROOT}/sklearn/metrics"')
        load_index = (
            source.index("check_nf_generalize_fig2_dit_resume.py")
            if path == PRECHECK_SCRIPT
            else source.index("run_cosmodiff_train_with_dit_resume.py")
        )
        assert stub_index < load_index


def test_slurm_jobs_pin_the_exact_audited_cosmodiff_runtime():
    submit = SUBMIT_SCRIPT.read_text()
    for path in (PRECHECK_SCRIPT, TRAIN_SCRIPT):
        source = path.read_text()
        assert "EXPECTED_COSMODIFF_BASE_REVISION" in source
        assert "COSMODIFF_PIN_ROOT" in source
        assert "COSMODIFF_PIN_MANIFEST" in source
        assert "verify_cosmodiff_seed_restart_runtime.py" in source
        assert 'git -C "${COSMODIFF_DIR}" rev-parse HEAD' not in source
    assert "EXPECTED_COSMODIFF_BASE_REVISION" in submit
    assert "COSMODIFF_PIN_ROOT" in submit
    assert "COSMODIFF_PIN_MANIFEST" in submit
    train = TRAIN_SCRIPT.read_text()
    assert "rev-parse --abbrev-ref" not in train
    assert "build_cosmodiff_seed_restart_pin.py" not in train
    assert "verify_cosmodiff_seed_restart_runtime.py" in PRECHECK_SCRIPT.read_text()
    assert "--resume-ema-step" in train
    assert "--target-ema-step" in train
    assert "--upgrade-existing-manifest" in submit


def test_source_metadata_writer_remains_available_but_jobs_use_the_audited_pin(tmp_path):
    module = load_source_metadata_module()
    metadata_root = tmp_path / "python_stubs"
    dist_info = module.write_distribution_metadata(
        metadata_root,
        distribution="cosmodiff",
        version="0+source.58c77eb",
    )

    env = dict(os.environ)
    env["PYTHONPATH"] = str(metadata_root)
    completed = subprocess.run(
        [
            sys.executable,
            "-S",
            "-c",
            (
                "from importlib.metadata import version; "
                "print(version('cosmodiff'))"
            ),
        ],
        check=True,
        capture_output=True,
        text=True,
        env=env,
    )

    assert completed.stdout.strip() == "0+source.58c77eb"
    assert dist_info == metadata_root / "cosmodiff-0+source.58c77eb.dist-info"
    source_root = tmp_path / "source_checkout"
    package = source_root / "cosmodiff"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text(
        "from importlib.metadata import version\n"
        "__version__ = version('cosmodiff')\n"
    )
    env["PYTHONPATH"] = os.pathsep.join((str(metadata_root), str(source_root)))
    imported = subprocess.run(
        [
            sys.executable,
            "-S",
            "-c",
            "import cosmodiff; print(cosmodiff.__version__)",
        ],
        check=True,
        capture_output=True,
        text=True,
        env=env,
    )
    assert imported.stdout.strip() == "0+source.58c77eb"

    for script in (PRECHECK_SCRIPT, TRAIN_SCRIPT):
        source = script.read_text()
        assert "write_source_checkout_metadata.py" not in source
        assert "build_cosmodiff_seed_restart_pin.py" not in source
        assert "cosmodiff-0+source" not in source
        assert "COSMODIFF_PIN_ROOT" in source


def test_manifest_upgrade_reuses_only_untouched_seed_directories(tmp_path):
    module = load_module()
    proposed = module.build_seed_restart_rows(
        tmp_path,
        source_rows(tmp_path),
        checkpoint_root=tmp_path / "new_checkpoints",
    )
    old = []
    new_fields = {
        "microbatches_per_optimizer_step",
        "source_total_microbatches",
        "source_ema_step",
        "previous_actual_total_updates",
        "previous_total_microbatches",
        "previous_expected_ema_step",
        "actual_total_microbatches",
        "expected_ema_step",
        "first_resumed_optimizer_step",
        "first_resumed_microbatch_step",
    }
    for row in proposed:
        legacy = {key: value for key, value in row.items() if key not in new_fields}
        legacy["manifest_version"] = 1
        old.append(legacy)

    module.assert_safe_manifest_upgrade(old, proposed)

    Path(proposed[0]["expected_checkpoint"]).mkdir(parents=True)
    with pytest.raises(ValueError, match="training target already exists"):
        module.assert_safe_manifest_upgrade(old, proposed)
