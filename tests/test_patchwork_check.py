from __future__ import annotations

import hashlib
import importlib.util
import os
from pathlib import Path
import subprocess

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "patchwork_check.py"
SBATCH = ROOT / "scripts/slurm/evaluate_dit_patchwork_check.sbatch"
RUNTIME_REVISION = "555f350f82c913ff06150c96e66709719856cd83"


def load_module():
    spec = importlib.util.spec_from_file_location("patchwork_check", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_real_copies_and_synthetic_patchworks_separate():
    module = load_module()
    rng = np.random.default_rng(8)
    reference = rng.normal(size=(16, 1, 32, 32)).astype(np.float32)

    real_assign = module.patch_assignments(reference[:8], reference)
    real_majority, real_distinct, _ = module.summarize(real_assign)
    assert np.all(real_majority == 1.0)
    assert np.all(real_distinct == 1)

    patchwork = module.synthetic_patchworks(reference, count=64, rng=rng)
    patch_assign = module.patch_assignments(patchwork, reference)
    patch_majority, patch_distinct, _ = module.summarize(patch_assign)
    assert np.median(patch_majority) < 0.5
    assert np.median(patch_distinct) > 4


def test_corrupted_single_map_has_requested_whole_image_cosine():
    module = load_module()
    rng = np.random.default_rng(9)
    reference = rng.normal(size=(8, 1, 16, 16)).astype(np.float32)
    corrupted = module.corrupted_single_maps(reference, cosine=0.5, rng=rng)
    source = reference.reshape(len(reference), -1)
    result = corrupted.reshape(len(corrupted), -1)
    cosine = (source * result).sum(axis=1) / (
        np.linalg.norm(source, axis=1) * np.linalg.norm(result, axis=1)
    )
    np.testing.assert_allclose(cosine, 0.5, atol=1e-6)


def test_provenance_is_frozen_to_existing_control_samples():
    module = load_module()
    reference = np.ones((4, 1, 8, 8), dtype=np.float32)
    mean_hash = hashlib.sha256(
        reference.mean(axis=0, keepdims=True).tobytes()
    ).hexdigest()
    record = {
        "training_mean_sha256": np.asarray(mean_hash),
        "late_start_actual": np.asarray(499),
        "late_start_steps_run": np.asarray(50),
        "late_start_init": np.asarray("mean"),
        "seed": np.asarray(123),
        "num_steps": np.asarray(50),
        "ema_sigma_rel": np.asarray(-1.0),
        "scheduler_class": np.asarray("DPMSolverMultistepScheduler"),
    }
    module.validate_provenance(record, reference)
    changed = dict(record)
    changed["late_start_actual"] = np.asarray(399)
    with pytest.raises(ValueError, match="late_start_actual"):
        module.validate_provenance(changed, reference)


def test_great_lakes_patchwork_preflight_has_no_analysis_side_effect(tmp_path):
    runtime = tmp_path / "runtime"
    code = tmp_path / "code"
    project = tmp_path / "project"
    for path in (runtime, code, project):
        path.mkdir()
    result_dir = project / "results/dit_l16_late_start_screen"
    result_dir.mkdir(parents=True)
    for name in (
        "dit_l16_fresh300k_t499_n128.npz",
        "dit_l16_seed456_500k_t499_n128.npz",
        "dit_l8_200k_t499_n128.npz",
    ):
        (result_dir / name).write_bytes(b"fixture")

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
        "if [[ \"$*\" == *resolve_dit_high_noise_models.py* ]]; then\n"
        "  echo '{\"config\": \"fixture.yaml\"}'\n"
        "elif [[ \"$*\" == *patchwork_check.py* && \"$*\" == *--help* ]]; then\n"
        "  exit 0\n"
        "elif [[ \"$1\" == - ]]; then\n"
        "  sed -n 's/.*config.*fixture.yaml.*/fixture.yaml/p' >/dev/null\n"
        "  echo fixture.yaml\n"
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
        PREFLIGHT_ONLY="1",
    )
    result = subprocess.run(
        ["bash", str(SBATCH)], env=env, text=True, capture_output=True
    )
    assert result.returncode == 0, result.stderr
    assert "PATCHWORK PREFLIGHT PASSED; NO ANALYSIS" in result.stdout
    assert not (project / "results/patchwork").exists()
