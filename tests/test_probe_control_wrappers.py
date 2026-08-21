from __future__ import annotations

import os
import subprocess
import sys
from argparse import Namespace
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
for path in (ROOT, SCRIPTS):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))


def test_path_sanitizer_removes_only_configured_incompatible_paths():
    from probe_controls_runtime import sanitize_sys_path

    paths = [
        "/job/runtime",
        "/home/jiamingp/venvs/cosmodiff_nf/lib/python3.10/site-packages",
        "/home/jiamingp/venvs/cosmodiff_nf_class/lib/python3.10/site-packages",
        "/home/jiamingp/venvs/cosmodiff_nf/lib/python3.10/dist-packages-extra",
    ]

    assert sanitize_sys_path(
        paths,
        incompatible_paths=(
            "/home/jiamingp/venvs/cosmodiff_nf/lib/python3.10/site-packages",
        ),
    ) == [
        "/job/runtime",
        "/home/jiamingp/venvs/cosmodiff_nf_class/lib/python3.10/site-packages",
        "/home/jiamingp/venvs/cosmodiff_nf/lib/python3.10/dist-packages-extra",
    ]


def test_preflight_wrapper_has_frozen_roots_and_required_artifacts():
    text = (SCRIPTS / "preflight_probe_controls.sh").read_text()

    for required in (
        "EXPECTED_COMMIT=${EXPECTED_COMMIT:-}",
        "/scratch/huterer_root/huterer0/jiamingp/probe_controls_code_456f01a",
        "/home/jiamingp/diffusion_models_repo",
        "vgg_mlp_encoder.npz",
        "vgg_mlp_encoder.pkl",
        "local/nf_conditional_bias_probe/manifest.json",
        "nf_cond_bias_hi_u128_d2p07_n128_200k_seed123_dpm50_heldout_k64.npz",
        "nf_cond_bias_hi_u128_d2p14_n16384_200k_seed123_dpm50_heldout_k64.npz",
        "vgg16-397923af.pth",
        "evaluate_probe_transform_controls.py",
        "evaluate_probe_degradation_control.py",
    ):
        assert required in text


def test_preflight_requires_explicit_expected_commit(tmp_path):
    env = os.environ.copy()
    env.pop("EXPECTED_COMMIT", None)
    env.update(
        CODE_ROOT=str(tmp_path / "missing-code-root"),
        PROJECT_DIR=str(tmp_path),
        PYTHON_BIN=sys.executable,
    )
    result = subprocess.run(
        ["bash", str(SCRIPTS / "preflight_probe_controls.sh")],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode != 0
    assert "EXPECTED_COMMIT" in result.stderr


def test_preflight_refuses_code_root_commit_mismatch(tmp_path):
    env = os.environ.copy()
    env.update(
        CODE_ROOT=str(ROOT),
        PROJECT_DIR=str(tmp_path),
        PYTHON_BIN=sys.executable,
        EXPECTED_COMMIT="not-the-current-commit",
        TORCH_HOME=str(tmp_path / "torch"),
    )
    result = subprocess.run(
        ["bash", str(SCRIPTS / "preflight_probe_controls.sh")],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode != 0
    assert "Expected CODE_ROOT commit not-the-current-commit" in result.stderr


def test_mismatched_checkout_cannot_execute_runtime_helper(tmp_path):
    code_root = tmp_path / "mismatched-checkout"
    scripts_root = code_root / "scripts"
    scripts_root.mkdir(parents=True)
    marker = tmp_path / "sourced-marker"
    (scripts_root / "probe_controls_runtime.sh").write_text(
        "#!/bin/bash\n"
        f"touch '{marker}'\n"
        "probe_controls_prepare_runtime() { :; }\n"
    )
    subprocess.run(["git", "init", "-q", str(code_root)], check=True)
    subprocess.run(["git", "-C", str(code_root), "config", "user.email", "test@example.invalid"], check=True)
    subprocess.run(["git", "-C", str(code_root), "config", "user.name", "Test"], check=True)
    subprocess.run(["git", "-C", str(code_root), "add", "scripts/probe_controls_runtime.sh"], check=True)
    subprocess.run(["git", "-C", str(code_root), "commit", "-qm", "runtime helper"], check=True)

    env = os.environ.copy()
    env.update(
        CODE_ROOT=str(code_root),
        PROJECT_DIR=str(tmp_path),
        PYTHON_BIN=sys.executable,
        EXPECTED_COMMIT="not-the-current-commit",
        TORCH_HOME=str(tmp_path / "torch"),
    )
    result = subprocess.run(
        ["bash", str(SCRIPTS / "preflight_probe_controls.sh")],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode != 0
    assert not marker.exists()


def test_generated_sitecustomize_filters_paths_on_fresh_python(tmp_path):
    from probe_controls_runtime import write_sitecustomize

    runtime = tmp_path / "runtime"
    bad = str(tmp_path / "incompatible-site-packages")
    neighbor = str(tmp_path / "neighbor-site-packages")
    write_sitecustomize(runtime / "sitecustomize.py", (bad,))
    env = os.environ.copy()
    env.update(
        PYTHONPATH=os.pathsep.join((str(runtime), bad, neighbor)),
        PROBE_CONTROLS_INCOMPATIBLE_PATHS=bad,
    )
    result = subprocess.run(
        [sys.executable, "-c", "import sys; print('\\n'.join(sys.path))"],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert bad not in result.stdout.splitlines()
    assert neighbor in result.stdout.splitlines()


def test_transform_manifest_uses_source_project_dir(monkeypatch, tmp_path):
    import evaluate_nf_conditional_bias_probe as probe_script
    import evaluate_probe_transform_controls as controls
    from simdiff_eval import probe_eval
    import pandas as pd
    import numpy as np

    artifact_root = tmp_path / "artifacts"
    source_root = tmp_path / "source"
    artifact_root.mkdir()
    source_root.mkdir()
    encoder_path = artifact_root / "encoder.npz"
    np.savez(
        encoder_path,
        normalization=np.array({}, dtype=object),
        heldout_indices=np.arange(900, 932, dtype=np.int64),
    )
    args = Namespace(
        project_dir=str(artifact_root),
        source_project_dir=str(source_root),
        data_root=str(tmp_path / "data"),
        encoder=str(encoder_path),
        device="cpu",
        embedding_batch_size=2,
        output_dir=tmp_path / "out-transform",
        bootstrap=1,
        bootstrap_seed=123,
        roll_seed=123,
        control=None,
        k_cut=None,
    )
    captured = {}

    class FakeSpec:
        def manifest_record(self):
            return {"name": "identity"}

    class FakeEncoder:
        model_path = source_root / "head.pkl"

    monkeypatch.setattr(controls, "parse_args", lambda: args)
    monkeypatch.setattr(
        probe_eval,
        "load_heldout_real_slices",
        lambda *unused, **kwargs: (
            np.zeros((32, 1, 2, 2), dtype=np.float32),
            np.zeros((32, 6), dtype=np.float32),
            np.arange(900, 932, dtype=np.int64),
            np.zeros(32, dtype=np.int64),
        ),
    )
    monkeypatch.setattr(probe_script, "load_vgg_encoder", lambda *unused: FakeEncoder())
    monkeypatch.setattr(controls, "build_requested_specs", lambda *unused, **kwargs: ([FakeSpec()], []))
    monkeypatch.setattr(controls, "evaluate_transform_specs", lambda *unused, **kwargs: pd.DataFrame())
    monkeypatch.setattr(controls, "aggregate_prediction_table", lambda *unused, **kwargs: {})
    monkeypatch.setattr(
        controls,
        "build_run_manifest",
        lambda **kwargs: captured.update(kwargs) or {},
    )

    controls.main()

    assert captured["project_dir"] == source_root.resolve()


def test_degradation_manifest_uses_source_project_dir(monkeypatch, tmp_path):
    import evaluate_nf_conditional_bias_probe as probe_script
    import evaluate_probe_degradation_control as controls
    from simdiff_eval import probe_eval
    import numpy as np
    import pandas as pd

    artifact_root = tmp_path / "artifacts"
    source_root = tmp_path / "source"
    artifact_root.mkdir()
    source_root.mkdir()
    encoder_path = artifact_root / "encoder.npz"
    heldout = np.arange(900, 932, dtype=np.int64)
    np.savez(
        encoder_path,
        normalization=np.array({}, dtype=object),
        heldout_indices=heldout,
    )
    sample_path = artifact_root / "sample.npz"
    np.savez(
        sample_path,
        samples=np.zeros((32, 1, 2, 2), dtype=np.float32),
        theta_raw=np.zeros((32, 6), dtype=np.float32),
        heldout_indices=heldout,
        samples_per_cosmology=1,
    )
    args = Namespace(
        project_dir=str(artifact_root),
        source_project_dir=str(source_root),
        manifest=None,
        run_name=None,
        data_root=str(tmp_path / "data"),
        encoder=str(encoder_path),
        device="cpu",
        seed=123,
        samples_per_cosmology=1,
        embedding_batch_size=2,
        pk_nbins=1,
        bootstrap=1,
        bootstrap_seed=123,
        split_seed=123,
        output_dir=tmp_path / "out-degradation",
    )
    captured = {}

    class FakeEncoder:
        model_path = source_root / "head.pkl"

    monkeypatch.setattr(controls, "parse_args", lambda: args)
    monkeypatch.setattr(
        probe_script,
        "load_manifest",
        lambda *unused, **kwargs: [{"run_name": "run", "dataset_size": 128}],
    )
    monkeypatch.setattr(probe_script, "selected_rows", lambda rows, unused: rows)
    monkeypatch.setattr(probe_script, "load_vgg_encoder", lambda *unused: FakeEncoder())
    monkeypatch.setattr(probe_script, "output_path_for", lambda *unused: sample_path)
    monkeypatch.setattr(
        probe_eval,
        "load_heldout_real_slices",
        lambda *unused, **kwargs: (
            np.zeros((32, 1, 2, 2), dtype=np.float32),
            np.zeros((32, 6), dtype=np.float32),
            heldout,
            np.zeros(32, dtype=np.int64),
        ),
    )
    monkeypatch.setattr(controls, "deterministic_cosmology_split", lambda *unused, **kwargs: (np.array([900]), np.array([901])))
    monkeypatch.setattr(controls, "subset_generated_cosmologies", lambda generated, theta, heldout, **kwargs: (generated, theta, np.array([900, 901]), np.array([0, 0])))
    monkeypatch.setattr(controls, "power_ratio_transfer", lambda *unused, **kwargs: (np.array([1.0]), np.array([1.0]), np.array([1.0]), np.array([1.0]), np.array([1.0])))
    monkeypatch.setattr(controls, "fit_gaussian_smoothing", lambda *unused, **kwargs: 1.0)
    monkeypatch.setattr(controls, "transfer_transform", lambda *unused, **kwargs: (lambda images: (images, None)))
    monkeypatch.setattr(controls, "gaussian_smoothing_transform", lambda *unused, **kwargs: (lambda images: (images, None)))
    monkeypatch.setattr(controls, "field_histogram", lambda *unused, **kwargs: {})
    monkeypatch.setattr(
        controls,
        "_prediction_table",
        lambda **kwargs: pd.DataFrame(
            [{
                "transform": kwargs["transform_name"],
                "source": kwargs["source"],
                "run_name": kwargs["run_name"],
                "dataset_size": kwargs["dataset_size"],
            }]
        ),
    )
    monkeypatch.setattr(controls, "aggregate_prediction_table", lambda *unused, **kwargs: {"metrics": [{"transform": "identity"}]})
    monkeypatch.setattr(
        controls,
        "build_run_manifest",
        lambda **kwargs: captured.update(kwargs) or {},
    )

    controls.main()

    assert captured["project_dir"] == source_root.resolve()


def test_job_wrapper_runs_preflight_then_c0_c1_then_c4():
    text = (SCRIPTS / "slurm/run_probe_controls.sbatch").read_text()

    assert "#SBATCH --partition=spgpu" in text
    assert "#SBATCH --gres=gpu:1" in text
    assert "#SBATCH --cpus-per-task=4" in text
    assert "#SBATCH --mem=80gb" in text
    assert "#SBATCH --time=12:00:00" in text
    assert "#SBATCH -A huterer2" in text
    assert "preflight_probe_controls.sh" in text
    assert "--control c0" in text
    assert "--control c1" in text
    assert "--source-project-dir \"${CODE_ROOT}\"" in text
    assert "evaluate_probe_degradation_control.py" in text
    assert text.index("preflight_probe_controls.sh") < text.index("evaluate_probe_transform_controls.py")
    assert text.index("evaluate_probe_transform_controls.py") < text.index("evaluate_probe_degradation_control.py")
    assert "/scratch/huterer_root/huterer0/CAMELS/CMD/3d_grids/IllustrisTNG" in text
    assert "/scratch/huterer_root/huterer0/jiamingp/torch_cache" in text
    assert "EXPECTED_COMMIT" in text


def test_commit_pin_is_propagated_through_submit_and_job_wrappers():
    submit = (SCRIPTS / "slurm/submit_probe_controls.sh").read_text()
    job = (SCRIPTS / "slurm/run_probe_controls.sbatch").read_text()

    assert "EXPECTED_COMMIT=\"${EXPECTED_COMMIT}\"" in submit
    assert "EXPECTED_COMMIT" in job


def test_submit_wrapper_does_not_call_sbatch_after_failed_preflight(tmp_path):
    preflight = tmp_path / "preflight.sh"
    preflight.write_text("#!/bin/bash\nexit 17\n")
    preflight.chmod(0o755)
    sbatch_log = tmp_path / "sbatch.log"
    sbatch = tmp_path / "sbatch"
    sbatch.write_text(f"#!/bin/bash\necho called >> {sbatch_log}\necho 123\n")
    sbatch.chmod(0o755)

    env = os.environ.copy()
    env.update(
        PREFLIGHT_SCRIPT=str(preflight),
        SBATCH_BIN=str(sbatch),
        PROJECT_DIR=str(tmp_path),
        EXPECTED_COMMIT="expected-commit",
    )
    result = subprocess.run(
        ["bash", str(SCRIPTS / "slurm/submit_probe_controls.sh")],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 17
    assert not sbatch_log.exists()


def test_submit_wrapper_creates_log_directory_before_sbatch(tmp_path):
    preflight = tmp_path / "preflight.sh"
    preflight.write_text(
        "#!/bin/bash\n"
        "test \"${EXPECTED_COMMIT}\" = expected-commit\n"
        "exit 0\n"
    )
    preflight.chmod(0o755)
    sbatch = tmp_path / "sbatch"
    sbatch.write_text(
        "#!/bin/bash\n"
        "test -d \"${PROJECT_DIR}/logs/nf_conditional_bias_probe\" || exit 42\n"
        "case \"$*\" in *EXPECTED_COMMIT=expected-commit*) ;; *) exit 43 ;; esac\n"
        "echo 123\n"
    )
    sbatch.chmod(0o755)
    env = os.environ.copy()
    env.update(
        PREFLIGHT_SCRIPT=str(preflight),
        SBATCH_BIN=str(sbatch),
        PROJECT_DIR=str(tmp_path),
        CODE_ROOT="/code",
        EXPECTED_COMMIT="expected-commit",
    )

    result = subprocess.run(
        ["bash", str(SCRIPTS / "slurm/submit_probe_controls.sh")],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert (tmp_path / "logs/nf_conditional_bias_probe").is_dir()


def test_probe_wrappers_never_mutate_shared_venv_pth():
    paths = [
        SCRIPTS / "preflight_probe_controls.sh",
        SCRIPTS / "slurm/run_probe_controls.sbatch",
        SCRIPTS / "slurm/submit_probe_controls.sh",
    ]
    forbidden = (
        "00-cosmodiff-base-venv.pth",
        "mv \"$PTH",
        "rm -f",
        "sed -i",
    )
    for path in paths:
        text = path.read_text()
        assert not any(marker in text for marker in forbidden), path
