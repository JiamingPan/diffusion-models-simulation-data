from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
WRITER = REPO_ROOT / "scripts/write_diffusers_runtime_sitecustomize.py"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _write_fake_torch_and_diffusers(root: Path) -> None:
    (root / "diffusers").mkdir(parents=True)
    (root / "torch.py").write_text(
        "float16 = 'float16'\n"
        "uint8 = 'uint8'\n"
    )


def _write_runtime_audit_fixture(tmp_path: Path):
    from simdiff_eval.seed_restart_runtime import write_runtime_assets

    runtime_root = tmp_path / "seed_restart_runtime"
    pin_root = tmp_path / "cosmodiff_pin"
    fixture_root = tmp_path / "third_party"
    pin_package = pin_root / "cosmodiff"
    pin_package.mkdir(parents=True)
    fixture_root.mkdir()

    (pin_package / "__init__.py").write_text('__version__ = "0+source.fixture"\n')
    for name in ("optim", "utils", "augment", "transform"):
        (pin_package / f"{name}.py").write_text(f'VALUE = "{name}"\n')

    (fixture_root / "torch.py").write_text(
        "__version__ = '2.1.2+cu118'\n"
        "float16 = 'float16'\n"
        "uint8 = 'uint8'\n"
    )
    diffusers = fixture_root / "diffusers"
    diffusers.mkdir()
    (diffusers / "__init__.py").write_text(
        "import torch\n"
        "__version__ = '0.35.0.fixture'\n"
        "DEVICE_EMPTY_CACHE = {'xpu': torch.xpu.empty_cache}\n"
        "class DDPMScheduler: pass\n"
        "class DiTTransformer2DModel: pass\n"
    )
    hub = fixture_root / "huggingface_hub"
    hub.mkdir()
    (hub / "__init__.py").write_text(
        "__version__ = '0.25.0.fixture'\n"
        "def hf_hub_download(*args, **kwargs): return None\n"
        "def snapshot_download(*args, **kwargs): return None\n"
    )
    write_runtime_assets(
        runtime_root,
        code_root=REPO_ROOT,
        entry_point="tests.runtime_audit.sitecustomize",
    )
    return runtime_root, pin_root, fixture_root
    (root / "diffusers/__init__.py").write_text(
        "import torch\n"
        "DEVICE_EMPTY_CACHE = {'xpu': torch.xpu.empty_cache}\n"
    )


def _run_child(program: str, *python_paths: Path) -> subprocess.CompletedProcess[str]:
    env = dict(os.environ)
    env["PYTHONNOUSERSITE"] = "1"
    env["PYTHONPATH"] = os.pathsep.join(str(path) for path in python_paths)
    return subprocess.run(
        [sys.executable, "-c", program],
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )


def test_runtime_assets_are_deterministic_and_provide_a_narrow_sklearn_stub(
    tmp_path,
):
    from simdiff_eval.seed_restart_runtime import write_runtime_assets

    runtime_root = tmp_path / "seed_restart_runtime"

    first = write_runtime_assets(
        runtime_root,
        code_root=REPO_ROOT,
        entry_point="tests.runtime_assets",
    )
    second = write_runtime_assets(
        runtime_root,
        code_root=REPO_ROOT,
        entry_point="tests.runtime_assets",
    )

    assert first == second
    assert sorted(
        path.relative_to(runtime_root).as_posix()
        for path in runtime_root.rglob("*")
        if path.is_file()
    ) == [
        "sitecustomize.py",
        "sklearn/__init__.py",
        "sklearn/metrics/__init__.py",
    ]
    assert all(len(row["sha256"]) == 64 for row in first["files"].values())

    completed = _run_child(
        "import json, sklearn, sklearn.metrics\n"
        "try:\n"
        "    sklearn.metrics.roc_curve([], [])\n"
        "except RuntimeError as exc:\n"
        "    print(json.dumps({'version': sklearn.__version__, "
        "'kind': sklearn.RUNTIME_KIND, 'error': str(exc)}))\n",
        runtime_root,
        REPO_ROOT,
    )
    assert completed.returncode == 0, completed.stderr
    payload = json.loads(completed.stdout.strip().splitlines()[-1])
    assert payload == {
        "version": "0+simdiff-seed-restart-stub",
        "kind": "simdiff-seed-restart-stub",
        "error": (
            "sklearn.metrics.roc_curve is unavailable in the DiT seed-restart "
            "stub; this runtime cannot load sklearn estimators"
        ),
    }


def test_runtime_assets_refuse_to_replace_different_existing_bytes(tmp_path):
    from simdiff_eval.seed_restart_runtime import write_runtime_assets

    runtime_root = tmp_path / "seed_restart_runtime"
    runtime_root.mkdir()
    (runtime_root / "sitecustomize.py").write_text("foreign runtime\n")

    with pytest.raises(RuntimeError, match="different existing runtime asset"):
        write_runtime_assets(
            runtime_root,
            code_root=REPO_ROOT,
            entry_point="tests.conflict",
        )


def test_child_environment_has_audited_order_and_drops_unapproved_paths(tmp_path):
    from simdiff_eval.seed_restart_runtime import build_child_env

    runtime_root = tmp_path / "runtime"
    code_root = tmp_path / "code"
    pin_root = tmp_path / "pin"
    bad = tmp_path / "bad"
    residual = tmp_path / "approved"
    for path in (runtime_root, code_root, pin_root, bad, residual):
        path.mkdir()

    env = build_child_env(
        {"PATH": "/bin", "PYTHONPATH": os.pathsep.join((str(bad), "/discard"))},
        runtime_root=runtime_root,
        code_root=code_root,
        pin_root=pin_root,
        incompatible_paths=(bad,),
        approved_residual_paths=(residual,),
    )

    assert env["PATH"] == "/bin"
    assert env["PYTHONNOUSERSITE"] == "1"
    assert env["PYTHONPATH"].split(os.pathsep) == [
        str(runtime_root.resolve()),
        str(code_root.resolve()),
        str(pin_root.resolve()),
        str(residual.resolve()),
    ]
    assert str(bad) not in env["PYTHONPATH"]
    assert "/discard" not in env["PYTHONPATH"]


def test_generated_sitecustomize_installs_the_canonical_shim(tmp_path):
    from simdiff_eval.seed_restart_runtime import write_runtime_assets

    runtime_root = tmp_path / "runtime"
    fixture_root = tmp_path / "fixture"
    fixture_root.mkdir()
    _write_fake_torch_and_diffusers(fixture_root)
    write_runtime_assets(
        runtime_root,
        code_root=REPO_ROOT,
        entry_point="tests.generated_sitecustomize",
    )

    completed = _run_child(
        "import json, pathlib, sitecustomize, torch\n"
        "import diffusers\n"
        "print(json.dumps({'sitecustomize': str(pathlib.Path(sitecustomize.__file__).resolve()), "
        "'xpu': torch.xpu.is_available()}))\n",
        runtime_root,
        fixture_root,
        REPO_ROOT,
    )

    assert completed.returncode == 0, completed.stderr
    payload = json.loads(completed.stdout.strip().splitlines()[-1])
    assert payload["sitecustomize"] == str(
        (runtime_root / "sitecustomize.py").resolve()
    )
    assert payload["xpu"] is False


def test_explicit_shim_survives_a_decoy_sitecustomize(tmp_path):
    decoy_root = tmp_path / "decoy"
    fixture_root = tmp_path / "fixture"
    decoy_root.mkdir()
    fixture_root.mkdir()
    (decoy_root / "sitecustomize.py").write_text(
        "import os\nos.environ['DECOY_LOADED'] = '1'\n"
    )
    _write_fake_torch_and_diffusers(fixture_root)

    completed = _run_child(
        "import os\n"
        "from simdiff_eval.torch_compat import install_torch_backend_compat\n"
        "install_torch_backend_compat(entry_point='tests.decoy_child')\n"
        "import diffusers\n"
        "print('DECOY_LOADED=' + os.environ.get('DECOY_LOADED', '0'))\n"
        "print('SHIMMED_IMPORT_PASSED')\n",
        decoy_root,
        fixture_root,
        REPO_ROOT,
    )

    assert completed.returncode == 0, completed.stderr
    assert "DECOY_LOADED=1" in completed.stdout
    assert "SHIMMED_IMPORT_PASSED" in completed.stdout


def test_sitecustomize_writer_cli_delegates_to_the_runtime_adapter(tmp_path):
    output = tmp_path / "sitecustomize.py"

    completed = subprocess.run(
        [
            sys.executable,
            str(WRITER),
            str(output),
            "--code-root",
            str(REPO_ROOT),
            "--entry-point",
            "tests.writer_cli",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert output.is_file()
    spec = importlib.util.spec_from_file_location("generated_adapter", output)
    assert spec is not None
    assert "tests.writer_cli" in output.read_text()
    assert str((REPO_ROOT / "simdiff_eval/torch_compat.py").resolve()) in output.read_text()


def test_runtime_audit_imports_exact_pinned_modules_and_records_versions(tmp_path):
    from simdiff_eval.seed_restart_runtime import run_runtime_audit

    runtime_root, pin_root, fixture_root = _write_runtime_audit_fixture(tmp_path)

    report = run_runtime_audit(
        Path(sys.executable),
        pin_root=pin_root,
        runtime_root=runtime_root,
        code_root=REPO_ROOT,
        expected_torch_prefix=fixture_root,
        approved_residual_paths=(fixture_root,),
    )

    assert report["python"]["executable"] == sys.executable
    assert report["sitecustomize"]["file"] == str(
        (runtime_root / "sitecustomize.py").resolve()
    )
    assert report["torch"]["version"] == "2.1.2+cu118"
    assert report["torch"]["compat"]["schema_version"] == 1
    assert report["sklearn"] == {
        "file": str((runtime_root / "sklearn/__init__.py").resolve()),
        "runtime_kind": "simdiff-seed-restart-stub",
        "version": "0+simdiff-seed-restart-stub",
    }
    assert report["diffusers"]["symbols"] == [
        "DDPMScheduler",
        "DiTTransformer2DModel",
    ]
    assert report["huggingface_hub"]["symbols"] == [
        "hf_hub_download",
        "snapshot_download",
    ]
    assert set(report["cosmodiff"]["modules"]) == {
        "cosmodiff",
        "cosmodiff.optim",
        "cosmodiff.utils",
        "cosmodiff.augment",
        "cosmodiff.transform",
    }
    assert report["cosmodiff"]["version"] == "0+source.fixture"


def test_runtime_audit_rejects_torch_outside_expected_prefix(tmp_path):
    from simdiff_eval.seed_restart_runtime import run_runtime_audit

    runtime_root, pin_root, fixture_root = _write_runtime_audit_fixture(tmp_path)

    with pytest.raises(RuntimeError, match="torch.*expected prefix") as caught:
        run_runtime_audit(
            Path(sys.executable),
            pin_root=pin_root,
            runtime_root=runtime_root,
            code_root=REPO_ROOT,
            expected_torch_prefix=tmp_path / "wrong_torch_prefix",
            approved_residual_paths=(fixture_root,),
        )

    assert str((fixture_root / "torch.py").resolve()) in str(caught.value)


def test_runtime_audit_rejects_missing_huggingface_hub_symbol(tmp_path):
    from simdiff_eval.seed_restart_runtime import run_runtime_audit

    runtime_root, pin_root, fixture_root = _write_runtime_audit_fixture(tmp_path)
    (fixture_root / "huggingface_hub/__init__.py").write_text(
        "__version__ = '0.25.0.fixture'\n"
        "def hf_hub_download(*args, **kwargs): return None\n"
    )

    with pytest.raises(RuntimeError, match="snapshot_download"):
        run_runtime_audit(
            Path(sys.executable),
            pin_root=pin_root,
            runtime_root=runtime_root,
            code_root=REPO_ROOT,
            expected_torch_prefix=fixture_root,
            approved_residual_paths=(fixture_root,),
        )


def test_runtime_audit_child_uses_site_and_honors_pythonpath_order(tmp_path):
    from simdiff_eval.seed_restart_runtime import run_runtime_audit

    runtime_root, pin_root, fixture_root = _write_runtime_audit_fixture(tmp_path)
    argv_path = tmp_path / "python_argv.json"
    wrapper = tmp_path / "python-wrapper"
    wrapper.write_text(
        "#!/bin/sh\n"
        f"printf '%s\\n' \"$@\" > {str(argv_path)!r}\n"
        f"exec {sys.executable!r} \"$@\"\n"
    )
    wrapper.chmod(0o755)

    report = run_runtime_audit(
        wrapper,
        pin_root=pin_root,
        runtime_root=runtime_root,
        code_root=REPO_ROOT,
        expected_torch_prefix=fixture_root,
        approved_residual_paths=(fixture_root,),
    )

    argv = argv_path.read_text().splitlines()
    assert "-S" not in argv
    assert "-E" not in argv
    assert Path(argv[0]).name == "check_cosmodiff_seed_restart_imports.py"
    assert report["python"]["pythonpath"][:3] == [
        str(runtime_root.resolve()),
        str(REPO_ROOT.resolve()),
        str(pin_root.resolve()),
    ]
    assert str((REPO_ROOT / "scripts").resolve()) in report["python"]["path"]
