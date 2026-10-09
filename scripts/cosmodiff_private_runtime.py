#!/usr/bin/env python
"""Build a private, patched copy of the cosmodiff runtime (never patches the shared checkout).

Same recipe as scripts/preflight_nf_cond_omsigc_d4shift.py build_runtime, with the patch list as an
argument so new patch modules (e.g. scripts/patch_cosmodiff_cfg_null_flag.py) can be added per study:
copy cosmodiff/*.py, cosmodiff/data and the two entry scripts from the shared checkout, apply
scripts/patch_cosmodiff_<name>.py for each name in order, write the sklearn stub and the diffusers
sitecustomize. Used by the preflight scripts of nf_cond_omsigc_cfg and nf_cond_omsigc_n256_aug_finetune
and by tests/test_cfg_null_flag.py.
"""
from __future__ import annotations

import hashlib
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
COSMODIFF_SRC = Path(os.environ.get("COSMODIFF_SHARED_SRC", "/home/jiamingp/Diffusion_model/cosmo_diffusion_main"))

# Patch lists of the existing continuous-label launchers (train_/sample_nf_conditional_omsig_continuous_array).
BASE_PATCH_SETS = {
    "train": ["continuous_labels", "direct_unet_checkpoint", "numpy_compat"],
    "sample": ["package_metadata", "continuous_labels", "direct_unet_checkpoint", "numpy_compat",
               "sampling_compat"],
}


def sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def git(args, cwd) -> str:
    try:
        return subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True, check=True).stdout.strip()
    except Exception as exc:  # noqa: BLE001
        return f"unavailable: {exc}"


def runtime_hashes(src: Path) -> dict:
    files = sorted(src.glob("cosmodiff/*.py")) + sorted(src.glob("scripts/cosmodiff_*.py"))
    return {str(f.relative_to(src)): sha256(f) for f in files}


def build_runtime(dest: Path, patches: list[str]) -> dict:
    dest = Path(dest)
    if dest.exists():
        raise SystemExit(f"Refusing to reuse existing runtime dir {dest}")
    src = dest / "cosmodiff_src"
    stubs = dest / "stubs"
    (src / "cosmodiff").mkdir(parents=True)
    (src / "scripts").mkdir()
    for f in sorted((COSMODIFF_SRC / "cosmodiff").glob("*.py")):
        shutil.copy2(f, src / "cosmodiff" / f.name)
    shutil.copytree(COSMODIFF_SRC / "cosmodiff/data", src / "cosmodiff/data")
    for name in ("cosmodiff_train.py", "cosmodiff_sample.py"):
        shutil.copy2(COSMODIFF_SRC / "scripts" / name, src / "scripts" / name)
    patch_log = []
    for p in patches:
        r = subprocess.run([sys.executable, str(ROOT / f"scripts/patch_cosmodiff_{p}.py"), str(src)],
                           capture_output=True, text=True)
        patch_log.append({"patch": p, "rc": r.returncode, "stdout": r.stdout.strip(), "stderr": r.stderr.strip()})
        if r.returncode != 0:
            raise SystemExit(f"patch {p} failed on {src}: {r.stderr}")
    (stubs / "sklearn/metrics").mkdir(parents=True)
    (stubs / "sklearn/__init__.py").write_text("from . import metrics\n")
    (stubs / "sklearn/metrics/__init__.py").write_text(
        "def roc_curve(*args, **kwargs):\n    raise RuntimeError('sklearn.metrics.roc_curve is stubbed')\n")
    subprocess.run([sys.executable, str(ROOT / "scripts/write_diffusers_runtime_sitecustomize.py"),
                    str(stubs / "sitecustomize.py")], check=True, capture_output=True, text=True)
    return {"dest": str(dest), "patch_list": list(patches), "patches": patch_log,
            "cosmodiff_source": str(COSMODIFF_SRC),
            "cosmodiff_source_git_head": git(["rev-parse", "HEAD"], COSMODIFF_SRC),
            "cosmodiff_source_git_status": git(["status", "--short", "--untracked-files=no"], COSMODIFF_SRC),
            "sha256": runtime_hashes(src)}


def runtime_env(dest: Path) -> dict:
    dest = Path(dest)
    env = dict(os.environ)
    keep = [e for e in env.get("PYTHONPATH", "").split(":") if e and "site-packages" not in e
            and "dist-packages" not in e]
    env["PYTHONPATH"] = ":".join([str(dest / "stubs"), str(dest / "cosmodiff_src"), str(ROOT),
                                  str(ROOT / "scripts"), *keep])
    env["COSMODIFF_DIR"] = str(dest / "cosmodiff_src")
    env["COSMODIFF_STUB_SKLEARN"] = "1"
    env["TORCHDYNAMO_DISABLE"] = "1"
    env.setdefault("OMP_NUM_THREADS", "8")
    return env


def verify_against_receipt(prov: dict, receipt: dict, key: str) -> list[str]:
    """Names of runtime files whose hash differs from receipt['runtime'][key]['sha256']."""
    want = receipt["runtime"][key]["sha256"]
    got = prov["sha256"]
    return sorted(k for k in set(want) | set(got) if want.get(k) != got.get(k))
