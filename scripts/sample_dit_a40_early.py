#!/usr/bin/env python
"""Sample only completed A40 arms 0/1 into a separate folder. Never submits jobs."""
from contextlib import contextmanager
import argparse
import gc
import importlib
import json
import os
from pathlib import Path
import subprocess
import sys

TRAIN_REVISION = "50b685f5999b90a212f696ab38b172dca5d968ab"
PLAN_SHA = "e814a42ad2026c2b9963a43b017220ff0326800c21d920c4234523e16b06b21f"
RUNTIME_REVISION = "555f350f82c913ff06150c96e66709719856cd83"
EARLY_FOLDER = "samples_early_nick_v1"
ARMS = ("p8_zero", "p4_native")


def clean_checkout(path, revision):
    path = Path(path)
    actual = subprocess.check_output(["git", "-C", str(path), "rev-parse", "HEAD"], text=True).strip()
    dirty = subprocess.check_output(["git", "-C", str(path), "status", "--porcelain"], text=True).strip()
    if actual != revision or dirty:
        raise ValueError(f"Checkout is not clean at the expected revision: {path}")


def frozen_adapter(root):
    """Verify immutable code/runtime before importing the unchanged sampling implementation."""
    code = Path(os.environ["CODE_ROOT"])
    runtime = Path(os.environ["RUNTIME_CODE_ROOT"])
    early = Path(os.environ["EARLY_CODE_ROOT"])
    if (os.environ.get("EXPECTED_COMMIT") != TRAIN_REVISION
            or os.environ.get("ABLATION_PLAN_SHA256") != PLAN_SHA
            or root.resolve() != Path(os.environ["ABLATION_OUT_DIR"]).resolve()
            or Path(__file__).resolve().parents[1] != early.resolve()
            or Path(sys.executable).resolve() != Path(os.environ["PYTHON_BIN"]).resolve()):
        raise ValueError("Early sampler environment differs from the frozen experiment")
    clean_checkout(code, TRAIN_REVISION)
    clean_checkout(runtime, RUNTIME_REVISION)
    clean_checkout(early, os.environ["EARLY_EXPECTED_COMMIT"])
    pin = Path(os.environ["COSMODIFF_PIN_ROOT"])
    os.environ.update(PYTHONNOUSERSITE="1", PYTHONUNBUFFERED="1", PYTHONHASHSEED="123",
                      COSMODIFF_DIR=str(pin), OMP_NUM_THREADS="4", MKL_NUM_THREADS="4",
                      OPENBLAS_NUM_THREADS="4")
    os.environ["PYTHONPATH"] = os.pathsep.join(map(str, (pin / "seed_restart_runtime", runtime, pin)))
    # Reuse the existing pin verifier and tiny native CPU API smoke. PREFLIGHT_ONLY exits
    # before every training/sampling/Test-A branch; no Slurm submission occurs here.
    subprocess.run(["bash", str(code / "scripts/slurm/dit_a40_ablation.sbatch")],
                   env={**os.environ, "PREFLIGHT_ONLY": "1", "ABLATION_MODE": "sample"}, check=True)
    sys.path[:0] = list(map(str, (code / "scripts", pin / "seed_restart_runtime", runtime, pin)))
    adapter = importlib.import_module("dit_a40_ablation")
    if Path(adapter.__file__).resolve() != (code / "scripts/dit_a40_ablation.py").resolve():
        raise ValueError("Sampling adapter escaped the frozen training checkout")
    return adapter


def preflight(adapter, root):
    """Both arms must be validated complete before committing an additional GPU job."""
    plan, _ = adapter.load_plan(root)
    adapter.require_data_preflight(root, plan)
    for index, name in enumerate(ARMS):
        row = plan["arms"][index]
        report = json.loads((Path(row["checkpoint_dir"]) / "complete_record/complete.json").read_text())
        steps = report.get("successful_optimizer_steps", 0)
        if (row["name"] != name or report.get("status") != "complete" or report.get("arm") != row
                or report.get("plan_sha256") != adapter.file_hash(root / "plan/plan.json")
                or report.get("nominal_optimizer_steps") != 300000
                or not isinstance(steps, int) or not 0 < steps <= 300000
                or report.get("amp_skipped_optimizer_steps") != 300000 - steps
                or not report.get("checkpoint_validation")):
            raise ValueError(f"Training completion is missing/invalid: {name}")
        final = Path(row["expected_checkpoint"])
        hashes = {str(p.relative_to(final)): adapter.file_hash(p) for p in final.rglob("*") if p.is_file()}
        if not hashes or hashes != report.get("checkpoint_files_sha256"):
            raise ValueError(f"Validated final checkpoint changed: {name}")
        destination = root / EARLY_FOLDER / name
        if (destination.exists() or destination.is_symlink()
                or destination.with_name(f".{name}.lock").exists()
                or list(destination.parent.glob(f".{name}.pending.*"))):
            raise FileExistsError(f"Prior early output/lock/partial retained; do not automatically retry: {name}")
        print(f"EARLY ARM READY: {name}; successful_updates={steps}; checkpoint={final}", flush=True)
    print("EARLY PREFLIGHT PASSED: arms 0/1 only; no sampling or job submission", flush=True)


@contextmanager
def redirect_outputs(trace, root):
    """Change only the output location; keep frozen scheduler/noises/loader/scoring intact."""
    original = trace.atomic_output
    destinations = {(root / "samples" / name).resolve(): root / EARLY_FOLDER / name for name in ARMS}

    @contextmanager
    def isolated(output):
        key = Path(output).resolve()
        if key not in destinations:
            raise ValueError(f"Unexpected write rejected by early sampler: {output}")
        with original(destinations[key]) as pending:
            yield pending

    trace.atomic_output = isolated
    try:
        yield
    finally:
        trace.atomic_output = original


def run(adapter, root):
    plan, _ = adapter.load_plan(root)
    # Compatibility must be installed in THIS process before trace imports torch.
    torch, _ = adapter.runtime(plan["runtime_versions"])
    adapter.single_a40(torch)
    # Import from the same immutable checkout as adapter.sample, which imports
    # atomic_output at call time. No files in either checkout are modified.
    trace = importlib.import_module("trace_trajectories")
    if Path(trace.__file__).resolve().parent != Path(adapter.__file__).resolve().parent:
        raise ValueError("Atomic output helper escaped the frozen code root")
    with redirect_outputs(trace, root):
        for index, name in enumerate(ARMS):
            adapter.sample(root, index)
            gc.collect()
            torch.cuda.empty_cache()
            print(f"EARLY RESULTS READY: {root / EARLY_FOLDER / name}", flush=True)
    print("EARLY DPM50 COMPLETE: 2 arms, 128 paired raw samples each; arm2 and queued sampler unchanged", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--preflight", action="store_true")
    mode.add_argument("--run", action="store_true")
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    adapter = frozen_adapter(args.out_dir)
    preflight(adapter, args.out_dir)
    if args.run:
        run(adapter, args.out_dir)


if __name__ == "__main__":
    main()
