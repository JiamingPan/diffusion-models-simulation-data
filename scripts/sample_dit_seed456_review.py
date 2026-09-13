"""DPM-50 adapter for the two seed-456 runs and their original 300k baselines.

Default mode prints an exact plan only. --execute requires a Slurm allocation
and a separately verified immutable runtime supplied by the batch wrapper.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

from review_dit_seed456 import EXPERIMENT, review


def make_plan(project, code_root, dataset, checkpoint_kind):
    rows = json.loads((project / "local" / EXPERIMENT / "manifest.json").read_text())
    selected = [r for r in rows if r["dataset_tag"] == dataset and r["continue_stage"] == 5]
    if len(selected) != 1:
        raise ValueError("Expected one stage-5 manifest row")
    row = selected[0]
    if "resume456" not in row["run_name"]:
        raise ValueError("Not a seed-456 run")
    finals = review(project)
    if any(r["errors"] for r in finals):
        raise ValueError(f"Final checkpoint review failed: {finals}")
    final = next(r for r in finals if r["dataset"] == dataset)
    if Path(row["expected_checkpoint"]).resolve() != Path(final["checkpoint"]).resolve():
        raise ValueError("Manifest disagrees with completion record")
    checkpoint = Path(row["source_checkpoint"] if checkpoint_kind == "300k" else final["checkpoint"])
    config = Path(row["source_config"] if checkpoint_kind == "300k" else row["config"])
    if not config.is_absolute():
        config = project / config
    if not checkpoint.is_dir() or not config.is_file():
        raise FileNotFoundError("Exact checkpoint or config is absent")
    output = project / "results" / EXPERIMENT / "review_samples" / f"{dataset}_{checkpoint_kind}_raw_seed123_dpm50.npz"
    command = [sys.executable, str(code_root / "scripts/sample_cosmodiff.py"),
               "--checkpoint", str(checkpoint), "--config", str(config),
               "--output", str(output), "--num-samples", "512", "--batch-size", "8",
               "--image-size", "128", "--scheduler", "DPMSolverMultistepScheduler",
               "--num-steps", "50", "--seed", "123", "--class-label", "0", "--device", "cuda"]
    return dict(dataset=dataset, checkpoint_kind=checkpoint_kind, checkpoint=str(checkpoint),
                config=str(config), config_sha256=hashlib.sha256(config.read_bytes()).hexdigest(),
                output=str(output), weights="raw", sampling_seed=123, command=command)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-dir", type=Path, required=True)
    parser.add_argument("--code-root", type=Path, required=True)
    parser.add_argument("--dataset", choices=["d2p08", "d2p10"], required=True)
    parser.add_argument("--checkpoint-kind", choices=["300k", "500k"], required=True)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    plan = make_plan(args.project_dir, args.code_root, args.dataset, args.checkpoint_kind)
    print(json.dumps(plan, indent=2), flush=True)
    if not args.execute:
        return
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("Refusing sampling outside a Slurm allocation")
    from validate_nf_generalize_fig2_dit_sample import validate_sample_file
    output = Path(plan["output"])
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    # Keep startup in the pin's original code root: its sitecustomize binds
    # the canonical shim to that directory, not to this adapter checkout.
    runtime_code_root = Path(os.environ["RUNTIME_CODE_ROOT"])
    subprocess.run(plan["command"], check=True, cwd=runtime_code_root)
    report = validate_sample_file(output, requested_checkpoint=Path(plan["checkpoint"]),
                                  scheduler="DPMSolverMultistepScheduler", requested_steps=50)
    with output.with_suffix(".validated.json").open("x") as handle:
        json.dump(dict(plan=plan, validation=report), handle, indent=2)


if __name__ == "__main__":
    main()
