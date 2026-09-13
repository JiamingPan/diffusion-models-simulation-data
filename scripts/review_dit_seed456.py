"""Read-only final-checkpoint inventory. Does not import torch or submit jobs."""
import argparse
import json
import math
from pathlib import Path

EXPERIMENT = "nf_generalize_fig2_dit_l16_seed_restart500k_v1"


def inspect_record(completion_path, project):
    completion = json.loads(completion_path.read_text())
    audit_path = project / "results" / EXPERIMENT / "resume_audits" / Path(completion["resume_audit"]).name
    audit = json.loads(audit_path.read_text())
    state = audit["target_checkpoint_state"]
    target = Path(completion["target_checkpoint"])
    errors = []
    if completion["continue_stage"] != 5 or completion["resume_seed"] != 456:
        errors.append("Wrong stage or experiment seed")
    if target.resolve() != Path(state["checkpoint"]).resolve():
        errors.append("Completion and audit checkpoint paths disagree")
    if completion["code_revision"] != audit["code_revision"]:
        errors.append("Code revisions disagree")
    if not target.is_dir():
        errors.append("Final checkpoint directory is absent")
    updates = state["absolute_updates"]
    if not 500000 <= updates < 500000 + audit["optimizer_steps_per_epoch"]:
        errors.append("Final update count outside target epoch boundary")
    if not math.isfinite(audit["first_resumed_loss"]):
        errors.append("Nonfinite first resumed loss")
    if audit["ema_restore"]["step"] != audit["expected_ema_step"]:
        errors.append("EMA restore step disagrees")
    snapshots = state["ema_snapshots"]
    if len(snapshots) != 2 or any(not Path(p).is_file() for p in snapshots):
        errors.append("Final EMA snapshots absent or incomplete")
    return dict(dataset=completion["dataset_tag"], updates=updates,
                checkpoint=str(target), audit=str(audit_path), errors=errors,
                status="FAILED" if errors else "FILES AND RECORDS CONSISTENT",
                limitation="No tensor-load, sample-quality, or stage-1 seed verification")


def review(project):
    paths = sorted((project / "local" / EXPERIMENT / "completions").glob("stage5_*.json"))
    if not paths:
        raise RuntimeError("No final-stage completion records found")
    rows = [inspect_record(p, project) for p in paths]
    if sorted(r["dataset"] for r in rows) != ["d2p08", "d2p10"]:
        raise RuntimeError("Expected exactly one final record for each dataset; resolve duplicates explicitly")
    return rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-dir", type=Path, required=True)
    args = parser.parse_args()
    rows = review(args.project_dir)
    print(json.dumps(rows, indent=2))
    raise SystemExit(1 if any(r["errors"] for r in rows) else 0)
