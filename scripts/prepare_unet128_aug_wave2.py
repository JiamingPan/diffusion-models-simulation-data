#!/usr/bin/env python
"""Plan augmentation-sweep wave 2 (ledger unet128_aug_wave2): which part of D4 x shift matters.

9 fresh UNet-128 runs = 3 arms x N in {64, 128, 256}, each reusing the unchanged
nf_fig2_u128_<tag>_noaug_200k config (same maps, fitted normalization, scheduler,
batch 32, 200k updates), training seed 123, per-fetch augmentation only:

  shiftonly : random periodic roll, no D4          (simdiff_eval.aug_wave2_transforms.ShiftOnly)
  d4only    : random D4 element, no roll           (D4Only)
  fliponly  : identity or left-right flip, p = 0.5 (FlipOnly)

Array-task order: all N = 64 tasks first, then 128, then 256 (arm order as above).
Dataset hashes come from local/unet128_aug_sweep/data_reference.json (same configs).

Writes local/unet128_aug_wave2/runs.json (training/sampling) and eval_runs.json
(evaluate_run.py --runs-json: the 9 new runs + existing noaug and d4shift rows at
the same N). Refuses to overwrite either file. No jobs.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

ARMS = {"shiftonly": "shift_only", "d4only": "d4_only", "fliponly": "flip_only"}
TAGS = {64: "d2p06", 128: "d2p07", 256: "d2p08"}
WAVE = "unet128_aug_wave2"
REF_SWEEP = "unet128_aug_sweep"
CHECKPOINT_ROOT = f"/scratch/huterer_root/huterer0/jiamingp/saved_runs/{WAVE}"
SEED = 123


def sample_label(arm_tag: str) -> str:
    return f"dpm50_{arm_tag}"


def build(project_dir: Path) -> tuple[list[dict], list[dict]]:
    fig2 = json.loads((project_dir / "local/nf_generalize_fig2/manifest.json").read_text())
    u128 = {int(r["dataset_size"]): r for r in fig2 if r["arch"] == "u128"}
    ref = json.loads((project_dir / f"local/{REF_SWEEP}/data_reference.json").read_text())
    d4shift = {int(r["dataset_size"]): r for r in json.loads((project_dir / f"local/{REF_SWEEP}/runs.json").read_text())
               if r["arm_tag"] == "d4shift"}
    runs, evals = [], []
    for n, tag in TAGS.items():
        base = u128[n]
        assert base["dataset_tag"] == tag and base["variant_tag"] == "noaug_fixed_updates", base["run_name"]
        sha = hashlib.sha256((project_dir / base["config"]).read_bytes()).hexdigest()
        assert d4shift[n]["config"] == base["config"] and d4shift[n]["config_sha256"] == sha, n
        for arm_tag, arm in ARMS.items():
            name = f"nf_fig2_u128_{tag}_{arm_tag}_200k"
            sample = f"results/{WAVE}/samples/{name}_seed{{seed}}_{sample_label(arm_tag)}.npz"
            runs.append({"run_name": name, "arm": arm, "arm_tag": arm_tag, "dataset_size": n, "dataset_tag": tag,
                         "baseline_run_name": base["run_name"], "config": base["config"], "config_sha256": sha,
                         "dataset_sha256": ref[str(n)]["dataset_sha256"],
                         "output_dir": f"{CHECKPOINT_ROOT}/{name}_checkpoints", "training_seed": SEED,
                         "augmentation_seed": SEED + 1000,
                         "sample_label": sample_label(arm_tag), "sample_path": sample,
                         "baseline_sample_path": f"results/nf_generalize_fig2/samples/{base['run_name']}_seed{SEED}_dpm50.npz"})
            evals.append(dict(run_name=name, arm=arm_tag, dataset_size=n, config=base["config"], sample_path=sample))
        evals.append(dict(run_name=base["run_name"], arm="noaug", dataset_size=n, config=base["config"],
                          sample_path=f"results/nf_generalize_fig2/samples/{base['run_name']}_seed{{seed}}_dpm50.npz"))
        evals.append(dict(run_name=d4shift[n]["run_name"], arm="d4shift", dataset_size=n, config=base["config"],
                          sample_path=d4shift[n]["sample_path"]))
    return runs, evals


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--project-dir", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--write", action="store_true", help=f"Write local/{WAVE}/{{runs,eval_runs}}.json (new files only)")
    args = parser.parse_args()
    project_dir = args.project_dir.resolve()
    runs, evals = build(project_dir)
    if args.write:
        out = project_dir / "local" / WAVE
        out.mkdir(parents=True, exist_ok=True)
        for fname, obj in (("runs.json", runs), ("eval_runs.json", evals)):
            with (out / fname).open("x") as f:
                f.write(json.dumps(obj, indent=2) + "\n")
        print("Wrote", out / "runs.json", "and", out / "eval_runs.json")
    for i, r in enumerate(runs):
        print(i, r["arm_tag"], r["dataset_size"], r["run_name"], r["config_sha256"][:12], r["dataset_sha256"][:12],
              r["sample_label"])


if __name__ == "__main__":
    main()
