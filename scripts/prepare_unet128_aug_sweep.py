#!/usr/bin/env python
"""Plan the width-128 augmentation sweep (Nick's "just augment" question).

18 fresh runs = 3 arms x N in {64,...,2048}, each reusing the existing
nf_fig2_u128_<tag>_noaug_200k config unchanged (same selected maps, fitted
normalization, scheduler, 200k updates); only the arm differs:

  d4shift : random D4 element + random periodic shift per fetch (UNet2DModel)
  warp    : smooth periodic warp, a novelty-only distortion control (UNet2DModel)
  d4equiv : D4-equivariant UNet, no data augmentation (D4ScalarUNet2DModel)

Writes local/unet128_aug_sweep/{runs.json,manifest.json}. manifest.json keeps
the 10 baseline u128 rows so compute_nf_generalize_pca_full_nn.py fits the same
PCA basis as the published fig2 table (largest N = u128 d2p15). No jobs.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

ARMS = {"d4shift": "d4_periodic_shift", "warp": "smooth_periodic_warp", "d4equiv": "d4_equivariant_unet"}
TAGS = {64: "d2p06", 128: "d2p07", 256: "d2p08", 512: "d2p09", 1024: "d2p10", 2048: "d2p11"}
SWEEP = "unet128_aug_sweep"
CHECKPOINT_ROOT = "/scratch/huterer_root/huterer0/jiamingp/saved_runs/unet128_aug_sweep"
SEED = 123


def sample_label(arm_tag: str) -> str:
    return f"dpm50_{arm_tag}"


def build(project_dir: Path) -> tuple[list[dict], list[dict]]:
    baseline = json.loads((project_dir / "local/nf_generalize_fig2/manifest.json").read_text())
    u128 = {int(r["dataset_size"]): r for r in baseline if r["arch"] == "u128"}
    runs, manifest = [], [dict(r) for r in sorted(u128.values(), key=lambda r: int(r["dataset_size"]))]
    for arm_tag, arm in ARMS.items():
        for n, tag in TAGS.items():
            base = u128[n]
            assert base["dataset_tag"] == tag and base["variant_tag"] == "noaug_fixed_updates", base["run_name"]
            config = project_dir / base["config"]
            name = f"nf_fig2_u128_{tag}_{arm_tag}_200k"
            sample = f"results/{SWEEP}/samples/{name}_seed{{seed}}_{sample_label(arm_tag)}.npz"
            runs.append({"run_name": name, "arm": arm, "arm_tag": arm_tag, "dataset_size": n, "dataset_tag": tag,
                         "baseline_run_name": base["run_name"], "config": base["config"],
                         "config_sha256": hashlib.sha256(config.read_bytes()).hexdigest(),
                         "output_dir": f"{CHECKPOINT_ROOT}/{name}_checkpoints", "training_seed": SEED,
                         "sample_label": sample_label(arm_tag), "sample_path": sample,
                         "baseline_sample_path": f"results/nf_generalize_fig2/samples/{base['run_name']}_seed{SEED}_dpm50.npz"})
            row = {k: v for k, v in base.items() if k not in ("sample_path",)}
            # No 'sample_label' key: the PCA script would then reject the shared --sample-label dpm50.
            row.update(run_name=name, arch=f"u128_{arm_tag}", arch_label=f"UNet-128 {arm_tag}",
                       variant_tag=arm_tag, sample_path=sample, method_sample_label=sample_label(arm_tag),
                       checkpoint_dir=f"{CHECKPOINT_ROOT}/{name}_checkpoints")
            manifest.append(row)
    return runs, manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-dir", type=Path, default=Path("."))
    parser.add_argument("--write", action="store_true", help="Write local/unet128_aug_sweep/*.json")
    parser.add_argument("--print-runs", metavar="ARM_TAG", choices=sorted(ARMS),
                        help="Print run names for one arm in array-task order")
    args = parser.parse_args()
    project_dir = args.project_dir.resolve()
    runs, manifest = build(project_dir)
    if args.print_runs:
        print("\n".join(r["run_name"] for r in runs if r["arm_tag"] == args.print_runs))
        return
    if args.write:
        out = project_dir / "local" / SWEEP
        out.mkdir(parents=True, exist_ok=True)
        (out / "runs.json").write_text(json.dumps(runs, indent=2) + "\n")
        (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        print("Wrote", out / "runs.json", "and", out / "manifest.json")
    for r in runs:
        print(r["arm_tag"], r["dataset_size"], r["run_name"], r["config_sha256"][:12], r["sample_label"])


if __name__ == "__main__":
    main()
