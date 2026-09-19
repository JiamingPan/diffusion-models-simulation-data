#!/usr/bin/env python
"""Draft isolated fresh zero-init DiT sweep configs; never submit or train.

The output is deliberately NOT a launch-ready plan. The pinned runtime must
first gain and verify the opt-in initialization factory and slice-first data
contract. Existing native/continued checkpoints are not reinitialized.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path

import yaml

import prepare_nf_generalize_fig2_dit_configs as base


def draft_runs(out_dir: Path, checkpoint_root: Path, target_updates: int):
    if target_updates < 1:
        raise ValueError("target-updates must be positive")
    if not checkpoint_root.is_absolute():
        raise ValueError("checkpoint-root must be an explicit absolute path")
    if checkpoint_root == Path(base.CHECKPOINT_ROOT) or checkpoint_root in [Path(base.CHECKPOINT_ROOT).parent,Path("/")]:
        raise ValueError("new checkpoints must not target the old sweep or a broad root")
    rows, configs = [], {}
    for source in base.iter_runs():
        depth = base.DIT_VARIANTS[source["arch"]]["model_kwargs"]["num_layers"]
        name = f"nf_fig2_dit_l{depth}_{source['dataset_tag']}_p8_adalnzero_fresh_{target_updates}u_seed123"
        cfg = base.build_config(source["run_name"],source["arch"],deepcopy(source["source_counts"]),source["dataset_size"])
        cfg["io"]["output_dir"] = str(checkpoint_root / f"{name}_checkpoints")
        cfg["model"]["initialization"] = "adaln_zero"
        cfg["model"]["kwargs"]["norm_type"] = "ada_norm_zero"
        spe = source["optimizer_steps_per_epoch"]
        epochs = math.ceil(target_updates/spe)
        cfg["train"]["num_epochs"] = epochs
        config_path = out_dir / "configs" / f"{name}.yaml"
        text = yaml.safe_dump(cfg,sort_keys=False)
        configs[config_path] = text
        rows.append({
            "run_name":name,"source_run_name":source["run_name"],"arch":source["arch"],
            "num_layers":depth,"dataset_size":source["dataset_size"],"dataset_tag":source["dataset_tag"],
            "initialization":"adaln_zero","patch_size":8,"training_seed":123,
            "seed_policy":"launcher must seed Python/NumPy/PyTorch before fresh construction",
            "conditioning":"constant_label_0; NOT continuous cosmology conditioning",
            "target_updates":target_updates,"nominal_updates":epochs*spe,
            "update_policy":"native epochs with AMP skips recorded, not assumed zero",
            "epochs":epochs,"checkpoint_epoch":epochs-1,"effective_batch_size":source["effective_batch_size"],
            "config":str(config_path),"config_sha256":hashlib.sha256(text.encode()).hexdigest(),
            "checkpoint_dir":cfg["io"]["output_dir"],
            "sample_path":str(out_dir / "samples" / f"{name}_seed{{seed}}_{{sample_label}}.npz"),
        })
    plan = {
        "status":"draft_not_launch_ready","fresh_runs":True,"runs":rows,
        "required_gates":[
            "verified opt-in cosmodiff initialization factory (prepared PR commit 816d693a629ed45052db08dd3108be757846ab74)",
            "runtime-version pin and real native CPU initialization/gradient smoke",
            "slice-first exact-subset and normalization audit for every N, with persisted hashes",
            "seeded fresh construction, no checkpoint reuse; original experiments preserved",
            "A40 forward/backward memory and throughput checks before long runs",
            "user-approved exact jobs, cost, budget and concurrency after final preview",
        ],
        "plot_policy":{
            "unconditional":["novelty using exact training subset","one-point PDF","P(k)","grid-boundary diagnostics"],
            "exclude":"do not pool new zero-init and old native runs in one depth curve",
            "posterior_recovery":"blocked for these constant-label models; requires a separate continuous-conditioning model",
        },
    }
    return plan, configs


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir",type=Path,required=True)
    parser.add_argument("--checkpoint-root",type=Path,required=True)
    parser.add_argument("--target-updates",type=int,required=True)
    parser.add_argument("--depth", type=int, action="append", choices=[8, 12, 16])
    parser.add_argument("--dataset-tag", action="append",
                        help="Restrict the plan, e.g. d2p09. Repeatable.")
    args=parser.parse_args()
    if args.out_dir.exists():
        raise FileExistsError("preserve the existing plan; choose a new out-dir")
    root=args.out_dir.resolve()
    plan, configs=draft_runs(root,args.checkpoint_root,args.target_updates)
    depths = set(args.depth or [8, 12, 16])
    tags = set(args.dataset_tag or [row["dataset_tag"] for row in plan["runs"]])
    plan["runs"] = [row for row in plan["runs"]
                    if row["num_layers"] in depths and row["dataset_tag"] in tags]
    if not plan["runs"]:
        raise ValueError("selection produced no runs")
    selected_configs = {Path(row["config"]): configs[Path(row["config"])] for row in plan["runs"]}
    configs = selected_configs
    plan["selection"] = {"depths": sorted(depths), "dataset_tags": sorted(tags)}
    (root/"configs").mkdir(parents=True,exist_ok=False)
    for path,text in configs.items():
        path.write_text(text)
    (root/"plan.json").write_text(json.dumps(plan,indent=2)+"\n")
    print(f"DRAFT ONLY: {len(plan['runs'])} fresh runs; no preflight, data loading, training or submission")


if __name__ == "__main__":
    main()
