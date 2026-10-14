#!/usr/bin/env python
"""Prepare (no GPU) ledger nf_cond_omsigc_n1024_n4096_aug_finetune: continue the continuous (Omega_m, sigma_8)
N = 1024 and N = 4096 baselines for about +50k updates with RandomDihedral2D(p=1) + RandomRoll(size 128), each into
a NEW output dir. Mirrors scripts/prepare_nf_cond_omsigc_n256_aug_finetune.py (ledger nf_cond_omsigc_n256_aug_finetune),
whose exact-divisibility check rejects this schedule: 25000 / 32 and 25000 / 128 are not integers, so the checkpoints
are the nearest ones the cosmodiff save rule can produce.

Save rule (cosmodiff/optim.py:580, start_epoch = checkpoint epoch + 1, optim.py:360):
    save when (epoch + 1) % checkpoint_every_n_epochs == 0  (ABSOLUTE epoch)  or  epoch == start_epoch + num_epochs - 1
Schedule (ledger entry, planner):
    N = 1024  from checkpoint-epoch-6249  (spe 32,  200,000 updates)  num_epochs 1563, every 7031
              -> checkpoint-epoch-7030 (+24,992) and checkpoint-epoch-7812 (+50,016)
    N = 4096  from checkpoint-epoch-1562  (spe 128, 200,064 updates)  num_epochs 391,  every 1758
              -> checkpoint-epoch-1757 (+24,960) and checkpoint-epoch-1953 (+50,048)
This script simulates the rule and refuses unless the saved set is exactly the two planned checkpoints.

Writes under local/nf_cond_omsigc_n1024_n4096_aug_finetune/ (refuses any existing target):
* configs/<run>.yaml per N = baseline yaml with exactly these changes:
      io.output_dir                     -> /scratch/.../saved_runs/nf_cond_omsigc_n1024_n4096_aug_finetune/<run>_checkpoints
      train.num_epochs                  -> 1563 (N = 1024) / 391 (N = 4096)   (epochs AFTER the checkpoint)
      train.checkpoint_every_n_epochs   -> 7031 (N = 1024) / 1758 (N = 4096)
      augmentations                     (new) {RandomDihedral2D: {dims: [-2, -1], p: 1.0},
                                               RandomRoll: {size: 128, dims: [-2, -1]}}
* manifest.json: four sampling/eval rows, in this order (= sample array tasks 0..3):
      N1024 ft025k, N1024 ft050k, N4096 ft025k, N4096 ft050k.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
LEDGER_RUN = "nf_cond_omsigc_n1024_n4096_aug_finetune"
BASE_MANIFEST = ROOT / "local/nf_conditional_omsig_continuous_200k/manifest.json"
LOCAL = ROOT / f"local/{LEDGER_RUN}"
CKPT_ROOT = Path(f"/scratch/huterer_root/huterer0/jiamingp/saved_runs/{LEDGER_RUN}")
RESULTS = ROOT / f"results/{LEDGER_RUN}"
SAMPLES_REL = f"results/{LEDGER_RUN}/samples"
AUG = {"RandomDihedral2D": {"dims": [-2, -1], "p": 1.0}, "RandomRoll": {"size": 128, "dims": [-2, -1]}}

# base run -> (training run name, source checkpoint epoch, steps/epoch, num_epochs, every, {tag: (epoch, updates)})
PLAN = {
    "nf_cond_omsigc_hi_u128_d2p10_n1024_fresh200k": (
        "nf_cond_omsigc_hi_u128_d2p10_n1024_d4shift_ft50k_from200k", 6249, 32, 1563, 7031,
        {"ft025k": (7030, 24992), "ft050k": (7812, 50016)}),
    "nf_cond_omsigc_hi_u128_d2p12_n4096_fresh200k": (
        "nf_cond_omsigc_hi_u128_d2p12_n4096_d4shift_ft50k_from200k", 1562, 128, 391, 1758,
        {"ft025k": (1757, 24960), "ft050k": (1953, 50048)}),
}


def saved_epochs(src_epoch: int, num_epochs: int, every: int) -> list[int]:
    """Epochs at which cosmodiff/optim.py:580 writes checkpoint-epoch-<epoch> on a resume from src_epoch."""
    start = src_epoch + 1
    last = start + num_epochs - 1
    return [e for e in range(start, start + num_epochs) if (e + 1) % every == 0 or e == last]


def main() -> None:
    for p in (LOCAL, CKPT_ROOT, RESULTS / "samples", RESULTS / "evaluate_run_v1"):
        if p.exists():
            raise SystemExit(f"Refusing: {p} exists")
    base = {r["run_name"]: r for r in json.loads(BASE_MANIFEST.read_text())}
    rows, cfg_paths = [], []
    for base_run, (run, src_epoch, spe_plan, n_ep, every, save_at) in PLAN.items():
        b = base[base_run]
        src = Path(b["requested_checkpoint"])
        if int(b["checkpoint_epoch"]) != src_epoch or src.name != f"checkpoint-epoch-{src_epoch}":
            raise SystemExit(f"{base_run}: baseline final checkpoint is {src.name}, plan says {src_epoch}")
        if int(b["steps_per_epoch"]) != spe_plan:
            raise SystemExit(f"{base_run}: steps_per_epoch {b['steps_per_epoch']} != plan {spe_plan}")
        if not src.is_dir() or (src / "augmentations.pkl").exists():
            raise SystemExit(f"{src}: missing, or already has augmentations.pkl")
        saved = saved_epochs(src_epoch, n_ep, every)
        want = sorted(ep for ep, _ in save_at.values())
        if saved != want:
            raise SystemExit(f"{base_run}: save rule gives {saved}, plan wants {want}")
        for tag, (ep, upd) in save_at.items():
            if (ep - src_epoch) * spe_plan != upd:
                raise SystemExit(f"{base_run} {tag}: epoch {ep} is +{(ep - src_epoch) * spe_plan} updates, plan {upd}")

        cfg = yaml.safe_load((ROOT / b["config"]).read_text())
        if "augmentations" in cfg:
            raise SystemExit(f"{base_run}: baseline config already has augmentations")
        out_dir = CKPT_ROOT / f"{run}_checkpoints"
        cfg["io"]["output_dir"] = str(out_dir)
        cfg["train"]["num_epochs"] = n_ep
        cfg["train"]["checkpoint_every_n_epochs"] = every
        cfg["augmentations"] = AUG
        cfg_path = LOCAL / f"configs/{run}.yaml"
        cfg_path.parent.mkdir(parents=True, exist_ok=True)
        with open(cfg_path, "x") as f:
            f.write(yaml.safe_dump(cfg, sort_keys=False))
        cfg_paths.append(cfg_path)

        for tag, (ep, upd) in save_at.items():
            row = copy.deepcopy(b)
            name = run.replace("_ft50k_from200k", f"_{tag}_from200k")
            row.update({
                "run_name": name, "config": str(cfg_path.relative_to(ROOT)), "checkpoint_dir": str(out_dir),
                "checkpoint_epoch": ep, "requested_checkpoint": f"{out_dir}/checkpoint-epoch-{ep}",
                "sample_path": f"{SAMPLES_REL}/{name}_seed{{seed}}_dpm50_heldout_k{{k}}.npz",
                "source_run_name": base_run, "reuse_existing_checkpoint": True,
                "initialization": f"resume:{src}", "resume_from_checkpoint": str(src),
                "training_run_name": run, "finetune_updates": upd,
                "actual_updates": int(b["actual_updates"]) + upd, "target_updates": int(b["actual_updates"]) + upd,
                "epochs": n_ep, "augmentations": AUG, "baseline_run_name": base_run,
                "note": b["note"] + f" Fine-tune +{upd} updates (tag {tag}; nearest checkpoint the save rule allows) "
                                    f"with D4 + periodic roll from {src.name}; ledger {LEDGER_RUN}.",
            })
            rows.append(row)
    with open(LOCAL / "manifest.json", "x") as f:
        f.write(json.dumps(rows, indent=2) + "\n")
    print("Wrote " + ", ".join(str(p) for p in cfg_paths) + f" and {LOCAL / 'manifest.json'} ({len(rows)} rows: "
          + ", ".join(f"{r['run_name']} @ epoch {r['checkpoint_epoch']} (+{r['finetune_updates']})" for r in rows) + ")")


if __name__ == "__main__":
    main()
