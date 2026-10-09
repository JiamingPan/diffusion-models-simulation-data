#!/usr/bin/env python
"""Prepare (no GPU) ledger nf_cond_omsigc_n256_aug_finetune: continue the continuous (Omega_m, sigma_8)
N = 256 baseline nf_cond_omsigc_hi_u128_d2p08_n256_fresh200k from checkpoint-epoch-24999 (200k updates) for
+100k updates with RandomDihedral2D(p=1) + RandomRoll(size 128), into a NEW output dir.

Writes under local/nf_cond_omsigc_n256_aug_finetune/ (refuses any existing target):
* configs/<run>.yaml = baseline N = 256 yaml with exactly these changes:
      io.output_dir                     -> /scratch/.../saved_runs/nf_cond_omsigc_n256_aug_finetune/<run>_checkpoints
      train.num_epochs                  25000 -> 12500   (epochs AFTER the checkpoint; 12500 x 8 = 100k updates)
      train.checkpoint_every_n_epochs   2500 -> 3125     (checkpoints at +25k, +50k, +75k, +100k updates)
      augmentations                     (new) {RandomDihedral2D: {dims: [-2, -1], p: 1.0},
                                               RandomRoll: {size: 128, dims: [-2, -1]}}
* manifest.json: three sampling/eval rows (+25k, +50k, +100k = checkpoint-epoch-28124, 31249, 37499).
"""
from __future__ import annotations

import copy
import json
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
BASE_MANIFEST = ROOT / "local/nf_conditional_omsig_continuous_200k/manifest.json"
BASE_RUN = "nf_cond_omsigc_hi_u128_d2p08_n256_fresh200k"
LOCAL = ROOT / "local/nf_cond_omsigc_n256_aug_finetune"
CKPT_ROOT = Path("/scratch/huterer_root/huterer0/jiamingp/saved_runs/nf_cond_omsigc_n256_aug_finetune")
SAMPLES_REL = "results/nf_cond_omsigc_n256_aug_finetune/samples"
RUN = "nf_cond_omsigc_hi_u128_d2p08_n256_d4shift_ft100k_from200k"
AUG = {"RandomDihedral2D": {"dims": [-2, -1], "p": 1.0}, "RandomRoll": {"size": 128, "dims": [-2, -1]}}
EXTRA_EPOCHS, EVERY = 12500, 3125
SAVE_AT = {"ft025k": 25000, "ft050k": 50000, "ft100k": 100000}


def main() -> None:
    for p in (LOCAL, CKPT_ROOT, ROOT / "results/nf_cond_omsigc_n256_aug_finetune"):
        if p.exists():
            raise SystemExit(f"Refusing: {p} exists")
    b = {r["run_name"]: r for r in json.loads(BASE_MANIFEST.read_text())}[BASE_RUN]
    src = Path(b["requested_checkpoint"])
    if not src.is_dir() or (src / "augmentations.pkl").exists():
        raise SystemExit(f"{src}: missing, or already has augmentations.pkl")
    spe, start = int(b["steps_per_epoch"]), int(b["checkpoint_epoch"]) + 1
    cfg = yaml.safe_load((ROOT / b["config"]).read_text())
    if "augmentations" in cfg:
        raise SystemExit("baseline config already has augmentations")
    out_dir = CKPT_ROOT / f"{RUN}_checkpoints"
    cfg["io"]["output_dir"] = str(out_dir)
    cfg["train"]["num_epochs"] = EXTRA_EPOCHS
    cfg["train"]["checkpoint_every_n_epochs"] = EVERY
    cfg["augmentations"] = AUG
    cfg_path = LOCAL / f"configs/{RUN}.yaml"
    cfg_path.parent.mkdir(parents=True)
    cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False))

    rows = []
    for tag, upd in SAVE_AT.items():
        if upd % (spe * EVERY) != 0:
            raise SystemExit(f"+{upd} updates is not a checkpoint")
        ep = start + upd // spe - 1
        row = copy.deepcopy(b)
        name = f"{RUN[: -len('_ft100k_from200k')]}_{tag}_from200k"
        row.update({
            "run_name": name, "config": str(cfg_path.relative_to(ROOT)), "checkpoint_dir": str(out_dir),
            "checkpoint_epoch": ep, "requested_checkpoint": f"{out_dir}/checkpoint-epoch-{ep}",
            "sample_path": f"{SAMPLES_REL}/{name}_seed{{seed}}_dpm50_heldout_k{{k}}.npz",
            "source_run_name": BASE_RUN, "reuse_existing_checkpoint": True,
            "initialization": f"resume:{src}", "resume_from_checkpoint": str(src),
            "training_run_name": RUN, "finetune_updates": upd,
            "actual_updates": int(b["actual_updates"]) + upd, "target_updates": int(b["actual_updates"]) + upd,
            "epochs": EXTRA_EPOCHS, "augmentations": AUG, "baseline_run_name": BASE_RUN,
            "note": b["note"] + f" Fine-tune +{upd} updates with D4 + periodic roll from {src.name}; "
                                "ledger nf_cond_omsigc_n256_aug_finetune.",
        })
        rows.append(row)
    (LOCAL / "manifest.json").write_text(json.dumps(rows, indent=2) + "\n")
    print(f"Wrote {cfg_path} and {LOCAL / 'manifest.json'} ({len(rows)} rows: "
          + ", ".join(f"{r['run_name']} @ epoch {r['checkpoint_epoch']}" for r in rows) + ")")


if __name__ == "__main__":
    main()
