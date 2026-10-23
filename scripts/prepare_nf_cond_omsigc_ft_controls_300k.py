#!/usr/bin/env python
"""Prepare (no GPU) ledger nf_cond_omsigc_ft_controls_300k: controls for the N = 1024 / 4096 D4+shift fine-tune
(ledger nf_cond_omsigc_n1024_n4096_aug_finetune). Mirrors scripts/prepare_nf_cond_omsigc_n1024_n4096_aug_finetune.py.

Arms (ledger entry, 2026-10-14):
  A noaug    no augmentations block; N = 1024 and N = 4096, from the 200k baselines.
  B idaug    N = 1024 only; {RandomDihedral2D dims [-2,-1] p 0.0, RandomRoll size 1 dims [-2,-1]}: same classes and
             ArrayDataset.__getitem__ path as the aug arm, output == input (still draws torch.rand / torch.randint).
  C d4shift  continue the aug-arm checkpoints 7812 (N = 1024) / 1953 (N = 4096), with their own D4 + roll pipeline,
             for one more segment to 9374 / 2343.

Save rule (cosmodiff/optim.py:580, start_epoch = checkpoint epoch + 1, optim.py:360):
    save when (epoch + 1) % checkpoint_every_n_epochs == 0 (ABSOLUTE epoch) or epoch == start_epoch + num_epochs - 1
No single modulus hits both 7030 and 7812, so A and B run in two segments matching the aug arm exactly:
    N = 1024 seg1 from 6249: num_epochs 1563, every 7031 -> 7030 (+24,992), 7812 (+50,016)
             seg2 from 7812: num_epochs 1562, every 7031 -> 9374 (+100,000; total 300,000)
    N = 4096 seg1 from 1562: num_epochs 391,  every 1758 -> 1757 (+24,960), 1953 (+50,048)
             seg2 from 1953: num_epochs 390,  every 1758 -> 2343 (+99,968; total 300,032)
    C uses seg2 only (from the aug-arm checkpoints). Update counts are relative to the 200k baselines.
This script simulates the rule and refuses unless each segment saves exactly the planned checkpoints.

Each segment is one training run with its own NEW output dir and config (scripts/finetune_cosmodiff_resume.py
refuses an existing output dir). Training-run order (= train array task id):
    0 A N1024 seg1   1 A N4096 seg1   2 B N1024 seg1
    3 A N1024 seg2 (source: task 0 checkpoint-epoch-7812)   4 A N4096 seg2 (task 1, 1953)   5 B N1024 seg2 (task 2, 7812)
    6 C N1024        7 C N4096
Manifest rows (= sample array task id 0..10): A1024 ft025k/ft050k/ft100k, A4096 x3, B1024 x3, C1024 ft100k,
C4096 ft100k.

Writes under local/nf_cond_omsigc_ft_controls_300k/ (refuses any existing target): configs/<training run>.yaml
and manifest.json.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
LEDGER_RUN = "nf_cond_omsigc_ft_controls_300k"
BASE_MANIFEST = ROOT / "local/nf_conditional_omsig_continuous_200k/manifest.json"
AUG_RUN = "nf_cond_omsigc_n1024_n4096_aug_finetune"
AUG_MANIFEST = ROOT / f"local/{AUG_RUN}/manifest.json"
LOCAL = ROOT / f"local/{LEDGER_RUN}"
CKPT_ROOT = Path(f"/scratch/huterer_root/huterer0/jiamingp/saved_runs/{LEDGER_RUN}")
RESULTS = ROOT / f"results/{LEDGER_RUN}"
SAMPLES_REL = f"results/{LEDGER_RUN}/samples"

D4SHIFT = {"RandomDihedral2D": {"dims": [-2, -1], "p": 1.0}, "RandomRoll": {"size": 128, "dims": [-2, -1]}}
IDAUG = {"RandomDihedral2D": {"dims": [-2, -1], "p": 0.0}, "RandomRoll": {"size": 1, "dims": [-2, -1]}}
AUGS = {"noaug": None, "idaug": IDAUG, "d4shift": D4SHIFT}
FLAG_NOAUG = "--allow-no-augmentations"
FLAG_CKAUG = "--allow-checkpoint-augmentations"

# N -> (baseline run, stem, baseline final epoch, steps/epoch, every, seg1 epochs, seg2 epochs,
#       seg1 saves {tag: epoch}, seg2 saves {tag: epoch})
BASES = {
    1024: ("nf_cond_omsigc_hi_u128_d2p10_n1024_fresh200k", "nf_cond_omsigc_hi_u128_d2p10_n1024", 6249, 32, 7031,
           1563, 1562, {"ft025k": 7030, "ft050k": 7812}, {"ft100k": 9374}),
    4096: ("nf_cond_omsigc_hi_u128_d2p12_n4096_fresh200k", "nf_cond_omsigc_hi_u128_d2p12_n4096", 1562, 128, 1758,
           391, 390, {"ft025k": 1757, "ft050k": 1953}, {"ft100k": 2343}),
}
# Planned update counts (ledger entry), relative to the 200k baseline checkpoint.
PLANNED_UPDATES = {(1024, "ft025k"): 24992, (1024, "ft050k"): 50016, (1024, "ft100k"): 100000,
                   (4096, "ft025k"): 24960, (4096, "ft050k"): 50048, (4096, "ft100k"): 99968}
# (arm, N, segment) in train-task order.
TRAIN_ORDER = [("noaug", 1024, 1), ("noaug", 4096, 1), ("idaug", 1024, 1),
               ("noaug", 1024, 2), ("noaug", 4096, 2), ("idaug", 1024, 2),
               ("d4shift", 1024, 2), ("d4shift", 4096, 2)]
# Sample-row order: (arm, N).
ROW_ORDER = [("noaug", 1024), ("noaug", 4096), ("idaug", 1024), ("d4shift", 1024), ("d4shift", 4096)]


def saved_epochs(src_epoch: int, num_epochs: int, every: int) -> list[int]:
    """Epochs at which cosmodiff/optim.py:580 writes checkpoint-epoch-<epoch> on a resume from src_epoch."""
    start = src_epoch + 1
    last = start + num_epochs - 1
    return [e for e in range(start, start + num_epochs) if (e + 1) % every == 0 or e == last]


def training_run_name(stem: str, arm: str, seg: int) -> str:
    return f"{stem}_{arm}_ft{'000to050k' if seg == 1 else '050to100k'}_from200k"


def flags_for(arm: str, seg: int) -> list[str]:
    if arm == "noaug":
        return [FLAG_NOAUG]
    if arm == "idaug" and seg == 1:
        return []  # source has no augmentations.pkl, config has a pipeline: the wrapper's default mode
    return [FLAG_CKAUG]  # idaug seg2 (own seg1 checkpoint carries the identity pkl), d4shift (aug-arm pkl)


def plan() -> list[dict]:
    """Training segments in task order, with sources, schedules and planned saves (no file I/O except reads)."""
    base = {r["run_name"]: r for r in json.loads(BASE_MANIFEST.read_text())}
    aug_rows = {r["run_name"]: r for r in json.loads(AUG_MANIFEST.read_text())}
    segs = []
    by_key = {}
    for task, (arm, n, seg) in enumerate(TRAIN_ORDER):
        base_run, stem, src0, spe, every, ep1, ep2, saves1, saves2 = BASES[n]
        b = base[base_run]
        if int(b["checkpoint_epoch"]) != src0 or int(b["steps_per_epoch"]) != spe:
            raise SystemExit(f"{base_run}: baseline epoch/spe {b['checkpoint_epoch']}/{b['steps_per_epoch']} "
                             f"!= plan {src0}/{spe}")
        run = training_run_name(stem, arm, seg)
        out_dir = CKPT_ROOT / f"{run}_checkpoints"
        depends = None
        if seg == 1:
            src = Path(b["requested_checkpoint"])
            src_epoch, n_ep, saves = src0, ep1, saves1
            template_cfg = ROOT / b["config"]
        elif arm == "d4shift":
            ar = aug_rows[f"{stem}_d4shift_ft050k_from200k"]
            src = Path(ar["requested_checkpoint"])
            src_epoch, n_ep, saves = int(ar["checkpoint_epoch"]), ep2, saves2
            template_cfg = ROOT / ar["config"]
        else:
            prev = by_key[(arm, n, 1)]
            src = Path(prev["output_dir"]) / f"checkpoint-epoch-{prev['last_epoch']}"
            src_epoch, n_ep, saves = prev["last_epoch"], ep2, saves2
            template_cfg = ROOT / b["config"]
            depends = prev["training_run_name"]
        if src.name != f"checkpoint-epoch-{src_epoch}":
            raise SystemExit(f"{run}: source {src} does not end in checkpoint-epoch-{src_epoch}")
        saved = saved_epochs(src_epoch, n_ep, every)
        if saved != sorted(saves.values()):
            raise SystemExit(f"{run}: save rule gives {saved}, plan wants {sorted(saves.values())}")
        upd = {tag: (ep - src0) * spe for tag, ep in saves.items()}
        for tag, u in upd.items():
            if u != PLANNED_UPDATES[(n, tag)]:
                raise SystemExit(f"{run} {tag}: +{u} updates, ledger plans +{PLANNED_UPDATES[(n, tag)]}")
        s = {"train_task": task, "arm": arm, "dataset_size": n, "segment": seg, "training_run_name": run,
             "baseline_run_name": base_run, "stem": stem, "source_checkpoint": str(src), "source_epoch": src_epoch,
             "template_config": str(template_cfg.relative_to(ROOT)), "output_dir": str(out_dir),
             "num_epochs": n_ep, "checkpoint_every_n_epochs": every, "steps_per_epoch": spe,
             "saves": saves, "updates_after_baseline": upd, "last_epoch": saved[-1],
             "finetune_flags": flags_for(arm, seg), "depends_on_training_run": depends,
             "augmentations": AUGS[arm], "baseline_actual_updates": int(b["actual_updates"])}
        segs.append(s)
        by_key[(arm, n, seg)] = s
    return segs


def main() -> None:
    for p in (LOCAL, CKPT_ROOT, RESULTS):
        if p.exists():
            raise SystemExit(f"Refusing: {p} exists")
    segs = plan()
    base = {r["run_name"]: r for r in json.loads(BASE_MANIFEST.read_text())}
    for s in segs:
        src = Path(s["source_checkpoint"])
        if s["segment"] == 1 and (not src.is_dir() or (src / "augmentations.pkl").exists()):
            raise SystemExit(f"{src}: missing, or already has augmentations.pkl")
        if s["arm"] == "d4shift" and not ((src / "augmentations.pkl").is_file()
                                          and (src / "random_states_0.pkl").is_file()):
            raise SystemExit(f"{src}: aug-arm source lacks augmentations.pkl / random_states_0.pkl")
        if s["segment"] == 2 and s["arm"] != "d4shift" and src.exists():
            raise SystemExit(f"{src}: seg2 source already exists before seg1 ran")

    cfg_paths = {}
    for s in segs:
        cfg = yaml.safe_load((ROOT / s["template_config"]).read_text())
        if s["arm"] == "d4shift":
            if cfg.get("augmentations") != D4SHIFT:
                raise SystemExit(f"{s['template_config']}: augmentations != D4 + roll")
        elif "augmentations" in cfg:
            raise SystemExit(f"{s['template_config']}: baseline config already has augmentations")
        cfg["io"]["output_dir"] = s["output_dir"]
        cfg["train"]["num_epochs"] = s["num_epochs"]
        cfg["train"]["checkpoint_every_n_epochs"] = s["checkpoint_every_n_epochs"]
        if s["arm"] == "idaug":
            cfg["augmentations"] = IDAUG
        cfg_path = LOCAL / f"configs/{s['training_run_name']}.yaml"
        cfg_path.parent.mkdir(parents=True, exist_ok=True)
        with open(cfg_path, "x") as f:
            f.write(yaml.safe_dump(cfg, sort_keys=False))
        cfg_paths[s["training_run_name"]] = cfg_path

    rows = []
    for arm, n in ROW_ORDER:
        for s in [s for s in segs if s["arm"] == arm and s["dataset_size"] == n]:
            b = base[s["baseline_run_name"]]
            for tag, ep in sorted(s["saves"].items(), key=lambda kv: kv[1]):
                name = f"{s['stem']}_{arm}_{tag}_from200k"
                upd = s["updates_after_baseline"][tag]
                row = copy.deepcopy(b)
                row.update({
                    "run_name": name, "config": str(cfg_paths[s["training_run_name"]].relative_to(ROOT)),
                    "checkpoint_dir": s["output_dir"], "checkpoint_epoch": ep,
                    "requested_checkpoint": f"{s['output_dir']}/checkpoint-epoch-{ep}",
                    "sample_path": f"{SAMPLES_REL}/{name}_seed{{seed}}_dpm50_heldout_k{{k}}.npz",
                    "source_run_name": s["baseline_run_name"], "reuse_existing_checkpoint": True,
                    "initialization": f"resume:{s['source_checkpoint']}",
                    "resume_from_checkpoint": s["source_checkpoint"], "training_run_name": s["training_run_name"],
                    "train_task": s["train_task"], "segment": s["segment"], "arm": arm,
                    "depends_on_training_run": s["depends_on_training_run"],
                    "finetune_flags": s["finetune_flags"], "finetune_updates": upd,
                    "actual_updates": int(b["actual_updates"]) + upd, "target_updates": int(b["actual_updates"]) + upd,
                    "epochs": s["num_epochs"], "augmentations": s["augmentations"],
                    "baseline_run_name": s["baseline_run_name"],
                    "note": b["note"] + f" Control fine-tune arm {arm}, segment {s['segment']}: +{upd} updates after "
                                        f"the 200k baseline (tag {tag}); ledger {LEDGER_RUN}.",
                })
                rows.append(row)
    if len(rows) != 11:
        raise SystemExit(f"expected 11 manifest rows, got {len(rows)}")
    with open(LOCAL / "manifest.json", "x") as f:
        f.write(json.dumps(rows, indent=2) + "\n")
    print("Wrote " + ", ".join(str(p) for p in cfg_paths.values()) + f" and {LOCAL / 'manifest.json'}")
    for r in rows:
        print(f"  row {r['run_name']} <- task {r['train_task']} @ epoch {r['checkpoint_epoch']} "
              f"(+{r['finetune_updates']}, total {r['actual_updates']})")


if __name__ == "__main__":
    main()
