#!/usr/bin/env python
"""Prepare (no GPU, no training) ledger nf_cond_omsigc_cfg: CFG with a flagged null on the continuous
(Omega_m, sigma_8) UNet-128 sweep, N = 64, 256, 1024, 4096, 32768.

For every baseline row of local/nf_conditional_omsig_continuous_200k/manifest.json writes, under
local/nf_cond_omsigc_cfg_200k/ (refuses any existing target):

* labels/<run>_train_labels_flag.npy      baseline train labels + a third column of zeros (flag 0)
* heldout/heldout_params_norm_omsig_flag_k64.npy   baseline held-out labels + a zero column
* configs/<run>.yaml   baseline yaml with exactly these changes:
      io.output_dir                 -> /scratch/.../saved_runs/nf_cond_omsigc_cfg_200k/<run>_checkpoints
      data.label_path               -> the flagged label file
      model.kwargs.encoder_hid_dim  2 -> 3
      train.cfg_dropout             0.0 -> 0.1
      train.cfg_null                (new) flag
      generate.continuous_labels    -> the flagged held-out label file
* manifest.json        one row per N, in training-array order N = 256, 32768, 1024, 64, 4096;
                       sample_path carries {guidance}
* sample_tasks.json    sampling array: N-major (same order) x guidance grid [None, 0, 0.25, 0.5, 0.75, 1, 1.5, 2, 3]
* eval_manifests/<run>.json   one row per guidance value with the concrete sample path (no {guidance}),
                       run_name <run>__<guidance_label>, for scripts/evaluate_run.py --mode conditional, which
                       always looks up the unguided path (output_path_for(..., None)).
"""
from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from sample_nf_conditional_bias_probe import guidance_label, output_path_for  # noqa: E402

BASE_MANIFEST = ROOT / "local/nf_conditional_omsig_continuous_200k/manifest.json"
LOCAL = ROOT / "local/nf_cond_omsigc_cfg_200k"
CKPT_ROOT = Path("/scratch/huterer_root/huterer0/jiamingp/saved_runs/nf_cond_omsigc_cfg_200k")
SAMPLES_REL = "results/nf_cond_omsigc_cfg_200k/samples"
EVAL_REL = "results/nf_cond_omsigc_cfg_200k/evaluate_run_v1"
TRAIN_ORDER = [256, 32768, 1024, 64, 4096]
GUIDANCE = [None, 0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0]
CFG_DROPOUT = 0.1
SEED, K = 123, 64


def new_run_name(base: str) -> str:
    assert base.endswith("_fresh200k"), base
    return base[: -len("_fresh200k")] + "_cfg10flag_fresh200k"


def with_flag(a: np.ndarray) -> np.ndarray:
    a = np.asarray(a, dtype=np.float32)
    if a.ndim != 2 or a.shape[1] != 2:
        raise SystemExit(f"expected (n, 2) labels, got {a.shape}")
    return np.concatenate([a, np.zeros((len(a), 1), np.float32)], axis=1)


def write_new(path: Path, write) -> None:
    if path.exists():
        raise SystemExit(f"Refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    write(path)


def main() -> None:
    if LOCAL.exists():
        raise SystemExit(f"Refusing: {LOCAL} exists")
    for p in (ROOT / SAMPLES_REL, ROOT / EVAL_REL, CKPT_ROOT):
        if p.exists():
            raise SystemExit(f"Refusing: {p} exists")
    base_rows = {int(r["dataset_size"]): r for r in json.loads(BASE_MANIFEST.read_text())}
    if sorted(base_rows) != sorted(TRAIN_ORDER):
        raise SystemExit(f"baseline sizes {sorted(base_rows)} != {sorted(TRAIN_ORDER)}")

    held_paths = {r["heldout_sample_params_norm_path"] for r in base_rows.values()}
    if len(held_paths) != 1:
        raise SystemExit(f"baseline rows use different held-out label files: {held_paths}")
    held_new = LOCAL / "heldout/heldout_params_norm_omsig_flag_k64.npy"
    write_new(held_new, lambda p: np.save(p, with_flag(np.load(held_paths.pop()))))

    rows, tasks, evals = [], [], {}
    for n in TRAIN_ORDER:
        b = base_rows[n]
        run = new_run_name(b["run_name"])
        lab_new = LOCAL / f"labels/{run}_train_labels_flag.npy"
        write_new(lab_new, lambda p, src=b["train_label_path"]: np.save(p, with_flag(np.load(src))))
        cfg = yaml.safe_load((ROOT / b["config"]).read_text())
        out_dir = CKPT_ROOT / f"{run}_checkpoints"
        cfg["io"]["output_dir"] = str(out_dir)
        cfg["data"]["label_path"] = str(lab_new)
        if cfg["model"]["kwargs"].get("encoder_hid_dim") != 2 or cfg["train"].get("cfg_dropout") != 0.0:
            raise SystemExit(f"{b['config']}: unexpected baseline encoder_hid_dim / cfg_dropout")
        if "cfg_null" in cfg["train"]:
            raise SystemExit(f"{b['config']}: baseline already has train.cfg_null")
        cfg["model"]["kwargs"]["encoder_hid_dim"] = 3
        cfg["train"]["cfg_dropout"] = CFG_DROPOUT
        cfg["train"]["cfg_null"] = "flag"
        cfg["generate"]["continuous_labels"] = str(held_new)
        cfg_path = LOCAL / f"configs/{run}.yaml"
        write_new(cfg_path, lambda p, c=cfg: p.write_text(yaml.safe_dump(c, sort_keys=False)))

        row = copy.deepcopy(b)
        row.update({
            "run_name": run,
            "config": str(cfg_path.relative_to(ROOT)),
            "checkpoint_dir": str(out_dir),
            "requested_checkpoint": f"{out_dir}/checkpoint-epoch-{int(b['checkpoint_epoch'])}",
            "sample_path": f"{SAMPLES_REL}/{run}_seed{{seed}}_dpm50_heldout_k{{k}}_{{guidance}}.npz",
            "train_label_path": str(lab_new),
            "heldout_sample_params_norm_path": str(held_new),
            "baseline_run_name": b["run_name"],
            "cfg_dropout": CFG_DROPOUT,
            "cfg_null": "flag",
            "label_dim": 3,
            "label_columns": ["Omega_m_z", "sigma_8_z", "null_flag"],
            "guidance_grid": GUIDANCE,
            "note": b["note"] + " CFG: cfg_dropout 0.1, null label [0, 0, 1] (flag column), ledger nf_cond_omsigc_cfg.",
        })
        rows.append(row)
        ev = []
        for w in GUIDANCE:
            path = output_path_for(ROOT, row, SEED, K, w)
            if path.exists():
                raise SystemExit(f"Refusing: sample file exists {path}")
            tasks.append({"task": len(tasks), "run_name": run, "dataset_size": n, "guidance_scale": w,
                          "guidance_label": guidance_label(w), "sample_path": str(path)})
            er = copy.deepcopy(row)
            er.update({"run_name": f"{run}__{guidance_label(w)}", "cfg_run_name": run,
                       "sample_path": str(path.relative_to(ROOT)), "guidance_scale": w,
                       "guidance_label": guidance_label(w)})
            ev.append(er)
        evals[run] = ev
    write_new(LOCAL / "manifest.json", lambda p: p.write_text(json.dumps(rows, indent=2) + "\n"))
    write_new(LOCAL / "sample_tasks.json", lambda p: p.write_text(json.dumps(tasks, indent=2) + "\n"))
    for run, ev in evals.items():
        write_new(LOCAL / f"eval_manifests/{run}.json", lambda p, e=ev: p.write_text(json.dumps(e, indent=2) + "\n"))
    print(f"Wrote {LOCAL}: {len(rows)} runs, {len(tasks)} sampling tasks, {len(evals)} eval manifests")


if __name__ == "__main__":
    main()
