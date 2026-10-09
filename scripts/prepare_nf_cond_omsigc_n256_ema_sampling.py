#!/usr/bin/env python
"""Configs + manifest for EMA sampling of the existing N = 256 (Omega_m, sigma_8) baseline (ledger
nf_cond_omsigc_n256_ema_sampling). Sampling only; no training.

Baseline checkpoint: nf_cond_omsigc_hi_u128_d2p08_n256_fresh200k, checkpoint-epoch-24999. It tracked post-hoc
EMA with ema_sigma_rels [0.02, 0.1] (ema/0.<step>.pt = 0.02, ema/1.<step>.pt = 0.1, one pair per checkpoint).
Existing samples used the raw weights (generate.ema_sigma_rel: null).

Each config is the baseline yaml with only generate.ema_sigma_rel changed. cosmodiff_sample.py reads that key
(CLI --ema_sigma_rel overrides it) and calls cosmodiff.optim.synthesize_ema_from_checkpoints over every
checkpoint-epoch-*/ema/*.pt in io.output_dir, which stays the baseline checkpoint directory (read only).
Samples get new paths (sample labels dpm50_ema0p10, dpm50_ema0p02).
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import yaml

PROJECT_DIR = Path(__file__).resolve().parents[1]
BASE_SWEEP = "nf_conditional_omsig_continuous_200k"
BASE_N = 256
STUDY = "nf_cond_omsigc_n256_ema_sampling"
EMA = [(0.10, "ema0p10"), (0.02, "ema0p02")]


def write_once(path: Path, text: str) -> None:
    if path.exists():
        if path.read_text() != text:
            raise SystemExit(f"{path} exists with different content; refusing to overwrite")
        return
    path.write_text(text)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--print-runs", action="store_true")
    args = parser.parse_args()

    base_row = next(r for r in json.loads((PROJECT_DIR / "local" / BASE_SWEEP / "manifest.json").read_text())
                    if r["dataset_size"] == BASE_N)
    names = [f"{base_row['run_name']}_{tag}" for _, tag in EMA]
    if args.print_runs:
        print("\n".join(names))
        return

    base_text = (PROJECT_DIR / base_row["config"]).read_text()
    base_cfg = yaml.safe_load(base_text)
    assert yaml.safe_dump(base_cfg, sort_keys=False) == base_text, "baseline yaml does not round-trip"
    assert base_cfg["generate"]["ema_sigma_rel"] is None
    assert base_cfg["train"]["ema_sigma_rels"] == [0.02, 0.1]
    ckpt = Path(base_row["requested_checkpoint"])
    for sigma_rel, _ in EMA:
        prof = base_cfg["train"]["ema_sigma_rels"].index(sigma_rel)
        assert list((ckpt / "ema").glob(f"{prof}.*.pt")), f"no EMA snapshot for profile {prof} in {ckpt / 'ema'}"
    local = PROJECT_DIR / "local" / STUDY
    (local / "configs").mkdir(parents=True, exist_ok=True)
    rows = []
    for name, (sigma_rel, tag) in zip(names, EMA):
        cfg = copy.deepcopy(base_cfg)
        cfg["generate"]["ema_sigma_rel"] = sigma_rel
        config_path = local / "configs" / f"{name}.yaml"
        write_once(config_path, yaml.safe_dump(cfg, sort_keys=False))
        row = copy.deepcopy(base_row)
        row.update({
            "run_name": name, "ledger_run": STUDY, "config": str(config_path.relative_to(PROJECT_DIR)),
            "ema_sigma_rel": sigma_rel, "sample_label": f"dpm50_{tag}",
            "sample_path": f"results/{STUDY}/samples/{name}_seed{{seed}}_dpm50_{tag}_heldout_k{{k}}.npz",
            "baseline_run_name": base_row["run_name"],
            "note": (f"Sampling-only: baseline checkpoint {ckpt.name} with post-hoc EMA sigma_rel={sigma_rel} "
                     "synthesized from all checkpoint-epoch-*/ema snapshots (cosmodiff synthesize_ema_from_checkpoints); "
                     "held-out CAMELS simulations 900-931."),
        })
        rows.append(row)
    write_once(local / "manifest.json", json.dumps(rows, indent=1))
    print(f"wrote {len(rows)} configs and {local / 'manifest.json'}")


if __name__ == "__main__":
    main()
