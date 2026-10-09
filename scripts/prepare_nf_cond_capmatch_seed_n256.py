#!/usr/bin/env python
"""Configs + manifests for two N = 256 controls of the continuous (Omega_m, sigma_8) baseline.

Baseline: nf_cond_omsigc_hi_u128_d2p08_n256_fresh200k (labels via single-token cross-attention, seed 123).
Both runs reuse the baseline's training maps, labels, normalization, schedule, optimizer, 200k updates,
held-out set and sampling, and change only what their row says:

  A  ledger nf_cond_omsigc_n256_seed124 -> local/nf_cond_omsigc_n256_seed124/
     baseline yaml unchanged except io.output_dir; training seed 124 (set by the launcher's sitecustomize
     from TRAINING_SEED, not by the yaml; recorded in the manifest row).
  B  ledger nf_condmech_n256_capmatch_tembss -> local/nf_condmech_n256_capmatch/
     baseline model (CrossAttn down/up blocks, encoder_hid_dim 2, cross_attention_dim 32, default
     UNetMidBlock2DCrossAttn) PLUS class_embed_type="projection" (2-d labels -> timestep embedding) and
     resnet_time_scale_shift="scale_shift". Labels reach both routes: cross-attention as before, and
     class_labels via simdiff_eval.timeemb_conditioning (installed from sitecustomize). That patch rebuilds the
     middle block only when mid_block_type == "UNetMidBlock2D", so this model's cross-attention middle block is
     untouched. Training seed 123.

Each run gets its own local/<study>/{configs/<run>.yaml, manifest.json}, scratch checkpoint dir and
results/<study>/ outputs. Refuses to change existing files.
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
SCRATCH_ROOT = Path("/scratch/huterer_root/huterer0/jiamingp/saved_runs")

# (run_name, study dir == ledger-facing directory, ledger run, training seed, arch tag, description, model.kwargs additions)
RUNS = [
    ("nf_cond_omsigc_hi_u128_d2p08_n256_seed124_fresh200k", "nf_cond_omsigc_n256_seed124", "nf_cond_omsigc_n256_seed124",
     124, "u128", "baseline single-token cross-attention, training seed 124", {}),
    ("nf_condmech_n256_capmatch_tembss", "nf_condmech_n256_capmatch", "nf_condmech_n256_capmatch_tembss",
     123, "u128_capmatch_tembss",
     "baseline cross-attention + time-embedding projection with scale-shift (both label routes)",
     {"class_embed_type": "projection", "projection_class_embeddings_input_dim": 2,
      "resnet_time_scale_shift": "scale_shift"}),
]


def write_once(path: Path, text: str) -> None:
    if path.exists():
        if path.read_text() != text:
            raise SystemExit(f"{path} exists with different content; refusing to overwrite")
        return
    path.write_text(text)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--print-train-runs", action="store_true", help="one line per run: run_name study seed ledger_run")
    args = parser.parse_args()

    if args.print_train_runs:
        print("\n".join(f"{r[0]} {r[1]} {r[3]} {r[2]}" for r in RUNS))
        return

    base_row = next(r for r in json.loads((PROJECT_DIR / "local" / BASE_SWEEP / "manifest.json").read_text())
                    if r["dataset_size"] == BASE_N)
    base_text = (PROJECT_DIR / base_row["config"]).read_text()
    base_cfg = yaml.safe_load(base_text)
    assert yaml.safe_dump(base_cfg, sort_keys=False) == base_text, "baseline yaml does not round-trip"
    kw = base_cfg["model"]["kwargs"]
    assert base_cfg["model"]["class"] == "UNet2DConditionModel" and kw["encoder_hid_dim"] == 2
    assert kw["cross_attention_dim"] == 32 and "mid_block_type" not in kw and "class_embed_type" not in kw
    for name, study, ledger_run, seed, arch, description, model_extra in RUNS:
        local = PROJECT_DIR / "local" / study
        (local / "configs").mkdir(parents=True, exist_ok=True)
        cfg = copy.deepcopy(base_cfg)
        cfg["model"]["kwargs"].update(model_extra)
        cfg["io"]["output_dir"] = str(SCRATCH_ROOT / study / f"{name}_checkpoints")
        config_path = local / "configs" / f"{name}.yaml"
        write_once(config_path, yaml.safe_dump(cfg, sort_keys=False))
        row = copy.deepcopy(base_row)
        row.update({
            "run_name": name, "ledger_run": ledger_run, "training_seed": seed, "arch": arch,
            "config": str(config_path.relative_to(PROJECT_DIR)),
            "checkpoint_dir": cfg["io"]["output_dir"],
            "requested_checkpoint": f"{cfg['io']['output_dir']}/checkpoint-epoch-{base_row['checkpoint_epoch']}",
            "sample_path": f"results/{study}/samples/{name}_seed{{seed}}_dpm50_heldout_k{{k}}.npz",
            "baseline_run_name": base_row["run_name"], "conditioning_mechanism": description,
            "note": (f"N={BASE_N} control; same data/labels/held-out set (CAMELS 900-931) as {base_row['run_name']}. "
                     "Training seed set by the launcher (TRAINING_SEED -> sitecustomize), recorded here."),
        })
        write_once(local / "manifest.json", json.dumps([row], indent=1))
        print(f"wrote {config_path.relative_to(PROJECT_DIR)} and {(local / 'manifest.json').relative_to(PROJECT_DIR)}")


if __name__ == "__main__":
    main()
