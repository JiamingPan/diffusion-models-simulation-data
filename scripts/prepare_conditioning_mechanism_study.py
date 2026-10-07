#!/usr/bin/env python
"""Configs for the conditioning-mechanism study at N = 256 (continuous Omega_m, sigma_8).

Baseline: nf_cond_omsigc_hi_u128_d2p08_n256_fresh200k (labels via single-token cross-attention at 32x32 only).
Every run reuses the baseline's training maps, labels, normalization, schedule, optimizer, seed, 200k updates,
held-out set and sampling, and changes only what its row says:

  temb          labels -> MLP added to the timestep embedding (class_embed_type="projection"), reaching every
                ResNet block; self-attention at the bottom level as in the unconditional u128 model; no cross-attn.
  tembss        temb + resnet_time_scale_shift="scale_shift" (AdaGN / FiLM in every block).
  tembss_nosnr  tembss with min-SNR loss weighting off (plain v-prediction MSE).

Labels reach class_labels through simdiff_eval.timeemb_conditioning (installed from sitecustomize by the sbatch files).
That patch also rebuilds the middle block as ResBlock + self-attention + ResBlock (diffusers' UNet2DConditionModel
otherwise builds mid_block_type="UNetMidBlock2D" as one ResBlock without attention).
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
STUDY = "nf_conditioning_mechanism_n256"
SCRATCH = Path("/scratch/huterer_root/huterer0/jiamingp/saved_runs") / STUDY

TEMB_MODEL = {
    "sample_size": 128,
    "in_channels": 1,
    "out_channels": 1,
    "layers_per_block": 2,
    "block_out_channels": [32, 64, 128],
    "down_block_types": ["DownBlock2D", "DownBlock2D", "AttnDownBlock2D"],
    "up_block_types": ["AttnUpBlock2D", "UpBlock2D", "UpBlock2D"],
    "mid_block_type": "UNetMidBlock2D",
    "norm_num_groups": 32,
    "class_embed_type": "projection",
    "projection_class_embeddings_input_dim": 2,
}
RUNS = [
    ("temb", "time-embedding addition (Diffusion-HMC style)", {}, {}),
    ("tembss", "time-embedding + scale-shift (AdaGN/FiLM)", {"resnet_time_scale_shift": "scale_shift"}, {}),
    ("tembss_nosnr", "tembss with min-SNR weighting off", {"resnet_time_scale_shift": "scale_shift"}, {"min_snr_gamma": None}),
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--print-train-runs", action="store_true")
    args = parser.parse_args()

    names = [f"nf_condmech_n256_{tag}" for tag, *_ in RUNS]
    if args.print_train_runs:
        print("\n".join(names))
        return

    base_row = next(r for r in json.loads((PROJECT_DIR / "local" / BASE_SWEEP / "manifest.json").read_text())
                    if r["dataset_size"] == BASE_N)
    base_cfg = yaml.safe_load((PROJECT_DIR / base_row["config"]).read_text())
    assert base_cfg["model"]["class"] == "UNet2DConditionModel" and base_cfg["model"]["kwargs"]["encoder_hid_dim"] == 2
    local = PROJECT_DIR / "local" / STUDY
    (local / "configs").mkdir(parents=True, exist_ok=True)
    rows = []
    for name, (tag, description, model_extra, train_extra) in zip(names, RUNS):
        cfg = copy.deepcopy(base_cfg)
        cfg["model"] = {"class": "UNet2DConditionModel", "kwargs": {**TEMB_MODEL, **model_extra}}
        train_key = next(k for k, v in cfg.items() if isinstance(v, dict) and "min_snr_gamma" in v)
        cfg[train_key].update(train_extra)
        out_key = next(k for k, v in cfg.items() if isinstance(v, dict) and "output_dir" in v)
        cfg[out_key]["output_dir"] = str(SCRATCH / f"{name}_checkpoints")
        config_path = local / "configs" / f"{name}.yaml"
        if config_path.exists() and yaml.safe_load(config_path.read_text()) != cfg:
            raise SystemExit(f"{config_path} exists with different content; refusing to overwrite")
        config_path.write_text(yaml.safe_dump(cfg, sort_keys=False))
        row = copy.deepcopy(base_row)
        row.update({
            "run_name": name, "arch": f"u128_{tag}", "config": str(config_path.relative_to(PROJECT_DIR)),
            "checkpoint_dir": cfg[out_key]["output_dir"],
            "requested_checkpoint": f"{cfg[out_key]['output_dir']}/checkpoint-epoch-{base_row['checkpoint_epoch']}",
            "sample_path": f"results/{STUDY}/samples/{name}_seed{{seed}}_dpm50_heldout_k{{k}}.npz",
            "baseline_run_name": base_row["run_name"], "conditioning_mechanism": description,
            "note": (f"Conditioning-mechanism study at N={BASE_N}; same data/labels/held-out set as {base_row['run_name']}. "
                     "Middle block rebuilt as ResBlock + self-attention + ResBlock by simdiff_eval.timeemb_conditioning."),
        })
        rows.append(row)
    (local / "manifest.json").write_text(json.dumps(rows, indent=1))
    print(f"wrote {len(rows)} configs and {local / 'manifest.json'}")


if __name__ == "__main__":
    main()
