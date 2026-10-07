#!/usr/bin/env python
"""Frozen VGG probe applied to the training maps of the small-N conditional models.

Memorization-regime diagnostic: generated maps at N <= 1024 are near-copies of
training maps, so the probe reading of a generated copy should be compared with
the probe reading of the training map it copies, not only with its label.
Writes one row per (model, N, training map): label Omega_m and probe Omega_m.
CPU is enough (the sweep evaluation sbatch also runs the probe on CPU).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))
sys.path.insert(0, str(PROJECT_DIR / "scripts"))

from simdiff_eval.io import _normalize_reference_slices  # noqa: E402

SWEEPS = {
    "Six-parameter continuous": "nf_conditional_bias_fresh_full_sweep_200k",
    "Continuous Ωm–σ8": "nf_conditional_omsig_continuous_200k",
    "36-class Ωm–σ8": "nf_conditional_omsig_class36_200k",
}


def train_images_path(row: dict) -> Path:
    p = row["prepared_image_path"]
    return Path(p if p.endswith(".npy") else p + "_images.npy")


def model_space_train_maps(row: dict) -> np.ndarray:
    """Training maps exactly as the model saw them (config log + tanh normalization)."""
    data_cfg = yaml.safe_load((PROJECT_DIR / row["config"]).read_text())["data"]
    raw = np.asarray(np.load(train_images_path(row), mmap_mode="r")[:, 0], dtype=np.float32)
    kw = data_cfg["norm_kwargs"]
    return _normalize_reference_slices(raw, data_cfg, kw["center"], kw["xmax"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, required=True, help="Fresh directory; must not exist")
    parser.add_argument("--dataset-size", type=int, action="append", help="Default: 64, 256, 1024")
    parser.add_argument("--vgg-encoder", type=Path, default=PROJECT_DIR / "results/nf_conditional_bias_probe/encoder/vgg_mlp_encoder.npz")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--limit", type=int, default=None, help="Smoke test: first maps only")
    args = parser.parse_args()

    from simdiff_eval.torch_compat import install_torch_backend_compat
    install_torch_backend_compat(entry_point="probe_train_maps_memorization")
    from evaluate_nf_conditional_bias_probe import load_vgg_encoder

    sizes = args.dataset_size or [64, 256, 1024]
    args.out_dir.mkdir(parents=True, exist_ok=False)
    encoder = load_vgg_encoder(PROJECT_DIR, args.vgg_encoder, args.device)
    rows, cache = [], {}  # all three sweeps share the training maps at a given N; probe each distinct set once
    for label, sweep in SWEEPS.items():
        manifest = {r["dataset_size"]: r for r in json.loads((PROJECT_DIR / "local" / sweep / "manifest.json").read_text())}
        for n in sizes:
            row = manifest[n]
            maps = model_space_train_maps(row)[: args.limit]
            truth = np.load(row["train_raw_params_path"])[: len(maps), 0]
            key = hashlib.sha256(np.ascontiguousarray(maps).tobytes()).hexdigest()
            if key not in cache:
                cache[key] = encoder.norm_to_raw(encoder.predict_norm(maps, batch_size=args.batch_size))[:, 0]
            pred = cache[key]
            sel = pd.read_csv(row["selected_pairs_path"]).iloc[: len(maps)]
            rows.append(pd.DataFrame({"model": label, "sweep": sweep, "run_name": row["run_name"], "dataset_size": n,
                                      "train_index": np.arange(len(maps)), "maps_sha256": key, "simulation_index": sel.simulation_index.to_numpy(),
                                      "z_index": sel.z_index.to_numpy(), "Omega_m_label": truth, "Omega_m_probe": pred}))
            d = rows[-1]
            print(f"{label} N={n}: probe-label median {np.median(d.Omega_m_probe - d.Omega_m_label):+.4f} on {len(d)} maps", flush=True)
    out = args.out_dir / "probe_on_train_maps.csv"
    pd.concat(rows, ignore_index=True).to_csv(out, index=False)
    (args.out_dir / "metadata.json").write_text(json.dumps({"vgg_encoder": str(args.vgg_encoder), "sizes": sizes,
                                                            "limit": args.limit, "device": args.device}, indent=1))
    print("Wrote", out)


if __name__ == "__main__":
    main()
