#!/usr/bin/env python
"""Build conditioning-label files for the theta-interpolation superposition test.

Two label sets for one conditional run (default N=64):

* ``theta_interp``: straight lines in raw parameter space between pairs of
  mutually nearest training cosmologies, ``s in {0, .25, .5, .75, 1}``,
  ``--seeds`` draws per point (pair-major, then s, then seed).
* ``train_theta``: the run's own training cosmologies, ``--seeds`` draws each
  (cosmology-major).

Labels are normalized with the frozen ``param_norm_stats.json`` used for training.
The script writes only under ``local/<sweep>/labels/`` and never touches results,
configs, or the manifest. No sampling happens here.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from simdiff_eval.conditional_unet_superposition import (  # noqa: E402
    interpolation_labels, mutually_nearest_pairs,
)

SWEEP = "nf_conditional_bias_fresh_full_sweep_200k"
SCALES = "local/nf_conditional_bias_probe/heldout/param_norm_stats.json"
S_VALUES = np.array([0.0, 0.25, 0.5, 0.75, 1.0])


def resolve(project: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else project / path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-dir", default=".")
    parser.add_argument("--sweep", default=SWEEP)
    parser.add_argument("--dataset-size", type=int, default=64)
    parser.add_argument("--pairs", type=int, default=4)
    parser.add_argument("--seeds", type=int, default=4)
    args = parser.parse_args()

    project = Path(args.project_dir).resolve()
    manifest = json.loads((project / "local" / args.sweep / "manifest.json").read_text())
    rows = [r for r in manifest if int(r["dataset_size"]) == args.dataset_size]
    if len(rows) != 1:
        raise SystemExit(f"expected one manifest row for N={args.dataset_size}, found {len(rows)}")
    row = rows[0]
    stats = json.loads(resolve(project, SCALES).read_text())
    mean = np.asarray(stats["mean"], dtype=np.float64)
    std = np.asarray(stats["std"], dtype=np.float64)
    theta_raw = np.load(resolve(project, row["train_raw_params_path"]), allow_pickle=False).astype(np.float64)
    pairs_csv = pd.read_csv(resolve(project, row["selected_pairs_path"]))
    ids = pairs_csv.simulation_index.to_numpy(np.int64)
    if theta_raw.shape != (len(ids), 6) or not np.array_equal(pairs_csv.row, np.arange(len(ids))):
        raise SystemExit("training labels and selected slices disagree")
    # sanity: saved normalized training labels must equal (raw-mean)/std
    saved_norm = np.load(resolve(project, row["train_label_path"]), allow_pickle=False)
    if not np.allclose(saved_norm, (theta_raw - mean) / std, rtol=0, atol=1e-5):
        raise SystemExit("frozen parameter normalization does not reproduce the saved training labels")

    out_dir = project / "local" / args.sweep / "labels"
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{row['run_name']}"

    pairs = mutually_nearest_pairs(theta_raw, ids, std, args.pairs)
    labels_raw, index = interpolation_labels(theta_raw, pairs, S_VALUES, args.seeds)
    labels_norm = ((labels_raw - mean) / std).astype(np.float32)
    np.save(out_dir / f"{stem}_theta_interp_raw.npy", labels_raw)
    np.save(out_dir / f"{stem}_theta_interp_norm.npy", labels_norm)
    (out_dir / f"{stem}_theta_interp_meta.json").write_text(json.dumps({
        "run_name": row["run_name"], "dataset_size": int(row["dataset_size"]),
        "pairs": pairs, "s_values": S_VALUES.tolist(), "seeds": args.seeds,
        "n_samples": int(len(labels_raw)), "order": "pair-major, then s, then seed",
        "index": index, "param_norm_stats": SCALES,
        "prepared_image_path": row["prepared_image_path"],
        "selected_pairs_path": row["selected_pairs_path"],
    }, indent=2) + "\n")

    unique, first = np.unique(ids, return_index=True)
    train_theta = theta_raw[first]
    train_labels_raw = np.repeat(train_theta, args.seeds, axis=0).astype(np.float32)
    train_labels_norm = ((train_labels_raw - mean) / std).astype(np.float32)
    np.save(out_dir / f"{stem}_train_theta_raw.npy", train_labels_raw)
    np.save(out_dir / f"{stem}_train_theta_norm.npy", train_labels_norm)
    (out_dir / f"{stem}_train_theta_meta.json").write_text(json.dumps({
        "run_name": row["run_name"], "dataset_size": int(row["dataset_size"]),
        "simulation_ids": unique.tolist(), "training_rows": first.tolist(),
        "seeds": args.seeds, "n_samples": int(len(train_labels_raw)),
        "order": "cosmology-major, then seed", "param_norm_stats": SCALES,
        "prepared_image_path": row["prepared_image_path"],
    }, indent=2) + "\n")

    print(f"theta_interp: {len(pairs)} pairs x {len(S_VALUES)} s x {args.seeds} seeds = {len(labels_raw)} labels")
    for p in pairs:
        print(f"  pair sims {p['sim_a']}-{p['sim_b']} rows {p['row_a']}-{p['row_b']} distance {p['distance']:.3f}")
    print(f"train_theta: {len(unique)} cosmologies x {args.seeds} seeds = {len(train_labels_raw)} labels")
    print(f"wrote under {out_dir}")


if __name__ == "__main__":
    main()
