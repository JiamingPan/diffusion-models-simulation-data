#!/usr/bin/env python
"""Sample a conditional run at custom continuous labels (theta-interpolation test).

Mirrors ``sample_nf_conditional_bias_probe.py`` exactly (same checkpoint epoch,
scheduler, steps, seed, batch size, no guidance) but overrides the label file and
sample count on the ``cosmodiff_sample.py`` command line, which takes precedence
over the config's ``generate:`` block. Output goes to
``results/<sweep>/samples_theta_interp/<run>_<set>_seed<seed>_dpm50.npz`` and is
annotated with the label metadata so the analysis can verify provenance.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import yaml

SWEEP = "nf_conditional_bias_fresh_full_sweep_200k"


def resolve(project: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else project / path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-dir", default=".")
    parser.add_argument("--sweep", default=SWEEP)
    parser.add_argument("--dataset-size", type=int, default=64)
    parser.add_argument("--label-set", choices=("theta_interp", "train_theta"), required=True)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-steps", type=int, default=50)
    parser.add_argument("--scheduler", default="DPMSolverMultistepScheduler")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--cosmodiff-sample", default=None)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    project = Path(args.project_dir).resolve()
    manifest = json.loads((project / "local" / args.sweep / "manifest.json").read_text())
    rows = [r for r in manifest if int(r["dataset_size"]) == args.dataset_size]
    if len(rows) != 1:
        raise SystemExit(f"expected one manifest row for N={args.dataset_size}, found {len(rows)}")
    row = rows[0]
    config_path = resolve(project, row["config"])
    config = yaml.safe_load(config_path.read_text())
    if config["train"].get("conditioning") != "continuous":
        raise SystemExit("this test is only defined for continuous conditioning")

    label_dir = project / "local" / args.sweep / "labels"
    stem = f"{row['run_name']}_{args.label_set}"
    labels_norm = label_dir / f"{stem}_norm.npy"
    meta_path = label_dir / f"{stem}_meta.json"
    if not labels_norm.is_file() or not meta_path.is_file():
        raise SystemExit(f"run prepare_nf_conditional_theta_interp_labels.py first; missing {labels_norm}")
    meta = json.loads(meta_path.read_text())
    n_samples = int(np.load(labels_norm, allow_pickle=False).shape[0])
    if n_samples != int(meta["n_samples"]):
        raise SystemExit("label file and metadata disagree on sample count")

    cosmodiff_dir = Path(os.environ.get("COSMODIFF_DIR", "/home/jiamingp/Diffusion_model/cosmo_diffusion_main"))
    sampler = Path(args.cosmodiff_sample) if args.cosmodiff_sample else cosmodiff_dir / "scripts" / "cosmodiff_sample.py"
    if not sampler.exists():
        raise FileNotFoundError(sampler)

    out_dir = project / "results" / args.sweep / "samples_theta_interp"
    out_dir.mkdir(parents=True, exist_ok=True)
    output = out_dir / f"{stem}_seed{args.seed}_dpm{args.num_steps}.npz"
    if output.exists() and not args.overwrite:
        print(f"Skipping existing {output}")
        return

    cmd = [sys.executable, str(sampler), "--config", str(config_path), "--filepath", str(output),
           "--n_samples", str(n_samples), "--batch_size", str(args.batch_size),
           "--scheduler", args.scheduler, "--num_steps", str(args.num_steps),
           "--seed", str(args.seed), "--device", args.device,
           "--conditioning", "continuous", "--continuous_labels", str(labels_norm), "--verbose"]
    if row.get("checkpoint_epoch") is not None:
        cmd += ["--checkpoint_epoch", str(int(row["checkpoint_epoch"]))]
    print("Running:", " ".join(cmd), flush=True)
    rc = subprocess.call(cmd)
    if rc != 0:
        raise SystemExit(rc)

    with np.load(output, allow_pickle=True) as data:
        payload = {key: data[key] for key in data.files}
    samples = np.asarray(payload["samples"])
    if len(samples) != n_samples:
        raise SystemExit(f"sampler wrote {len(samples)} samples, expected {n_samples}")
    payload.update({
        "run_name": np.array(row["run_name"]), "dataset_size": np.array(int(row["dataset_size"])),
        "label_set": np.array(args.label_set), "seed": np.array(int(args.seed)),
        "guidance_label": np.array("noguidance"), "num_steps": np.array(int(args.num_steps)),
        "scheduler": np.array(args.scheduler),
        "checkpoint_epoch": np.array(int(row["checkpoint_epoch"]) if row.get("checkpoint_epoch") is not None else -1),
        "labels_norm": np.load(labels_norm, allow_pickle=False),
        "labels_raw": np.load(label_dir / f"{stem}_raw.npy", allow_pickle=False),
        "label_meta_json": np.array(meta_path.read_text()),
    })
    np.savez(output, **payload)
    print(f"Wrote annotated sample file: {output}")


if __name__ == "__main__":
    main()
