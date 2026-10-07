#!/usr/bin/env python
"""Report for the training-theta test of one conditional run (default N=64).

Reads the sample file written by ``sample_nf_conditional_theta_interp.py
--label-set train_theta`` and the run's own training maps, verifies provenance,
and reports per-cosmology (draws aggregated first):

* centered pixel cosine to the corresponding training map,
* maximum cosine over all N training maps and whether the nearest is the own map,
* within-cosmology draw-to-draw cosine,
* frozen-VGG Omega_m recovery bias for generated fields and for the corresponding
  real training maps, and the paired generated-minus-real prediction.

The frozen probe is the one recorded in the sweep's evaluation metadata; VGG16
weights must already be cached (downloads are blocked). No sampling, training,
or writes outside ``results/<sweep>/samples_theta_interp/``.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from urllib.parse import urlparse

import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
for p in (PROJECT_ROOT, PROJECT_ROOT / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from simdiff_eval.conditional_unet_diagnostics import normalize_raw_hi  # noqa: E402
from simdiff_eval.train_theta_report import train_theta_report  # noqa: E402

SWEEP = "nf_conditional_bias_fresh_full_sweep_200k"
SCALES = "local/nf_conditional_bias_probe/heldout/param_norm_stats.json"


def resolve(project: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else project / path


def load_probe(project: Path, metadata_path: Path, device: str):
    """Same recipe as the notebook's Test 4: recorded descriptor, cached weights, no download."""
    import torch
    from torchvision.models import VGG16_Weights
    from unittest.mock import patch
    import matplotlib
    matplotlib.use("Agg")

    meta = json.loads(metadata_path.read_text())
    if meta.get("encoder_type") != "vgg" or not meta.get("encoder_path"):
        raise SystemExit("evaluation metadata does not record a frozen VGG probe")
    encoder_path = resolve(project, meta["encoder_path"])
    with np.load(encoder_path, allow_pickle=False) as d:
        weights_name = str(d["vgg_weights"].item())
        head_path = resolve(project, str(d["model_path"].item()))
    url = getattr(VGG16_Weights, weights_name).url
    cache = Path(torch.hub.get_dir()) / "checkpoints" / Path(urlparse(url).path).name
    if not head_path.is_file() or not cache.is_file():
        raise SystemExit(f"probe head or cached VGG weights missing: {head_path}; {cache}. No download attempted.")
    from evaluate_nf_conditional_bias_probe import load_vgg_encoder
    with patch("torch.hub.download_url_to_file", side_effect=RuntimeError("downloads prohibited")):
        probe = load_vgg_encoder(project, encoder_path, device=device)
    return probe, {"encoder_path": str(encoder_path), "head_path": str(head_path), "vgg_weights_cache": str(cache),
                   "device": device}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-dir", default=".")
    parser.add_argument("--sweep", default=SWEEP)
    parser.add_argument("--dataset-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--num-steps", type=int, default=50)
    parser.add_argument("--probe-device", default="cpu")
    parser.add_argument("--probe-batch-size", type=int, default=8)
    parser.add_argument("--skip-probe", action="store_true")
    parser.add_argument("--probe-metadata", default="results/nf_conditional_bias_fresh_full_sweep_200k/calibration_vgg/bias_probe_eval_metadata.json",
                        help="evaluation metadata naming the frozen VGG probe (default: the full sweep's, same probe for every sweep)")
    args = parser.parse_args()

    project = Path(args.project_dir).resolve()
    manifest = json.loads((project / "local" / args.sweep / "manifest.json").read_text())
    rows = [r for r in manifest if int(r["dataset_size"]) == args.dataset_size]
    if len(rows) != 1:
        raise SystemExit(f"expected one manifest row for N={args.dataset_size}")
    row = rows[0]
    cfg = yaml.safe_load(resolve(project, row["config"]).read_text())
    checks = {
        "conditioning_continuous": cfg["train"].get("conditioning") == "continuous",
        "transform_log_tanh": cfg["data"].get("transform") == ["log"] and cfg["data"].get("normalization") == "tanh",
        "generate_ema_none": cfg["generate"].get("ema_sigma_rel") is None,
        "generate_guidance_none": cfg["generate"].get("guidance_scale") is None,
        "checkpoint_epoch_is_final": int(row.get("checkpoint_epoch", -1)) == int(row["epochs"]) - 1,
    }
    checkpoint_epoch = int(row["checkpoint_epoch"])
    if not all(checks.values()):
        raise SystemExit(f"run provenance checks failed: {checks}")

    label_dir = project / "local" / args.sweep / "labels"
    stem = f"{row['run_name']}_train_theta"
    meta = json.loads((label_dir / f"{stem}_meta.json").read_text())
    sample_file = project / "results" / args.sweep / "samples_theta_interp" / f"{stem}_seed{args.seed}_dpm{args.num_steps}.npz"
    if not sample_file.is_file():
        raise SystemExit(f"missing samples: {sample_file}")
    with np.load(sample_file, allow_pickle=False) as a:
        expected = {"run_name": row["run_name"], "dataset_size": args.dataset_size, "label_set": "train_theta",
                    "seed": args.seed, "guidance_label": "noguidance", "num_steps": args.num_steps,
                    "scheduler": "DPMSolverMultistepScheduler", "checkpoint_epoch": checkpoint_epoch}
        for key, value in expected.items():
            got = a[key].item() if a[key].ndim == 0 else a[key]
            if str(got) != str(value):
                raise SystemExit(f"sample provenance mismatch: {key}={got!r}, expected {value!r}")
        samples = np.asarray(a["samples"], dtype=np.float32)
        labels_raw, labels_norm = np.asarray(a["labels_raw"], float), np.asarray(a["labels_norm"], float)
    stats = json.loads(resolve(project, SCALES).read_text())
    mean, std = np.asarray(stats["mean"], float), np.asarray(stats["std"], float)
    theta_raw = np.load(resolve(project, row["train_raw_params_path"]), allow_pickle=False).astype(float)
    training_rows = np.asarray(meta["training_rows"], int)
    n_seeds = int(meta["seeds"])
    if not np.allclose(labels_raw, np.repeat(theta_raw[training_rows], n_seeds, axis=0), atol=1e-5):
        raise SystemExit("sample labels are not the training cosmologies repeated per seed")
    if not np.allclose(labels_norm, (labels_raw - mean) / std, atol=1e-5):
        raise SystemExit("normalized labels do not match the frozen parameter normalization")
    if len(samples) != len(training_rows) * n_seeds:
        raise SystemExit("sample count does not match cosmologies x seeds")

    train_raw = np.load(resolve(project, row["prepared_image_path"]), mmap_mode="r", allow_pickle=False)
    train_all = normalize_raw_hi(np.asarray(train_raw), cfg["data"]["norm_kwargs"])
    # cosmology index of every training map: maps sharing a simulation share a label (multiplicity m > 1)
    sim_ids = np.asarray(meta["simulation_ids"], int)
    pairs_csv = np.genfromtxt(resolve(project, row["selected_pairs_path"]), delimiter=",", names=True, dtype=int)
    row_sims = np.asarray(pairs_csv["simulation_index"], int)
    lookup = {int(s): i for i, s in enumerate(sim_ids)}
    if set(row_sims.tolist()) != set(lookup):
        raise SystemExit("training-map simulations do not match the training-theta label cosmologies")
    group_index = np.array([lookup[int(s)] for s in row_sims])
    if not np.array_equal(row_sims[training_rows], sim_ids):
        raise SystemExit("training_rows in the label metadata do not point at the expected simulations")
    true_omega = theta_raw[training_rows, 0]

    predict = None
    probe_info = {"used": False}
    if not args.skip_probe:
        metadata_path = resolve(project, args.probe_metadata)
        probe, probe_info = load_probe(project, metadata_path, args.probe_device)
        probe_info["used"] = True

        def predict(maps: np.ndarray) -> np.ndarray:
            out = probe.norm_to_raw(probe.predict_norm(np.asarray(maps, np.float32)[:, None], batch_size=args.probe_batch_size))
            return np.asarray(out)[:, 0]

    report = train_theta_report(samples, train_all, n_seeds, true_omega, predict, group_index=group_index)
    report["provenance"] = {"sample_file": str(sample_file), "run_name": row["run_name"], "checks": checks,
                            "seed": args.seed, "num_steps": args.num_steps, "n_seeds": n_seeds,
                            "probe": probe_info, "param_norm_stats": SCALES}
    out_dir = sample_file.parent
    (out_dir / f"{stem}_report.json").write_text(json.dumps(report, indent=2) + "\n")
    per = report["per_cosmology"]
    with (out_dir / f"{stem}_per_cosmology.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(per[0].keys()))
        writer.writeheader(); writer.writerows(per)

    print(f"\nTraining-theta test: {report['n_cosmologies']} cosmologies x {n_seeds} draws = {report['n_fields']} fields; "
          f"{report['n_training_maps']} training maps ({report['n_training_maps']/report['n_cosmologies']:.1f} per cosmology); checkpoint epoch {checkpoint_epoch}")
    print(f"probe: {probe_info}")
    print(f"{'metric (per cosmology, then over 64)':48s} {'median':>8s} {'q16':>8s} {'q84':>8s}")
    for key, s in report["summary"].items():
        if isinstance(s, dict):
            print(f"{key:48s} {s['median']:8.3f} {s['q16']:8.3f} {s['q84']:8.3f}")
        else:
            print(f"{key:48s} {s:8.3f}")
    print(f"\nwrote {out_dir / (stem + '_report.json')} and {out_dir / (stem + '_per_cosmology.csv')}")


if __name__ == "__main__":
    main()
