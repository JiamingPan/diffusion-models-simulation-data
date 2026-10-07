#!/usr/bin/env python
"""Score nearest-parameter training fields with the existing frozen probe.

CPU-only by default. No diffusion model is loaded; no fitting or sampling.
Inputs are explicit manifests and real-data files. Results go to a new directory.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import yaml

from simdiff_eval.parameter_neighbor_control import (
    PARAM_NAMES, ensemble_points, nearest_parameter_matches, parameter_matrix,
    residual_decomposition, summarize_points,
)


def resolve(project, path):
    path = Path(path)
    return path if path.is_absolute() else project / path


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def audit_probe(saved, stats, cfg, heldout):
    """Check the frozen probe's labels, preprocessing and held-out split."""
    if tuple(saved["param_names"]) != PARAM_NAMES:
        raise ValueError("frozen probe parameter order disagrees with the control")
    for key, stat_key in [("param_mean", "mean"), ("param_std", "std")]:
        value = np.asarray(saved[key], dtype=float)
        if value.shape != (6,) or not np.allclose(value, stats[stat_key], rtol=0, atol=1e-6):
            raise ValueError("frozen probe parameter normalization disagrees with the control")
    excluded = np.asarray(saved["heldout_indices"], dtype=np.int64)
    if not np.all(np.isin(heldout, excluded)):
        raise ValueError("requested cosmologies were not excluded from the frozen probe")
    for key in ["train_sims", "val_sims"]:
        if np.intersect1d(heldout, np.asarray(saved[key], dtype=np.int64)).size:
            raise ValueError("held-out cosmology leaked into probe fitting or model selection")
    norm = saved["normalization"].item()
    if norm.get("transform") != ["log"] or norm.get("method") != "tanh":
        raise ValueError("frozen probe has an incompatible image transform")
    defaults = {"alpha": .8, "beta": 10., "gamma": 1., "delta": 1., "sigma": 1.5}
    for key in ["center", "xmax", *defaults]:
        probe_value = norm.get(key, defaults.get(key))
        run_value = cfg["data"]["norm_kwargs"].get(key, defaults.get(key))
        if (probe_value is None or run_value is None
                or not np.isfinite(probe_value) or not np.isfinite(run_value)
                or not np.isclose(probe_value, run_value, rtol=0, atol=1e-7)):
            raise ValueError("frozen probe image normalization disagrees with the generator")


def generated_points(table, run, requested, heldout):
    sub = table[table.run_name == run].copy()
    if sub.empty:
        raise ValueError(f"no generated predictions for {run}")
    if "guidance_label" in sub:
        sub = sub[sub.guidance_label == "noguidance"]
    if sub.empty or ("cfg_dropout" in sub and sub.cfg_dropout.nunique() != 1):
        raise ValueError("select exactly one no-guidance generator condition")
    keys = ["heldout_sim", "seed_index", "parameter"]
    if sub.duplicated(keys).any():
        raise ValueError("duplicate generated predictions; do not pool sampling seeds/conditions")
    if set(sub.heldout_sim) != set(heldout):
        raise ValueError("generated predictions do not match held-out simulation IDs")
    ensembles = []
    for h, sim in enumerate(heldout):
        group = sub[sub.heldout_sim == sim]
        if not set(group.parameter) == set(PARAM_NAMES):
            raise ValueError("generated predictions must contain all six parameters")
        for p, name in enumerate(PARAM_NAMES):
            if not np.allclose(group[group.parameter == name].theta_in, requested[h, p], rtol=0, atol=1e-6):
                raise ValueError("generated requested parameters disagree with the manifest")
        matrix = group.pivot(index="seed_index", columns="parameter", values="theta_rec").reindex(columns=PARAM_NAMES)
        ensembles.append(parameter_matrix(matrix.to_numpy(), "generated predictions"))
    return ensemble_points(ensembles, requested, heldout, "generated")


def evaluate_manifest(project, manifest_path, stats_path, params_path, grid_path, encoder,
                      *, generated=None, run_names=None, real_k=64, batch_size=64,
                      probe_metadata=None):
    from train_nf_conditional_bias_encoder import load_raw_slices, preprocess_real_slices

    rows = json.loads(manifest_path.read_text())
    if run_names:
        wanted = set(run_names)
        rows = [row for row in rows if row["run_name"] in wanted]
        if {row["run_name"] for row in rows} != wanted:
            raise ValueError("requested run not found in manifest")
    if not rows or len({r["run_name"] for r in rows}) != len(rows):
        raise ValueError("manifest is empty or has duplicate run names")
    stats = json.loads(stats_path.read_text())
    if tuple(stats["param_names"]) != PARAM_NAMES:
        raise ValueError("parameter order mismatch in frozen scale metadata")
    all_theta = parameter_matrix(np.loadtxt(params_path, dtype=np.float64)[:, :6], "CAMELS params")
    grid = np.load(grid_path, mmap_mode="r", allow_pickle=False)
    if grid.ndim != 4 or not 1 <= real_k <= grid.shape[1]:
        raise ValueError("real grid must be (simulation,z,H,W); real-k must fit its z dimension")
    z_real = np.unique(np.linspace(0, grid.shape[1]-1, real_k, dtype=np.int64))
    points, match_tables, provenance = [], [], []
    for row in rows:
        run = row["run_name"]
        cfg_path = resolve(project, row["config"])
        cfg = yaml.safe_load(cfg_path.read_text())
        if cfg["train"].get("conditioning") != "continuous" or tuple(row.get("param_names", [])) != PARAM_NAMES:
            raise ValueError(f"{run} is not six-parameter continuous conditioning")
        data = cfg["data"]
        if data.get("transform") != ["log"] or data.get("normalization") != "tanh":
            raise ValueError("this control supports only the existing log+tanh data recipe")
        if data.get("reshape") is not None or data.get("zthin") != 1 or data.get("n_samples") is not None:
            raise ValueError("prepared training rows must not be reshaped, thinned, or reselected")
        pair_path = resolve(project, row["selected_pairs_path"])
        raw_path = resolve(project, row["train_raw_params_path"])
        image_path = resolve(project, row["prepared_image_path"])
        if resolve(project, data["img_path"]).resolve() != image_path.resolve():
            raise ValueError("config and manifest training image paths disagree")
        pairs = pd.read_csv(pair_path)
        if not np.array_equal(pairs.row, np.arange(len(pairs))) or pairs.duplicated(["simulation_index", "z_index"]).any():
            raise ValueError("training slice IDs are not a unique ordered row mapping")
        if not all(pd.api.types.is_integer_dtype(pairs[col]) for col in ["simulation_index", "z_index"]):
            raise ValueError("slice mapping must contain integer indices")
        sims = pairs.simulation_index.to_numpy(np.int64)
        zs = pairs.z_index.to_numpy(np.int64)
        if np.any(sims < 0) or np.any(sims >= min(len(all_theta), len(grid))) or np.any(zs < 0) or np.any(zs >= grid.shape[1]):
            raise ValueError("training pair is outside the real grid")
        train = parameter_matrix(np.load(raw_path, allow_pickle=False), "training labels")
        if len(train) != int(row["dataset_size"]) or len(train) != len(pairs) or not np.allclose(train, all_theta[sims], rtol=0, atol=1e-6):
            raise ValueError("selected slice mapping and raw training labels disagree")
        norm_labels = np.load(resolve(project, data["label_path"]), allow_pickle=False)
        if norm_labels.shape != train.shape or not np.allclose(norm_labels, (train-np.asarray(stats["mean"]))/np.asarray(stats["std"]), rtol=0, atol=2e-5):
            raise ValueError("training label normalization disagrees with frozen metadata")
        heldout_path = resolve(project, row["heldout_indices_path"])
        heldout = np.atleast_1d(np.loadtxt(heldout_path, dtype=np.int64))
        if np.any(heldout < 0) or np.any(heldout >= min(len(all_theta), len(grid))):
            raise ValueError("held-out index is outside the real grid")
        if probe_metadata is not None:
            audit_probe(probe_metadata, stats, cfg, heldout)
        requested = all_theta[heldout]
        if row.get("heldout_raw_params_path"):
            saved = np.load(resolve(project, row["heldout_raw_params_path"]), allow_pickle=False)
            if saved.shape != requested.shape or not np.allclose(saved, requested, rtol=0, atol=1e-6):
                raise ValueError("held-out raw parameters disagree with the source table")
        matches = nearest_parameter_matches(train, sims, requested, heldout, stats["std"])
        prepared = np.load(image_path, mmap_mode="r", allow_pickle=False)
        if prepared.shape != (len(train), 1, grid.shape[-2], grid.shape[-1]):
            raise ValueError("prepared training images have an unexpected shape")
        if row.get("prepared_image_sha256") and file_hash(image_path) != row["prepared_image_sha256"]:
            raise ValueError("prepared image hash disagrees with manifest")
        selected_rows = np.flatnonzero(np.isin(sims, matches.nearest_sim))
        subset = np.asarray(prepared[selected_rows], dtype=np.float32)
        source_subset = load_raw_slices(grid_path, np.column_stack([sims[selected_rows], zs[selected_rows]]))
        if not np.allclose(subset, source_subset, rtol=1e-6, atol=1e-8):
            raise ValueError("nearest training images do not match their declared source slices")
        normalized = preprocess_real_slices(subset, data["norm_kwargs"])
        pred = parameter_matrix(encoder.norm_to_raw(encoder.predict_norm(normalized, batch_size=batch_size)), "probe predictions")
        if len(pred) != len(subset):
            raise ValueError("probe output row count mismatch")
        nearest_ensembles = [pred[sims[selected_rows] == sim] for sim in matches.nearest_sim]
        neighbor_theta = [train[sims == sim][:1] for sim in matches.nearest_sim]
        real_pairs = np.array([(sim,z) for sim in heldout for z in z_real], dtype=np.int64)
        real = preprocess_real_slices(load_raw_slices(grid_path, real_pairs), data["norm_kwargs"])
        real_pred = parameter_matrix(encoder.norm_to_raw(encoder.predict_norm(real, batch_size=batch_size)), "held-out probe predictions")
        if len(real_pred) != len(real):
            raise ValueError("held-out probe output row count mismatch")
        tables = [ensemble_points(neighbor_theta, requested, heldout, "nearest_theta"),
                  ensemble_points(nearest_ensembles, requested, heldout, "nearest_training_field"),
                  ensemble_points(list(real_pred.reshape(len(heldout),len(z_real),6)), requested, heldout, "heldout_real")]
        if generated is not None:
            tables.append(generated_points(generated, run, requested, heldout))
        for table in tables:
            table["run_name"] = run
            table["dataset_size"] = int(row["dataset_size"])
        points.extend(tables)
        matches["run_name"] = run
        match_tables.append(matches)
        provenance.append({"run_name": run, "config": str(cfg_path), "config_sha256": file_hash(cfg_path),
                           "pairs": str(pair_path), "pairs_sha256": file_hash(pair_path),
                           "raw_labels_sha256": file_hash(raw_path), "prepared_images": str(image_path),
                           "encoded_training_rows": selected_rows.tolist(), "real_z_indices": z_real.tolist()})
    return pd.concat(points, ignore_index=True), pd.concat(match_tables, ignore_index=True), provenance


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-dir", type=Path, default=Path.cwd())
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--param-stats", type=Path, required=True)
    parser.add_argument("--params-table", type=Path, required=True)
    parser.add_argument("--grid-path", type=Path, required=True)
    parser.add_argument("--encoder", type=Path, required=True)
    parser.add_argument("--encoder-type", choices=["vgg", "ridge", "mlp"], default="vgg")
    parser.add_argument("--generated-predictions", type=Path)
    parser.add_argument("--generated-metadata", type=Path,
                        help="Original bias_probe_eval_metadata.json; required with generated predictions")
    parser.add_argument("--run-name", action="append")
    parser.add_argument("--real-k", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    project = args.project_dir.resolve()
    if args.out_dir.exists():
        raise FileExistsError("use a new output directory; existing results are preserved")
    if args.batch_size < 1:
        raise ValueError("batch-size must be positive")
    if bool(args.generated_predictions) != bool(args.generated_metadata):
        raise ValueError("generated predictions and their original evaluation metadata must be supplied together")
    from evaluate_nf_conditional_bias_probe import load_encoder, load_mlp_encoder, load_vgg_encoder
    encoder_path = resolve(project, args.encoder)
    with np.load(encoder_path, allow_pickle=True) as saved:
        probe_metadata = {key: saved[key] for key in saved.files}
    dependencies = {}
    for key in ["model_path", "pca_basis_path"]:
        if key in probe_metadata:
            dependency = resolve(project, str(probe_metadata[key].item()))
            dependencies[key] = {"path": str(dependency), "sha256": file_hash(dependency)}
            if key == "pca_basis_path" and "pca_basis_sha256" in probe_metadata:
                if dependencies[key]["sha256"] != str(probe_metadata["pca_basis_sha256"].item()):
                    raise ValueError("frozen PCA basis hash disagrees with the probe descriptor")
    generated_metadata_path = resolve(project, args.generated_metadata) if args.generated_metadata else None
    if generated_metadata_path:
        original = json.loads(generated_metadata_path.read_text())
        if (original.get("encoder_type") != args.encoder_type
                or tuple(original.get("param_names", [])) != PARAM_NAMES
                or resolve(project, original["encoder_path"]).resolve() != encoder_path.resolve()):
            raise ValueError("generated evaluation used a different probe; regenerate predictions with the selected frozen probe")
    if args.encoder_type == "vgg":
        # Refuse an implicit download into a nearly full home directory.
        import torch
        from torchvision.models import VGG16_Weights
        from urllib.parse import urlparse
        with np.load(encoder_path, allow_pickle=True) as saved:
            weights = str(saved["vgg_weights"].item())
        if weights.lower() in {"none", "random"}:
            raise ValueError("this review requires the existing pretrained VGG probe, not random features")
        url = getattr(VGG16_Weights, weights).url
        cached = Path(torch.hub.get_dir()) / "checkpoints" / Path(urlparse(url).path).name
        if not cached.is_file():
            raise FileNotFoundError(f"existing pretrained VGG weights must already be cached; no download attempted: {cached}")
        dependencies["vgg_weights"] = {"path": str(cached), "sha256": file_hash(cached)}
    if args.encoder_type == "vgg":
        encoder = load_vgg_encoder(project, encoder_path, "cpu")
    else:
        encoder = (load_encoder if args.encoder_type == "ridge" else load_mlp_encoder)(project, encoder_path)
    generated_path = resolve(project, args.generated_predictions) if args.generated_predictions else None
    generated = pd.read_csv(generated_path) if generated_path else None
    inputs = [resolve(project, p) for p in [args.manifest, args.param_stats, args.params_table, args.grid_path]]
    points, matches, provenance = evaluate_manifest(project, *inputs, encoder, generated=generated,
                                                    run_names=args.run_name, real_k=args.real_k,
                                                    batch_size=args.batch_size, probe_metadata=probe_metadata)
    args.out_dir.mkdir(parents=True, exist_ok=False)
    points.to_csv(args.out_dir / "control_points.csv", index=False)
    matches.to_csv(args.out_dir / "parameter_matches.csv", index=False)
    summarize_points(points).to_csv(args.out_dir / "control_summary.csv", index=False)
    decomposition = residual_decomposition(points)
    if not decomposition.empty:
        decomposition.to_csv(args.out_dir / "residual_decomposition.csv", index=False)
    metadata = {"distance": "Euclidean in six raw parameters divided by frozen training-only std",
                "intervals": "field/probe recovery spread; not Bayesian posterior coverage",
                "tie_policy": "lowest simulation ID; all its selected training fields",
                "manifest_sha256": file_hash(inputs[0]), "param_stats_sha256": file_hash(inputs[1]),
                "params_sha256": file_hash(inputs[2]), "grid_path": str(inputs[3]),
                "encoder_sha256": file_hash(encoder_path), "encoder_type": args.encoder_type,
                "encoder_dependencies": dependencies,
                "generated_predictions_sha256": file_hash(generated_path) if generated_path else None,
                "generated_metadata_sha256": file_hash(generated_metadata_path) if generated_metadata_path else None,
                "legacy_probe_provenance_limit": "old evaluation metadata identifies the probe path but does not pin its historical contents",
                "runs": provenance}
    (args.out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"NEAREST-PARAMETER CONTROL COMPLETE: {len(provenance)} runs; {args.out_dir}")


if __name__ == "__main__":
    main()
