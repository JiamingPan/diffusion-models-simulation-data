#!/usr/bin/env python
"""Fit one UMAP to frozen VGG/MLP inputs for the saved C4 controls."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import pickle
import subprocess
import sys
import warnings
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from simdiff_eval.probe_c4_umap import (  # noqa: E402
    balanced_real_slice_pairs,
    classify_transform_identity,
    compare_source_to_reference,
    deterministic_balanced_real_split,
    frozen_mlp_inputs,
    generated_identity_diagnostics,
    measured_power_deficit_depth,
    perfect_mixing_expectation,
    real_split_mixing_baseline,
    source_metadata,
)
from simdiff_eval.probe_controls import json_safe, subset_generated_cosmologies  # noqa: E402
from simdiff_eval.terminal_reports import (  # noqa: E402
    start_report,
    update_incomplete_report,
)
from simdiff_eval.probe_transforms import (  # noqa: E402
    gaussian_smoothing_transform,
    transfer_transform,
)


EXPECTED_HELDOUT = np.arange(900, 932, dtype=np.int64)
EXPECTED_RUNS = (
    "nf_cond_bias_hi_u128_d2p07_n128_200k",
    "nf_cond_bias_hi_u128_d2p14_n16384_200k",
)
SOURCES = (
    "real_original",
    "real_measured_transfer",
    "real_gaussian",
    "generated",
)
SOURCE_COLORS = {
    "real_original": "#303030",
    "real_measured_transfer": "#168aad",
    "real_gaussian": "#f28e2b",
    "generated": "#8f5fbf",
}


def sha256_file(path: str | Path, block_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(block_size), b""):
            digest.update(block)
    return digest.hexdigest()


def artifact_record(path: str | Path) -> dict[str, Any]:
    resolved = Path(path).resolve()
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    return {
        "path": str(resolved),
        "sha256": sha256_file(resolved),
        "bytes": resolved.stat().st_size,
    }


def large_input_record(path: str | Path) -> dict[str, Any]:
    """Record a large immutable input without an expensive full-file reread."""
    resolved = Path(path).resolve()
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    stat = resolved.stat()
    return {
        "path": str(resolved),
        "bytes": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "sha256": None,
        "sha256_reason": "large raw grid; exact selected indices and file stat are recorded",
    }


def git_state(root: Path) -> dict[str, Any]:
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    dirty = bool(
        subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    )
    return {"revision": revision, "dirty": dirty}


def load_frozen_vgg_and_head(
    *, weights_path: Path, head_path: Path, device: str
) -> tuple[Any, Any, dict[str, str]]:
    """Load the exact explicit VGG and head artifacts, without symbolic lookup."""
    import torch
    from torchvision.models import vgg16

    resolved_device = (
        "cuda" if device == "auto" and torch.cuda.is_available()
        else "cpu" if device == "auto"
        else device
    )
    weights_path = Path(weights_path).resolve()
    head_path = Path(head_path).resolve()
    try:
        state = torch.load(weights_path, map_location="cpu", weights_only=True)
    except TypeError:
        state = torch.load(weights_path, map_location="cpu")
    model = vgg16(weights=None)
    model.load_state_dict(state, strict=True)
    model = model.eval().to(resolved_device)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    with head_path.open("rb") as handle:
        head = pickle.load(handle)
    return model.features, head, {
        "weights_path": str(weights_path),
        "head_path": str(head_path),
        "weights_loading": "explicit torch.load + strict VGG16 state_dict",
        "head_loading": "explicit pickle path",
        "device": str(resolved_device),
    }


def generated_sample_path(
    results_root: Path,
    row: dict[str, Any],
    *,
    seed: int,
    samples_per_cosmology: int,
) -> Path:
    """Resolve only the manifest filename, pinned under the supplied results root."""
    template = str(row["sample_path"])
    values = {
        "seed": int(seed),
        "sample_label": "dpm50",
        "k": int(samples_per_cosmology),
        "guidance": "noguidance",
    }
    try:
        rendered = template.format(**values)
    except KeyError as exc:
        raise ValueError(f"unknown sample-path template field in {template!r}") from exc
    filename = Path(rendered).name
    if not filename.endswith(".npz"):
        raise ValueError(f"generated sample template did not produce an NPZ: {rendered}")
    return Path(results_root).resolve() / "samples" / filename


def validate_c4_probe_artifacts(
    c4_manifest: dict[str, Any],
    *,
    encoder_record: dict[str, Any],
    head_record: dict[str, Any],
) -> None:
    """Require the explicit probe files to equal those used by completed C4."""
    for label, actual in (("encoder", encoder_record), ("head", head_record)):
        expected = c4_manifest.get(label, {}).get("sha256")
        if not expected or expected != actual.get("sha256"):
            raise RuntimeError(
                f"explicit frozen {label} does not match completed C4 manifest: "
                f"expected {expected!r}, found {actual.get('sha256')!r}"
            )


def validate_saved_c4_parameters(
    power: dict[str, Any],
    c4_manifest: dict[str, Any],
    rows: list[dict[str, Any]],
) -> None:
    """Cross-check both frozen C4 parameter stores run by run."""
    transforms = c4_manifest.get("transforms", [])
    by_name: dict[str, dict[str, Any]] = {}
    for record in transforms:
        name = str(record.get("name"))
        if name in by_name:
            raise RuntimeError(f"duplicate C4 transform record: {name}")
        by_name[name] = record
    for row in rows:
        run_name = str(row["run_name"])
        dataset_size = int(row["dataset_size"])
        saved = power.get("runs", {}).get(run_name)
        if saved is None or int(saved.get("dataset_size", -1)) != dataset_size:
            raise RuntimeError(f"missing or mismatched C4 power record for {run_name}")
        measured_name = f"transfer_Tk__{run_name}__N{dataset_size}"
        gaussian_name = f"gaussian_smoothing__{run_name}__N{dataset_size}"
        measured = by_name.get(measured_name)
        gaussian = by_name.get(gaussian_name)
        if measured is None or gaussian is None:
            raise RuntimeError(f"completed C4 manifest lacks saved transforms for {run_name}")
        for label, left, right in (
            ("k bins", saved.get("k_bins"), measured.get("k_bins")),
            (
                "measured transfer",
                saved.get("measured_transfer"),
                measured.get("transfer_values"),
            ),
        ):
            if not np.array_equal(
                np.asarray(left, dtype=np.float64), np.asarray(right, dtype=np.float64)
            ):
                raise RuntimeError(f"saved C4 {label} mismatch for {run_name}")
        if float(saved.get("gaussian_sigma_pixels")) != float(
            gaussian.get("sigma_pixels")
        ):
            raise RuntimeError(f"saved C4 Gaussian sigma mismatch for {run_name}")


def validate_c4_sample_path(saved: dict[str, Any], actual_path: Path) -> None:
    """Require the generated maps to be the file used to derive saved C4."""
    recorded = saved.get("sample_path")
    if not recorded:
        raise RuntimeError("saved C4 power record lacks its generated sample path")
    recorded_path = Path(recorded).resolve()
    actual_path = Path(actual_path).resolve()
    same = (
        os.path.samefile(recorded_path, actual_path)
        if recorded_path.exists() and actual_path.exists()
        else recorded_path == actual_path
    )
    if not same:
        raise RuntimeError(
            f"generated sample differs from saved C4 derivation: {actual_path} != {recorded_path}"
        )


def _analysis_config(args: argparse.Namespace, rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "analysis": "c4_frozen_vgg_umap_seed123_v3",
        "heldout_indices": EXPECTED_HELDOUT.astype(int).tolist(),
        "run_names": [row["run_name"] for row in rows],
        "sources": list(SOURCES),
        "real_slice_selection": {
            "rule": "fixed_even_indices",
            "slice_indices": list(range(0, 128, 2)),
            "slices_per_simulation": 64,
        },
        "generated_sample_selection": {
            "rule": "all_saved_samples_in_stored_order",
            "samples_per_simulation": 64,
        },
        "pooled_reference_policy": (
            "real_original is included once in the pooled fit and reused in both run plots; "
            "duplicating identical reference rows would distort the UMAP neighbour graph"
        ),
        "feature_layer": (
            "torchvision VGG16 .features through features[30] (final MaxPool2d); "
            "adaptive average and maximum pooling concatenated to 1024 raw features; "
            "the frozen fitted StandardScaler.transform output is the 1024-value "
            "input to the first frozen MLP dense layer"
        ),
        "probe_policy": "transform_only; no fit, refit, calibration, or prediction-based selection",
        "c4_policy": "apply saved transfer_values and saved gaussian_sigma_pixels; never rederive",
        "umap": {
            "n_components": 2,
            "n_neighbors": int(args.umap_neighbors),
            "min_dist": float(args.umap_min_dist),
            "metric": "euclidean",
            "random_state": int(args.seed),
            "transform_seed": int(args.seed),
            "n_jobs": 1,
        },
        "metrics": {
            "k": int(args.knn_k),
            "knn_definition": (
                "For a balanced source+real_original pool, fraction of each point's "
                "k nearest neighbours carrying the other source label; 0 means separated, "
                "about 0.5 means well mixed. Point scores are averaged within simulation."
            ),
            "bootstrap": "simulation-block percentile bootstrap",
            "bootstrap_replicates": int(args.bootstrap),
            "bootstrap_seed": int(args.seed),
            "headline_feature_space": "frozen_standardized_vgg_mlp_input_1024d",
            "umap_policy": "visualization only; no 2-D headline metric",
            "perfect_mixing_formula": "2*n_a*n_b/((n_a+n_b)*(n_a+n_b-1))",
            "real_split_baseline": {
                "seed": int(args.seed),
                "rule": "seeded within-simulation balanced half split",
            },
            "identity": {
                "array_dtype": "float32",
                "rtol": 1e-6,
                "atol": 1e-7,
                "criterion": "arrays allclose and both bootstrap intervals have zero width",
            },
        },
    }


def build_c4_metric_tables(
    *,
    metadata: pd.DataFrame,
    standardized: np.ndarray,
    original_images: np.ndarray,
    transformed_images: dict[tuple[str, str], np.ndarray],
    run_parameters: dict[str, dict[str, Any]],
    k: int,
    n_boot: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Build auditable 1024-D complete/headline tables and identity report."""
    values = np.asarray(standardized)
    if len(values) != len(metadata):
        raise ValueError("standardized features and metadata are misaligned")
    reference_mask = metadata["source"].eq("real_original").to_numpy()
    reference_values = values[reference_mask]
    reference_sim = metadata.loc[reference_mask, "sim_index"].to_numpy()
    baseline = real_split_mixing_baseline(
        reference_values,
        reference_sim,
        k=int(k),
        n_boot=int(n_boot),
        seed=int(seed),
    )

    complete_rows: list[dict[str, Any]] = []
    identity_rows: list[dict[str, Any]] = []
    for run_name, parameters in run_parameters.items():
        for source in SOURCES[1:]:
            source_mask = (
                metadata["run_name"].eq(run_name)
                & metadata["source"].eq(source)
            ).to_numpy()
            source_values = values[source_mask]
            source_sim = metadata.loc[source_mask, "sim_index"].to_numpy()
            metric = compare_source_to_reference(
                source_values,
                reference_values,
                source_sim=source_sim,
                reference_sim=reference_sim,
                k=int(k),
                n_boot=int(n_boot),
                seed=int(seed),
            )
            if source == "generated":
                identity = generated_identity_diagnostics()
            else:
                identity = classify_transform_identity(
                    original_images,
                    transformed_images[(run_name, source)],
                    metric,
                )
            transforms = metadata.loc[source_mask, "transform"].unique().tolist()
            if len(transforms) != 1:
                raise RuntimeError(
                    f"expected one transform for {run_name}/{source}; found {transforms}"
                )
            source_count = int(len(source_values))
            reference_count = int(len(reference_values))
            row = {
                "run_name": run_name,
                "dataset_size": int(parameters["dataset_size"]),
                "source": source,
                "reference": "real_original",
                "transform": transforms[0],
                "feature_space": "frozen_standardized_vgg_mlp_input_1024d",
                "gaussian_sigma_pixels": float(parameters["gaussian_sigma_pixels"]),
                "measured_power_deficit_depth": float(
                    parameters["measured_power_deficit_depth"]
                ),
                "source_count": source_count,
                "reference_count": reference_count,
                "perfect_mixing_expectation": perfect_mixing_expectation(
                    source_count, reference_count
                ),
                "real_split_mixing_baseline": baseline[
                    "real_split_mixing_baseline"
                ],
                "real_split_mixing_baseline_ci_low": baseline[
                    "real_split_mixing_baseline_ci_low"
                ],
                "real_split_mixing_baseline_ci_high": baseline[
                    "real_split_mixing_baseline_ci_high"
                ],
                **metric,
                **identity,
            }
            complete_rows.append(row)
            if source != "generated":
                identity_rows.append(
                    {
                        "run_name": run_name,
                        "dataset_size": int(parameters["dataset_size"]),
                        "source": source,
                        "transform": transforms[0],
                        "gaussian_sigma_pixels": float(
                            parameters["gaussian_sigma_pixels"]
                        ),
                        "measured_power_deficit_depth": float(
                            parameters["measured_power_deficit_depth"]
                        ),
                        "message": identity["identity_reason"],
                        **identity,
                    }
                )
    complete = pd.DataFrame(complete_rows)
    headline = complete.loc[~complete["transform_is_identity"].astype(bool)].copy()
    return complete, headline, pd.DataFrame(identity_rows), baseline


def build_umap_visual_only_table(
    metadata: pd.DataFrame, embedding: np.ndarray
) -> pd.DataFrame:
    """Return a clearly named 2-D layout diagnostic, never a scientific metric."""
    values = np.asarray(embedding, dtype=np.float64)
    if values.shape != (len(metadata), 2):
        raise ValueError("UMAP embedding and metadata are misaligned")
    reference = values[metadata["source"].eq("real_original").to_numpy()]
    reference_center = reference.mean(axis=0)
    rows = []
    for run_name in sorted(
        set(metadata.loc[~metadata["source"].eq("real_original"), "run_name"])
    ):
        for source in SOURCES[1:]:
            mask = (
                metadata["run_name"].eq(run_name)
                & metadata["source"].eq(source)
            ).to_numpy()
            source_values = values[mask]
            rows.append(
                {
                    "run_name": run_name,
                    "source": source,
                    "reference": "real_original",
                    "umap_layout_separation_visual_only": float(
                        np.linalg.norm(source_values.mean(axis=0) - reference_center)
                    ),
                }
            )
    return pd.DataFrame(rows)


def fit_umap_with_connectivity_report(
    reducer: Any, standardized: np.ndarray
) -> tuple[np.ndarray, dict[str, Any]]:
    """Capture only UMAP's disconnected-graph warning and re-emit the rest."""
    disconnected = []
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        embedding = np.asarray(
            reducer.fit_transform(standardized), dtype=np.float32
        )
    for warning in caught:
        message = str(warning.message)
        if "Graph is not fully connected" in message:
            disconnected.append(
                {
                    "category": warning.category.__name__,
                    "message": message,
                }
            )
        else:
            warnings.warn_explicit(
                warning.message,
                warning.category,
                warning.filename,
                warning.lineno,
            )
    return embedding, {
        "umap_graph_not_fully_connected": bool(disconnected),
        "warnings": disconnected,
    }


def configure_known_warning_filters() -> None:
    """Suppress only the known PyTorch TypedStorage deprecation noise."""
    warnings.filterwarnings(
        "ignore",
        message=r"^TypedStorage is deprecated.*",
        category=UserWarning,
    )


def threadpool_compatibility_report(info_callable: Any | None = None) -> dict[str, Any]:
    """Record only the known threadpoolctl callback failure; re-raise all else."""
    if info_callable is None:
        from threadpoolctl import threadpool_info

        info_callable = threadpool_info
    try:
        pools = info_callable()
    except (AttributeError, TypeError) as exc:
        message = str(exc)
        if "NoneType" in message and "split" in message:
            return {
                "status": "known_callback_failure",
                "exception_type": type(exc).__name__,
                "message": message,
            }
        raise
    return {"status": "ok", "threadpools": pools}


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(json_safe(payload), indent=2, sort_keys=True, allow_nan=False) + "\n"
    )


def _load_real_images(
    data_root: Path,
    heldout: np.ndarray,
    normalization: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    from prepare_nf_conditional_u128_config import (
        N_TRAIN_SIMS,
        image_path,
        load_params,
        params_path,
    )
    from train_nf_conditional_bias_encoder import load_raw_slices, preprocess_real_slices

    pairs = balanced_real_slice_pairs(heldout, slices_per_sim=64)
    images = preprocess_real_slices(
        load_raw_slices(image_path(data_root), pairs), normalization
    ).astype(np.float32, copy=False)
    parameters = load_params(params_path(data_root), N_TRAIN_SIMS)
    theta = parameters[pairs[:, 0]].astype(np.float32, copy=False)
    ordinal = np.tile(np.arange(64, dtype=np.int64), len(heldout))
    return images, theta, pairs[:, 0], ordinal, pairs[:, 1]


def _plot_run(table: pd.DataFrame, run_name: str, output: Path) -> None:
    run_table = pd.concat(
        [
            table[table["source"] == "real_original"],
            table[(table["run_name"] == run_name) & (table["source"] != "real_original")],
        ],
        ignore_index=True,
    )
    fig, axes = plt.subplots(1, 2, figsize=(15.5, 6.7), constrained_layout=True)
    for source in SOURCES:
        subset = run_table[run_table["source"] == source]
        axes[0].scatter(
            subset["umap_1"],
            subset["umap_2"],
            s=8,
            alpha=0.32,
            linewidths=0,
            color=SOURCE_COLORS[source],
            label=source.replace("_", " "),
            rasterized=True,
        )
    axes[0].legend(frameon=False, markerscale=2.1, fontsize=9)
    axes[0].set_title("Same UMAP coordinates, coloured by map source")

    omega = axes[1].scatter(
        run_table["umap_1"],
        run_table["umap_2"],
        c=run_table["Omega_m"],
        cmap="viridis",
        s=8,
        alpha=0.40,
        linewidths=0,
        rasterized=True,
    )
    fig.colorbar(omega, ax=axes[1], label=r"true $\Omega_m$")
    axes[1].set_title(r"Same points, coloured only by true $\Omega_m$")
    for ax in axes:
        ax.set_xlabel("UMAP coordinate 1")
        ax.set_ylabel("UMAP coordinate 2")
        ax.grid(alpha=0.14)
    fig.suptitle(
        f"Frozen-probe C4 feature space: {run_name}\n"
        "One pooled UMAP for both runs; no probe or transfer was fitted here",
        fontsize=15,
    )
    fig.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_mixing_baselines(table: pd.DataFrame, output: Path) -> None:
    """Plot observed frozen-space mixing against exact and empirical references."""
    ordered = table.reset_index(drop=True)
    x = np.arange(len(ordered), dtype=float)
    observed = ordered["knn_cross_source_fraction"].to_numpy(dtype=float)
    lower = observed - ordered[
        "knn_cross_source_fraction_ci_low"
    ].to_numpy(dtype=float)
    upper = ordered[
        "knn_cross_source_fraction_ci_high"
    ].to_numpy(dtype=float) - observed
    fig, ax = plt.subplots(figsize=(max(8.5, 1.25 * len(ordered)), 5.8))
    for index, row in ordered.iterrows():
        identity = bool(row["transform_is_identity"])
        ax.errorbar(
            x[index],
            observed[index],
            yerr=np.array([[lower[index]], [upper[index]]]),
            fmt="*" if identity else "o",
            markersize=11 if identity else 7,
            color=SOURCE_COLORS.get(str(row["source"]), "0.25"),
            markerfacecolor="none" if identity else None,
            capsize=4,
        )
    perfect = float(ordered["perfect_mixing_expectation"].iloc[0])
    empirical = float(ordered["real_split_mixing_baseline"].iloc[0])
    ax.axhline(perfect, color="0.25", linestyle="--", label="exact perfect mixing")
    ax.axhline(
        empirical,
        color="#2a9d8f",
        linestyle=":",
        linewidth=2,
        label="real-vs-real split baseline",
    )
    labels = [
        f"N={int(row.dataset_size)}\n{str(row.source).replace('_', ' ')}"
        for row in ordered.itertuples()
    ]
    ax.set_xticks(x, labels, rotation=28, ha="right")
    ax.set_ylabel("kNN cross-source mixing fraction")
    ax.set_title("C4 mixing in the frozen standardized 1024-D probe space")
    ax.grid(axis="y", alpha=0.2)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-dir", default=".")
    parser.add_argument("--code-root", required=True, type=Path)
    parser.add_argument("--expected-code-revision", required=True)
    parser.add_argument(
        "--expected-results-revision",
        default="dced4f8928efe248d819a72560ef61a099d0c4a3",
    )
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--results-root", required=True, type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--encoder", type=Path)
    parser.add_argument("--head", type=Path)
    parser.add_argument("--vgg-weights-path", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--embedding-batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--umap-neighbors", type=int, default=30)
    parser.add_argument("--umap-min-dist", type=float, default=0.1)
    parser.add_argument("--knn-k", type=int, default=15)
    parser.add_argument("--bootstrap", type=int, default=2000)
    return parser.parse_args()


def main() -> None:
    configure_known_warning_filters()
    args = parse_args()
    project_dir = Path(args.project_dir).resolve()
    code_root = args.code_root.resolve()
    data_root = args.data_root.resolve()
    results_root = args.results_root.resolve()
    output_dir = args.output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {output_dir}")
    staging = output_dir.with_name(
        output_dir.name + f".tmp-{os.environ.get('SLURM_JOB_ID', os.getpid())}"
    )
    if staging.exists():
        raise FileExistsError(f"refusing to overwrite staging output: {staging}")

    code_state = git_state(code_root)
    if code_state != {"revision": args.expected_code_revision, "dirty": False}:
        raise RuntimeError(f"code provenance mismatch: {code_state}")
    staging.mkdir(parents=True)
    start_report(
        staging / "manifest.json",
        payload={
            "analysis": "c4_frozen_vgg_umap_seed123_v3",
            "code_git": code_state,
        },
        producer_job_id=os.environ.get("SLURM_JOB_ID"),
    )

    from evaluate_nf_conditional_bias_probe import VGGEncoder, load_manifest, selected_rows
    from prepare_nf_conditional_u128_config import (
        PARAM_NAMES,
        image_path,
        params_path,
    )
    from train_nf_conditional_bias_vgg_encoder import vgg_embed

    manifest_path = args.manifest or project_dir / "local/nf_conditional_bias_probe/manifest.json"
    rows = selected_rows(load_manifest(project_dir, manifest_path), list(EXPECTED_RUNS))
    if tuple(row["run_name"] for row in rows) != EXPECTED_RUNS:
        raise RuntimeError(f"expected exactly {EXPECTED_RUNS}; found {[row['run_name'] for row in rows]}")

    encoder_path = (args.encoder or results_root / "encoder/vgg_mlp_encoder.npz").resolve()
    head_path = (args.head or results_root / "encoder/vgg_mlp_encoder.pkl").resolve()
    encoder_artifact = artifact_record(encoder_path)
    head_artifact = artifact_record(head_path)
    weights_artifact = artifact_record(args.vgg_weights_path)
    if not str(weights_artifact["sha256"]).startswith("397923af"):
        raise RuntimeError(
            "explicit VGG16 weights fail the official 397923af SHA256 prefix: "
            f"{weights_artifact['sha256']}"
        )
    power_path = results_root / "degradation_controls/power_transfer_curves.json"
    c4_manifest_path = results_root / "degradation_controls/manifest.json"
    power = json.loads(power_path.read_text())
    c4_manifest = json.loads(c4_manifest_path.read_text())
    if c4_manifest.get("git") != {
        "dirty": False,
        "revision": args.expected_results_revision,
    }:
        raise RuntimeError(f"C4 provenance mismatch: {c4_manifest.get('git')}")
    validate_c4_probe_artifacts(
        c4_manifest,
        encoder_record=encoder_artifact,
        head_record=head_artifact,
    )
    if set(power.get("runs", {})) != set(EXPECTED_RUNS):
        raise RuntimeError("saved C4 transfer file does not contain exactly the two expected runs")
    validate_saved_c4_parameters(power, c4_manifest, rows)

    with np.load(encoder_path, allow_pickle=True) as payload:
        heldout = payload["heldout_indices"].astype(np.int64)
        normalization = payload["normalization"].item()
        encoder_meta = {
            "encoder_type": str(payload["encoder_type"].item()),
            "model_path": str(payload["model_path"].item()),
            "vgg_weights": str(payload["vgg_weights"].item()),
            "vgg_image_size": int(payload["vgg_image_size"]),
            "vgg_value_min": float(payload["vgg_value_min"]),
            "vgg_value_max": float(payload["vgg_value_max"]),
            "vgg_pool": str(payload["vgg_pool"].item()),
            "feature_dim": int(payload["feature_dim"]),
        }
    if not np.array_equal(heldout, EXPECTED_HELDOUT):
        raise RuntimeError(f"heldout simulations are not exactly 900..931: {heldout.tolist()}")
    if encoder_meta["vgg_pool"] != "avgmax" or encoder_meta["feature_dim"] != 1024:
        raise RuntimeError(f"unexpected frozen VGG feature contract: {encoder_meta}")

    vgg, head, explicit_load_report = load_frozen_vgg_and_head(
        weights_path=args.vgg_weights_path,
        head_path=head_path,
        device=args.device,
    )
    encoder = VGGEncoder(
        vgg=vgg,
        head=head,
        param_mean=np.empty(0, dtype=np.float32),
        param_std=np.empty(0, dtype=np.float32),
        model_path=head_path,
        weights=encoder_meta["vgg_weights"],
        image_size=encoder_meta["vgg_image_size"],
        value_min=encoder_meta["vgg_value_min"],
        value_max=encoder_meta["vgg_value_max"],
        pool=encoder_meta["vgg_pool"],
        feature_dim=encoder_meta["feature_dim"],
        device=args.device,
    )
    real_images, real_theta, real_sim, real_sample, real_slice = _load_real_images(
        data_root, heldout, normalization
    )
    omega_index = list(PARAM_NAMES).index("Omega_m")

    config = _analysis_config(args, rows)
    _, _, split_membership = deterministic_balanced_real_split(
        real_sim, seed=args.seed
    )
    config["metrics"]["real_split_baseline"]["membership"] = split_membership
    config_bytes = (
        json.dumps(config, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode()
    config_sha256 = hashlib.sha256(config_bytes).hexdigest()
    config["sha256"] = config_sha256

    raw_blocks: list[np.ndarray] = []
    metadata_blocks: list[pd.DataFrame] = []
    source_artifacts: list[dict[str, Any]] = []
    transformed_images: dict[tuple[str, str], np.ndarray] = {}
    run_parameters: dict[str, dict[str, Any]] = {}

    def add_block(
        images: np.ndarray,
        *,
        run_name: str,
        dataset_size: int,
        source: str,
        transform: str,
        sim_index: np.ndarray,
        sample_index: np.ndarray,
        slice_index: np.ndarray,
        omega_m: np.ndarray,
    ) -> None:
        raw = vgg_embed(
            images,
            encoder.vgg,
            device=encoder.device,
            batch_size=args.embedding_batch_size,
            image_size=encoder.image_size,
            value_min=encoder.value_min,
            value_max=encoder.value_max,
            pool=encoder.pool,
        )
        if raw.shape != (len(images), 1024):
            raise RuntimeError(f"unexpected VGG feature matrix: {raw.shape}")
        raw_blocks.append(raw)
        metadata_blocks.append(
            source_metadata(
                run_name=run_name,
                dataset_size=dataset_size,
                source=source,
                transform=transform,
                sim_index=sim_index,
                sample_index=sample_index,
                slice_index=slice_index,
                omega_m=omega_m,
                code_revision=args.expected_code_revision,
                config_sha256=config_sha256,
            )
        )

    add_block(
        real_images,
        run_name="shared_real_original",
        dataset_size=0,
        source="real_original",
        transform="identity",
        sim_index=real_sim,
        sample_index=real_sample,
        slice_index=real_slice,
        omega_m=real_theta[:, omega_index],
    )

    for row in rows:
        run_name = str(row["run_name"])
        dataset_size = int(row["dataset_size"])
        saved = power["runs"][run_name]
        if int(saved["dataset_size"]) != dataset_size:
            raise RuntimeError(f"C4 dataset size mismatch for {run_name}")
        k_bins = np.asarray(saved["k_bins"], dtype=np.float64)
        measured_transfer = np.asarray(saved["measured_transfer"], dtype=np.float64)
        sigma = float(saved["gaussian_sigma_pixels"])
        measured_images, _ = transfer_transform(k_bins, measured_transfer)(real_images)
        gaussian_images, _ = gaussian_smoothing_transform(sigma)(real_images)
        transformed_images[(run_name, "real_measured_transfer")] = measured_images
        transformed_images[(run_name, "real_gaussian")] = gaussian_images
        run_parameters[run_name] = {
            "dataset_size": dataset_size,
            "gaussian_sigma_pixels": sigma,
            "measured_power_deficit_depth": measured_power_deficit_depth(
                saved["power_ratio"]
            ),
        }

        sample_path = generated_sample_path(
            results_root,
            row,
            seed=args.seed,
            samples_per_cosmology=64,
        )
        validate_c4_sample_path(saved, sample_path)
        source_artifacts.append(artifact_record(sample_path))
        with np.load(sample_path, allow_pickle=True) as generated_payload:
            generated = generated_payload["samples"].astype(np.float32)
            generated_theta = generated_payload["theta_raw"].astype(np.float32)
            generated_heldout = generated_payload["heldout_indices"].astype(np.int64)
            sample_count = int(generated_payload["samples_per_cosmology"])
        if sample_count != 64 or not np.array_equal(generated_heldout, heldout):
            raise RuntimeError(f"generated sample contract mismatch: {sample_path}")
        generated, generated_theta_rows, generated_sim, generated_sample = (
            subset_generated_cosmologies(
                generated,
                generated_theta,
                generated_heldout,
                samples_per_cosmology=64,
                selected_simulations=heldout,
            )
        )
        transforms = {
            "real_measured_transfer": f"transfer_Tk__{run_name}__N{dataset_size}",
            "real_gaussian": f"gaussian_smoothing__{run_name}__N{dataset_size}",
            "generated": f"generated__{run_name}__N{dataset_size}",
        }
        for images, source in (
            (measured_images, "real_measured_transfer"),
            (gaussian_images, "real_gaussian"),
        ):
            add_block(
                images,
                run_name=run_name,
                dataset_size=dataset_size,
                source=source,
                transform=transforms[source],
                sim_index=real_sim,
                sample_index=real_sample,
                slice_index=real_slice,
                omega_m=real_theta[:, omega_index],
            )
        add_block(
            generated,
            run_name=run_name,
            dataset_size=dataset_size,
            source="generated",
            transform=transforms["generated"],
            sim_index=generated_sim,
            sample_index=generated_sample,
            slice_index=np.full(len(generated), -1, dtype=np.int64),
            omega_m=generated_theta_rows[:, omega_index],
        )

    raw_features = np.concatenate(raw_blocks).astype(np.float32, copy=False)
    metadata = pd.concat(metadata_blocks, ignore_index=True)
    if len(raw_features) != len(metadata) or len(metadata) != 14_336:
        raise RuntimeError(
            f"expected 14,336 balanced feature rows; got {len(raw_features)}"
        )
    standardized, scaler_report = frozen_mlp_inputs(encoder.head, raw_features)

    import joblib
    import umap

    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=args.umap_neighbors,
        min_dist=args.umap_min_dist,
        metric="euclidean",
        random_state=args.seed,
        transform_seed=args.seed,
        n_jobs=1,
        low_memory=True,
    )
    embedding, umap_connectivity = fit_umap_with_connectivity_report(
        reducer, standardized
    )
    if embedding.shape != (len(metadata), 2) or not np.isfinite(embedding).all():
        raise RuntimeError(f"UMAP returned invalid embedding: {embedding.shape}")
    metadata["umap_1"] = embedding[:, 0]
    metadata["umap_2"] = embedding[:, 1]

    complete_metrics, headline_metrics, identity_report, mixing_baseline = (
        build_c4_metric_tables(
            metadata=metadata,
            standardized=standardized,
            original_images=real_images,
            transformed_images=transformed_images,
            run_parameters=run_parameters,
            k=args.knn_k,
            n_boot=args.bootstrap,
            seed=args.seed,
        )
    )
    if mixing_baseline["split_membership"] != split_membership:
        raise RuntimeError("real-split baseline membership changed after config hashing")
    visual_only = build_umap_visual_only_table(metadata, embedding)

    try:
        _write_json(staging / "analysis_config.json", config)
        np.savez_compressed(
            staging / "frozen_vgg_features_and_umap.npz",
            raw_vgg_features=raw_features,
            standardized_mlp_inputs=standardized,
            umap_embedding=embedding,
        )
        metadata.to_csv(staging / "sample_provenance.csv", index=False)
        complete_metrics.to_csv(staging / "c4_feature_metrics_complete.csv", index=False)
        _write_json(
            staging / "c4_feature_metrics_complete.json",
            {"metrics": complete_metrics.to_dict("records")},
        )
        headline_metrics.to_csv(staging / "c4_feature_metrics_headline.csv", index=False)
        _write_json(
            staging / "c4_feature_metrics_headline.json",
            {"metrics": headline_metrics.to_dict("records")},
        )
        identity_report.to_csv(staging / "c4_transform_identity_report.csv", index=False)
        _write_json(
            staging / "c4_transform_identity_report.json",
            {"transforms": identity_report.to_dict("records")},
        )
        visual_only.to_csv(
            staging / "umap_layout_separation_visual_only.csv", index=False
        )
        _write_json(
            staging / "umap_layout_separation_visual_only.json",
            {
                "warning": "UMAP coordinates are visualization-only, not a metric space",
                "rows": visual_only.to_dict("records"),
            },
        )
        joblib.dump(reducer, staging / "pooled_umap_reducer.joblib")
        for row in rows:
            _plot_run(
                metadata,
                str(row["run_name"]),
                staging / f"{row['run_name']}_c4_frozen_probe_umap.png",
            )
        plot_mixing_baselines(
            complete_metrics, staging / "c4_knn_mixing_baselines.png"
        )

        full_metrics = complete_metrics
        findings = []
        for row in rows:
            run_metrics = full_metrics[full_metrics["run_name"].eq(row["run_name"])].set_index("source")
            measured = run_metrics.loc["real_measured_transfer"]
            generated = run_metrics.loc["generated"]
            findings.append(
                {
                    "run_name": row["run_name"],
                    "measured_centroid_distance": float(measured["centroid_distance"]),
                    "generated_centroid_distance": float(generated["centroid_distance"]),
                    "measured_knn_cross_source_fraction": float(measured["knn_cross_source_fraction"]),
                    "generated_knn_cross_source_fraction": float(generated["knn_cross_source_fraction"]),
                    "prominent_if_true": bool(
                        measured["centroid_distance"] >= generated["centroid_distance"]
                        or measured["knn_cross_source_fraction"] <= generated["knn_cross_source_fraction"]
                    ),
                    "flag_definition": (
                        "measured-transfer real is at least as far by centroid distance OR "
                        "no better mixed by kNN than generated, in the full 1024D standardized space"
                    ),
                }
            )
        _write_json(staging / "measured_transfer_sanity_check.json", {"runs": findings})

        manifest = {
            "code_git": code_state,
            "expected_results_revision": args.expected_results_revision,
            "analysis_config_sha256": config_sha256,
            "analysis_config": config,
            "feature_contract": scaler_report,
            "threadpool_compatibility": threadpool_compatibility_report(),
            "umap_connectivity": umap_connectivity,
            "identity_tolerances": {"dtype": "float32", "rtol": 1e-6, "atol": 1e-7},
            "run_parameters": run_parameters,
            "mixing_baseline": mixing_baseline,
            "explicit_frozen_artifact_loading": explicit_load_report,
            "encoder_metadata": encoder_meta,
            "environment": {
                "python": sys.version,
                "numpy": np.__version__,
                "pandas": pd.__version__,
                "torch": importlib.metadata.version("torch"),
                "torchvision": importlib.metadata.version("torchvision"),
                "scikit_learn": importlib.metadata.version("scikit-learn"),
                "umap_learn": importlib.metadata.version("umap-learn"),
                "joblib": importlib.metadata.version("joblib"),
            },
            "artifacts_read": {
                "encoder": encoder_artifact,
                "frozen_head": head_artifact,
                "vgg_weights": weights_artifact,
                "c4_power_transfer": artifact_record(power_path),
                "c4_manifest": artifact_record(c4_manifest_path),
                "generated_samples": source_artifacts,
                "run_manifest": artifact_record(manifest_path),
                "raw_grid": large_input_record(image_path(data_root)),
                "raw_parameters": artifact_record(params_path(data_root)),
            },
            "row_count": int(len(metadata)),
            "output_files": sorted(
                {path.name for path in staging.iterdir()} | {"manifest.json"}
            ),
        }
        update_incomplete_report(
            staging / "manifest.json",
            payload=manifest,
            producer_job_id=os.environ.get("SLURM_JOB_ID"),
        )
        output_dir.parent.mkdir(parents=True, exist_ok=True)
        os.replace(staging, output_dir)
    except Exception:
        print(f"Partial staging output retained for diagnosis: {staging}", file=sys.stderr)
        raise

    print(f"Wrote frozen-probe C4 v3 analysis to {output_dir}; awaiting job finalization")
    print(f"features={raw_features.shape} embedding={embedding.shape}")
    print(complete_metrics.to_string(index=False))


if __name__ == "__main__":
    main()
