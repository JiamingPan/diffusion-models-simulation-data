#!/usr/bin/env python
"""Full-reference PCA nearest-neighbor diagnostic for nf_generalize_nick_data.

This computes the metric Nick asked for:

    s_j = max_i cosine(z_j, z_train_i)

where ``z`` is a normalized PCA embedding.  Unlike the notebook quick-check,
this script uses the full training/reference set for each dataset size and
computes nearest-neighbor similarities row-by-row so the largest run does not
need an in-memory all-pairs matrix.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml


SWEEP_NAME = "nf_generalize_nick_data"
DEFAULT_SAMPLE_LABEL = "raw_train_full"


@dataclass
class PCAEncoder:
    mean: np.ndarray
    scale: np.ndarray
    components: np.ndarray
    explained_variance_ratio: np.ndarray

    def transform_images(self, images: np.ndarray, batch_size: int = 512) -> np.ndarray:
        """Project ``(N,1,H,W)`` or ``(N,H,W)`` images into L2-normalized PCA space."""
        images = as_nchw(images)
        rows: list[np.ndarray] = []
        for start in range(0, len(images), batch_size):
            batch = images[start:start + batch_size].reshape(len(images[start:start + batch_size]), -1)
            batch = (batch.astype(np.float32, copy=False) - self.mean) / self.scale
            emb = batch @ self.components.T
            emb = emb.astype(np.float32, copy=False)
            norm = np.linalg.norm(emb, axis=1, keepdims=True)
            emb = emb / np.maximum(norm, 1e-12)
            rows.append(emb)
        return np.concatenate(rows, axis=0) if rows else np.empty((0, self.components.shape[0]), dtype=np.float32)


def as_nchw(images: np.ndarray) -> np.ndarray:
    arr = np.asarray(images)
    if arr.ndim == 3:
        return arr[:, None, :, :]
    if arr.ndim == 4 and arr.shape[1] == 1:
        return arr
    raise ValueError(f"Expected (N,H,W) or (N,1,H,W), got {arr.shape}.")


def evenly_limit(arr: np.ndarray, n: int | None) -> np.ndarray:
    if n is None or n <= 0 or len(arr) <= n:
        return arr
    idx = np.linspace(0, len(arr) - 1, int(n), dtype=np.int64)
    return arr[idx]


def load_npz_array(path: Path) -> np.ndarray:
    with np.load(path) as data:
        if "samples" in data:
            arr = data["samples"]
        elif "arr_0" in data:
            arr = data["arr_0"]
        else:
            arr = data[data.files[0]]
    return as_nchw(np.asarray(arr, dtype=np.float32))


def load_manifest(project_dir: Path, manifest_path: Path | None) -> list[dict[str, Any]]:
    path = manifest_path or project_dir / "local" / SWEEP_NAME / "manifest.json"
    with path.open() as f:
        return json.load(f)


def load_config(project_dir: Path, row: dict[str, Any]) -> dict[str, Any]:
    config_path = project_dir / row["config"]
    with config_path.open() as f:
        return yaml.safe_load(f)


def load_source_slices(path: str | Path, start: int, count: int, zthin: int) -> np.ndarray:
    if count <= 0:
        return np.empty((0, 128, 128), dtype=np.float32)
    arr = np.load(path, mmap_mode="r")
    stop = min(start + count, len(arr))
    if stop <= start:
        return np.empty((0, arr.shape[-2], arr.shape[-1]), dtype=np.float32)
    # Read only the z-thinned slices, not the full 3D cube volume.
    slices = np.asarray(arr[start:stop, ::zthin], dtype=np.float32)
    return slices.reshape(-1, slices.shape[-2], slices.shape[-1])


def normalize_like_training(
    train_images: np.ndarray,
    val_images: np.ndarray,
    config: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    data_cfg = config["data"]
    transform = data_cfg.get("transform", None)
    use_log = bool(data_cfg.get("log", False)) or (
        isinstance(transform, (list, tuple)) and "log" in transform
    )

    train = train_images.astype(np.float32, copy=False)
    val = val_images.astype(np.float32, copy=False)
    if use_log:
        np.log(train, out=train)
        if len(val):
            np.log(val, out=val)

    normalization = data_cfg.get("normalization", None)
    norm_kwargs = dict(data_cfg.get("norm_kwargs") or {})
    info: dict[str, float] = {}

    if normalization in {"tanh", "centermax", "center-max", "centered_maxabs"}:
        center = norm_kwargs.get("center", None)
        if center is None:
            center = float(train.mean())
        train -= np.float32(center)
        if len(val):
            val -= np.float32(center)

        xmax = norm_kwargs.get("xmax", None)
        if xmax is None:
            xmax = float(np.abs(train).max())
        xmax = max(float(xmax), 1e-30)
        train /= np.float32(xmax)
        if len(val):
            val /= np.float32(xmax)
        info.update({"center": float(center), "xmax": float(xmax)})

    if normalization == "tanh":
        alpha = float(norm_kwargs.get("alpha", 1.0))
        beta = float(norm_kwargs.get("beta", 1.0))
        gamma = float(norm_kwargs.get("gamma", 1.0))
        delta = float(norm_kwargs.get("delta", 1.0))
        sigma = float(norm_kwargs.get("sigma", 1.0))
        mu = float(norm_kwargs.get("mu", 0.0))

        train = tanh_transform(train, alpha, beta, gamma, delta, sigma, mu)
        if len(val):
            val = tanh_transform(val, alpha, beta, gamma, delta, sigma, mu)
    elif normalization in {None, "none", "centermax", "center-max", "centered_maxabs"}:
        pass
    else:
        raise ValueError(f"This script supports tanh/centered normalization, got {normalization!r}.")

    return as_nchw(train), as_nchw(val), info


def tanh_transform(
    images: np.ndarray,
    alpha: float,
    beta: float,
    gamma: float,
    delta: float,
    sigma: float,
    mu: float,
) -> np.ndarray:
    shifted = images - np.float32(mu)
    pos = alpha * np.tanh((gamma * shifted) / alpha)
    neg = beta * np.tanh((delta * shifted) / beta)
    return (np.where(shifted >= 0, pos, neg) * sigma).astype(np.float32, copy=False)


def load_train_and_validation(
    row: dict[str, Any],
    config: dict[str, Any],
    val_raw_per_source: int,
    max_val_slices: int | None,
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    zthin = int(row.get("zthin", config["data"].get("zthin", 1)))
    train_parts: list[np.ndarray] = []
    val_parts: list[np.ndarray] = []
    for source in row["source_counts"]:
        n_train = int(source["n_samples"])
        source_path = source["path"]
        train_parts.append(load_source_slices(source_path, 0, n_train, zthin))
        val_parts.append(load_source_slices(source_path, n_train, int(val_raw_per_source), zthin))

    train_raw = np.concatenate(train_parts, axis=0)
    nonempty_val = [x for x in val_parts if len(x)]
    if nonempty_val:
        val_raw = np.concatenate(nonempty_val, axis=0)
    else:
        val_raw = np.empty((0, train_raw.shape[-2], train_raw.shape[-1]), dtype=np.float32)
    val_raw = evenly_limit(val_raw, max_val_slices)
    train, val, norm_info = normalize_like_training(train_raw, val_raw, config)
    norm_info.update({
        "n_train_loaded": float(len(train)),
        "n_val_loaded": float(len(val)),
    })
    return train, val, norm_info


def fit_pca_encoder(
    images: np.ndarray,
    n_components: int,
    max_fit: int | None,
    target_variance: float | None = None,
) -> PCAEncoder:
    fit_images = evenly_limit(as_nchw(images), max_fit)
    x = fit_images.reshape(len(fit_images), -1).astype(np.float32, copy=False)
    mean = x.mean(axis=0, keepdims=True)
    scale = x.std(axis=0, keepdims=True)
    scale = np.maximum(scale, 1e-6).astype(np.float32, copy=False)
    x = (x - mean) / scale

    rank = min(int(n_components), x.shape[0] - 1, x.shape[1])
    if rank < 1:
        raise ValueError("Need at least two images to fit PCA.")

    try:
        from sklearn.decomposition import PCA

        pca = PCA(n_components=rank, svd_solver="randomized", random_state=0)
        pca.fit(x)
        components = pca.components_.astype(np.float32, copy=False)
        evr = pca.explained_variance_ratio_.astype(np.float32, copy=False)
    except Exception as exc:
        print(f"sklearn PCA failed or unavailable; falling back to numpy SVD: {exc!r}")
        _, s, vt = np.linalg.svd(x, full_matrices=False)
        components = vt[:rank].astype(np.float32, copy=False)
        ev = (s[:rank] ** 2) / max(x.shape[0] - 1, 1)
        total = float(np.var(x, axis=0, ddof=1).sum())
        evr = (ev / max(total, 1e-12)).astype(np.float32, copy=False)

    if target_variance is not None and target_variance > 0:
        cumulative = np.cumsum(evr)
        if cumulative[-1] + 1e-6 < target_variance:
            raise RuntimeError(
                f"PCA rank cap {rank} only explains {cumulative[-1]:.4f}, "
                f"below requested target {target_variance:.4f}. Increase --pca-components."
            )
        keep = int(np.searchsorted(cumulative, target_variance, side="left") + 1)
        components = components[:keep].astype(np.float32, copy=False)
        evr = evr[:keep].astype(np.float32, copy=False)
        print(
            f"PCA target_variance={target_variance:.4f}: selected rank {keep} "
            f"from cap {rank} with explained_variance_sum={float(evr.sum()):.4f}",
            flush=True,
        )

    return PCAEncoder(
        mean=mean.squeeze(0).astype(np.float32, copy=False),
        scale=scale.squeeze(0).astype(np.float32, copy=False),
        components=components,
        explained_variance_ratio=evr,
    )


def max_similarity(query: np.ndarray, reference: np.ndarray, batch_size: int = 512) -> np.ndarray:
    out = np.empty(len(query), dtype=np.float32)
    for start in range(0, len(query), batch_size):
        stop = min(start + batch_size, len(query))
        sims = query[start:stop] @ reference.T
        out[start:stop] = sims.max(axis=1)
    return out


def leave_one_out_max_similarity(reference: np.ndarray, batch_size: int = 512) -> np.ndarray:
    out = np.empty(len(reference), dtype=np.float32)
    for start in range(0, len(reference), batch_size):
        stop = min(start + batch_size, len(reference))
        sims = reference[start:stop] @ reference.T
        rows = np.arange(stop - start)
        cols = np.arange(start, stop)
        sims[rows, cols] = -np.inf
        out[start:stop] = sims.max(axis=1)
    return out


def summarize(values: np.ndarray, prefix: str) -> dict[str, float]:
    if len(values) == 0:
        return {
            f"{prefix}_median": math.nan,
            f"{prefix}_q90": math.nan,
            f"{prefix}_q95": math.nan,
            f"{prefix}_q99": math.nan,
            f"{prefix}_mean": math.nan,
        }
    return {
        f"{prefix}_median": float(np.median(values)),
        f"{prefix}_q90": float(np.quantile(values, 0.90)),
        f"{prefix}_q95": float(np.quantile(values, 0.95)),
        f"{prefix}_q99": float(np.quantile(values, 0.99)),
        f"{prefix}_mean": float(np.mean(values)),
    }


def threshold_label(value: float) -> str:
    return f"{value:.3f}".rstrip("0").rstrip(".").replace(".", "p")


def parse_float_list(value: str) -> list[float]:
    return [float(x.strip()) for x in value.split(",") if x.strip()]


def sample_path_for(project_dir: Path, row: dict[str, Any], seed: int, sample_label: str) -> Path:
    if row.get("sample_path"):
        raw = str(row["sample_path"])
        if sample_label != DEFAULT_SAMPLE_LABEL and "{sample_label}" not in raw:
            raw = raw.replace(DEFAULT_SAMPLE_LABEL, sample_label)
        return project_dir / raw.format(seed=seed, sample_label=sample_label)
    return project_dir / "results" / SWEEP_NAME / "samples" / f"{row['run_name']}_seed{seed}_{sample_label}.npz"


def selected_rows(rows: list[dict[str, Any]], args: argparse.Namespace) -> list[dict[str, Any]]:
    out = rows
    if getattr(args, "arch", None):
        wanted = set(args.arch)
        out = [row for row in out if row.get("arch") in wanted]
    if args.dataset_tag:
        wanted = set(args.dataset_tag)
        out = [row for row in out if row["dataset_tag"] in wanted]
    if args.run_name:
        wanted = set(args.run_name)
        out = [row for row in out if row["run_name"] in wanted]
    return sorted(out, key=lambda row: (str(row.get("arch", "")), int(row["dataset_size"])))


def _grouped_curve_inputs(df: pd.DataFrame) -> list[tuple[str | None, pd.DataFrame]]:
    if "arch_label" in df.columns and df["arch_label"].nunique(dropna=True) > 1:
        return list(df.groupby("arch_label", sort=False))
    if "arch" in df.columns and df["arch"].nunique(dropna=True) > 1:
        return list(df.groupby("arch", sort=False))
    return [(None, df)]


def _label_for(group_name: str | None, label: str) -> str:
    return label if group_name is None else f"{group_name} {label}"


def plot_outputs(df: pd.DataFrame, output_dir: Path, out_prefix: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    groups = _grouped_curve_inputs(df)

    with plt.rc_context({
        "font.size": 13,
        "axes.titlesize": 15,
        "axes.labelsize": 14,
        "legend.fontsize": 11,
        "figure.titlesize": 17,
    }):
        fig, axes = plt.subplots(1, 2, figsize=(15, 5.2), sharex=True)
        for group_name, sub in groups:
            x = sub["dataset_size"].astype(float).to_numpy()
            axes[0].plot(x, sub["gen_nn_median"], "o-", label=_label_for(group_name, "generated -> train"))
            axes[0].plot(x, sub["val_nn_median"], "o-", label=_label_for(group_name, "held-out real -> train"))
            axes[0].plot(x, sub["ref_nn_median"], "o--", label=_label_for(group_name, "train real leave-one-out"))
        axes[0].set_ylabel("median nearest-neighbor cosine")
        axes[0].set_title("median full-reference NN similarity")

        for group_name, sub in groups:
            x = sub["dataset_size"].astype(float).to_numpy()
            axes[1].plot(x, sub["gen_nn_q99"], "o-", label=_label_for(group_name, "generated q99"))
            axes[1].plot(x, sub["val_nn_q99"], "o-", label=_label_for(group_name, "held-out real q99"))
            axes[1].plot(x, sub["threshold_q99"], "s:", label=_label_for(group_name, "train real q99 threshold"))
        axes[1].set_ylabel("nearest-neighbor cosine")
        axes[1].set_title("q99 versus real-real threshold")

        for ax in axes:
            ax.set_xscale("log", base=2)
            ax.set_xlabel("training dataset size")
            ax.grid(alpha=0.25)
            ax.legend()

        fig.suptitle("PCA full-reference nearest-neighbor diagnostic")
        fig.tight_layout(rect=(0, 0, 1, 0.93))
        out = output_dir / f"{out_prefix}_similarity_curves.png"
        fig.savefig(out, dpi=180, bbox_inches="tight")
        print("wrote", out)
        plt.close(fig)

        fig, axes = plt.subplots(1, 2, figsize=(15, 5.2), sharex=True)
        for group_name, sub in groups:
            x = sub["dataset_size"].astype(float).to_numpy()
            axes[0].plot(x, sub["gen_copy_fraction_q95"], "o-", label=_label_for(group_name, "generated > q95"))
            axes[0].plot(x, sub["gen_copy_fraction_q99"], "o-", label=_label_for(group_name, "generated > q99"))
        axes[0].set_ylabel("copy-like fraction")
        axes[0].set_title("generated above real-real thresholds")

        for group_name, sub in groups:
            x = sub["dataset_size"].astype(float).to_numpy()
            axes[1].plot(x, sub["val_copy_fraction_q95"], "o-", label=_label_for(group_name, "held-out real > q95"))
            axes[1].plot(x, sub["val_copy_fraction_q99"], "o-", label=_label_for(group_name, "held-out real > q99"))
        axes[1].set_ylabel("near-real-neighbor fraction")
        axes[1].set_title("held-out real above real-real thresholds")

        for ax in axes:
            ax.set_xscale("log", base=2)
            ax.set_xlabel("training dataset size")
            ax.grid(alpha=0.25)
            ax.legend()

        fig.suptitle("PCA full-reference threshold crossings")
        fig.tight_layout(rect=(0, 0, 1, 0.93))
        out = output_dir / f"{out_prefix}_copy_fraction_curves.png"
        fig.savefig(out, dpi=180, bbox_inches="tight")
        print("wrote", out)
        plt.close(fig)

        fig, axes = plt.subplots(1, 2, figsize=(15, 5.2), sharex=True)
        for group_name, sub in groups:
            x = sub["dataset_size"].astype(float).to_numpy()
            for suffix in ("q95", "q99"):
                gen_col = f"gen_gl_{suffix}"
                val_col = f"val_gl_{suffix}"
                if gen_col in sub:
                    axes[0].plot(x, sub[gen_col], "o-", label=_label_for(group_name, f"generated, train-real {suffix}"))
                if val_col in sub:
                    axes[0].plot(x, sub[val_col], "o--", alpha=0.7, label=_label_for(group_name, f"held-out real, train-real {suffix}"))
        axes[0].set_ylabel("PCA GL = 1 - fraction above threshold")
        axes[0].set_title("adaptive real-real threshold")

        fixed_cols = sorted(
            (c for c in df.columns if c.startswith("gen_gl_fixed_")),
            key=lambda c: float(c.rsplit("_", 1)[-1].replace("p", ".")),
        )
        for group_name, sub in groups:
            x = sub["dataset_size"].astype(float).to_numpy()
            for col in fixed_cols:
                label = col.rsplit("_", 1)[-1].replace("p", ".")
                axes[1].plot(x, sub[col], "o-", label=_label_for(group_name, f"generated, tau={label}"))
        axes[1].set_ylabel("PCA GL = 1 - fraction above fixed tau")
        axes[1].set_title("fixed PCA similarity threshold")

        for ax in axes:
            ax.set_xscale("log", base=2)
            ax.set_ylim(-0.03, 1.03)
            ax.set_xlabel("training dataset size")
            ax.grid(alpha=0.25)
            ax.legend()

        fig.suptitle("PCA paper-style generalizability score")
        fig.tight_layout(rect=(0, 0, 1, 0.93))
        out = output_dir / f"{out_prefix}_paper_style_gl_curves.png"
        fig.savefig(out, dpi=180, bbox_inches="tight")
        print("wrote", out)
        plt.close(fig)


def reproducibility_rows(
    generated_embeddings: dict[tuple[str, int], np.ndarray],
    thresholds: list[float],
    adaptive_thresholds: dict[tuple[str, int], dict[str, float]] | None = None,
    similarity_batch_size: int = 512,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    arches = sorted({arch for arch, _ in generated_embeddings})
    if len(arches) < 2:
        return rows
    for i, arch_a in enumerate(arches):
        for arch_b in arches[i + 1:]:
            sizes = sorted(
                size
                for size in {size for arch, size in generated_embeddings if arch == arch_a}
                if (arch_b, size) in generated_embeddings
            )
            for size in sizes:
                a = generated_embeddings[(arch_a, size)]
                b = generated_embeddings[(arch_b, size)]
                n = min(len(a), len(b))
                if n == 0:
                    continue
                paired = np.sum(a[:n] * b[:n], axis=1)
                a_to_b = max_similarity(a, b, batch_size=similarity_batch_size)
                b_to_a = max_similarity(b, a, batch_size=similarity_batch_size)
                unpaired = np.concatenate([a_to_b, b_to_a])
                rec: dict[str, Any] = {
                    "arch_pair": f"{arch_a}_vs_{arch_b}",
                    "arch_a": arch_a,
                    "arch_b": arch_b,
                    "dataset_size": int(size),
                    "n_paired": int(n),
                    "n_unpaired_query": int(len(unpaired)),
                    "paired_similarity_mean": float(np.mean(paired)),
                    "paired_similarity_median": float(np.median(paired)),
                    "paired_similarity_q90": float(np.quantile(paired, 0.90)),
                    "paired_similarity_q99": float(np.quantile(paired, 0.99)),
                    "unpaired_nn_mean": float(np.mean(unpaired)),
                    "unpaired_nn_median": float(np.median(unpaired)),
                    "unpaired_nn_q90": float(np.quantile(unpaired, 0.90)),
                    "unpaired_nn_q99": float(np.quantile(unpaired, 0.99)),
                    "a_to_b_nn_mean": float(np.mean(a_to_b)),
                    "a_to_b_nn_median": float(np.median(a_to_b)),
                    "a_to_b_nn_q90": float(np.quantile(a_to_b, 0.90)),
                    "a_to_b_nn_q99": float(np.quantile(a_to_b, 0.99)),
                    "b_to_a_nn_mean": float(np.mean(b_to_a)),
                    "b_to_a_nn_median": float(np.median(b_to_a)),
                    "b_to_a_nn_q90": float(np.quantile(b_to_a, 0.90)),
                    "b_to_a_nn_q99": float(np.quantile(b_to_a, 0.99)),
                }
                for threshold in thresholds:
                    suffix = threshold_label(threshold)
                    rec[f"rp_fixed_{suffix}"] = float(np.mean(paired > threshold))
                    rec[f"rp_unpaired_fixed_{suffix}"] = float(np.mean(unpaired > threshold))
                    rec[f"rp_unpaired_a_to_b_fixed_{suffix}"] = float(np.mean(a_to_b > threshold))
                    rec[f"rp_unpaired_b_to_a_fixed_{suffix}"] = float(np.mean(b_to_a > threshold))
                if adaptive_thresholds:
                    for suffix in ("q95", "q99"):
                        pair_thresholds = [
                            adaptive_thresholds.get((arch_a, size), {}).get(suffix, np.nan),
                            adaptive_thresholds.get((arch_b, size), {}).get(suffix, np.nan),
                        ]
                        finite_thresholds = [float(x) for x in pair_thresholds if np.isfinite(x)]
                        if finite_thresholds:
                            # Use the stricter of the two train-real thresholds.  For
                            # the u64/u128 Fig. 2 runs these should be nearly identical,
                            # because the real training reference set is the same.
                            adaptive_threshold = max(finite_thresholds)
                            rec[f"rp_threshold_{suffix}"] = adaptive_threshold
                            rec[f"rp_{suffix}"] = float(np.mean(paired > adaptive_threshold))
                            rec[f"rp_unpaired_{suffix}"] = float(np.mean(unpaired > adaptive_threshold))
                            rec[f"rp_unpaired_a_to_b_{suffix}"] = float(np.mean(a_to_b > adaptive_threshold))
                            rec[f"rp_unpaired_b_to_a_{suffix}"] = float(np.mean(b_to_a > adaptive_threshold))
                rows.append(rec)
    return rows


def plot_reproducibility(rp_df: pd.DataFrame, output_dir: Path, out_prefix: str, thresholds: list[float]) -> None:
    if rp_df.empty:
        return
    output_dir.mkdir(parents=True, exist_ok=True)
    with plt.rc_context({
        "font.size": 13,
        "axes.titlesize": 15,
        "axes.labelsize": 14,
        "legend.fontsize": 10,
        "figure.titlesize": 17,
    }):
        fig, axes = plt.subplots(1, 2, figsize=(15, 5.2), sharex=True)
        for pair, sub in rp_df.groupby("arch_pair", sort=False):
            x = sub["dataset_size"].astype(float).to_numpy()
            if "unpaired_nn_median" in sub and "unpaired_nn_q90" in sub:
                axes[0].plot(x, sub["unpaired_nn_median"], "o-", label=f"{pair} unpaired NN median")
                axes[0].plot(x, sub["unpaired_nn_q90"], "o--", label=f"{pair} unpaired NN q90")
            else:
                axes[0].plot(x, sub["paired_similarity_median"], "o-", label=f"{pair} paired median")
                axes[0].plot(x, sub["paired_similarity_q90"], "o--", label=f"{pair} paired q90")
        axes[0].set_ylabel("generated-to-generated cosine")
        axes[0].set_title("unpaired nearest-neighbor generated similarity")

        for pair, sub in rp_df.groupby("arch_pair", sort=False):
            x = sub["dataset_size"].astype(float).to_numpy()
            for threshold in thresholds:
                suffix = threshold_label(threshold)
                col = f"rp_unpaired_fixed_{suffix}" if f"rp_unpaired_fixed_{suffix}" in sub else f"rp_fixed_{suffix}"
                if col in sub:
                    label_kind = "unpaired NN" if col.startswith("rp_unpaired_") else "paired"
                    axes[1].plot(x, sub[col], "o-", label=f"{pair} {label_kind} tau={threshold:g}")
        axes[1].set_ylabel("RP = fraction above tau")
        axes[1].set_ylim(-0.03, 1.03)
        axes[1].set_title("nearest-neighbor reproducibility")

        for ax in axes:
            ax.set_xscale("log", base=2)
            ax.set_xlabel("training dataset size")
            ax.grid(alpha=0.25)
            ax.legend()
        fig.suptitle("PCA generated-set reproducibility")
        fig.tight_layout(rect=(0, 0, 1, 0.93))
        out = output_dir / f"{out_prefix}_reproducibility_curves.png"
        fig.savefig(out, dpi=180, bbox_inches="tight")
        print("wrote", out)
        plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-dir", default=".", help="Repository root.")
    parser.add_argument("--manifest", type=Path, help="Optional manifest path.")
    parser.add_argument("--run-name", action="append", help="Restrict to one run name. Repeatable.")
    parser.add_argument("--arch", action="append", help="Restrict to architecture tag like u64/u128. Repeatable.")
    parser.add_argument("--dataset-tag", action="append", help="Restrict to dataset tag like d2p11. Repeatable.")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--sample-label", default=DEFAULT_SAMPLE_LABEL)
    parser.add_argument("--pca-components", type=int, default=32)
    parser.add_argument("--pca-fit-max-slices", type=int, default=4096)
    parser.add_argument(
        "--pca-target-variance",
        type=float,
        default=0.0,
        help=(
            "If >0, fit up to --pca-components and then keep the smallest rank whose "
            "cumulative explained variance reaches this target. Use --pca-fit-max-slices 0 "
            "to fit on all loaded training slices."
        ),
    )
    parser.add_argument("--max-generated", type=int, default=512)
    parser.add_argument("--val-raw-per-source", type=int, default=8)
    parser.add_argument("--max-val-slices", type=int, default=512)
    parser.add_argument("--embedding-batch-size", type=int, default=512)
    parser.add_argument("--similarity-batch-size", type=int, default=512)
    parser.add_argument("--output-dir", type=Path, help="Figure output directory.")
    parser.add_argument("--table-dir", type=Path, help="CSV output directory.")
    parser.add_argument("--out-prefix", default="nf_generalize_nick_data_pca_full_nn")
    parser.add_argument(
        "--pca-fit-only",
        action="store_true",
        help="Fit the PCA encoder, print the selected rank/explained variance, and exit before scoring samples.",
    )
    parser.add_argument(
        "--fixed-similarity-thresholds",
        default="0.5,0.6,0.7,0.8,0.9,0.95",
        help="Comma-separated fixed PCA cosine thresholds for paper-style GL curves.",
    )
    parser.add_argument("--skip-missing-samples", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project_dir = Path(args.project_dir).resolve()
    output_dir = args.output_dir or project_dir / "results" / SWEEP_NAME / "quickcheck"
    table_dir = args.table_dir or project_dir / "results" / SWEEP_NAME / "tables"
    output_dir.mkdir(parents=True, exist_ok=True)
    table_dir.mkdir(parents=True, exist_ok=True)

    rows = selected_rows(load_manifest(project_dir, args.manifest), args)
    if not rows:
        raise SystemExit("No rows selected.")

    available_rows: list[dict[str, Any]] = []
    for row in rows:
        sample_path = sample_path_for(project_dir, row, args.seed, args.sample_label)
        if sample_path.exists():
            available_rows.append(row)
        elif args.skip_missing_samples:
            print(f"skipping missing sample: {sample_path}")
        else:
            raise FileNotFoundError(f"Missing sample: {sample_path}")
    rows = available_rows
    if not rows:
        raise SystemExit("No rows with samples selected.")
    fixed_thresholds = parse_float_list(args.fixed_similarity_thresholds)

    fit_row = max(rows, key=lambda row: int(row["dataset_size"]))
    fit_config = load_config(project_dir, fit_row)
    print(f"Fitting PCA on {fit_row['run_name']} with max {args.pca_fit_max_slices} slices")
    fit_train, _, fit_norm = load_train_and_validation(
        fit_row,
        fit_config,
        val_raw_per_source=0,
        max_val_slices=0,
    )
    encoder = fit_pca_encoder(
        fit_train,
        args.pca_components,
        args.pca_fit_max_slices,
        target_variance=args.pca_target_variance,
    )
    pca_evr_sum = float(np.sum(encoder.explained_variance_ratio))
    print(f"PCA rank={len(encoder.explained_variance_ratio)} explained_variance_sum={pca_evr_sum:.4f}")
    if args.pca_fit_only:
        print("PCA fit-only requested; exiting before nearest-neighbor scoring.")
        return
    del fit_train
    gc.collect()

    metric_rows: list[dict[str, Any]] = []
    generated_embeddings: dict[tuple[str, int], np.ndarray] = {}
    for row in rows:
        config = load_config(project_dir, row)
        sample_path = sample_path_for(project_dir, row, args.seed, args.sample_label)
        generated = evenly_limit(load_npz_array(sample_path), args.max_generated)
        print(
            f"scoring {row['run_name']} "
            f"N={row['dataset_size']} generated={len(generated)} full-reference"
        )

        real_ref, real_val, norm_info = load_train_and_validation(
            row,
            config,
            val_raw_per_source=args.val_raw_per_source,
            max_val_slices=args.max_val_slices,
        )
        ref_z = encoder.transform_images(real_ref, batch_size=args.embedding_batch_size)
        val_z = encoder.transform_images(real_val, batch_size=args.embedding_batch_size)
        gen_z = encoder.transform_images(generated, batch_size=args.embedding_batch_size)
        generated_embeddings[(str(row.get("arch", "")), int(row["dataset_size"]))] = gen_z.copy()

        ref_nn = leave_one_out_max_similarity(ref_z, batch_size=args.similarity_batch_size)
        val_nn = max_similarity(val_z, ref_z, batch_size=args.similarity_batch_size)
        gen_nn = max_similarity(gen_z, ref_z, batch_size=args.similarity_batch_size)

        thresholds = {
            "threshold_q90": float(np.quantile(ref_nn, 0.90)),
            "threshold_q95": float(np.quantile(ref_nn, 0.95)),
            "threshold_q99": float(np.quantile(ref_nn, 0.99)),
        }
        metrics: dict[str, Any] = {
            "run_name": row["run_name"],
            "arch": row.get("arch", ""),
            "arch_label": row.get("arch_label", row.get("arch", "")),
            "dataset_tag": row["dataset_tag"],
            "dataset_size": int(row["dataset_size"]),
            "n_train_ref": int(len(real_ref)),
            "n_val_real": int(len(real_val)),
            "n_generated": int(len(generated)),
            "sample_path": str(sample_path),
            "pca_fit_run_name": fit_row["run_name"],
            "pca_fit_dataset_size": int(fit_row["dataset_size"]),
            "pca_fit_max_slices": int(args.pca_fit_max_slices),
            "pca_target_variance": float(args.pca_target_variance),
            "pca_rank": int(len(encoder.explained_variance_ratio)),
            "pca_explained_variance_sum": pca_evr_sum,
            "normalization_center": norm_info.get("center", math.nan),
            "normalization_xmax": norm_info.get("xmax", math.nan),
            **thresholds,
            **summarize(ref_nn, "ref_nn"),
            **summarize(val_nn, "val_nn"),
            **summarize(gen_nn, "gen_nn"),
        }
        for key, threshold in thresholds.items():
            suffix = key.replace("threshold_", "")
            gen_copy = float(np.mean(gen_nn > threshold))
            val_copy = float(np.mean(val_nn > threshold))
            metrics[f"gen_copy_fraction_{suffix}"] = gen_copy
            metrics[f"val_copy_fraction_{suffix}"] = val_copy
            metrics[f"gen_gl_{suffix}"] = 1.0 - gen_copy
            metrics[f"val_gl_{suffix}"] = 1.0 - val_copy
        for threshold in fixed_thresholds:
            suffix = threshold_label(threshold)
            gen_copy = float(np.mean(gen_nn > threshold))
            val_copy = float(np.mean(val_nn > threshold))
            ref_copy = float(np.mean(ref_nn > threshold))
            metrics[f"fixed_threshold_{suffix}"] = float(threshold)
            metrics[f"gen_copy_fraction_fixed_{suffix}"] = gen_copy
            metrics[f"val_copy_fraction_fixed_{suffix}"] = val_copy
            metrics[f"ref_copy_fraction_fixed_{suffix}"] = ref_copy
            metrics[f"gen_gl_fixed_{suffix}"] = 1.0 - gen_copy
            metrics[f"val_gl_fixed_{suffix}"] = 1.0 - val_copy
            metrics[f"ref_gl_fixed_{suffix}"] = 1.0 - ref_copy
        metric_rows.append(metrics)

        del real_ref, real_val, generated, ref_z, val_z, gen_z, ref_nn, val_nn, gen_nn
        gc.collect()

    sort_cols = ["dataset_size"]
    if "arch" in metric_rows[0]:
        sort_cols = ["arch", "dataset_size"]
    df = pd.DataFrame(metric_rows).sort_values(sort_cols)
    metrics_path = table_dir / f"{args.out_prefix}_metrics.csv"
    df.to_csv(metrics_path, index=False)
    print("wrote", metrics_path)
    print(
        df[
            [
                "arch",
                "dataset_tag",
                "dataset_size",
                "n_train_ref",
                "n_val_real",
                "n_generated",
                "gen_nn_median",
                "val_nn_median",
                "threshold_q99",
                "gen_copy_fraction_q99",
                "val_copy_fraction_q99",
            ]
        ].to_string(index=False)
    )
    plot_outputs(df, output_dir, args.out_prefix)

    adaptive_thresholds = {
        (str(row["arch"]), int(row["dataset_size"])): {
            "q95": float(row["threshold_q95"]),
            "q99": float(row["threshold_q99"]),
        }
        for row in metric_rows
    }
    rp_rows = reproducibility_rows(
        generated_embeddings,
        fixed_thresholds,
        adaptive_thresholds,
        similarity_batch_size=args.similarity_batch_size,
    )
    if rp_rows:
        rp_df = pd.DataFrame(rp_rows).sort_values(["arch_pair", "dataset_size"])
        rp_path = table_dir / f"{args.out_prefix}_reproducibility.csv"
        rp_df.to_csv(rp_path, index=False)
        print("wrote", rp_path)
        print(rp_df.to_string(index=False))
        plot_reproducibility(rp_df, output_dir, args.out_prefix, fixed_thresholds)


if __name__ == "__main__":
    main()
