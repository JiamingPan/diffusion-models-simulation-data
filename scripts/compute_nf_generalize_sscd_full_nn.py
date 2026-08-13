#!/usr/bin/env python
"""Full-reference SSCD nearest-neighbor diagnostic for nf_generalize_nick_data.

This computes the paper-style generalizability score

    GL(tau) = 1 - P(max_i sim_sscd(x_j, y_i) > tau)

for each generated sample ``x_j`` against all training slices ``y_i`` used by
that run.  It also reports held-out-real and train-real leave-one-out baselines
so the SSCD score can be interpreted beside one-point/P(k) diagnostics.
"""

from __future__ import annotations

import argparse
import gc
import math
import os
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from compute_nf_generalize_pca_full_nn import (
    DEFAULT_SAMPLE_LABEL,
    SWEEP_NAME,
    evenly_limit,
    load_config,
    load_manifest,
    load_npz_array,
    load_train_and_validation,
    parse_float_list,
    sample_path_for,
    selected_rows,
    summarize,
    threshold_label,
)


def load_sscd_helpers(project_dir: Path):
    import sys

    if str(project_dir) not in sys.path:
        sys.path.insert(0, str(project_dir))
    from simdiff_eval.sscd import load_sscd_torchscript, sscd_embeddings

    return load_sscd_torchscript, sscd_embeddings


def cache_suffix(args: argparse.Namespace, kind: str) -> str:
    return (
        f"{kind}_{args.sample_label}_seed{args.seed}_"
        f"img{args.image_size}_{args.render_mode}_{args.device}"
    ).replace("/", "_")


def embedding_cache_path(
    cache_dir: Path,
    row: dict[str, Any],
    args: argparse.Namespace,
    kind: str,
) -> Path:
    return cache_dir / f"{row['run_name']}_{cache_suffix(args, kind)}.pt"


def embed_with_cache(
    images: np.ndarray,
    *,
    model: torch.nn.Module,
    embed_fn,
    row: dict[str, Any],
    args: argparse.Namespace,
    cache_dir: Path,
    kind: str,
    source_id: str | None = None,
) -> torch.Tensor:
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = embedding_cache_path(cache_dir, row, args, kind)
    if path.exists() and not args.refresh_cache:
        payload = torch.load(path, map_location="cpu")
        emb = payload["embeddings"] if isinstance(payload, dict) else payload
        cached_source_id = payload.get("source_id") if isinstance(payload, dict) else None
        source_matches = source_id is None or cached_source_id == source_id
        if len(emb) == len(images) and source_matches:
            print(f"  loaded cached {kind} embeddings: {path}")
            return F.normalize(emb.float(), dim=1)
        reason = "source mismatch" if not source_matches else "length mismatch"
        print(f"  cache {reason} for {path}; recomputing")

    emb = embed_fn(
        images,
        model,
        device=args.device,
        batch_size=args.embedding_batch_size,
        image_size=args.image_size,
        render_mode=args.render_mode,
        value_range=(args.value_min, args.value_max),
    )
    emb = F.normalize(emb.float().cpu(), dim=1)
    torch.save(
        {
            "embeddings": emb,
            "run_name": row["run_name"],
            "dataset_size": int(row["dataset_size"]),
            "kind": kind,
            "n_images": int(len(images)),
            "image_size": int(args.image_size),
            "render_mode": args.render_mode,
            "value_range": (float(args.value_min), float(args.value_max)),
            "source_id": source_id,
        },
        path,
    )
    print(f"  wrote cached {kind} embeddings: {path}")
    return emb


@torch.no_grad()
def max_similarity(query: torch.Tensor, reference: torch.Tensor, batch_size: int) -> np.ndarray:
    query = F.normalize(query.float(), dim=1)
    reference = F.normalize(reference.float(), dim=1)
    rows: list[torch.Tensor] = []
    for start in range(0, len(query), batch_size):
        stop = min(start + batch_size, len(query))
        sim = query[start:stop] @ reference.T
        rows.append(sim.max(dim=1).values.cpu())
    return torch.cat(rows).numpy().astype(np.float32, copy=False)


@torch.no_grad()
def leave_one_out_max_similarity(reference: torch.Tensor, batch_size: int) -> np.ndarray:
    reference = F.normalize(reference.float(), dim=1)
    rows_out: list[torch.Tensor] = []
    for start in range(0, len(reference), batch_size):
        stop = min(start + batch_size, len(reference))
        sim = reference[start:stop] @ reference.T
        rows = torch.arange(stop - start)
        cols = torch.arange(start, stop)
        sim[rows, cols] = -float("inf")
        rows_out.append(sim.max(dim=1).values.cpu())
    return torch.cat(rows_out).numpy().astype(np.float32, copy=False)


def plot_outputs(df: pd.DataFrame, output_dir: Path, out_prefix: str, primary_threshold: float) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    primary_suffix = threshold_label(primary_threshold)

    group_col = None
    if "arch_label" in df.columns and df["arch_label"].nunique(dropna=True) > 1:
        group_col = "arch_label"
    elif "arch" in df.columns and df["arch"].nunique(dropna=True) > 1:
        group_col = "arch"

    groups = [(None, df)] if group_col is None else list(df.groupby(group_col, sort=False))

    def label_for(group_name: str | None, label: str) -> str:
        return label if group_name is None else f"{group_name} {label}"

    with plt.rc_context({
        "font.size": 13,
        "axes.titlesize": 15,
        "axes.labelsize": 14,
        "legend.fontsize": 10,
        "figure.titlesize": 17,
    }):
        fig, axes = plt.subplots(1, 2, figsize=(15, 5.2), sharex=True)
        col = f"gen_gl_fixed_{primary_suffix}"
        copy_col = f"gen_copy_fraction_fixed_{primary_suffix}"
        for group_name, sub in groups:
            x = sub["dataset_size"].astype(float).to_numpy()
            if col in sub:
                axes[0].plot(
                    x,
                    sub[col],
                    "o-",
                    lw=2.5,
                    label=label_for(group_name, f"generated GL, tau={primary_threshold:g}"),
                )
            if f"val_gl_fixed_{primary_suffix}" in sub:
                axes[0].plot(
                    x,
                    sub[f"val_gl_fixed_{primary_suffix}"],
                    "o--",
                    alpha=0.75,
                    label=label_for(group_name, "held-out real GL baseline"),
                )
        axes[0].set_ylabel("SSCD GL = 1 - fraction above tau")
        axes[0].set_title("paper-style SSCD generalizability")

        fixed_cols = sorted(
            (c for c in df.columns if c.startswith("gen_gl_fixed_")),
            key=lambda c: float(c.rsplit("_", 1)[-1].replace("p", ".")),
        )
        for group_name, sub in groups:
            x = sub["dataset_size"].astype(float).to_numpy()
            for fixed_col in fixed_cols:
                label = fixed_col.rsplit("_", 1)[-1].replace("p", ".")
                axes[1].plot(x, sub[fixed_col], "o-", label=label_for(group_name, f"tau={label}"))
        axes[1].set_ylabel("SSCD GL")
        axes[1].set_title("fixed-threshold sensitivity")

        for ax in axes:
            ax.set_xscale("log", base=2)
            ax.set_ylim(-0.03, 1.03)
            ax.set_xlabel("training dataset size N")
            ax.grid(alpha=0.25)
            ax.legend()

        fig.suptitle("SSCD full-reference generalizability")
        fig.tight_layout(rect=(0, 0, 1, 0.93))
        out = output_dir / f"{out_prefix}_paper_style_gl_curves.png"
        fig.savefig(out, dpi=180, bbox_inches="tight")
        print("wrote", out)
        plt.close(fig)

        fig, axes = plt.subplots(1, 2, figsize=(15, 5.2), sharex=True)
        for group_name, sub in groups:
            x = sub["dataset_size"].astype(float).to_numpy()
            axes[0].plot(x, sub["gen_nn_median"], "o-", label=label_for(group_name, "generated -> train"))
            axes[0].plot(x, sub["val_nn_median"], "o-", label=label_for(group_name, "held-out real -> train"))
            axes[0].plot(x, sub["ref_nn_median"], "o--", label=label_for(group_name, "train real leave-one-out"))
        axes[0].set_ylabel("median nearest-neighbor SSCD cosine")
        axes[0].set_title("median full-reference NN similarity")

        for group_name, sub in groups:
            x = sub["dataset_size"].astype(float).to_numpy()
            axes[1].plot(x, sub["gen_nn_q99"], "o-", label=label_for(group_name, "generated q99"))
            axes[1].plot(x, sub["val_nn_q99"], "o-", label=label_for(group_name, "held-out real q99"))
            axes[1].plot(x, sub["threshold_q99"], "s:", label=label_for(group_name, "train real q99 threshold"))
        axes[1].set_ylabel("nearest-neighbor SSCD cosine")
        axes[1].set_title("q99 versus train-real threshold")

        for ax in axes:
            ax.set_xscale("log", base=2)
            ax.set_xlabel("training dataset size N")
            ax.grid(alpha=0.25)
            ax.legend()

        fig.suptitle("SSCD full-reference nearest-neighbor diagnostic")
        fig.tight_layout(rect=(0, 0, 1, 0.93))
        out = output_dir / f"{out_prefix}_similarity_curves.png"
        fig.savefig(out, dpi=180, bbox_inches="tight")
        print("wrote", out)
        plt.close(fig)

        if copy_col in df:
            fig, ax = plt.subplots(figsize=(7.5, 5.2))
            for group_name, sub in groups:
                x = sub["dataset_size"].astype(float).to_numpy()
                ax.plot(
                    x,
                    sub[copy_col],
                    "o-",
                    lw=2.5,
                    label=label_for(group_name, f"generated > tau={primary_threshold:g}"),
                )
                if f"val_copy_fraction_fixed_{primary_suffix}" in sub:
                    ax.plot(
                        x,
                        sub[f"val_copy_fraction_fixed_{primary_suffix}"],
                        "o--",
                        alpha=0.75,
                        label=label_for(group_name, "held-out real > tau"),
                    )
            ax.set_xscale("log", base=2)
            ax.set_ylim(-0.03, 1.03)
            ax.set_xlabel("training dataset size N")
            ax.set_ylabel("copy-like fraction")
            ax.set_title("SSCD threshold crossings")
            ax.grid(alpha=0.25)
            ax.legend()
            fig.tight_layout()
            out = output_dir / f"{out_prefix}_copy_fraction_curves.png"
            fig.savefig(out, dpi=180, bbox_inches="tight")
            print("wrote", out)
            plt.close(fig)


def reproducibility_rows(
    generated_embeddings: dict[tuple[str, int], torch.Tensor],
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
                a = F.normalize(generated_embeddings[(arch_a, size)].float(), dim=1)
                b = F.normalize(generated_embeddings[(arch_b, size)].float(), dim=1)
                n = min(len(a), len(b))
                if n == 0:
                    continue
                paired = (a[:n] * b[:n]).sum(dim=1).cpu().numpy()
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


def plot_reproducibility(
    rp_df: pd.DataFrame,
    output_dir: Path,
    out_prefix: str,
    thresholds: list[float],
    primary_threshold: float,
) -> None:
    if rp_df.empty:
        return
    primary_suffix = threshold_label(primary_threshold)
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
        axes[0].set_ylabel("generated-to-generated SSCD cosine")
        axes[0].set_title("unpaired nearest-neighbor generated similarity")

        for pair, sub in rp_df.groupby("arch_pair", sort=False):
            x = sub["dataset_size"].astype(float).to_numpy()
            primary_col = (
                f"rp_unpaired_fixed_{primary_suffix}"
                if f"rp_unpaired_fixed_{primary_suffix}" in sub
                else f"rp_fixed_{primary_suffix}"
            )
            if primary_col in sub:
                primary_kind = "unpaired NN" if primary_col.startswith("rp_unpaired_") else "paired"
                axes[1].plot(
                    x,
                    sub[primary_col],
                    "o-",
                    lw=2.5,
                    label=f"{pair} {primary_kind} tau={primary_threshold:g}",
                )
            for threshold in thresholds:
                if threshold == primary_threshold:
                    continue
                suffix = threshold_label(threshold)
                col = f"rp_unpaired_fixed_{suffix}" if f"rp_unpaired_fixed_{suffix}" in sub else f"rp_fixed_{suffix}"
                if col in sub:
                    label_kind = "unpaired NN" if col.startswith("rp_unpaired_") else "paired"
                    axes[1].plot(
                        x,
                        sub[col],
                        "o--",
                        alpha=0.45,
                        label=f"{pair} {label_kind} tau={threshold:g}",
                    )
        axes[1].set_ylabel("RP = fraction above tau")
        axes[1].set_ylim(-0.03, 1.03)
        axes[1].set_title("nearest-neighbor reproducibility")

        for ax in axes:
            ax.set_xscale("log", base=2)
            ax.set_xlabel("training dataset size N")
            ax.grid(alpha=0.25)
            ax.legend()
        fig.suptitle("SSCD generated-set reproducibility")
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
    parser.add_argument(
        "--sscd-path",
        type=Path,
        default=Path(os.environ.get("SSCD_PATH", "~/.cache/torch/hub/sscd_disc_mixup.torchscript.pt")).expanduser(),
    )
    parser.add_argument("--threshold", type=float, default=0.6)
    parser.add_argument("--fixed-similarity-thresholds", default="0.5,0.6,0.7,0.8,0.9,0.95")
    parser.add_argument("--device", default=os.environ.get("SSCD_DEVICE", "cpu"))
    parser.add_argument("--image-size", type=int, default=320)
    parser.add_argument("--render-mode", choices=("fixed", "per_image"), default="fixed")
    parser.add_argument("--value-min", type=float, default=-1.0)
    parser.add_argument("--value-max", type=float, default=1.0)
    parser.add_argument("--max-generated", type=int, default=512)
    parser.add_argument("--val-raw-per-source", type=int, default=8)
    parser.add_argument("--max-val-slices", type=int, default=512)
    parser.add_argument("--embedding-batch-size", type=int, default=16)
    parser.add_argument("--similarity-batch-size", type=int, default=512)
    parser.add_argument("--output-dir", type=Path, help="Figure output directory.")
    parser.add_argument("--table-dir", type=Path, help="CSV output directory.")
    parser.add_argument("--cache-dir", type=Path, help="Embedding cache directory.")
    parser.add_argument("--out-prefix", default="nf_generalize_nick_data_sscd_full_nn")
    parser.add_argument("--refresh-cache", action="store_true")
    parser.add_argument("--skip-missing-samples", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project_dir = Path(args.project_dir).resolve()
    output_dir = args.output_dir or project_dir / "results" / SWEEP_NAME / "quickcheck"
    table_dir = args.table_dir or project_dir / "results" / SWEEP_NAME / "tables"
    cache_dir = args.cache_dir or project_dir / "results" / SWEEP_NAME / "cache" / "sscd_full_nn"
    output_dir.mkdir(parents=True, exist_ok=True)
    table_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    if not args.sscd_path.exists():
        raise FileNotFoundError(
            f"Missing SSCD model: {args.sscd_path}\n"
            "Download with:\n"
            "mkdir -p ~/.cache/torch/hub\n"
            "curl -L -o ~/.cache/torch/hub/sscd_disc_mixup.torchscript.pt "
            "https://dl.fbaipublicfiles.com/sscd-copy-detection/sscd_disc_mixup.torchscript.pt"
        )

    load_model, embed_fn = load_sscd_helpers(project_dir)
    model = load_model(args.sscd_path, device=args.device)
    print(f"loaded SSCD model on {args.device}: {args.sscd_path}")

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
    if args.threshold not in fixed_thresholds:
        fixed_thresholds = sorted([*fixed_thresholds, float(args.threshold)])

    metric_rows: list[dict[str, Any]] = []
    generated_embeddings: dict[tuple[str, int], torch.Tensor] = {}
    for row in rows:
        config = load_config(project_dir, row)
        sample_path = sample_path_for(project_dir, row, args.seed, args.sample_label)
        generated = evenly_limit(load_npz_array(sample_path), args.max_generated)
        print(
            f"scoring {row['run_name']} N={row['dataset_size']} "
            f"generated={len(generated)} full-reference SSCD"
        )

        real_ref, real_val, norm_info = load_train_and_validation(
            row,
            config,
            val_raw_per_source=args.val_raw_per_source,
            max_val_slices=args.max_val_slices,
        )

        ref_z = embed_with_cache(
            real_ref, model=model, embed_fn=embed_fn, row=row, args=args, cache_dir=cache_dir, kind="train"
        )
        val_z = embed_with_cache(
            real_val, model=model, embed_fn=embed_fn, row=row, args=args, cache_dir=cache_dir, kind="heldout"
        )
        gen_z = embed_with_cache(
            generated,
            model=model,
            embed_fn=embed_fn,
            row=row,
            args=args,
            cache_dir=cache_dir,
            kind="generated",
            source_id=str(sample_path.resolve()),
        )
        generated_embeddings[(str(row.get("arch", "")), int(row["dataset_size"]))] = gen_z.cpu().clone()

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
            "sscd_path": str(args.sscd_path),
            "sscd_device": str(args.device),
            "sscd_image_size": int(args.image_size),
            "render_mode": args.render_mode,
            "value_min": float(args.value_min),
            "value_max": float(args.value_max),
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
                "gen_nn_q99",
                f"gen_gl_fixed_{threshold_label(args.threshold)}",
                f"gen_copy_fraction_fixed_{threshold_label(args.threshold)}",
            ]
        ].to_string(index=False)
    )
    plot_outputs(df, output_dir, args.out_prefix, args.threshold)

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
        plot_reproducibility(rp_df, output_dir, args.out_prefix, fixed_thresholds, args.threshold)


if __name__ == "__main__":
    main()
