#!/usr/bin/env python
"""Train an SSCD + Ridge cosmology encoder for the HI bias probe.

This is a comparison encoder for the PCA+Ridge probe.  It uses real HI fields
only, excludes the fixed held-out cosmologies, embeds the normalized fields
with SSCD, and fits the same small linear Ridge head to the six normalized
CAMELS parameters.
"""

from __future__ import annotations

import argparse
import json
import sys
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

from prepare_nf_conditional_u128_config import DATA_ROOT, PARAM_NAMES, image_path, load_params, params_path
from simdiff_eval.sscd import load_sscd_torchscript, sscd_embeddings
from train_nf_conditional_bias_encoder import (
    SWEEP_NAME,
    file_sha256,
    fit_ridge,
    load_manifest,
    load_raw_slices,
    metric_rows,
    predict_ridge,
    preprocess_real_slices,
    select_slice_pairs,
)


DEFAULT_ENCODER_PATH = "results/nf_conditional_bias_probe/encoder/sscd_ridge_encoder.npz"
DEFAULT_SSCD_PATH = "~/.cache/torch/hub/sscd_disc_mixup.torchscript.pt"


def embed_sscd(
    images: np.ndarray,
    model: Any,
    *,
    device: str,
    batch_size: int,
    image_size: int,
    render_mode: str,
    value_min: float,
    value_max: float,
) -> np.ndarray:
    emb = sscd_embeddings(
        images,
        model,
        device=device,
        batch_size=batch_size,
        image_size=image_size,
        render_mode=render_mode,
        value_range=(float(value_min), float(value_max)),
    )
    return emb.numpy().astype(np.float32, copy=False)


def save_diagnostic_plot(df: pd.DataFrame, out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    order = PARAM_NAMES
    sub = df[df["split"] == "val"].set_index("parameter").loc[order]
    axes[0].bar(np.arange(len(order)), sub["mae"], color="#4c78a8")
    axes[0].set_xticks(np.arange(len(order)), order, rotation=35, ha="right")
    axes[0].set_ylabel("MAE")
    axes[0].set_title("Validation MAE")
    axes[1].bar(np.arange(len(order)), sub["r2"], color="#f58518")
    axes[1].set_xticks(np.arange(len(order)), order, rotation=35, ha="right")
    axes[1].set_ylabel(r"$R^2$")
    axes[1].set_title("Validation R2")
    fig.suptitle("SSCD ridge encoder sanity check")
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-dir", default=".")
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--data-root", default=DATA_ROOT)
    parser.add_argument("--sscd-path", default=DEFAULT_SSCD_PATH)
    parser.add_argument("--encoder-out", default=DEFAULT_ENCODER_PATH)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--image-size", type=int, default=320)
    parser.add_argument("--render-mode", choices=("fixed", "per_image"), default="fixed")
    parser.add_argument("--value-min", type=float, default=-1.0)
    parser.add_argument("--value-max", type=float, default=1.0)
    parser.add_argument("--embedding-batch-size", type=int, default=16)
    parser.add_argument("--head-train-slices", type=int, default=16384)
    parser.add_argument("--head-val-slices", type=int, default=4096)
    parser.add_argument("--val-sims", type=int, default=64)
    parser.add_argument("--ridge-alpha", type=float, default=10.0)
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args()

    project_dir = Path(args.project_dir).resolve()
    rows = load_manifest(project_dir, args.manifest)
    if not rows:
        raise SystemExit("empty manifest")
    norm = rows[0]["normalization"]
    heldout = np.loadtxt(rows[0]["heldout_indices_path"], dtype=np.int64)
    encoder_path = project_dir / args.encoder_out
    output_dir = encoder_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    sscd_path = Path(args.sscd_path).expanduser()
    if not sscd_path.exists():
        raise FileNotFoundError(
            f"Missing SSCD checkpoint: {sscd_path}\n"
            "Download it with:\n"
            "  curl -L -o ~/.cache/torch/hub/sscd_disc_mixup.torchscript.pt "
            "https://dl.fbaipublicfiles.com/sscd-copy-detection/sscd_disc_mixup.torchscript.pt"
        )

    raw_params_all = load_params(params_path(args.data_root), 1000)
    stats_path = project_dir / "local" / SWEEP_NAME / "heldout" / "param_norm_stats.json"
    stats = json.loads(stats_path.read_text())
    param_mean = np.asarray(stats["mean"], dtype=np.float32)
    param_std = np.asarray(stats["std"], dtype=np.float32)
    params_norm_all = ((raw_params_all - param_mean) / param_std).astype(np.float32)

    rng = np.random.default_rng(args.seed)
    all_sims = np.arange(len(raw_params_all), dtype=np.int64)
    allowed = np.setdiff1d(all_sims, heldout, assume_unique=False)
    val_sims = np.sort(rng.choice(allowed, size=min(args.val_sims, len(allowed) // 4), replace=False))
    train_sims = np.setdiff1d(allowed, val_sims, assume_unique=False)

    grid_path = image_path(args.data_root)
    train_pairs = select_slice_pairs(train_sims, args.head_train_slices)
    val_pairs = select_slice_pairs(val_sims, args.head_val_slices)
    train_images = preprocess_real_slices(load_raw_slices(grid_path, train_pairs), norm)
    val_images = preprocess_real_slices(load_raw_slices(grid_path, val_pairs), norm)
    y_train = params_norm_all[train_pairs[:, 0]]
    y_val = params_norm_all[val_pairs[:, 0]]

    print(f"Loading SSCD: {sscd_path}", flush=True)
    model = load_sscd_torchscript(sscd_path, device=args.device)
    x_train = embed_sscd(
        train_images,
        model,
        device=args.device,
        batch_size=args.embedding_batch_size,
        image_size=args.image_size,
        render_mode=args.render_mode,
        value_min=args.value_min,
        value_max=args.value_max,
    )
    x_val = embed_sscd(
        val_images,
        model,
        device=args.device,
        batch_size=args.embedding_batch_size,
        image_size=args.image_size,
        render_mode=args.render_mode,
        value_min=args.value_min,
        value_max=args.value_max,
    )

    coef, intercept = fit_ridge(x_train, y_train, args.ridge_alpha)
    pred_train = predict_ridge(x_train, coef, intercept)
    pred_val = predict_ridge(x_val, coef, intercept)
    metrics = pd.DataFrame(metric_rows(y_train, pred_train, "train") + metric_rows(y_val, pred_val, "val"))
    metrics_path = output_dir / "sscd_encoder_val_metrics.csv"
    metrics.to_csv(metrics_path, index=False)

    sscd_hash = file_sha256(sscd_path)
    np.savez(
        encoder_path,
        encoder_type=np.array("sscd_ridge"),
        sscd_path=np.array(str(sscd_path)),
        sscd_sha256=np.array(sscd_hash),
        sscd_image_size=np.array(int(args.image_size)),
        sscd_render_mode=np.array(str(args.render_mode)),
        sscd_value_min=np.array(float(args.value_min)),
        sscd_value_max=np.array(float(args.value_max)),
        feature_dim=np.array(int(x_train.shape[1])),
        coef=coef,
        intercept=intercept,
        ridge_alpha=np.array(float(args.ridge_alpha)),
        param_names=np.array(PARAM_NAMES, dtype=object),
        param_mean=param_mean,
        param_std=param_std,
        heldout_indices=heldout,
        val_sims=val_sims,
        train_sims=train_sims,
        normalization=np.array(norm, dtype=object),
    )

    split_path = output_dir / "sscd_encoder_real_split.json"
    split_path.write_text(
        json.dumps(
            {
                "heldout_indices": heldout.astype(int).tolist(),
                "val_sims": val_sims.astype(int).tolist(),
                "train_sims": train_sims.astype(int).tolist(),
                "sscd_path": str(sscd_path),
                "sscd_sha256": sscd_hash,
                "sscd_image_size": int(args.image_size),
                "sscd_render_mode": str(args.render_mode),
                "sscd_value_range": [float(args.value_min), float(args.value_max)],
                "feature_dim": int(x_train.shape[1]),
                "ridge_alpha": float(args.ridge_alpha),
            },
            indent=2,
        )
        + "\n"
    )
    save_diagnostic_plot(metrics, output_dir / "sscd_encoder_val_metrics.png")

    print("heldout simulations:", ",".join(str(int(x)) for x in heldout))
    print(f"SSCD path: {sscd_path}")
    print(f"SSCD sha256: {sscd_hash}")
    print(f"SSCD feature_dim={x_train.shape[1]} ridge_alpha={float(args.ridge_alpha):g}")
    print(f"Wrote SSCD encoder: {encoder_path}")
    print(f"Wrote metrics: {metrics_path}")
    print(metrics[metrics["split"] == "val"].round(4).to_string(index=False))


if __name__ == "__main__":
    main()
