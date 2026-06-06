#!/usr/bin/env python
"""Train a VGG-feature + MLP cosmology encoder for the HI bias probe.

This is a no-diffusion real-data sanity check.  It embeds normalized HI slices
with a frozen TorchVision VGG16 feature extractor, trains a small MLP regression
head on non-held-out real slices, and evaluates on the fixed held-out CAMELS
simulations.
"""

from __future__ import annotations

import argparse
import json
import pickle
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
from train_nf_conditional_bias_encoder import (
    SWEEP_NAME,
    fit_ridge,
    load_manifest,
    load_raw_slices,
    predict_ridge,
    preprocess_real_slices,
    select_slice_pairs,
)
from test_nf_conditional_bias_pca_mlp_encoder import PARAM_DISPLAY_LABELS, summarize_by_sim


DEFAULT_ENCODER_PATH = "results/nf_conditional_bias_probe/encoder/vgg_mlp_encoder.npz"
DEFAULT_MODEL_PATH = "results/nf_conditional_bias_probe/encoder/vgg_mlp_encoder.pkl"


def metric_rows(y_true: np.ndarray, y_pred: np.ndarray, split: str, grain: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for j, name in enumerate(PARAM_NAMES):
        err = y_pred[:, j] - y_true[:, j]
        denom = float(np.sum((y_true[:, j] - y_true[:, j].mean()) ** 2))
        r2 = 1.0 - float(np.sum(err**2)) / max(denom, 1.0e-30)
        rows.append(
            {
                "split": split,
                "grain": grain,
                "parameter": name,
                "label": PARAM_DISPLAY_LABELS.get(name, name),
                "n": int(len(y_true)),
                "mae": float(np.mean(np.abs(err))),
                "rmse": float(np.sqrt(np.mean(err**2))),
                "bias": float(np.mean(err)),
                "r2": float(r2),
            }
        )
    return rows


def torch_device(name: str) -> str:
    import torch

    if name == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return name


def load_vgg_features(weights_name: str, device: str) -> tuple[Any, str]:
    import torch
    from torchvision.models import VGG16_Weights, vgg16

    if weights_name.lower() in {"none", "random"}:
        weights = None
        weights_label = "none"
    else:
        weights = getattr(VGG16_Weights, weights_name)
        weights_label = weights_name
    model = vgg16(weights=weights).features.eval().to(device)
    for p in model.parameters():
        p.requires_grad_(False)
    return model, weights_label


def vgg_embed(
    images: np.ndarray,
    model: Any,
    *,
    device: str,
    batch_size: int,
    image_size: int,
    value_min: float,
    value_max: float,
    pool: str,
) -> np.ndarray:
    import torch
    import torch.nn.functional as F

    arr = np.asarray(images, dtype=np.float32)
    if arr.ndim == 3:
        arr = arr[:, None]
    if arr.ndim != 4 or arr.shape[1] != 1:
        raise ValueError(f"Expected (N,1,H,W), got {arr.shape}")

    mean = torch.tensor([0.485, 0.456, 0.406], device=device, dtype=torch.float32).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device, dtype=torch.float32).view(1, 3, 1, 1)
    rows = []
    denom = max(float(value_max) - float(value_min), 1.0e-6)
    with torch.no_grad():
        for start in range(0, len(arr), int(batch_size)):
            batch = torch.as_tensor(arr[start:start + int(batch_size)], device=device, dtype=torch.float32)
            batch = torch.clamp((batch - float(value_min)) / denom, 0.0, 1.0)
            batch = batch.repeat(1, 3, 1, 1)
            batch = F.interpolate(batch, size=(int(image_size), int(image_size)), mode="bilinear", align_corners=False)
            batch = (batch - mean) / std
            feat = model(batch)
            avg = F.adaptive_avg_pool2d(feat, 1).flatten(1)
            if pool == "avg":
                out = avg
            elif pool == "avgmax":
                mx = F.adaptive_max_pool2d(feat, 1).flatten(1)
                out = torch.cat([avg, mx], dim=1)
            else:
                raise ValueError(f"Unknown VGG pool mode: {pool}")
            rows.append(out.detach().cpu().numpy().astype(np.float32, copy=False))
    return np.concatenate(rows, axis=0) if rows else np.empty((0, 0), dtype=np.float32)


def inverse_param_norm(theta_norm: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return theta_norm.astype(np.float32, copy=False) * std + mean


def save_one_to_one(summary: pd.DataFrame, metrics: pd.DataFrame, out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    metric_lookup = {
        row["parameter"]: row
        for row in metrics[(metrics["split"] == "test") & (metrics["grain"] == "per_cosmology")].to_dict("records")
    }
    fig, axes = plt.subplots(2, 3, figsize=(15.5, 9.0), constrained_layout=True)
    for ax, name in zip(axes.ravel(), PARAM_NAMES):
        x = summary[f"{name}_true"].to_numpy(float)
        y = summary[f"{name}_pred_median"].to_numpy(float)
        y16 = summary[f"{name}_pred_q16"].to_numpy(float)
        y84 = summary[f"{name}_pred_q84"].to_numpy(float)
        yerr = np.vstack([np.maximum(y - y16, 0.0), np.maximum(y84 - y, 0.0)])
        lo = float(min(x.min(), y16.min()))
        hi = float(max(x.max(), y84.max()))
        pad = 0.08 * max(hi - lo, 1.0e-6)
        lo -= pad
        hi += pad
        ax.plot([lo, hi], [lo, hi], color="0.25", lw=1.8, ls="--", label="ideal")
        ax.errorbar(x, y, yerr=yerr, fmt="o", ms=5.5, capsize=2.5, color="#6f4aa8", ecolor="#b8a5d6")
        row = metric_lookup[name]
        ax.set_title(f"{PARAM_DISPLAY_LABELS.get(name, name)}   $R^2$={float(row['r2']):.2f}", fontsize=16)
        ax.set_xlabel("true parameter", fontsize=13)
        ax.set_ylabel("VGG prediction", fontsize=13)
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.grid(alpha=0.22)
        ax.tick_params(labelsize=11)
    handles, labels = axes.ravel()[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False, fontsize=13)
    fig.suptitle("Real held-out HI fields: VGG features + MLP cosmology recovery", fontsize=22, y=1.03)
    fig.savefig(out, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-dir", default=".")
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--data-root", default=DATA_ROOT)
    parser.add_argument("--encoder-out", default=DEFAULT_ENCODER_PATH)
    parser.add_argument("--model-out", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--weights", default="IMAGENET1K_V1")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--value-min", type=float, default=-1.0)
    parser.add_argument("--value-max", type=float, default=1.0)
    parser.add_argument("--pool", choices=("avg", "avgmax"), default="avgmax")
    parser.add_argument("--embedding-batch-size", type=int, default=32)
    parser.add_argument("--head-type", choices=("mlp", "ridge"), default="mlp")
    parser.add_argument("--head-train-slices", type=int, default=32768)
    parser.add_argument("--head-val-slices", type=int, default=4096)
    parser.add_argument("--test-slices-per-sim", type=int, default=128)
    parser.add_argument("--val-sims", type=int, default=64)
    parser.add_argument("--hidden-layers", default="512,256")
    parser.add_argument("--mlp-alpha", type=float, default=1.0e-4)
    parser.add_argument("--mlp-lr", type=float, default=3.0e-4)
    parser.add_argument("--mlp-max-iter", type=int, default=700)
    parser.add_argument("--ridge-alpha", type=float, default=10.0)
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args()

    from sklearn.neural_network import MLPRegressor
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    project_dir = Path(args.project_dir).resolve()
    rows = load_manifest(project_dir, args.manifest)
    if not rows:
        raise SystemExit("empty manifest")
    norm = rows[0]["normalization"]
    heldout = np.loadtxt(rows[0]["heldout_indices_path"], dtype=np.int64)
    encoder_path = project_dir / args.encoder_out
    model_path = project_dir / args.model_out
    output_dir = encoder_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path.parent.mkdir(parents=True, exist_ok=True)

    raw_params_all = load_params(params_path(args.data_root), 1000)
    stats = json.loads((project_dir / "local" / SWEEP_NAME / "heldout" / "param_norm_stats.json").read_text())
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
    test_pairs = select_slice_pairs(heldout, int(len(heldout) * args.test_slices_per_sim))

    device = torch_device(args.device)
    vgg, weights_label = load_vgg_features(args.weights, device)
    print(f"Loaded VGG16 features weights={weights_label} device={device}", flush=True)

    def embed_pairs(pairs: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        images = preprocess_real_slices(load_raw_slices(grid_path, pairs), norm)
        x = vgg_embed(
            images,
            vgg,
            device=device,
            batch_size=args.embedding_batch_size,
            image_size=args.image_size,
            value_min=args.value_min,
            value_max=args.value_max,
            pool=args.pool,
        )
        y_norm = params_norm_all[pairs[:, 0]]
        y_raw = raw_params_all[pairs[:, 0]]
        return x, y_norm.astype(np.float32), y_raw.astype(np.float32)

    x_train, y_train_norm, y_train_raw = embed_pairs(train_pairs)
    x_val, y_val_norm, y_val_raw = embed_pairs(val_pairs)
    x_test, y_test_norm, y_test_raw = embed_pairs(test_pairs)

    if args.head_type == "ridge":
        coef, intercept = fit_ridge(x_train, y_train_norm, args.ridge_alpha)
        pred_train_raw = inverse_param_norm(predict_ridge(x_train, coef, intercept), param_mean, param_std)
        pred_val_raw = inverse_param_norm(predict_ridge(x_val, coef, intercept), param_mean, param_std)
        pred_test_raw = inverse_param_norm(predict_ridge(x_test, coef, intercept), param_mean, param_std)
        head_obj: Any = {"head_type": "ridge", "coef": coef, "intercept": intercept, "ridge_alpha": float(args.ridge_alpha)}
    else:
        hidden_layers = tuple(int(x) for x in args.hidden_layers.split(",") if x.strip())
        head = make_pipeline(
            StandardScaler(),
            MLPRegressor(
                hidden_layer_sizes=hidden_layers,
                activation="relu",
                solver="adam",
                alpha=float(args.mlp_alpha),
                learning_rate_init=float(args.mlp_lr),
                batch_size=256,
                max_iter=int(args.mlp_max_iter),
                early_stopping=True,
                validation_fraction=0.12,
                n_iter_no_change=25,
                random_state=int(args.seed),
            ),
        )
        head.fit(x_train, y_train_norm)
        pred_train_raw = inverse_param_norm(head.predict(x_train), param_mean, param_std)
        pred_val_raw = inverse_param_norm(head.predict(x_val), param_mean, param_std)
        pred_test_raw = inverse_param_norm(head.predict(x_test), param_mean, param_std)
        head_obj = head

    with model_path.open("wb") as f:
        pickle.dump(head_obj, f)

    summary, y_test_sim_raw, pred_test_sim_raw = summarize_by_sim(test_pairs, y_test_raw, pred_test_raw)
    summary_path = output_dir / "vgg_real_test_per_cosmology_predictions.csv"
    summary.to_csv(summary_path, index=False)

    metrics = pd.DataFrame(
        metric_rows(y_train_raw, pred_train_raw, "train", "per_slice")
        + metric_rows(y_val_raw, pred_val_raw, "val", "per_slice")
        + metric_rows(y_test_raw, pred_test_raw, "test", "per_slice")
        + metric_rows(y_test_sim_raw, pred_test_sim_raw, "test", "per_cosmology")
    )
    metrics_path = output_dir / "vgg_real_test_metrics.csv"
    metrics.to_csv(metrics_path, index=False)

    plot_path = output_dir / "vgg_real_test_1to1.png"
    save_one_to_one(summary, metrics, plot_path)

    np.savez(
        encoder_path,
        encoder_type=np.array(f"vgg_{args.head_type}"),
        model_path=np.array(str(model_path)),
        vgg_weights=np.array(weights_label),
        vgg_image_size=np.array(int(args.image_size)),
        vgg_value_min=np.array(float(args.value_min)),
        vgg_value_max=np.array(float(args.value_max)),
        vgg_pool=np.array(str(args.pool)),
        feature_dim=np.array(int(x_train.shape[1])),
        param_names=np.array(PARAM_NAMES, dtype=object),
        param_mean=param_mean,
        param_std=param_std,
        heldout_indices=heldout,
        val_sims=val_sims,
        train_sims=train_sims,
        normalization=np.array(norm, dtype=object),
    )

    meta_path = output_dir / "vgg_real_test_metadata.json"
    meta_path.write_text(
        json.dumps(
            {
                "description": "Frozen VGG16 feature encoder trained on real non-held-out HI slices.",
                "encoder_path": str(encoder_path),
                "model_path": str(model_path),
                "vgg_weights": weights_label,
                "vgg_image_size": int(args.image_size),
                "vgg_pool": args.pool,
                "feature_dim": int(x_train.shape[1]),
                "head_type": args.head_type,
                "head_train_slices": int(len(train_pairs)),
                "head_val_slices": int(len(val_pairs)),
                "test_slices": int(len(test_pairs)),
                "heldout_indices": heldout.astype(int).tolist(),
                "val_sims": val_sims.astype(int).tolist(),
                "train_sims": train_sims.astype(int).tolist(),
            },
            indent=2,
        )
        + "\n"
    )

    print("VGG real held-out test complete.")
    print(f"VGG weights={weights_label} feature_dim={x_train.shape[1]} head={args.head_type}")
    print("heldout simulations:", ",".join(str(int(x)) for x in heldout))
    print(f"wrote {encoder_path}")
    print(f"wrote {model_path}")
    print(f"wrote {metrics_path}")
    print(f"wrote {summary_path}")
    print(f"wrote {plot_path}")
    print(metrics[(metrics["split"] == "test") & (metrics["grain"] == "per_cosmology")].round(4).to_string(index=False))


if __name__ == "__main__":
    main()
