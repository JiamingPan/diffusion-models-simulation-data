#!/usr/bin/env python
"""Train the frozen-PCA + small-head cosmology encoder for the bias probe.

The encoder is trained on real HI slices only and excludes the fixed held-out
cosmologies used for generated-sample calibration.  Generated fields are never
used to fit either PCA or the regression head.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from prepare_nf_conditional_u128_config import DATA_ROOT, PARAM_NAMES, image_path, load_params, params_path


SWEEP_NAME = "nf_conditional_bias_probe"
DEFAULT_BASIS_PATH = "results/nf_conditional_bias_probe/encoder/pca_basis_98.npz"
DEFAULT_ENCODER_PATH = "results/nf_conditional_bias_probe/encoder/frozen_pca_ridge_encoder.npz"


@dataclass
class FrozenPCA:
    mean: np.ndarray
    scale: np.ndarray
    components: np.ndarray
    explained_variance_ratio: np.ndarray
    metadata: dict[str, Any] | None = None

    @property
    def rank(self) -> int:
        return int(self.components.shape[0])

    @property
    def explained_variance_sum(self) -> float:
        return float(np.sum(self.explained_variance_ratio))

    def transform(self, images: np.ndarray, batch_size: int = 512) -> np.ndarray:
        images = as_nchw(images)
        rows: list[np.ndarray] = []
        for start in range(0, len(images), batch_size):
            batch = images[start:start + batch_size].reshape(len(images[start:start + batch_size]), -1)
            batch = (batch.astype(np.float32, copy=False) - self.mean) / self.scale
            z = batch @ self.components.T
            rows.append(z.astype(np.float32, copy=False))
        return np.concatenate(rows, axis=0) if rows else np.empty((0, self.rank), dtype=np.float32)


def as_nchw(images: np.ndarray) -> np.ndarray:
    arr = np.asarray(images)
    if arr.ndim == 3:
        return arr[:, None, :, :]
    if arr.ndim == 4 and arr.shape[1] == 1:
        return arr
    raise ValueError(f"Expected (N,H,W) or (N,1,H,W), got {arr.shape}.")


def file_sha256(path: Path, block_size: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(block_size), b""):
            h.update(block)
    return h.hexdigest()


def load_manifest(project_dir: Path, manifest_path: Path | None) -> list[dict[str, Any]]:
    path = manifest_path or project_dir / "local" / SWEEP_NAME / "manifest.json"
    with path.open() as f:
        return json.load(f)


def select_slice_pairs(sim_indices: np.ndarray, n_slices: int, z_size: int = 128) -> np.ndarray:
    total = int(len(sim_indices)) * int(z_size)
    n_slices = min(int(n_slices), total)
    flat_idx = np.linspace(0, total - 1, n_slices, dtype=np.int64)
    sims = sim_indices[flat_idx // z_size]
    z = flat_idx % z_size
    return np.stack([sims.astype(np.int64), z.astype(np.int64)], axis=1)


def load_raw_slices(grid_path: Path, pairs: np.ndarray) -> np.ndarray:
    arr = np.load(grid_path, mmap_mode="r")
    out = np.empty((len(pairs), 1, arr.shape[-2], arr.shape[-1]), dtype=np.float32)
    for row, (sim_idx, z_idx) in enumerate(pairs):
        out[row, 0] = np.asarray(arr[int(sim_idx), int(z_idx)], dtype=np.float32)
    return out


def tanh_normalize_logged(log_images: np.ndarray, norm: dict[str, Any]) -> np.ndarray:
    center = np.float32(norm["center"])
    xmax = np.float32(norm["xmax"])
    alpha = np.float32(norm.get("alpha", 0.8))
    beta = np.float32(norm.get("beta", 10.0))
    gamma = np.float32(norm.get("gamma", 1.0))
    delta = np.float32(norm.get("delta", 1.0))
    sigma = np.float32(norm.get("sigma", 1.5))
    x = (log_images.astype(np.float32, copy=False) - center) / xmax
    pos = alpha * np.tanh((gamma * x) / alpha)
    neg = beta * np.tanh((delta * x) / beta)
    return (np.where(x >= 0, pos, neg) * sigma).astype(np.float32, copy=False)


def preprocess_real_slices(raw_images: np.ndarray, norm: dict[str, Any]) -> np.ndarray:
    logged = np.log(np.maximum(raw_images.astype(np.float32, copy=False), np.float32(1.0e-30)))
    return tanh_normalize_logged(logged, norm)


def fit_pca(images: np.ndarray, n_components: int, target_variance: float) -> FrozenPCA:
    x = as_nchw(images).reshape(len(images), -1).astype(np.float32, copy=False)
    mean = x.mean(axis=0, keepdims=True)
    scale = np.maximum(x.std(axis=0, keepdims=True), 1.0e-6).astype(np.float32, copy=False)
    x = (x - mean) / scale
    rank_cap = min(int(n_components), x.shape[0] - 1, x.shape[1])
    if rank_cap < 1:
        raise ValueError("Need at least two images to fit PCA.")
    try:
        from sklearn.decomposition import PCA

        pca = PCA(n_components=rank_cap, svd_solver="randomized", random_state=0)
        pca.fit(x)
        components = pca.components_.astype(np.float32, copy=False)
        evr = pca.explained_variance_ratio_.astype(np.float32, copy=False)
    except Exception as exc:
        print(f"sklearn PCA failed or unavailable; falling back to numpy SVD: {exc!r}", flush=True)
        _, s, vt = np.linalg.svd(x, full_matrices=False)
        components = vt[:rank_cap].astype(np.float32, copy=False)
        ev = (s[:rank_cap] ** 2) / max(x.shape[0] - 1, 1)
        total = float(np.var(x, axis=0, ddof=1).sum())
        evr = (ev / max(total, 1.0e-12)).astype(np.float32, copy=False)

    if target_variance > 0:
        cumulative = np.cumsum(evr)
        if cumulative[-1] + 1.0e-6 < target_variance:
            raise RuntimeError(
                f"PCA rank cap {rank_cap} explains {cumulative[-1]:.4f}, "
                f"below requested {target_variance:.4f}."
            )
        keep = int(np.searchsorted(cumulative, target_variance, side="left") + 1)
        components = components[:keep]
        evr = evr[:keep]
        print(f"PCA target {target_variance:.3f}: selected {keep} modes, EV={float(evr.sum()):.4f}")

    return FrozenPCA(
        mean=mean.squeeze(0).astype(np.float32, copy=False),
        scale=scale.squeeze(0).astype(np.float32, copy=False),
        components=components.astype(np.float32, copy=False),
        explained_variance_ratio=evr.astype(np.float32, copy=False),
        metadata=None,
    )


def save_pca(path: Path, pca: FrozenPCA, metadata: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        mean=pca.mean,
        scale=pca.scale,
        components=pca.components,
        explained_variance_ratio=pca.explained_variance_ratio,
        metadata=np.array(metadata, dtype=object),
    )


def load_pca(path: Path) -> FrozenPCA:
    with np.load(path, allow_pickle=True) as data:
        metadata = {}
        if "metadata" in data.files:
            metadata_obj = data["metadata"].item()
            if isinstance(metadata_obj, dict):
                metadata = metadata_obj
        return FrozenPCA(
            mean=data["mean"].astype(np.float32),
            scale=data["scale"].astype(np.float32),
            components=data["components"].astype(np.float32),
            explained_variance_ratio=data["explained_variance_ratio"].astype(np.float32),
            metadata=metadata,
        )


def metadata_matches(pca: FrozenPCA, expected: dict[str, Any]) -> tuple[bool, str]:
    metadata = pca.metadata or {}
    for key in ("source", "param_names", "heldout_indices", "pca_train_sims", "normalization"):
        if metadata.get(key) != expected.get(key):
            return False, f"{key} mismatch"
    return True, "ok"


def fit_ridge(x: np.ndarray, y: np.ndarray, alpha: float) -> tuple[np.ndarray, np.ndarray]:
    x64 = x.astype(np.float64, copy=False)
    y64 = y.astype(np.float64, copy=False)
    x_aug = np.concatenate([x64, np.ones((len(x64), 1), dtype=np.float64)], axis=1)
    reg = np.eye(x_aug.shape[1], dtype=np.float64) * float(alpha)
    reg[-1, -1] = 0.0
    weights = np.linalg.solve(x_aug.T @ x_aug + reg, x_aug.T @ y64)
    coef = weights[:-1].astype(np.float32)
    intercept = weights[-1].astype(np.float32)
    return coef, intercept


def predict_ridge(x: np.ndarray, coef: np.ndarray, intercept: np.ndarray) -> np.ndarray:
    return (x.astype(np.float32, copy=False) @ coef + intercept).astype(np.float32)


def metric_rows(y_true: np.ndarray, y_pred: np.ndarray, split: str) -> list[dict[str, Any]]:
    rows = []
    for j, name in enumerate(PARAM_NAMES):
        err = y_pred[:, j] - y_true[:, j]
        denom = float(np.sum((y_true[:, j] - y_true[:, j].mean()) ** 2))
        r2 = 1.0 - float(np.sum(err**2)) / max(denom, 1.0e-30)
        rows.append(
            {
                "split": split,
                "parameter": name,
                "mae": float(np.mean(np.abs(err))),
                "rmse": float(np.sqrt(np.mean(err**2))),
                "bias": float(np.mean(err)),
                "r2": float(r2),
                "n": int(len(y_true)),
            }
        )
    return rows


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
    fig.suptitle("Frozen-PCA ridge encoder sanity check")
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-dir", default=".")
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--data-root", default=DATA_ROOT)
    parser.add_argument("--pca-basis", default=DEFAULT_BASIS_PATH)
    parser.add_argument("--encoder-out", default=DEFAULT_ENCODER_PATH)
    parser.add_argument("--fit-pca-if-missing", action="store_true", help="Deprecated; PCA is fit by default unless --reuse-existing-pca is set.")
    parser.add_argument(
        "--reuse-existing-pca",
        action="store_true",
        help="Reuse an existing PCA file only if its metadata proves it excludes the current held-out cosmologies.",
    )
    parser.add_argument("--pca-components", type=int, default=8192)
    parser.add_argument("--pca-target-variance", type=float, default=0.98)
    parser.add_argument("--pca-fit-slices", type=int, default=16384)
    parser.add_argument("--head-train-slices", type=int, default=16384)
    parser.add_argument("--head-val-slices", type=int, default=4096)
    parser.add_argument("--val-sims", type=int, default=64)
    parser.add_argument("--ridge-alpha", type=float, default=10.0)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--embedding-batch-size", type=int, default=512)
    args = parser.parse_args()

    project_dir = Path(args.project_dir).resolve()
    rows = load_manifest(project_dir, args.manifest)
    if not rows:
        raise SystemExit("empty manifest")
    norm = rows[0]["normalization"]
    heldout = np.loadtxt(rows[0]["heldout_indices_path"], dtype=np.int64)
    basis_path = project_dir / args.pca_basis
    encoder_path = project_dir / args.encoder_out
    output_dir = encoder_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)

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

    pca_metadata = {
        "source": "real non-heldout HI fields",
        "target_variance": args.pca_target_variance,
        "pca_components_requested": int(args.pca_components),
        "pca_fit_slices_requested": int(args.pca_fit_slices),
        "param_names": PARAM_NAMES,
        "heldout_indices": heldout.astype(int).tolist(),
        "pca_train_sims": train_sims.astype(int).tolist(),
        "pca_val_sims_excluded_from_fit": val_sims.astype(int).tolist(),
        "normalization": norm,
    }

    if args.reuse_existing_pca and basis_path.exists():
        pca = load_pca(basis_path)
        ok, reason = metadata_matches(pca, pca_metadata)
        if not ok:
            raise RuntimeError(
                f"Refusing to reuse PCA basis {basis_path}: {reason}. "
                "Run without --reuse-existing-pca to refit a leakage-safe basis."
            )
        print(f"Loaded PCA basis: {basis_path}")
    else:
        if basis_path.exists() and not args.reuse_existing_pca:
            print(f"Refitting PCA and overwriting existing basis to avoid stale-basis leakage: {basis_path}")
        pca_pairs = select_slice_pairs(train_sims, args.pca_fit_slices)
        pca_images = preprocess_real_slices(load_raw_slices(grid_path, pca_pairs), norm)
        pca = fit_pca(pca_images, args.pca_components, args.pca_target_variance)
        pca.metadata = pca_metadata
        save_pca(basis_path, pca, pca_metadata)
        print(f"Wrote PCA basis: {basis_path}")

    train_pairs = select_slice_pairs(train_sims, args.head_train_slices)
    val_pairs = select_slice_pairs(val_sims, args.head_val_slices)
    train_images = preprocess_real_slices(load_raw_slices(grid_path, train_pairs), norm)
    val_images = preprocess_real_slices(load_raw_slices(grid_path, val_pairs), norm)
    y_train = params_norm_all[train_pairs[:, 0]]
    y_val = params_norm_all[val_pairs[:, 0]]

    x_train = pca.transform(train_images, batch_size=args.embedding_batch_size)
    x_val = pca.transform(val_images, batch_size=args.embedding_batch_size)
    coef, intercept = fit_ridge(x_train, y_train, args.ridge_alpha)
    pred_train = predict_ridge(x_train, coef, intercept)
    pred_val = predict_ridge(x_val, coef, intercept)

    metrics = pd.DataFrame(metric_rows(y_train, pred_train, "train") + metric_rows(y_val, pred_val, "val"))
    metrics_path = output_dir / "encoder_val_metrics.csv"
    metrics.to_csv(metrics_path, index=False)

    basis_hash = file_sha256(basis_path)
    np.savez(
        encoder_path,
        pca_basis_path=np.array(str(basis_path)),
        pca_basis_sha256=np.array(basis_hash),
        pca_rank=np.array(pca.rank),
        pca_explained_variance_sum=np.array(pca.explained_variance_sum),
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

    split_path = output_dir / "encoder_real_split.json"
    split_path.write_text(
        json.dumps(
            {
                "heldout_indices": heldout.astype(int).tolist(),
                "val_sims": val_sims.astype(int).tolist(),
                "train_sims": train_sims.astype(int).tolist(),
                "pca_basis_path": str(basis_path),
                "pca_basis_sha256": basis_hash,
                "pca_rank": pca.rank,
                "pca_explained_variance_sum": pca.explained_variance_sum,
            },
            indent=2,
        )
        + "\n"
    )
    save_diagnostic_plot(metrics, output_dir / "encoder_val_metrics.png")

    print("heldout simulations:", ",".join(str(int(x)) for x in heldout))
    print(f"PCA basis: {basis_path}")
    print(f"PCA basis sha256: {basis_hash}")
    print(f"PCA rank={pca.rank} explained_variance={pca.explained_variance_sum:.4f}")
    print(f"Wrote encoder: {encoder_path}")
    print(f"Wrote metrics: {metrics_path}")
    print(metrics[metrics["split"] == "val"].round(4).to_string(index=False))


if __name__ == "__main__":
    main()
