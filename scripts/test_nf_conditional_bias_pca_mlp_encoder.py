#!/usr/bin/env python
"""Test a frozen-PCA + MLP cosmology head on real held-out HI fields only.

This is a no-diffusion sanity check for the conditional bias probe.  The PCA
basis and train/validation split come from the real-only encoder training step.
The MLP head is fit on non-held-out real slices and evaluated on the fixed
held-out CAMELS simulations.
"""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from prepare_nf_conditional_u128_config import DATA_ROOT, PARAM_NAMES, image_path, load_params, params_path
from train_nf_conditional_bias_encoder import (
    DEFAULT_BASIS_PATH,
    SWEEP_NAME,
    file_sha256,
    load_manifest,
    load_pca,
    load_raw_slices,
    preprocess_real_slices,
    select_slice_pairs,
)


PARAM_DISPLAY_LABELS = {
    "Omega_m": r"$\Omega_\mathrm{m}$",
    "sigma_8": r"$\sigma_8$",
    "A_SN1": r"$A_{\mathrm{SN1}}$",
    "A_AGN1": r"$A_{\mathrm{AGN1}}$",
    "A_SN2": r"$A_{\mathrm{SN2}}$",
    "A_AGN2": r"$A_{\mathrm{AGN2}}$",
}


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


def inverse_param_norm(theta_norm: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return theta_norm.astype(np.float32, copy=False) * std + mean


def load_split(project_dir: Path, split_path: Path | None) -> dict[str, Any]:
    path = split_path or project_dir / "results" / SWEEP_NAME / "encoder" / "encoder_real_split.json"
    with path.open() as f:
        return json.load(f)


def pairs_to_dataframe(pairs: np.ndarray) -> pd.DataFrame:
    return pd.DataFrame({"sim_index": pairs[:, 0].astype(int), "z_index": pairs[:, 1].astype(int)})


def summarize_by_sim(pairs: np.ndarray, y_true: np.ndarray, y_pred: np.ndarray) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    rows = []
    for sim_idx in np.unique(pairs[:, 0]):
        mask = pairs[:, 0] == sim_idx
        true = y_true[mask][0]
        pred = np.median(y_pred[mask], axis=0)
        spread16 = np.quantile(y_pred[mask], 0.16, axis=0)
        spread84 = np.quantile(y_pred[mask], 0.84, axis=0)
        row: dict[str, Any] = {"sim_index": int(sim_idx), "n_slices": int(np.sum(mask))}
        for j, name in enumerate(PARAM_NAMES):
            row[f"{name}_true"] = float(true[j])
            row[f"{name}_pred_median"] = float(pred[j])
            row[f"{name}_pred_q16"] = float(spread16[j])
            row[f"{name}_pred_q84"] = float(spread84[j])
        rows.append(row)
    df = pd.DataFrame(rows).sort_values("sim_index").reset_index(drop=True)
    true_cols = [f"{name}_true" for name in PARAM_NAMES]
    pred_cols = [f"{name}_pred_median" for name in PARAM_NAMES]
    return df, df[true_cols].to_numpy(np.float32), df[pred_cols].to_numpy(np.float32)


def save_one_to_one_plot(summary: pd.DataFrame, metrics: pd.DataFrame, out: Path) -> None:
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
        ax.errorbar(
            x,
            y,
            yerr=yerr,
            fmt="o",
            color="#1f77b4",
            ecolor="#8bb9e0",
            elinewidth=1.3,
            capsize=2.5,
            ms=5.5,
            alpha=0.9,
        )
        r2 = float(metric_lookup[name]["r2"])
        mae = float(metric_lookup[name]["mae"])
        ax.set_title(f"{PARAM_DISPLAY_LABELS.get(name, name)}   $R^2$={r2:.2f}, MAE={mae:.3g}", fontsize=16)
        ax.set_xlabel("true parameter", fontsize=13)
        ax.set_ylabel("MLP prediction", fontsize=13)
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.grid(alpha=0.22)
        ax.tick_params(labelsize=11)
    handles, labels = axes.ravel()[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False, fontsize=13)
    fig.suptitle("Real held-out HI fields: PCA + MLP cosmology recovery", fontsize=22, y=1.03)
    fig.savefig(out, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-dir", default=".")
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--data-root", default=DATA_ROOT)
    parser.add_argument("--pca-basis", default=DEFAULT_BASIS_PATH)
    parser.add_argument("--split", type=Path)
    parser.add_argument("--output-dir", default=f"results/{SWEEP_NAME}/encoder")
    parser.add_argument("--model-out", default="pca_mlp_encoder.pkl")
    parser.add_argument("--encoder-out", default="pca_mlp_encoder.npz")
    parser.add_argument("--train-slices", type=int, default=32768)
    parser.add_argument("--val-slices", type=int, default=4096)
    parser.add_argument("--test-slices-per-sim", type=int, default=128)
    parser.add_argument("--hidden-layers", default="256,128")
    parser.add_argument("--alpha", type=float, default=1.0e-4)
    parser.add_argument("--learning-rate", type=float, default=1.0e-3)
    parser.add_argument("--max-iter", type=int, default=700)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--embedding-batch-size", type=int, default=512)
    args = parser.parse_args()

    from sklearn.neural_network import MLPRegressor
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    project_dir = Path(args.project_dir).resolve()
    output_dir = project_dir / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = load_manifest(project_dir, args.manifest)
    if not rows:
        raise SystemExit("empty manifest")
    norm = rows[0]["normalization"]
    split = load_split(project_dir, args.split)
    train_sims = np.asarray(split["train_sims"], dtype=np.int64)
    val_sims = np.asarray(split["val_sims"], dtype=np.int64)
    heldout = np.asarray(split["heldout_indices"], dtype=np.int64)

    stats = json.loads((project_dir / "local" / SWEEP_NAME / "heldout" / "param_norm_stats.json").read_text())
    param_mean = np.asarray(stats["mean"], dtype=np.float32)
    param_std = np.asarray(stats["std"], dtype=np.float32)
    raw_params_all = load_params(params_path(args.data_root), 1000)
    params_norm_all = ((raw_params_all - param_mean) / param_std).astype(np.float32)

    pca_path = project_dir / args.pca_basis
    pca = load_pca(pca_path)
    grid_path = image_path(args.data_root)
    hidden_layers = tuple(int(x) for x in args.hidden_layers.split(",") if x.strip())

    train_pairs = select_slice_pairs(train_sims, args.train_slices)
    val_pairs = select_slice_pairs(val_sims, args.val_slices)
    test_pairs = select_slice_pairs(heldout, int(len(heldout) * args.test_slices_per_sim))

    def embed_pairs(pairs: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        images = preprocess_real_slices(load_raw_slices(grid_path, pairs), norm)
        x = pca.transform(images, batch_size=args.embedding_batch_size)
        y_norm = params_norm_all[pairs[:, 0]]
        y_raw = raw_params_all[pairs[:, 0]]
        return x, y_norm.astype(np.float32), y_raw.astype(np.float32)

    x_train, y_train_norm, y_train_raw = embed_pairs(train_pairs)
    x_val, y_val_norm, y_val_raw = embed_pairs(val_pairs)
    x_test, y_test_norm, y_test_raw = embed_pairs(test_pairs)

    model = make_pipeline(
        StandardScaler(),
        MLPRegressor(
            hidden_layer_sizes=hidden_layers,
            activation="relu",
            solver="adam",
            alpha=float(args.alpha),
            learning_rate_init=float(args.learning_rate),
            batch_size=256,
            max_iter=int(args.max_iter),
            early_stopping=True,
            validation_fraction=0.12,
            n_iter_no_change=25,
            random_state=int(args.seed),
            verbose=False,
        ),
    )
    model.fit(x_train, y_train_norm)

    pred_train_raw = inverse_param_norm(model.predict(x_train), param_mean, param_std)
    pred_val_raw = inverse_param_norm(model.predict(x_val), param_mean, param_std)
    pred_test_raw = inverse_param_norm(model.predict(x_test), param_mean, param_std)

    test_summary, y_test_sim_raw, pred_test_sim_raw = summarize_by_sim(test_pairs, y_test_raw, pred_test_raw)
    test_summary_path = output_dir / "pca_mlp_real_test_per_cosmology_predictions.csv"
    test_summary.to_csv(test_summary_path, index=False)

    slice_df = pairs_to_dataframe(test_pairs)
    for j, name in enumerate(PARAM_NAMES):
        slice_df[f"{name}_true"] = y_test_raw[:, j]
        slice_df[f"{name}_pred"] = pred_test_raw[:, j]
    slice_path = output_dir / "pca_mlp_real_test_per_slice_predictions.csv"
    slice_df.to_csv(slice_path, index=False)

    metrics = pd.DataFrame(
        metric_rows(y_train_raw, pred_train_raw, "train", "per_slice")
        + metric_rows(y_val_raw, pred_val_raw, "val", "per_slice")
        + metric_rows(y_test_raw, pred_test_raw, "test", "per_slice")
        + metric_rows(y_test_sim_raw, pred_test_sim_raw, "test", "per_cosmology")
    )
    metrics_path = output_dir / "pca_mlp_real_test_metrics.csv"
    metrics.to_csv(metrics_path, index=False)

    plot_path = output_dir / "pca_mlp_real_test_1to1.png"
    save_one_to_one_plot(test_summary, metrics, plot_path)

    model_path = output_dir / args.model_out
    with model_path.open("wb") as f:
        pickle.dump(model, f)
    encoder_path = output_dir / args.encoder_out
    pca_hash = file_sha256(pca_path)
    np.savez(
        encoder_path,
        encoder_type=np.array("pca_mlp"),
        pca_basis_path=np.array(str(pca_path)),
        pca_basis_sha256=np.array(pca_hash),
        pca_rank=np.array(pca.rank),
        pca_explained_variance_sum=np.array(pca.explained_variance_sum),
        model_path=np.array(str(model_path)),
        param_names=np.array(PARAM_NAMES, dtype=object),
        param_mean=param_mean,
        param_std=param_std,
        train_sims=train_sims,
        val_sims=val_sims,
        heldout_indices=heldout,
    )

    metadata = {
        "description": "PCA + MLP real held-out sanity check, no diffusion samples used.",
        "pca_basis": str(pca_path),
        "pca_basis_sha256": pca_hash,
        "pca_rank": pca.rank,
        "pca_explained_variance_sum": pca.explained_variance_sum,
        "model_path": str(model_path),
        "encoder_path": str(encoder_path),
        "train_sims": train_sims.astype(int).tolist(),
        "val_sims": val_sims.astype(int).tolist(),
        "heldout_sims": heldout.astype(int).tolist(),
        "train_slices": int(len(train_pairs)),
        "val_slices": int(len(val_pairs)),
        "test_slices": int(len(test_pairs)),
        "test_slices_per_sim": int(args.test_slices_per_sim),
        "hidden_layers": hidden_layers,
        "alpha": float(args.alpha),
        "learning_rate": float(args.learning_rate),
        "max_iter": int(args.max_iter),
        "param_names": PARAM_NAMES,
    }
    meta_path = output_dir / "pca_mlp_real_test_metadata.json"
    meta_path.write_text(json.dumps(metadata, indent=2) + "\n")

    print("PCA + MLP real held-out test complete.")
    print(f"PCA basis: {pca_path}")
    print(f"PCA rank={pca.rank} explained_variance={pca.explained_variance_sum:.4f}")
    print(f"heldout simulations: {','.join(str(int(x)) for x in heldout)}")
    print(f"wrote {metrics_path}")
    print(f"wrote {test_summary_path}")
    print(f"wrote {slice_path}")
    print(f"wrote {plot_path}")
    print(f"wrote {model_path}")
    print(f"wrote {encoder_path}")
    print(metrics[(metrics["split"] == "test") & (metrics["grain"] == "per_cosmology")].round(4).to_string(index=False))


if __name__ == "__main__":
    main()
