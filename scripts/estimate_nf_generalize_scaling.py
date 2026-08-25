#!/usr/bin/env python3
"""Estimate the data-size scaling of the memorization-to-generalization turn-on.

This is a lightweight diagnostic for Nick's question:

    At what training-set size does the generalization score exceed 0.5,
    and how does that threshold scale with model parameter count?

The script reads the Fig. 2 full-nearest-neighbor metric tables, interpolates
N_50 where GL crosses 0.5 for each architecture, counts UNet parameters from
the repo's architecture templates when diffusers is available, then fits

    N_50 = C * P**alpha

where P is the number of trainable parameters. It also reports the equivalent
fit P = K * N_50**x, with x = 1 / alpha.

Interpretation caveat: with only UNet-64/128/256 this is a rough scaling
summary for the current experiment, not a universal law.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from simdiff_eval.torch_compat import install_torch_backend_compat


install_torch_backend_compat(entry_point=__name__)


ARCH_ORDER = ("u64", "u128", "u256")
DEFAULT_TABLES = {
    "PCA": Path("results/nf_generalize_fig2/tables/nf_generalize_fig2_pca_full_nn_metrics.csv"),
    "SSCD": Path("results/nf_generalize_fig2/tables/nf_generalize_fig2_sscd_full_nn_metrics.csv"),
}
DEFAULT_TEMPLATES = {
    "u64": Path("configs/templates/u64_lh_template.yaml"),
    "u128": Path("configs/templates/u128_lh_template.yaml"),
    "u256": Path("configs/templates/u256_lh_template.yaml"),
}


def fixed_tau_suffix(tau: float) -> str:
    return f"{float(tau):.3f}".rstrip("0").rstrip(".").replace(".", "p")


def parse_param_counts(text: str | None) -> dict[str, int]:
    if not text:
        return {}
    out: dict[str, int] = {}
    for part in text.split(","):
        if not part.strip():
            continue
        if "=" not in part:
            raise ValueError(f"Bad --param-counts entry {part!r}; expected arch=count.")
        arch, value = part.split("=", 1)
        out[arch.strip()] = int(float(value.strip()))
    return out


def load_model_kwargs_from_template(path: Path) -> dict[str, Any]:
    try:
        import yaml  # type: ignore
    except Exception as exc:  # pragma: no cover - environment dependent
        raise RuntimeError("PyYAML is required to count parameters from templates.") from exc
    cfg = yaml.safe_load(path.read_text())
    return dict(cfg["model"]["kwargs"])


def count_params_from_template(path: Path) -> int:
    try:
        from diffusers import UNet2DModel  # type: ignore
    except Exception as exc:  # pragma: no cover - environment dependent
        raise RuntimeError("diffusers is required to instantiate UNet2DModel.") from exc
    kwargs = load_model_kwargs_from_template(path)
    model = UNet2DModel(**kwargs)
    return int(sum(p.numel() for p in model.parameters()))


def get_param_counts(overrides: dict[str, int]) -> dict[str, int | None]:
    counts: dict[str, int | None] = {}
    for arch in ARCH_ORDER:
        if arch in overrides:
            counts[arch] = overrides[arch]
            continue
        template = DEFAULT_TEMPLATES[arch]
        try:
            counts[arch] = count_params_from_template(template)
        except Exception as exc:
            print(f"warning: could not count params for {arch} from {template}: {exc}")
            counts[arch] = None
    return counts


def choose_score_column(df: pd.DataFrame, mode: str, quantile: str, tau: float) -> str:
    if mode == "adaptive":
        candidates = [f"gen_gl_{quantile}"]
    elif mode == "fixed":
        candidates = [f"gen_gl_fixed_{fixed_tau_suffix(tau)}"]
    else:
        raise ValueError("--score-mode must be adaptive or fixed")

    candidates += ["generalization_score", "pca_generalization_score"]
    for col in candidates:
        if col in df.columns:
            return col
    available = [c for c in df.columns if "gl" in c.lower() or "general" in c.lower()]
    raise KeyError(f"No usable score column found. Tried {candidates}; available={available}")


def interpolate_n50(sub: pd.DataFrame, score_col: str, threshold: float) -> tuple[float, str]:
    rows = sub[["dataset_size", score_col]].dropna().sort_values("dataset_size")
    if rows.empty:
        return math.nan, "missing"
    x = rows["dataset_size"].astype(float).to_numpy()
    y = rows[score_col].astype(float).to_numpy()

    if np.nanmax(y) < threshold:
        return float(x[-1]), "right_censored"
    if np.nanmin(y) >= threshold:
        return float(x[0]), "left_censored"

    for i in range(1, len(x)):
        y0, y1 = y[i - 1], y[i]
        if (y0 < threshold <= y1) or (y1 < threshold <= y0):
            if y1 == y0:
                return float(x[i]), "crossing_flat"
            logx0 = math.log(float(x[i - 1]))
            logx1 = math.log(float(x[i]))
            frac = (threshold - float(y0)) / (float(y1) - float(y0))
            return float(math.exp(logx0 + frac * (logx1 - logx0))), "interpolated"

    # Fallback for nonmonotonic curves: nearest score to the threshold.
    idx = int(np.nanargmin(np.abs(y - threshold)))
    return float(x[idx]), "nearest_nonmonotonic"


def summarize_table(
    feature: str,
    path: Path,
    param_counts: dict[str, int | None],
    score_mode: str,
    quantile: str,
    tau: float,
    threshold: float,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    df = pd.read_csv(path)
    score_col = choose_score_column(df, score_mode, quantile, tau)
    rows: list[dict[str, Any]] = []
    for arch in ARCH_ORDER:
        if "arch" not in df.columns:
            continue
        sub = df[df["arch"].astype(str) == arch]
        if sub.empty:
            continue
        n50, status = interpolate_n50(sub, score_col, threshold)
        rows.append(
            {
                "feature": feature,
                "arch": arch,
                "score_col": score_col,
                "threshold": threshold,
                "n50_2d_images": n50,
                "n50_status": status,
                "model_params": param_counts.get(arch),
                "min_dataset_size": float(sub["dataset_size"].min()),
                "max_dataset_size": float(sub["dataset_size"].max()),
                "max_score": float(sub[score_col].max()),
            }
        )
    out = pd.DataFrame(rows)

    finite = out[
        np.isfinite(out["n50_2d_images"].astype(float))
        & out["model_params"].notna()
        & ~out["n50_status"].astype(str).str.contains("censored")
    ].copy()
    fit: dict[str, Any] = {
        "feature": feature,
        "score_col": score_col,
        "n_fit_points": int(len(finite)),
        "caveat": "Only three architectures are available, so treat this as a rough diagnostic.",
    }
    if len(finite) >= 2:
        log_p = np.log(finite["model_params"].astype(float).to_numpy())
        log_n = np.log(finite["n50_2d_images"].astype(float).to_numpy())
        alpha, log_c = np.polyfit(log_p, log_n, 1)
        x_equiv = float(1.0 / alpha) if alpha != 0 else math.inf
        k_equiv = float(math.exp(-log_c / alpha)) if alpha != 0 else math.nan
        fit.update(
            {
                "alpha_in_N50_equals_C_P_alpha": float(alpha),
                "C_in_N50_equals_C_P_alpha": float(math.exp(log_c)),
                "x_in_P_equals_K_N50_x": x_equiv,
                "K_in_P_equals_K_N50_x": k_equiv,
                "constant_N_to_x_over_P": float(1.0 / k_equiv) if k_equiv else math.nan,
            }
        )
    return out, fit


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-dir", type=Path, default=Path("."))
    parser.add_argument("--score-mode", choices=("adaptive", "fixed"), default="adaptive")
    parser.add_argument("--quantile", default="q95")
    parser.add_argument("--tau", type=float, default=0.9)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument(
        "--param-counts",
        default=None,
        help="Optional comma list if diffusers is unavailable, e.g. u64=...,u128=...,u256=...",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("results/nf_generalize_fig2/tables"),
    )
    args = parser.parse_args()

    project_dir = args.project_dir.resolve()
    param_counts = get_param_counts(parse_param_counts(args.param_counts))

    all_rows: list[pd.DataFrame] = []
    fits: list[dict[str, Any]] = []
    for feature, rel_path in DEFAULT_TABLES.items():
        path = project_dir / rel_path
        if not path.exists():
            print(f"missing {feature} table: {path}")
            continue
        rows, fit = summarize_table(
            feature,
            path,
            param_counts,
            args.score_mode,
            args.quantile,
            args.tau,
            args.threshold,
        )
        all_rows.append(rows)
        fits.append(fit)

    if not all_rows:
        raise SystemExit("No metric tables found; run the PCA/SSCD analyzers first.")

    out_dir = (project_dir / args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = pd.concat(all_rows, ignore_index=True)
    summary_path = out_dir / "nf_generalize_fig2_n50_scaling_summary.csv"
    fits_path = out_dir / "nf_generalize_fig2_n50_scaling_fits.json"
    summary.to_csv(summary_path, index=False)
    fits_path.write_text(json.dumps(fits, indent=2, sort_keys=True) + "\n")

    print("\nN_50 summary:")
    print(summary.to_string(index=False))
    print("\nScaling fits:")
    print(json.dumps(fits, indent=2, sort_keys=True))
    print("\nwrote", summary_path)
    print("wrote", fits_path)


if __name__ == "__main__":
    main()
