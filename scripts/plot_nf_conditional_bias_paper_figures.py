#!/usr/bin/env python
"""Build paper-ready conditional-recovery and companion verification figures."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
PAPER_DIR = ROOT / "paper" / "ai4science_verification"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(PAPER_DIR) not in sys.path:
    sys.path.insert(0, str(PAPER_DIR))

from paperstyle import FULL_W, INK, set_paper_style, style_axis
from simdiff_eval.paper_nearest_training import (
    load_npz_samples,
    load_sscd_embedding_cache,
    normalized_sscd_frechet,
    resolve_sscd_embedding_cache,
    summarize_exact_training_reference,
)


ALL_POWERS = tuple(range(6, 16))
ALL_DATASET_SIZES = tuple(2**power for power in ALL_POWERS)
REPRESENTATIVE_POWERS = (7, 14)
REGIME_COLORS = {
    "memorization": "#D55E00",
    "generalization": "#0072B2",
}
TRANSITION_COLOR = "#4D4D4D"
COVERAGE_LEVELS = (0.68, 0.95)
COVERAGE_CURVE_LEVELS = tuple(
    sorted(set(np.linspace(0.0, 1.0, 51).tolist()) | set(COVERAGE_LEVELS))
)
COVERAGE_POWERS = (7, 10, 14)
COVERAGE_MARKERS = {0.68: "o", 0.95: "s"}
COVERAGE_STYLES = {
    7: {
        "color": "#C23B2A",
        "label": r"$N_{2D}=2^{7}$ memorization regime",
    },
    10: {
        "color": "#E67E22",
        "label": r"$N_{2D}=2^{10}$",
    },
    14: {
        "color": "#0072B2",
        "label": r"$N_{2D}=2^{14}$ generalization regime",
    },
}
EXPECTED_HELDOUT_COSMOLOGIES = 32
EXPECTED_NEAREST_SAMPLES = 512
CAMELS_MAP_SIDE_HINV_MPC = 25.0
REGIME_LABELS = {
    7: "memorization regime",
    14: "generalization regime",
}
PARAM_ORDER = ("Omega_m", "sigma_8", "A_SN1", "A_AGN1", "A_SN2", "A_AGN2")
PARAM_LABELS = {
    "Omega_m": r"$\Omega_m$",
    "sigma_8": r"$\sigma_8$",
    "A_SN1": r"$A_{\rm SN1}$",
    "A_AGN1": r"$A_{\rm AGN1}$",
    "A_SN2": r"$A_{\rm SN2}$",
    "A_AGN2": r"$A_{\rm AGN2}$",
}
ARCHITECTURES = (
    ("u64", "U-Net-64", "#009E73", "^"),
    ("u128", "U-Net-128", "#D55E00", "o"),
    ("u256", "U-Net-256", "#0072B2", "s"),
)
NEAREST_POWERS = (6, 8, 10, 12, 15)
NEAREST_DATASET_SIZES = tuple(2**power for power in NEAREST_POWERS)
NEAREST_TABLE_COLUMNS = (
    "dataset_tag",
    "dataset_size",
    "cos_max",
    "pk_log10_mae",
    "generated_to_heldout_real_frechet",
    "heldout_real_split_frechet",
    "sscd_frechet_normalized",
    "n_generated",
    "n_real_split",
    "n_heldout_real_available",
    "sscd_reference_kind",
    "generated_cache_path",
    "heldout_real_cache_path",
    "config_path",
)


def format_slope_with_interval(slope: float, ci16: float, ci84: float) -> str:
    """Format one fitted slope with its 16th--84th percentile interval."""

    lower = max(float(slope) - float(ci16), 0.0)
    upper = max(float(ci84) - float(slope), 0.0)
    return rf"${float(slope):.3f}^{{+{upper:.3f}}}_{{-{lower:.3f}}}$"


def _single_protocol(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    if "guidance_label" in out and out["guidance_label"].nunique() > 1:
        if "noguidance" not in set(out["guidance_label"]):
            raise ValueError("multiple guidance protocols found without a noguidance baseline")
        out = out[out["guidance_label"] == "noguidance"].copy()
    if "cfg_dropout" in out and out["cfg_dropout"].nunique() > 1:
        out = out[np.isclose(out["cfg_dropout"].astype(float), 0.0)].copy()
    return out


def _require_sizes(frame: pd.DataFrame) -> None:
    if "dataset_size" not in frame:
        raise ValueError("table has no dataset_size column")
    present = {int(value) for value in frame["dataset_size"].dropna().unique()}
    missing = sorted(set(ALL_DATASET_SIZES) - present)
    extra = sorted(present - set(ALL_DATASET_SIZES))
    if missing:
        raise ValueError(f"missing dataset sizes: {missing}")
    if extra:
        raise ValueError(f"unexpected dataset sizes: {extra}")


def _training_colors() -> dict[int, np.ndarray]:
    colors = plt.get_cmap("viridis")(np.linspace(0.08, 0.92, len(ALL_POWERS)))
    return dict(zip(ALL_DATASET_SIZES, colors))


def style_training_size_axis(axis, *, label_every: int = 1) -> None:
    """Use the exact shared horizontal power-of-two axis for paper comparisons."""

    axis.set_xlim(5.65, 15.35)
    axis.set_xticks(ALL_POWERS)
    axis.set_xticklabels(
        [
            rf"$2^{{{power}}}$" if (power - ALL_POWERS[0]) % label_every == 0 else ""
            for power in ALL_POWERS
        ],
        rotation=0,
    )
    style_axis(axis)


def _omega_tables(
    points: pd.DataFrame,
    slopes: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    points = _single_protocol(points)
    slopes = _single_protocol(slopes)
    omega_points = points[points["parameter"] == "Omega_m"].copy()
    omega_slopes = slopes[slopes["parameter"] == "Omega_m"].copy()
    _require_sizes(omega_points)
    _require_sizes(omega_slopes)
    counts = omega_slopes.groupby("dataset_size", observed=True).size()
    invalid = counts[counts != 1]
    if not invalid.empty:
        raise ValueError(f"expected one Omega_m fit per size, found {invalid.to_dict()}")
    return omega_points, omega_slopes.sort_values("dataset_size").reset_index(drop=True)


def compute_conditional_coverage(
    samples: pd.DataFrame,
    *,
    nominal_coverages: tuple[float, ...] = COVERAGE_LEVELS,
    bootstrap: int = 2000,
    seed: int = 123,
) -> pd.DataFrame:
    """Compute central-interval coverage from individual generated-map recoveries."""

    samples = _single_protocol(samples)
    required = {
        "dataset_size",
        "parameter",
        "heldout_sim",
        "seed_index",
        "theta_in",
        "theta_rec",
    }
    missing = sorted(required - set(samples.columns))
    if missing:
        raise ValueError(f"per-sample recovery table is missing columns: {missing}")
    omega = samples[samples["parameter"] == "Omega_m"].copy()
    _require_sizes(omega)
    if omega.empty:
        raise ValueError("per-sample recovery table has no Omega_m rows")

    group_columns = ["dataset_size", "heldout_sim"]
    draw_counts = omega.groupby(group_columns, observed=True).size()
    if draw_counts.nunique() != 1:
        detail = draw_counts.groupby(level=0).agg(["min", "max"]).to_dict("index")
        raise ValueError(f"inconsistent posterior-draw counts: {detail}")
    draws_per_cosmology = int(draw_counts.iloc[0])
    if draws_per_cosmology < 2:
        raise ValueError("at least two posterior draws are required per held-out cosmology")
    if omega.duplicated(group_columns + ["seed_index"]).any():
        raise ValueError("duplicate seed_index values found within a held-out cosmology")

    truth_counts = omega.groupby(group_columns, observed=True)["theta_in"].nunique()
    if not bool((truth_counts == 1).all()):
        raise ValueError("theta_in is not constant within a held-out cosmology")
    heldout_counts = omega.groupby("dataset_size", observed=True)["heldout_sim"].nunique()
    if not bool((heldout_counts == EXPECTED_HELDOUT_COSMOLOGIES).all()):
        raise ValueError(
            "expected exactly "
            f"{EXPECTED_HELDOUT_COSMOLOGIES} held-out cosmologies per training size; "
            f"found {heldout_counts.to_dict()}"
        )

    nominal_coverages = tuple(float(value) for value in nominal_coverages)
    if not nominal_coverages:
        raise ValueError("at least one nominal coverage is required")
    if any(value < 0.0 or value > 1.0 for value in nominal_coverages):
        raise ValueError("nominal coverages must lie in [0, 1]")
    if len(set(nominal_coverages)) != len(nominal_coverages):
        raise ValueError("nominal coverages must be unique")

    interval_rows: list[dict[str, float | int | bool]] = []
    for (dataset_size, heldout_sim), group in omega.groupby(group_columns, sort=True):
        truth = float(group["theta_in"].iloc[0])
        draws = group["theta_rec"].to_numpy(float)
        for nominal in nominal_coverages:
            tail = (1.0 - nominal) / 2.0
            lower, upper = np.quantile(draws, [tail, 1.0 - tail])
            interval_rows.append(
                {
                    "dataset_size": int(dataset_size),
                    "heldout_sim": int(heldout_sim),
                    "nominal_coverage": float(nominal),
                    "covered": bool(lower <= truth <= upper),
                }
            )
    indicators = pd.DataFrame(interval_rows)

    rng = np.random.default_rng(seed)
    report_rows: list[dict[str, float | int]] = []
    for (dataset_size, nominal), group in indicators.groupby(
        ["dataset_size", "nominal_coverage"], sort=True
    ):
        covered = group["covered"].to_numpy(float)
        empirical = float(covered.mean())
        if bootstrap > 0:
            indices = rng.integers(0, len(covered), size=(int(bootstrap), len(covered)))
            bootstrap_values = covered[indices].mean(axis=1)
            ci16, ci84 = np.quantile(bootstrap_values, [0.16, 0.84])
        else:
            ci16 = ci84 = empirical
        report_rows.append(
            {
                "dataset_size": int(dataset_size),
                "nominal_coverage": float(nominal),
                "empirical_coverage": empirical,
                "coverage_ci16": float(ci16),
                "coverage_ci84": float(ci84),
                "n_heldout": int(len(covered)),
                "draws_per_cosmology": draws_per_cosmology,
            }
        )
    return pd.DataFrame(report_rows).sort_values(
        ["dataset_size", "nominal_coverage"]
    ).reset_index(drop=True)


def build_conditional_coverage_figure(
    samples: pd.DataFrame,
    *,
    bootstrap: int = 2000,
    seed: int = 123,
) -> tuple[plt.Figure, pd.DataFrame]:
    """Plot full calibration curves for representative training-set sizes."""

    set_paper_style()
    report = compute_conditional_coverage(
        samples,
        nominal_coverages=COVERAGE_CURVE_LEVELS,
        bootstrap=bootstrap,
        seed=seed,
    )
    selected_sizes = {2**power for power in COVERAGE_POWERS}
    report["plotted"] = report["dataset_size"].isin(selected_sizes)
    figure, axis = plt.subplots(figsize=(FULL_W, 3.25))
    figure.subplots_adjust(left=0.09, right=0.985, bottom=0.17, top=0.97)

    axis.plot(
        (0.0, 1.0),
        (0.0, 1.0),
        color="0.15",
        ls="--",
        lw=0.9,
        label="_nolegend_",
        zorder=0,
    )
    for power in COVERAGE_POWERS:
        dataset_size = 2**power
        plot_style = COVERAGE_STYLES[power]
        sub = report[report["dataset_size"] == dataset_size].sort_values(
            "nominal_coverage"
        )
        axis.plot(
            sub["nominal_coverage"],
            sub["empirical_coverage"],
            color=plot_style["color"],
            lw=1.45,
            label=plot_style["label"],
            zorder=2,
        )
        for nominal in COVERAGE_LEVELS:
            focal = sub[np.isclose(sub["nominal_coverage"], nominal)]
            axis.scatter(
                focal["nominal_coverage"],
                focal["empirical_coverage"],
                s=22,
                marker=COVERAGE_MARKERS[nominal],
                color=plot_style["color"],
                edgecolor="0.15",
                linewidth=0.45,
                zorder=3,
            )

    axis.text(0.45, 0.82, "underconfident", color="0.38", fontsize=8.5)
    axis.text(0.72, 0.25, "overconfident", color="0.38", fontsize=8.5)
    axis.set_xlabel("Nominal coverage")
    axis.set_ylabel("Empirical coverage")
    axis.set_xlim(0.0, 1.0)
    axis.set_ylim(0.0, 1.0)
    axis.set_xticks(np.linspace(0.0, 1.0, 6))
    axis.set_yticks(np.linspace(0.0, 1.0, 6))
    style_axis(axis)
    axis.legend(frameon=False, loc="upper left", ncol=1, handlelength=2.2)
    return figure, report


def fourier_bin_to_physical_k(
    fourier_bin: np.ndarray | float,
    *,
    side_length_hinv_mpc: float = CAMELS_MAP_SIDE_HINV_MPC,
) -> np.ndarray:
    """Convert radial Fourier-bin index to comoving ``h Mpc^-1``."""

    if float(side_length_hinv_mpc) <= 0.0:
        raise ValueError("map side length must be positive")
    return 2.0 * np.pi * np.asarray(fourier_bin, dtype=float) / float(
        side_length_hinv_mpc
    )


def build_conditional_recovery_figure(
    points: pd.DataFrame,
    slopes: pd.DataFrame,
) -> tuple[plt.Figure, pd.DataFrame]:
    """Return a two-regime calibration contrast and the full response transition."""

    set_paper_style()
    omega_points, report = _omega_tables(points, slopes)
    required = {
        "theta_in",
        "theta_rec_median",
        "theta_rec_q16",
        "theta_rec_q84",
    }
    missing = sorted(required - set(omega_points.columns))
    if missing:
        raise ValueError(f"Omega_m point table is missing columns: {missing}")

    lo = float(min(omega_points["theta_in"].min(), omega_points["theta_rec_q16"].min()))
    hi = float(max(omega_points["theta_in"].max(), omega_points["theta_rec_q84"].max()))
    pad = 0.055 * max(hi - lo, 1.0e-6)
    limits = (lo - pad, hi + pad)
    figure = plt.figure(figsize=(FULL_W, 2.4))
    grid = figure.add_gridspec(
        1,
        2,
        width_ratios=(1.0, 1.15),
        left=0.085,
        right=0.990,
        bottom=0.205,
        top=0.955,
        wspace=0.34,
    )
    calibration_axis = figure.add_subplot(grid[0, 0])
    transition_axis = figure.add_subplot(grid[0, 1])

    calibration_axis.plot(limits, limits, color="0.38", ls="--", lw=0.9, label="_nolegend_")
    for index, power in enumerate(REPRESENTATIVE_POWERS):
        dataset_size = 2**power
        sub = omega_points[omega_points["dataset_size"] == dataset_size].sort_values("theta_in")
        row = report[report["dataset_size"] == dataset_size].iloc[0]
        x = sub["theta_in"].to_numpy(float)
        y = sub["theta_rec_median"].to_numpy(float)
        q16 = sub["theta_rec_q16"].to_numpy(float)
        q84 = sub["theta_rec_q84"].to_numpy(float)
        yerr = np.vstack((np.maximum(y - q16, 0.0), np.maximum(q84 - y, 0.0)))
        regime = REGIME_LABELS[power].split()[0]
        color = REGIME_COLORS[regime]
        calibration_axis.errorbar(
            x,
            y,
            yerr=yerr,
            fmt="o" if index == 0 else "s",
            ms=3.0,
            capsize=1.4,
            elinewidth=0.7,
            color=color,
            ecolor=color,
            markeredgecolor="white",
            markeredgewidth=0.35,
            alpha=0.9,
            label=REGIME_LABELS[power],
        )
        fit_x = np.asarray(limits)
        calibration_axis.plot(
            fit_x,
            float(row["slope"]) * fit_x + float(row["intercept"]),
            color=color,
            lw=1.35,
        )
        calibration_axis.text(
            0.05,
            0.95 - 0.16 * index,
            REGIME_LABELS[power]
            + "\n"
            + rf"$N_{{2D}}=2^{{{power}}}$, slope = "
            + format_slope_with_interval(
                float(row["slope"]),
                float(row["slope_ci16"]),
                float(row["slope_ci84"]),
            ),
            transform=calibration_axis.transAxes,
            ha="left",
            va="top",
            fontsize=7.0,
            color=color,
        )

    calibration_axis.set_xlim(limits)
    calibration_axis.set_ylim(limits)
    calibration_axis.set_xlabel(r"Requested $\Omega_m$")
    calibration_axis.set_ylabel(r"Recovered $\Omega_m$")
    style_axis(calibration_axis)

    x = np.log2(report["dataset_size"].to_numpy(float))
    y = report["slope"].to_numpy(float)
    lower = np.maximum(y - report["slope_ci16"].to_numpy(float), 0.0)
    upper = np.maximum(report["slope_ci84"].to_numpy(float) - y, 0.0)
    transition_axis.axhline(1.0, color="0.38", ls="--", lw=0.9, label="_nolegend_")
    transition_axis.plot(x, y, color=TRANSITION_COLOR, lw=0.9, zorder=1)
    for xpos, value, low, high in zip(x, y, lower, upper):
        transition_axis.errorbar(
            xpos,
            value,
            yerr=np.array([[low], [high]]),
            fmt="o",
            ms=3.7,
            capsize=1.8,
            elinewidth=0.8,
            color=TRANSITION_COLOR,
            ecolor=TRANSITION_COLOR,
            markeredgecolor="white",
            markeredgewidth=0.35,
            zorder=2,
        )
    for power, regime in ((7, "memorization"), (14, "generalization")):
        value = float(report.loc[report["dataset_size"] == 2**power, "slope"].iloc[0])
        transition_axis.plot(
            [power],
            [value],
            marker="o" if power == 7 else "s",
            ms=5.2,
            color=REGIME_COLORS[regime],
            markeredgecolor="white",
            markeredgewidth=0.55,
            linestyle="none",
            zorder=4,
        )
        transition_axis.annotate(
            rf"$2^{{{power}}}$",
            xy=(power, value),
            xytext=(5, 7) if power == 7 else (-5, 7),
            textcoords="offset points",
            color=REGIME_COLORS[regime],
            fontsize=7.0,
            ha="left" if power == 7 else "right",
            va="bottom",
            zorder=5,
        )
    style_training_size_axis(transition_axis, label_every=2)
    transition_axis.set_xlabel(r"Training images $N_{2D}$")
    transition_axis.set_ylabel(r"$\Omega_m$ response slope")
    transition_axis.set_ylim(bottom=min(0.0, float(report["slope_ci16"].min()) - 0.04), top=1.05)
    return figure, report


def build_generalization_figure(metrics: pd.DataFrame) -> plt.Figure:
    """Restyle the existing PCA q95 generalization curve without changing values."""

    set_paper_style()
    required = {"arch", "dataset_size", "gen_gl_q95"}
    missing = sorted(required - set(metrics.columns))
    if missing:
        raise ValueError(f"generalization table is missing columns: {missing}")
    figure, axis = plt.subplots(figsize=(FULL_W, 2.35))
    figure.subplots_adjust(left=0.09, right=0.985, bottom=0.23, top=0.94)
    for arch, label, color, marker in ARCHITECTURES:
        sub = metrics[metrics["arch"] == arch].copy()
        _require_sizes(sub)
        sub["power"] = np.log2(sub["dataset_size"].astype(float))
        sub = sub.sort_values("power")
        axis.plot(
            sub["power"],
            sub["gen_gl_q95"],
            color=color,
            marker=marker,
            ms=3.5,
            lw=1.25,
            label=label,
        )
    style_training_size_axis(axis)
    axis.axhline(0.5, color="0.50", ls=":", lw=0.8, label="_nolegend_")
    axis.set_ylim(-0.03, 1.04)
    axis.set_xlabel(r"Training images $N_{2D}$")
    axis.set_ylabel("PCA q95 novelty score")
    axis.legend(frameon=False, ncol=3, loc="lower right", handlelength=1.8, columnspacing=1.1)
    return figure


def crop_nearest_training_image(
    image: np.ndarray,
    *,
    top_crop_fraction: float = 0.075,
) -> np.ndarray:
    if image.ndim not in (2, 3):
        raise ValueError(f"expected a 2D or 3D image, got shape {image.shape}")
    if not 0.0 <= top_crop_fraction < 0.5:
        raise ValueError("top_crop_fraction must be in [0, 0.5)")
    first_row = int(round(image.shape[0] * top_crop_fraction))
    return image[first_row:].copy()


def export_nearest_training_pdf(
    source: Path,
    output: Path,
    *,
    top_crop_fraction: float = 0.075,
) -> tuple[float, float]:
    """Wrap the exact raster audit in a title-free, full-width vector PDF page."""

    set_paper_style()
    source = Path(source)
    if not source.is_file():
        raise FileNotFoundError(f"missing exact U-Net-128 nearest-training figure: {source}")
    image = crop_nearest_training_image(
        plt.imread(source),
        top_crop_fraction=top_crop_fraction,
    )
    height = FULL_W * image.shape[0] / image.shape[1]
    figure = plt.figure(figsize=(FULL_W, height))
    axis = figure.add_axes((0.0, 0.0, 1.0, 1.0))
    axis.imshow(image, interpolation="nearest")
    axis.set_axis_off()
    save_figure(figure, output)
    plt.close(figure)
    return FULL_W, height


def _load_manifest(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"missing nf_generalize_fig2 manifest: {path}")
    payload = json.loads(path.read_text())
    rows = payload.get("rows", payload.get("runs", [])) if isinstance(payload, dict) else payload
    if not isinstance(rows, list):
        raise ValueError(f"manifest must contain a list of runs: {path}")
    return [dict(row) for row in rows]


def _project_path(project_dir: Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else project_dir / path


def _sample_path(project_dir: Path, row: dict[str, Any], *, seed: int, sample_label: str) -> Path:
    template = str(row.get("sample_path", ""))
    if not template:
        raise ValueError(f"manifest row has no sample_path: {row.get('run_name', '<unknown>')}")
    return _project_path(
        project_dir,
        template.format(seed=int(seed), sample_label=sample_label),
    )


def build_nearest_training_panels(
    project_dir: str | Path,
    manifest_path: str | Path,
    sscd_cache_dir: str | Path,
    *,
    seed: int = 123,
    sample_label: str = "dpm50",
    nbins: int = 30,
    k_max: float = 64.0,
) -> list[dict[str, Any]]:
    """Resolve all data products for the three-row U-Net-128 paper figure."""

    project_dir = Path(project_dir).resolve()
    manifest_path = _project_path(project_dir, manifest_path)
    cache_dir = _project_path(project_dir, sscd_cache_dir)
    selected = [
        row
        for row in _load_manifest(manifest_path)
        if row.get("arch") == "u128" and int(row.get("dataset_size", -1)) in NEAREST_DATASET_SIZES
    ]
    grouped: dict[int, list[dict[str, Any]]] = {}
    for row in selected:
        grouped.setdefault(int(row["dataset_size"]), []).append(row)
    missing_sizes = sorted(set(NEAREST_DATASET_SIZES) - set(grouped))
    if missing_sizes:
        raise ValueError(f"missing U-Net-128 runs for dataset sizes: {missing_sizes}")
    duplicate_sizes = sorted(size for size, rows in grouped.items() if len(rows) != 1)
    if duplicate_sizes:
        raise ValueError(f"duplicate U-Net-128 runs for dataset sizes: {duplicate_sizes}")
    by_size = {size: rows[0] for size, rows in grouped.items()}

    resolved_samples = {
        size: _sample_path(
            project_dir,
            row,
            seed=int(seed),
            sample_label=sample_label,
        )
        for size, row in by_size.items()
    }
    missing_samples = [
        f"dataset_size={size} run={by_size[size].get('run_name', '<unknown>')} path={path}"
        for size, path in resolved_samples.items()
        if not path.is_file()
    ]
    if missing_samples:
        raise FileNotFoundError(
            "missing requested U-Net-128 sample archives:\n" + "\n".join(missing_samples)
        )

    panels: list[dict[str, Any]] = []
    for dataset_size in NEAREST_DATASET_SIZES:
        row = by_size[dataset_size]
        run_name = str(row["run_name"])
        config_path = _project_path(project_dir, row["config"])
        sample_path = resolved_samples[dataset_size]
        generated = load_npz_samples(sample_path)
        if len(generated) != EXPECTED_NEAREST_SAMPLES:
            raise RuntimeError(
                "paper Fréchet audit requires exactly "
                f"{EXPECTED_NEAREST_SAMPLES} generated maps, found {len(generated)}: "
                f"{sample_path}"
            )
        reference = summarize_exact_training_reference(
            config_path,
            expected_slices=int(row.get("actual_2d", row["dataset_size"])),
            generated=generated,
            nbins=int(nbins),
            k_max=float(k_max),
        )

        heldout_cache = resolve_sscd_embedding_cache(
            cache_dir,
            run_name=run_name,
            kind="heldout",
            sample_label=sample_label,
            seed=int(seed),
        )
        generated_cache = resolve_sscd_embedding_cache(
            cache_dir,
            run_name=run_name,
            kind="generated",
            sample_label=sample_label,
            seed=int(seed),
        )
        heldout_features = load_sscd_embedding_cache(heldout_cache)
        generated_features = load_sscd_embedding_cache(generated_cache)
        if len(generated_features) != len(generated):
            raise RuntimeError(
                "generated sample/cache count mismatch: "
                f"samples={len(generated)}, cache={len(generated_features)}, "
                f"sample_path={sample_path}, cache_path={generated_cache}"
            )
        try:
            frechet = normalized_sscd_frechet(
                heldout_features,
                generated_features,
                seed=int(seed),
            )
        except ValueError as error:
            raise ValueError(
                f"{error}; heldout_real_cache={heldout_cache}; "
                f"generated_cache={generated_cache}"
            ) from error
        panels.append(
            {
                "dataset_tag": str(row["dataset_tag"]),
                "dataset_size": int(dataset_size),
                "generated_image": generated[0, 0],
                "nearest_training_image": reference["nearest_training_image"],
                "cos_max": float(reference["nearest_cosine"]),
                "k_bins": reference["k_bins"],
                "pk_ratio": reference["pk_ratio"],
                "pk_log10_mae": float(reference["pk_log10_mae"]),
                "generated_to_heldout_real_frechet": float(
                    frechet["generated_to_real_frechet"]
                ),
                "heldout_real_split_frechet": float(frechet["real_split_frechet"]),
                "sscd_frechet_normalized": float(frechet["sscd_frechet_normalized"]),
                "n_generated": int(frechet["n_generated"]),
                "n_real_split": int(frechet["n_real_split"]),
                "n_heldout_real_available": int(frechet["n_heldout_real_available"]),
                "sscd_reference_kind": str(frechet["reference_kind"]),
                "generated_cache_path": str(generated_cache),
                "heldout_real_cache_path": str(heldout_cache),
                "config_path": str(config_path),
            }
        )
    return panels


def build_nearest_training_figure(panels: list[dict[str, Any]]) -> plt.Figure:
    """Build the generated/nearest/P(k) three-row U-Net-128 paper figure."""

    if len(panels) != len(NEAREST_DATASET_SIZES):
        raise ValueError(
            f"expected {len(NEAREST_DATASET_SIZES)} nearest-training panels, found {len(panels)}"
        )
    ordered = sorted(panels, key=lambda row: int(row["dataset_size"]))
    if [int(row["dataset_size"]) for row in ordered] != list(NEAREST_DATASET_SIZES):
        raise ValueError(
            "nearest-training panels do not cover "
            f"N_2D={list(NEAREST_DATASET_SIZES)} exactly"
        )

    set_paper_style()
    figure, axes = plt.subplots(
        3,
        len(NEAREST_DATASET_SIZES),
        figsize=(FULL_W, 3.35),
        gridspec_kw={"height_ratios": (1.0, 1.0, 0.72)},
    )
    figure.subplots_adjust(
        left=0.085,
        right=0.980,
        bottom=0.16,
        top=0.925,
        wspace=0.17,
        hspace=0.12,
    )

    image_values = np.concatenate(
        [
            np.asarray(row[key], dtype=float).reshape(-1)
            for row in ordered
            for key in ("generated_image", "nearest_training_image")
        ]
    )
    finite_images = image_values[np.isfinite(image_values)]
    if not len(finite_images):
        raise ValueError("nearest-training panels contain no finite image values")
    image_limits = tuple(np.quantile(finite_images, (0.005, 0.995)))

    ratio_values = np.concatenate([np.asarray(row["pk_ratio"], dtype=float) for row in ordered])
    finite_ratios = ratio_values[np.isfinite(ratio_values)]
    if not len(finite_ratios):
        raise ValueError("nearest-training panels contain no finite power-spectrum ratios")
    ratio_low = min(1.0, float(finite_ratios.min()))
    ratio_high = max(1.0, float(finite_ratios.max()))
    ratio_pad = 0.07 * max(ratio_high - ratio_low, 0.1)
    shared_ratio_limits = (max(0.0, ratio_low - ratio_pad), ratio_high + ratio_pad)

    for column, row in enumerate(ordered):
        power = int(round(np.log2(int(row["dataset_size"]))))
        generated_axis = axes[0, column]
        nearest_axis = axes[1, column]
        spectrum_axis = axes[2, column]
        generated_axis.imshow(
            row["generated_image"],
            cmap="viridis",
            vmin=image_limits[0],
            vmax=image_limits[1],
            interpolation="none",
        )
        nearest_axis.imshow(
            row["nearest_training_image"],
            cmap="viridis",
            vmin=image_limits[0],
            vmax=image_limits[1],
            interpolation="none",
        )
        generated_axis.set_title(rf"$2^{{{power}}}$", pad=2.0)
        generated_axis.text(
            0.04,
            0.04,
            f"F={float(row['sscd_frechet_normalized']):.2f}",
            transform=generated_axis.transAxes,
            fontsize=6.3,
            color="white",
            ha="left",
            va="bottom",
            bbox={"facecolor": "black", "alpha": 0.55, "edgecolor": "none", "pad": 1.2},
        )
        nearest_axis.text(
            0.04,
            0.04,
            f"cos={float(row['cos_max']):.3f}",
            transform=nearest_axis.transAxes,
            fontsize=6.3,
            color="white",
            ha="left",
            va="bottom",
            bbox={"facecolor": "black", "alpha": 0.55, "edgecolor": "none", "pad": 1.2},
        )
        for image_axis in (generated_axis, nearest_axis):
            image_axis.set_xticks([])
            image_axis.set_yticks([])
            for spine in image_axis.spines.values():
                spine.set_linewidth(0.55)
                spine.set_color(INK)

        k_bins = np.asarray(row["k_bins"], dtype=float)
        ratio = np.asarray(row["pk_ratio"], dtype=float)
        if np.nanmax(k_bins) > 64.0 + 1.0e-9:
            raise ValueError(f"post-Nyquist k bin reached the paper figure: {np.nanmax(k_bins)}")
        spectrum_axis.axhline(1.0, color="0.42", ls="--", lw=0.7)
        physical_k = fourier_bin_to_physical_k(k_bins)
        physical_k_max = float(fourier_bin_to_physical_k(64.0))
        spectrum_axis.plot(physical_k, ratio, color="0.20", lw=0.9)
        spectrum_axis.set_xlim(left=0.0, right=physical_k_max)
        spectrum_axis.set_ylim(shared_ratio_limits)
        spectrum_axis.set_xticks((0.0, 8.0, 16.0))
        spectrum_axis.set_xticklabels(("0", "8", "16"))
        spectrum_axis.set_xlabel(
            r"$k\,[h\,\mathrm{Mpc}^{-1}]$",
            labelpad=2.0,
            fontsize=6.5,
        )
        style_axis(spectrum_axis)
        spectrum_axis.tick_params(axis="x", labelsize=7.0)
        if column:
            spectrum_axis.tick_params(labelleft=False)

    axes[0, 0].set_ylabel("Generated", labelpad=5.0)
    axes[1, 0].set_ylabel("Closest training", labelpad=5.0)
    axes[2, 0].set_ylabel(r"$R(k)=P_{\rm gen}/P_{\rm real}$", labelpad=5.0)
    return figure


def nearest_training_table(panels: list[dict[str, Any]]) -> pd.DataFrame:
    table = pd.DataFrame(
        [{column: row[column] for column in NEAREST_TABLE_COLUMNS} for row in panels]
    )
    return table.loc[:, list(NEAREST_TABLE_COLUMNS)].sort_values("dataset_size").reset_index(drop=True)


def export_nearest_training_outputs(
    panels: list[dict[str, Any]],
    pdf_path: str | Path,
    csv_path: str | Path,
    *,
    preview_path: str | Path | None = None,
) -> tuple[tuple[float, float], pd.DataFrame]:
    figure = build_nearest_training_figure(panels)
    try:
        dimensions = save_figure(figure, Path(pdf_path))
        if preview_path is not None:
            preview_path = Path(preview_path)
            preview_path.parent.mkdir(parents=True, exist_ok=True)
            figure.savefig(preview_path, format="png", dpi=300, facecolor="white")
    finally:
        plt.close(figure)
    table = nearest_training_table(panels)
    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(csv_path, index=False)
    return dimensions, table


def build_probe_summary_figure(summary: pd.DataFrame) -> plt.Figure:
    """Show held-out-real VGG+MLP response slope and R2 for all parameters."""

    set_paper_style()
    required = {"parameter", "slope", "r2"}
    missing = sorted(required - set(summary.columns))
    if missing:
        raise ValueError(f"probe summary is missing columns: {missing}")
    indexed = summary.set_index("parameter")
    missing_parameters = [parameter for parameter in PARAM_ORDER if parameter not in indexed.index]
    if missing_parameters:
        raise ValueError(f"probe summary is missing parameters: {missing_parameters}")
    ordered = indexed.loc[list(PARAM_ORDER)].reset_index()
    x = np.arange(len(PARAM_ORDER))
    colors = plt.get_cmap("viridis")(np.linspace(0.08, 0.92, len(PARAM_ORDER)))
    figure, axes = plt.subplots(1, 2, figsize=(FULL_W, 2.2))
    figure.subplots_adjust(left=0.08, right=0.985, bottom=0.25, top=0.95, wspace=0.24)
    for axis, metric, ylabel in zip(
        axes,
        ("slope", "r2"),
        ("Held-out-real response slope", r"Held-out-real $R^2$"),
    ):
        values = ordered[metric].to_numpy(float)
        axis.axhline(1.0, color="0.38", ls="--", lw=0.9, label="_nolegend_")
        axis.axhline(0.0, color="0.65", lw=0.65, label="_nolegend_")
        axis.plot(x, values, color="0.60", lw=0.75, zorder=1)
        axis.scatter(x, values, c=colors, s=24, marker="o", edgecolor="white", linewidth=0.4, zorder=2)
        for xpos, value in zip(x, values):
            axis.annotate(f"{value:.2f}", (xpos, value), xytext=(0, 4), textcoords="offset points", ha="center", fontsize=7)
        axis.set_xticks(x, [PARAM_LABELS[parameter] for parameter in PARAM_ORDER])
        axis.set_ylabel(ylabel)
        finite = values[np.isfinite(values)]
        axis.set_ylim(min(-0.15, float(finite.min()) - 0.12), max(1.05, float(finite.max()) + 0.12))
        style_axis(axis)
    axes[0].text(0.01, 1.02, "(a)", transform=axes[0].transAxes, fontweight="bold")
    axes[1].text(0.01, 1.02, "(b)", transform=axes[1].transAxes, fontweight="bold")
    return figure


def save_figure(figure: plt.Figure, output: Path) -> tuple[float, float]:
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, format="pdf", dpi=300, facecolor="white")
    width, height = figure.get_size_inches()
    return float(width), float(height)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=Path, required=True)
    parser.add_argument("--generalization", type=Path, required=True)
    parser.add_argument("--project-dir", type=Path, default=ROOT)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=ROOT / "local" / "nf_generalize_fig2" / "manifest.json",
    )
    parser.add_argument(
        "--sscd-cache-dir",
        type=Path,
        default=ROOT / "results" / "nf_generalize_fig2" / "cache" / "sscd_full_nn",
    )
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--sample-label", default="dpm50")
    parser.add_argument("--probe-summary", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=PAPER_DIR / "figures")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    outputs = {
        "conditional": args.output_dir / "conditional_recovery_transition.pdf",
        "conditional_coverage_table": (
            args.output_dir / "conditional_recovery_coverage_curves.csv"
        ),
        "generalization": args.output_dir / "generalization_transition.pdf",
        "nearest": args.output_dir / "nearest_training_u128.pdf",
        "nearest_preview": args.output_dir / "nearest_training_u128_preview.png",
        "nearest_table": args.output_dir / "nearest_training_u128.csv",
        "probe": args.output_dir / "vgg_probe_heldout_real.pdf",
    }
    samples = pd.read_csv(args.samples)
    conditional, report = build_conditional_coverage_figure(samples)
    dimensions = {"conditional": save_figure(conditional, outputs["conditional"])}
    report.to_csv(outputs["conditional_coverage_table"], index=False)
    plt.close(conditional)
    generalization = build_generalization_figure(pd.read_csv(args.generalization))
    dimensions["generalization"] = save_figure(generalization, outputs["generalization"])
    plt.close(generalization)
    nearest_panels = build_nearest_training_panels(
        args.project_dir,
        args.manifest,
        args.sscd_cache_dir,
        seed=args.seed,
        sample_label=args.sample_label,
    )
    dimensions["nearest"], nearest_table = export_nearest_training_outputs(
        nearest_panels,
        outputs["nearest"],
        outputs["nearest_table"],
        preview_path=outputs["nearest_preview"],
    )
    probe = build_probe_summary_figure(pd.read_csv(args.probe_summary))
    dimensions["probe"] = save_figure(probe, outputs["probe"])
    plt.close(probe)
    for name in ("conditional", "generalization", "nearest", "probe"):
        output = outputs[name]
        print(f"{name}: {output} ({dimensions[name][0]:.3f} x {dimensions[name][1]:.3f} in)")
    print(f"nearest_table: {outputs['nearest_table']} ({len(nearest_table)} rows)")
    print(f"nearest_preview: {outputs['nearest_preview']} (300 dpi)")
    print(
        "conditional_coverage_table: "
        f"{outputs['conditional_coverage_table']} ({len(report)} rows)"
    )
    print(
        report[
            [
                "dataset_size",
                "nominal_coverage",
                "empirical_coverage",
                "coverage_ci16",
                "coverage_ci84",
                "n_heldout",
                "draws_per_cosmology",
            ]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
