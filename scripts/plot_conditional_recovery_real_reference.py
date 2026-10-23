#!/usr/bin/env python
"""Paper Fig. 2 (conditional_recovery_transition.pdf) with the real-map coverage curve added as a grey reference.

The generated-map panels are built by plot_nf_conditional_bias_paper_figures.build_conditional_coverage_figure from
results/nf_conditional_bias_fresh_full_sweep_200k/calibration_vgg/bias_probe_per_sample_predictions.csv (same input
and numbers as the paper figure). The real-map curve scores all 128 z=0 slices of each held-out simulation 900-931
through the same frozen probe (plot_iaifi_statement_figure.load) exactly as the 64 generated draws are scored: is the
true Omega_m inside the central interval of the per-cosmology predictions. Writes a new timestamped folder under
results/; never touches paper/ai4science_verification/figures/.
"""
from __future__ import annotations

import shutil
import sys
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "scripts", ROOT / "paper" / "ai4science_verification"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
from plot_iaifi_statement_figure import load as load_real  # noqa: E402
from plot_nf_conditional_bias_paper_figures import (  # noqa: E402
    COVERAGE_CURVE_LEVELS,
    COVERAGE_LEVELS,
    COVERAGE_MARKERS,
    build_conditional_coverage_figure,
)

SAMPLES = ROOT / "results/nf_conditional_bias_fresh_full_sweep_200k/calibration_vgg/bias_probe_per_sample_predictions.csv"
REAL_COLOR = "0.55"


def real_curve(truth: np.ndarray, real: np.ndarray) -> pd.DataFrame:
    levels = np.asarray(COVERAGE_CURVE_LEVELS, dtype=float)
    lo = np.quantile(real, (1 - levels) / 2, axis=1)  # (levels, cosmologies)
    hi = np.quantile(real, (1 + levels) / 2, axis=1)
    cov = ((lo <= truth) & (truth <= hi)).mean(axis=1)
    return pd.DataFrame({"nominal_coverage": levels, "empirical_coverage": cov,
                         "n_heldout": len(truth), "slices_per_cosmology": real.shape[1]})


def main() -> None:
    out = ROOT / "results" / f"conditional_recovery_real_reference_{datetime.now():%Y%m%d_%H%M%S}"
    out.mkdir(parents=True, exist_ok=False)
    shutil.copy(__file__, out / Path(__file__).name)
    figure, report = build_conditional_coverage_figure(pd.read_csv(SAMPLES))
    truth, draws = load_real()
    real = real_curve(truth, draws["real"])
    cax = figure.axes[1]
    # Against a single-map spread, coverage above the diagonal is expected for an unbiased probe, so the paper
    # figure's "underconfident"/"overconfident" region labels would mislead next to the real-map reference.
    for t in list(cax.texts):
        if t.get_text() in {"underconfident", "overconfident"}:
            t.remove()
    cax.plot(real.nominal_coverage, real.empirical_coverage, color=REAL_COLOR, lw=1.3, zorder=1.5)
    for nominal in COVERAGE_LEVELS:
        f = real[np.isclose(real.nominal_coverage, nominal)]
        cax.scatter(f.nominal_coverage, f.empirical_coverage, s=22, marker=COVERAGE_MARKERS[nominal],
                    color=REAL_COLOR, edgecolor="0.15", linewidth=0.45, zorder=1.6)
    cax.text(0.19, 0.345, "real maps", color="0.35", fontsize=8.5, ha="center")
    stem = "conditional_recovery_transition_realref"
    figure.savefig(out / f"{stem}.pdf", format="pdf", dpi=300, facecolor="white")
    figure.savefig(out / f"{stem}.png", format="png", dpi=200, facecolor="white")
    plt.close(figure)
    report.to_csv(out / "generated_coverage_curves.csv", index=False)
    real.to_csv(out / "real_coverage_curve.csv", index=False)
    for lv in (0.68, 0.95):
        print(f"real coverage at {lv}:", float(real.empirical_coverage[np.isclose(real.nominal_coverage, lv)].iloc[0]))
    print("Wrote", out)


if __name__ == "__main__":
    main()
