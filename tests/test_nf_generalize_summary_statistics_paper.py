from __future__ import annotations

import json
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks" / "nf_generalize_fig2_partial_quickcheck.ipynb"

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _complete_metrics() -> pd.DataFrame:
    rows = []
    for exponent in range(6, 16):
        rows.append(
            {
                "arch": "u128",
                "dataset_size": 2**exponent,
                "hist_l1": 0.02 + 0.002 * (exponent - 6),
                "pk_log10_mae": 0.03 + 0.003 * (exponent - 6),
            }
        )
    rows.append(
        {
            "arch": "u64",
            "dataset_size": 64,
            "hist_l1": 99.0,
            "pk_log10_mae": 99.0,
        }
    )
    return pd.DataFrame(rows)


def _complete_curves() -> list[dict[str, object]]:
    centers = np.linspace(-1.0, 1.0, 41)
    k_bins = np.arange(30, dtype=float)
    curves = []
    for exponent in range(6, 16):
        width = 0.20 + 0.01 * (exponent - 6)
        real_hist = np.exp(-0.5 * (centers / width) ** 2)
        generated_hist = np.exp(-0.5 * ((centers - 0.01) / (width * 1.03)) ** 2)
        pk_ratio = 1.0 + 0.08 * np.sin(k_bins / 6.0 + exponent)
        curves.append(
            {
                "arch": "u128",
                "dataset_size": 2**exponent,
                "hist_centers": centers,
                "real_hist": real_hist,
                "generated_hist": generated_hist,
                "k_bins": k_bins,
                "pk_ratio": pk_ratio,
            }
        )
    return curves


def test_builds_complete_u128_paper_figure(tmp_path):
    from simdiff_eval.paper_figures import (
        build_summary_statistics_figure,
    )

    output = tmp_path / "summary_statistics_sweep.pdf"
    figure, plotted = build_summary_statistics_figure(_complete_metrics(), output)

    assert output.is_file()
    assert np.allclose(figure.get_size_inches(), [6.75, 2.35])
    assert plotted["arch"].eq("u128").all()
    assert plotted["dataset_size"].tolist() == [2**k for k in range(6, 16)]
    assert len(figure.axes) == 2
    assert all(not axis.xaxis._major_tick_kw.get("gridOn", False) for axis in figure.axes)
    plt.close(figure)


def test_refuses_incomplete_u128_sweep(tmp_path):
    from simdiff_eval.paper_figures import (
        build_summary_statistics_figure,
    )

    incomplete = _complete_metrics().query("dataset_size != 32768")
    with pytest.raises(ValueError, match="complete UNet-128 sweep"):
        build_summary_statistics_figure(
            incomplete,
            tmp_path / "summary_statistics_sweep.pdf",
        )


def test_notebook_exports_the_paper_figure():
    notebook = json.loads(NOTEBOOK.read_text())
    source = "\n".join(
        "".join(cell.get("source", [])) for cell in notebook.get("cells", [])
    )

    assert "build_summary_statistics_curve_figure" in source
    assert "summary_statistics_sweep.pdf" in source
    assert "UNet-128 summary-statistics sweep" in source


def test_builds_full_curve_u128_paper_figure(tmp_path):
    from simdiff_eval.paper_figures import build_summary_statistics_curve_figure

    output = tmp_path / "summary_statistics_sweep.pdf"
    figure, plotted = build_summary_statistics_curve_figure(_complete_curves(), output)

    assert output.is_file()
    assert np.allclose(figure.get_size_inches(), [6.75, 3.4])
    assert plotted["dataset_size"].tolist() == [2**k for k in range(6, 16)]
    assert len(figure.axes) == 2
    assert all(len(axis.get_yticklabels()) == 10 for axis in figure.axes)
    assert all(not axis.xaxis._major_tick_kw.get("gridOn", False) for axis in figure.axes)
    plt.close(figure)


def test_curve_figure_refuses_incomplete_u128_sweep(tmp_path):
    from simdiff_eval.paper_figures import build_summary_statistics_curve_figure

    with pytest.raises(ValueError, match="complete UNet-128 sweep"):
        build_summary_statistics_curve_figure(
            _complete_curves()[:-1],
            tmp_path / "summary_statistics_sweep.pdf",
        )
