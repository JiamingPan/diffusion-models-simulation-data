from __future__ import annotations

import importlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))


def _module():
    return importlib.import_module("plot_nf_conditional_bias_vgg_probe_validation")


def _predictions() -> pd.DataFrame:
    plotting = _module()
    x = np.linspace(0.1, 0.9, 16)
    data: dict[str, np.ndarray] = {}
    for index, parameter in enumerate(plotting.PARAM_ORDER):
        slope = 0.2 + 0.1 * index
        intercept = -0.03 + 0.01 * index
        y = intercept + slope * x
        data[f"{parameter}_true"] = x
        data[f"{parameter}_pred_median"] = y
        data[f"{parameter}_pred_q16"] = y - 0.02
        data[f"{parameter}_pred_q84"] = y + 0.02
    return pd.DataFrame(data)


def test_probe_summary_computes_slope_intercept_and_r2_from_same_points():
    plotting = _module()
    summary = plotting.compute_probe_validation_summary(_predictions())

    assert summary["parameter"].tolist() == plotting.PARAM_ORDER
    for index, row in summary.iterrows():
        expected_slope = 0.2 + 0.1 * index
        expected_intercept = -0.03 + 0.01 * index
        assert row["slope"] == pytest.approx(expected_slope)
        assert row["intercept"] == pytest.approx(expected_intercept)

        truth = _predictions()[f"{row['parameter']}_true"].to_numpy(float)
        prediction = _predictions()[f"{row['parameter']}_pred_median"].to_numpy(float)
        expected_r2 = 1.0 - np.sum((prediction - truth) ** 2) / np.sum((truth - truth.mean()) ** 2)
        assert row["r2"] == pytest.approx(expected_r2)
        assert int(row["n_heldout"]) == len(truth)


def test_probe_summary_fails_closed_when_a_parameter_column_is_missing():
    plotting = _module()
    predictions = _predictions().drop(columns=["A_AGN2_pred_median"])
    with pytest.raises(ValueError, match="A_AGN2_pred_median"):
        plotting.compute_probe_validation_summary(predictions)


def test_probe_validation_writes_all_parameter_figures_and_summary(tmp_path):
    plotting = _module()
    panel_path = tmp_path / "all_parameters_1to1.png"
    summary_path = tmp_path / "slope_r2_summary.png"
    table_path = tmp_path / "slope_r2_summary.csv"

    summary = plotting.write_probe_validation_outputs(
        _predictions(),
        panel_path=panel_path,
        summary_path=summary_path,
        table_path=table_path,
        probe_label="unit-test probe",
    )

    assert len(summary) == 6
    assert panel_path.exists() and panel_path.stat().st_size > 10_000
    assert summary_path.exists() and summary_path.stat().st_size > 10_000
    assert table_path.exists()
    saved = pd.read_csv(table_path)
    assert saved["parameter"].tolist() == plotting.PARAM_ORDER

