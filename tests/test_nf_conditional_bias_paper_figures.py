from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
PAPER_DIR = ROOT / "paper" / "ai4science_verification"
for path in (SCRIPTS, PAPER_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))


def _module():
    return importlib.import_module("plot_nf_conditional_bias_paper_figures")


def _conditional_tables() -> tuple[pd.DataFrame, pd.DataFrame]:
    point_rows: list[dict[str, float | int | str]] = []
    slope_rows: list[dict[str, float | int | str]] = []
    for power in range(6, 16):
        dataset_size = 2**power
        slope = 0.25 + 0.055 * (power - 6)
        intercept = 0.18 * (1.0 - slope)
        for theta_in in np.linspace(0.12, 0.50, 10):
            recovered = intercept + slope * theta_in
            point_rows.append(
                {
                    "dataset_size": dataset_size,
                    "parameter": "Omega_m",
                    "theta_in": theta_in,
                    "theta_rec_median": recovered,
                    "theta_rec_q16": recovered - 0.018,
                    "theta_rec_q84": recovered + 0.021,
                }
            )
        slope_rows.append(
            {
                "dataset_size": dataset_size,
                "parameter": "Omega_m",
                "slope": slope,
                "slope_ci16": slope - 0.035,
                "slope_ci84": slope + 0.045,
                "intercept": intercept,
            }
        )
    return pd.DataFrame(point_rows), pd.DataFrame(slope_rows)


def _posterior_sample_table(*, draws_per_cosmology: int = 64) -> pd.DataFrame:
    """Build nested synthetic intervals with 68% coverage=.5 and 95%=.75."""

    rows: list[dict[str, float | int | str]] = []
    base_draws = np.linspace(-3.0, 3.0, draws_per_cosmology)
    for power in range(6, 16):
        dataset_size = 2**power
        for heldout_sim in range(900, 932):
            if heldout_sim < 916:
                shift = 0.0
            elif heldout_sim < 924:
                shift = 2.2
            else:
                shift = 4.0
            truth = 0.3
            for seed_index, draw in enumerate(base_draws + shift + truth):
                rows.append(
                    {
                        "dataset_size": dataset_size,
                        "parameter": "Omega_m",
                        "heldout_sim": heldout_sim,
                        "seed_index": seed_index,
                        "theta_in": truth,
                        "theta_rec": float(draw),
                        "guidance_label": "noguidance",
                        "cfg_dropout": 0.0,
                    }
                )
    return pd.DataFrame(rows)


def _generalization_table() -> pd.DataFrame:
    rows = []
    offsets = {"u64": -2, "u128": 0, "u256": 1}
    for arch, offset in offsets.items():
        for power in range(6, 16):
            score = np.clip((power - (8 + offset)) / 2.0, 0.0, 1.0)
            rows.append(
                {
                    "arch": arch,
                    "dataset_size": 2**power,
                    "gen_gl_q95": score,
                }
            )
    return pd.DataFrame(rows)


def _probe_summary() -> pd.DataFrame:
    parameters = ["Omega_m", "sigma_8", "A_SN1", "A_AGN1", "A_SN2", "A_AGN2"]
    return pd.DataFrame(
        {
            "parameter": parameters,
            "slope": [0.91, 0.74, 0.28, 0.12, 0.19, 0.08],
            "r2": [0.91, 0.74, 0.18, -0.05, 0.07, -0.11],
        }
    )


def _assert_paper_axis(axis) -> None:
    assert not axis.spines["top"].get_visible()
    assert not axis.spines["right"].get_visible()
    assert not any(line.get_visible() for line in axis.get_xgridlines())
    assert not any(line.get_visible() for line in axis.get_ygridlines())


def test_paper_style_sets_vector_text_and_full_width():
    paperstyle = importlib.import_module("paperstyle")
    paperstyle.set_paper_style()
    assert paperstyle.FULL_W == pytest.approx(6.75)
    assert plt.rcParams["pdf.fonttype"] == 42
    assert plt.rcParams["font.family"][0] == "serif"
    assert plt.rcParams["font.size"] == pytest.approx(9.0)
    assert plt.rcParams["xtick.labelsize"] == pytest.approx(8.0)


def test_conditional_figure_contrasts_two_regimes_and_highlights_matching_curve_points(tmp_path):
    plotting = _module()
    points, slopes = _conditional_tables()
    output = tmp_path / "conditional_recovery_transition.pdf"

    figure, report = plotting.build_conditional_recovery_figure(points, slopes)
    try:
        assert figure.get_size_inches()[0] == pytest.approx(6.75)
        assert figure.get_size_inches()[1] == pytest.approx(2.4)
        assert figure._suptitle is None
        assert len(figure.axes) == 2
        assert all(axis.get_title() == "" for axis in figure.axes)
        for axis in figure.axes:
            _assert_paper_axis(axis)

        calibration_axis, transition_axis = figure.axes
        assert calibration_axis.get_xlabel() == r"Requested $\Omega_m$"
        assert calibration_axis.get_ylabel() == r"Recovered $\Omega_m$"
        annotation_text = "\n".join(text.get_text() for text in calibration_axis.texts)
        assert "memorization regime" in annotation_text
        assert "generalization regime" in annotation_text
        assert r"$N_{2D}=2^{7}$" in annotation_text
        assert r"$N_{2D}=2^{14}$" in annotation_text
        assert r"slope = $0.305^{+0.045}_{-0.035}$" in annotation_text
        assert r"slope = $0.690^{+0.045}_{-0.035}$" in annotation_text
        assert all(text.get_fontsize() == pytest.approx(7.0) for text in calibration_axis.texts)
        assert len(calibration_axis.collections) == 2
        fit_colors = {
            line.get_color()
            for line in calibration_axis.lines
            if line.get_linestyle() == "-"
        }
        assert set(plotting.REGIME_COLORS.values()).issubset(fit_colors)

        assert transition_axis.get_xlim() == pytest.approx((5.65, 15.35))
        assert transition_axis.get_xticks().tolist() == list(range(6, 16))
        assert [tick.get_text() for tick in transition_axis.get_xticklabels()] == [
            rf"$2^{{{power}}}$" if power % 2 == 0 else "" for power in range(6, 16)
        ]
        assert all(tick.get_rotation() == 0 for tick in transition_axis.get_xticklabels())
        assert transition_axis.get_ylabel() == r"$\Omega_m$ response slope"
        assert transition_axis.get_xlabel() == r"Training images $N_{2D}$"
        assert transition_axis.get_position().width > calibration_axis.get_position().width

        assert len(transition_axis.patches) == 0
        transition_text = "\n".join(text.get_text() for text in transition_axis.texts)
        assert "memorization" not in transition_text
        assert "generalization" not in transition_text
        assert set(transition_text.splitlines()) == {r"$2^{7}$", r"$2^{14}$"}

        axis_triangles = [
            line
            for line in transition_axis.lines
            if line.get_marker() == "v" and line.get_markersize() > 0
        ]
        assert axis_triangles == []
        neutral_points = [
            line
            for line in transition_axis.lines
            if line.get_marker() == "o"
            and line.get_markersize() == pytest.approx(3.7)
        ]
        assert len(neutral_points) == 10
        assert {line.get_color() for line in neutral_points} == {plotting.TRANSITION_COLOR}
        highlighted_points = [
            line
            for line in transition_axis.lines
            if line.get_markersize() == pytest.approx(5.2)
        ]
        assert {float(line.get_xdata()[0]) for line in highlighted_points} == {7.0, 14.0}
        assert {line.get_color() for line in highlighted_points} == set(
            plotting.REGIME_COLORS.values()
        )
        assert {
            text.get_color() for text in transition_axis.texts
        } == set(plotting.REGIME_COLORS.values())
        assert report["dataset_size"].tolist() == [2**power for power in range(6, 16)]
        assert report.loc[0, "slope"] == pytest.approx(0.25)
        assert report.loc[9, "slope_ci84"] == pytest.approx(0.25 + 0.055 * 9 + 0.045)
        plotting.save_figure(figure, output)
    finally:
        plt.close(figure)

    assert output.read_bytes().startswith(b"%PDF")


def test_conditional_figure_fails_closed_if_any_training_size_is_missing():
    plotting = _module()
    points, slopes = _conditional_tables()
    incomplete = slopes[slopes["dataset_size"] != 2**11]
    with pytest.raises(ValueError, match="missing dataset sizes"):
        plotting.build_conditional_recovery_figure(
            points,
            incomplete,
        )


def test_coverage_summary_uses_central_intervals_for_each_heldout_cosmology():
    plotting = _module()
    report = plotting.compute_conditional_coverage(
        _posterior_sample_table(),
        bootstrap=200,
        seed=17,
    )

    assert report["dataset_size"].drop_duplicates().tolist() == [2**power for power in range(6, 16)]
    assert set(report["nominal_coverage"]) == {0.68, 0.95}
    assert set(report["n_heldout"]) == {32}
    assert set(report["draws_per_cosmology"]) == {64}
    first = report[report["dataset_size"] == 2**6].set_index("nominal_coverage")
    assert first.loc[0.68, "empirical_coverage"] == pytest.approx(0.50)
    assert first.loc[0.95, "empirical_coverage"] == pytest.approx(0.75)
    assert np.all(report["coverage_ci16"] <= report["empirical_coverage"])
    assert np.all(report["empirical_coverage"] <= report["coverage_ci84"])


def test_coverage_summary_fails_closed_on_inconsistent_draw_counts():
    plotting = _module()
    samples = _posterior_sample_table()
    bad_row = samples.index[
        (samples["dataset_size"] == 2**6)
        & (samples["heldout_sim"] == 900)
        & (samples["seed_index"] == 0)
    ][0]
    samples = samples.drop(index=bad_row)
    with pytest.raises(ValueError, match="inconsistent posterior-draw counts"):
        plotting.compute_conditional_coverage(samples, bootstrap=20, seed=2)


def test_conditional_coverage_figure_pairs_recovery_scatter_with_coverage(tmp_path):
    plotting = _module()
    figure, report = plotting.build_conditional_coverage_figure(
        _posterior_sample_table(),
        bootstrap=200,
        seed=17,
    )
    try:
        assert figure.get_size_inches() == pytest.approx((6.75, 3.20))
        assert figure._suptitle is None
        assert len(figure.axes) == 2
        recovery_axis, coverage_axis = figure.axes
        for axis in figure.axes:
            _assert_paper_axis(axis)

        assert recovery_axis.get_xlabel() == r"Requested $\Omega_m$"
        assert recovery_axis.get_ylabel() == r"Recovered $\Omega_m$"
        assert recovery_axis.get_xlim() == pytest.approx(recovery_axis.get_ylim())
        assert recovery_axis.get_aspect() == pytest.approx(1.0)
        assert len(recovery_axis.collections) == 3
        recovery_reference = [
            line
            for line in recovery_axis.lines
            if line.get_linestyle() == "--"
            and np.allclose(line.get_xdata(), line.get_ydata())
        ]
        assert len(recovery_reference) == 1

        assert coverage_axis.get_xlabel() == "Nominal coverage"
        assert coverage_axis.get_ylabel() == "Empirical coverage"
        for axis in (recovery_axis, coverage_axis):
            assert axis.xaxis.label.get_fontsize() >= 11.0
            assert axis.yaxis.label.get_fontsize() >= 11.0
            assert all(label.get_fontsize() >= 9.5 for label in axis.get_xticklabels())
            assert all(label.get_fontsize() >= 9.5 for label in axis.get_yticklabels())
        assert coverage_axis.get_xlim() == pytest.approx((0.0, 1.0))
        assert coverage_axis.get_ylim() == pytest.approx((0.0, 1.0))
        labels = [text.get_text() for text in figure.legends[0].get_texts()]
        assert labels == [
            r"$N_{2D}=2^{7}$ memorization regime",
            r"$N_{2D}=2^{10}$",
            r"$N_{2D}=2^{14}$ generalization regime",
        ]
        calibration_lines = [
            line
            for line in coverage_axis.lines
            if np.allclose(line.get_xdata(), line.get_ydata())
        ]
        assert len(calibration_lines) == 1
        region_labels = {text.get_text(): text for text in coverage_axis.texts}
        assert set(region_labels) == {
            "underconfident",
            "overconfident",
        }
        assert all(text.get_fontsize() >= 9.5 for text in region_labels.values())
        assert all(text.get_ha() == "center" for text in region_labels.values())
        assert region_labels["overconfident"].get_position()[0] <= 0.68
        assert figure.legends[0].get_texts()
        assert all(
            text.get_fontsize() >= 9.0
            for text in figure.legends[0].get_texts()
        )
        assert plotting.COVERAGE_MARKERS == {0.68: "o", 0.95: "s"}
        plotted_sizes = set(report.loc[report["plotted"], "dataset_size"])
        assert plotted_sizes == {2**7, 2**10, 2**14}
        assert report["nominal_coverage"].nunique() >= 41
        output = tmp_path / "conditional_recovery_transition.pdf"
        plotting.save_figure(figure, output)
    finally:
        plt.close(figure)
    assert output.read_bytes().startswith(b"%PDF")
    assert len(report) >= 410


def test_generalization_figure_uses_the_identical_training_axis(tmp_path):
    plotting = _module()
    figure = plotting.build_generalization_figure(_generalization_table())
    try:
        axis = figure.axes[0]
        assert figure.get_size_inches()[0] == pytest.approx(6.75)
        assert figure._suptitle is None
        assert axis.get_title() == ""
        assert axis.get_xlim() == pytest.approx((5.65, 15.35))
        assert axis.get_xticks().tolist() == list(range(6, 16))
        assert [tick.get_text() for tick in axis.get_xticklabels()] == [
            rf"$2^{{{power}}}$" for power in range(6, 16)
        ]
        _assert_paper_axis(axis)
        output = tmp_path / "generalization_transition.pdf"
        plotting.save_figure(figure, output)
    finally:
        plt.close(figure)
    assert output.read_bytes().startswith(b"%PDF")


def test_nearest_training_export_crops_title_band_and_stays_full_width(tmp_path):
    plotting = _module()
    image = np.ones((200, 600, 3), dtype=float)
    image[:30, :, :] = 0.0
    image[30:, :, 1] = 0.5
    cropped = plotting.crop_nearest_training_image(image, top_crop_fraction=0.15)
    assert cropped.shape == (170, 600, 3)
    assert float(cropped[0].mean()) > 0.5

    source = tmp_path / "nearest.png"
    plt.imsave(source, image)
    output = tmp_path / "nearest_training_unet128.pdf"
    dimensions = plotting.export_nearest_training_pdf(
        source,
        output,
        top_crop_fraction=0.15,
    )
    assert dimensions[0] == pytest.approx(6.75)
    assert dimensions[1] < dimensions[0]
    assert output.read_bytes().startswith(b"%PDF")


def test_probe_summary_is_saved_as_a_paper_pdf(tmp_path):
    plotting = _module()
    figure = plotting.build_probe_summary_figure(_probe_summary())
    try:
        assert figure.get_size_inches()[0] == pytest.approx(6.75)
        assert len(figure.axes) == 2
        assert figure._suptitle is None
        assert all(axis.get_title() == "" for axis in figure.axes)
        for axis in figure.axes:
            _assert_paper_axis(axis)
        output = tmp_path / "vgg_probe_heldout_real.pdf"
        plotting.save_figure(figure, output)
    finally:
        plt.close(figure)
    assert output.read_bytes().startswith(b"%PDF")


def test_results_notebook_contains_idempotent_paper_figure_section(tmp_path):
    updater = importlib.import_module("update_nf_conditional_bias_paper_figures_notebook")
    source = ROOT / "notebooks" / "nf_conditional_bias_vgg_results.ipynb"
    target = tmp_path / source.name
    target.write_text(source.read_text())

    updater.update(target)
    once = target.read_text()
    updater.update(target)
    assert target.read_text() == once

    notebook = json.loads(once)
    tagged = [
        cell
        for cell in notebook["cells"]
        if updater.TAG in cell.get("metadata", {}).get("tags", [])
    ]
    assert [cell["cell_type"] for cell in tagged] == ["markdown", "code", "markdown"]
    source_text = "\n".join("".join(cell.get("source", [])) for cell in tagged)
    assert "conditional_recovery_transition.pdf" in source_text
    assert "conditional_recovery_coverage_curves.csv" in source_text
    assert "recovered-versus-requested" in source_text
    assert "left panel" in source_text
    assert "right panel" in source_text
    assert "generalization_transition.pdf" in source_text
    assert "nearest_training_u128.pdf" in source_text
    assert "nearest_training_u128_preview.png" in source_text
    assert "nearest_training_u128.csv" in source_text
    assert "nearest_training_u128_caption.tex" in source_text
    assert "vgg_probe_heldout_real.pdf" in source_text
    assert "bias_probe_per_sample_predictions.csv" in source_text
    assert "build_conditional_coverage_figure" in source_text
    assert "nominal_coverage" in source_text and "empirical_coverage" in source_text
    assert "conditional_coverage_report.to_csv" in source_text
    assert "conditional_coverage_report['plotted']" in source_text
    assert "paper_dimensions" in source_text
    assert "plot_nf_conditional_bias_paper_figures" in source_text
    assert "paper_generalization" in source_text
    assert "paper_samples" in source_text


def test_results_notebook_displays_every_paper_figure_inline_before_closing(tmp_path):
    updater = importlib.import_module("update_nf_conditional_bias_paper_figures_notebook")
    source = ROOT / "notebooks" / "nf_conditional_bias_vgg_results.ipynb"
    target = tmp_path / source.name
    target.write_text(source.read_text())

    updater.update(target)
    notebook = json.loads(target.read_text())
    code_cell = next(
        cell
        for cell in notebook["cells"]
        if cell["cell_type"] == "code"
        and updater.TAG in cell.get("metadata", {}).get("tags", [])
    )
    source_text = "".join(code_cell["source"])

    for figure_name in (
        "conditional_figure",
        "generalization_figure",
        "probe_figure",
    ):
        display_call = f"display({figure_name})"
        close_call = f"plt.close({figure_name})"
        assert display_call in source_text
        assert source_text.index(display_call) < source_text.index(close_call)

    assert "build_nearest_training_panels" in source_text
    assert "export_nearest_training_outputs" in source_text
    assert "display(nearest_training_report)" in source_text
    assert "display(Image(filename=str(paper_outputs['nearest_preview']), width=1100))" in source_text
    assert "display(nearest_figure)" not in source_text
