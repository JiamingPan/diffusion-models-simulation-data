from __future__ import annotations

import json
from pathlib import Path


NOTEBOOK = (
    Path(__file__).resolve().parents[1]
    / "notebooks"
    / "nf_probe_control_validation.ipynb"
)


def notebook_source() -> str:
    notebook = json.loads(NOTEBOOK.read_text())
    return "\n".join("".join(cell.get("source", [])) for cell in notebook["cells"])


def test_probe_validation_notebook_has_reader_facing_sections_and_compilable_code():
    notebook = json.loads(NOTEBOOK.read_text())
    assert notebook["nbformat"] == 4
    source = notebook_source()
    sections = ["## tl;dr", "## Context & Methods", "### Key Assumptions", "## Data", "## Results", "## Takeaways"]
    positions = [source.index(section) for section in sections]
    assert positions == sorted(positions)
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            compile("".join(cell["source"]), "probe-control-notebook-cell", "exec")


def test_probe_validation_notebook_freezes_inputs_and_uses_visible_overrides():
    source = notebook_source()
    for required in (
        "PROBE_CONTROLS_RESULTS_ROOT",
        "PROBE_CONTROLS_CODE_ROOT",
        "dced4f8928efe248d819a72560ef61a099d0c4a3",
        "/scratch/huterer_root/huterer0/jiamingp/probe_controls_code_456f01a",
        "nf_conditional_bias_probe",
        "vgg_mlp_encoder.npz",
        "vgg_mlp_encoder.pkl",
        "local/nf_conditional_bias_probe/manifest.json",
        "vgg16-397923af.pth",
    ):
        assert required in source


def test_probe_validation_notebook_has_strict_structural_gates_without_scientific_thresholds():
    source = notebook_source()
    for required in (
        "MissingInputError",
        "json.loads",
        "EXPECTED_TRANSFORM_LINES = 1_990_657",
        "EXPECTED_C4_LINES = 73_729",
        "heldout_indices",
        "900",
        "931",
        "duplicate analytical grains",
        "required_columns",
        "chunksize",
        "git rev-parse HEAD",
        "Ready to share",
        "Share with caveats",
        "Needs revision",
        "C4_LIMITATION",
        "one-point PDF",
        "higher-order",
    ):
        assert required in source
    assert "assert rmse" not in source
    assert "PASS_THRESHOLD" not in source


def test_probe_validation_notebook_covers_c0_c1_c4_visuals_and_non_color_encoding():
    source = notebook_source()
    for required in (
        "C0 symmetry and translation stability",
        "worst cases",
        "median_std_ratio_ci_low",
        "lowpass",
        "highpass",
        "sharp",
        "hann",
        "fft_roundtrip_null",
        "C4 grouped probe metrics",
        "real_original",
        "real_measured_transfer",
        "real_gaussian",
        "generated",
        "power_ratio",
        "measured_transfer",
        "field_histograms",
        "linestyle",
        "marker",
        "tab:blue",
        "tab:orange",
    ):
        assert required in source


def test_c1_uses_family_coverage_and_parameter_facets_with_uncertainty():
    source = notebook_source()
    for required in (
        "transform_families",
        "lowpass",
        "highpass",
        "fft_roundtrip_null",
        "parameter_facets",
        "assert_unique_grain",
        "rmse_ci_low",
        "rmse_ci_high",
        "bias_ci_low",
        "slope_ci_low",
        "out_of_range_fraction",
        "per_cosmology",
        "PARAMETER_ORDER = ('Omega_m', 'sigma_8', 'A_SN1', 'A_AGN1', 'A_SN2', 'A_AGN2')",
        "PARAMETERS = set(PARAMETER_ORDER)",
        "parameter_facets = list(PARAMETER_ORDER)",
    ):
        assert required in source
    assert "if not {'lowpass', 'highpass', 'fft_roundtrip_null'}.issubset(transform_names)" not in source


def test_c4_preserves_grains_baselines_uncertainty_and_saved_histogram_bins():
    source = notebook_source()
    assert "groupby(['run_name', 'source'], as_index=False).first()" not in source
    for required in (
        "assert_unique_grain(metrics",
        "real_original",
        "real_measured_transfer",
        "real_gaussian",
        "generated",
        "rmse_ci_low",
        "bias_ci_low",
        "slope_ci_low",
        "bin_edges",
        "axis.hist",
        "histtype",
        "measured_transfer",
        "gaussian_transfer",
        "linestyle='--'",
    ):
        assert required in source


def test_c4_chart_has_parameter_override_and_reader_safe_two_by_three_layout():
    source = notebook_source()
    for required in (
        "PROBE_CONTROLS_PARAMETER",
        "FOCUS_PARAMETER",
        "FOCUS_PARAMETER not in PARAMETERS",
        "plt.subplots(2, 3",
        "source_short_labels",
        "RUN_COLORS",
        "axhline(0",
        "axhline(1",
        "row_index",
        "column_index",
        "Omega_m",
    ):
        assert required in source
    assert "axis.set_xticks([])" not in source
    assert "axis.text(float(np.mean(x)), axis.get_ylim()[0]" not in source


def test_c4_power_histograms_and_execution_handoff_have_stable_labels():
    source = notebook_source()
    for required in (
        "RUN_COLORS[run_name]",
        "linestyle='--'",
        "linestyle=':'",
        "source_styles[source]",
        "label=source_short_labels[source]",
        "--output-dir",
        "nf_probe_control_validation.ipynb",
        "RUN_LABELS",
        "N=128, d=2.07",
        "N=16,384, d=2.14",
        "run_label = RUN_LABELS[run_name]",
        "comparison['parameter'] == FOCUS_PARAMETER",
    ):
        assert required in source
