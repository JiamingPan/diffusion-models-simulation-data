from __future__ import annotations

import json
import sys
import types
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


NOTEBOOK = (
    Path(__file__).resolve().parents[1]
    / "notebooks"
    / "nf_probe_control_validation.ipynb"
)


def notebook_source() -> str:
    notebook = json.loads(NOTEBOOK.read_text())
    return "\n".join("".join(cell.get("source", [])) for cell in notebook["cells"])


def test_notebook_setup_uses_production_c4_limitation():
    from simdiff_eval.probe_controls import C4_LIMITATION

    notebook = json.loads(NOTEBOOK.read_text())
    setup = next(
        cell for cell in notebook["cells"]
        if cell["cell_type"] == "code" and "PROJECT_DIR =" in "".join(cell["source"])
    )
    namespace = {}
    display_module = types.ModuleType("IPython.display")
    display_module.Markdown = lambda value: value
    display_module.display = lambda *args, **kwargs: None
    ipython_module = types.ModuleType("IPython")
    ipython_module.display = display_module
    previous = {name: sys.modules.get(name) for name in ("IPython", "IPython.display")}
    sys.modules["IPython"] = ipython_module
    sys.modules["IPython.display"] = display_module
    try:
        exec("".join(setup["source"]), namespace)
    finally:
        for name, module in previous.items():
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module
    assert namespace["C4_LIMITATION"] == C4_LIMITATION


def test_notebook_cells_have_unique_nonempty_ids():
    notebook = json.loads(NOTEBOOK.read_text())
    ids = [cell.get("id") for cell in notebook["cells"]]
    assert all(isinstance(cell_id, str) and cell_id for cell_id in ids)
    assert len(ids) == len(set(ids))


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


def test_reader_story_says_one_frozen_predictor_and_no_refitting():
    source = notebook_source()
    assert "same map -> one controlled change -> SAME frozen predictor -> compare Omega_m prediction" in source
    assert "No refitting" in source
    assert "trained once and frozen" in source


def test_deterministic_example_selection_is_not_error_based():
    notebook = json.loads(NOTEBOOK.read_text())
    helper_cell = next(
        cell for cell in notebook["cells"]
        if cell["cell_type"] == "code"
        and "def select_representative_example" in "".join(cell["source"])
    )
    namespace = {"np": np, "pd": pd}
    exec("".join(helper_cell["source"]), namespace)
    rows = pd.DataFrame(
        {
            "sim_index": [900, 900, 901, 901, 902, 902],
            "z_index": [3, 4, 3, 4, 3, 4],
            "parameter": ["Omega_m"] * 6,
            "theta_true": [0.20, 0.20, 0.30, 0.30, 0.40, 0.40],
            "theta_pred": [0.20, 0.20, 0.99, 0.99, 0.40, 0.40],
        }
    )
    selected = namespace["select_representative_example"](rows)
    assert selected["sim_index"] == 901
    assert selected["z_index"] == 3
    function_text = "".join(helper_cell["source"])
    function_text = function_text.split("def select_representative_example", 1)[1]
    assert "theta_pred" not in function_text
    assert "rmse" not in function_text
    assert "error" not in function_text.lower()


def test_gallery_reconstruction_uses_saved_predictions_without_inference():
    source = notebook_source()
    for required in (
        "read_saved_prediction_rows",
        "theta_pred",
        "apply_saved_transform_for_display",
        "load_heldout_real_slices",
        "get_transform",
        "SAME frozen predictor",
    ):
        assert required in source
    for forbidden in ("load_vgg_encoder", "predict_norm", "evaluate_transform_specs", "fit_ridge"):
        assert forbidden not in source


def test_gallery_panels_cover_exact_c0_c1_and_c4_story():
    source = notebook_source()
    for required in (
        "C0_GALLERY_COLUMNS = ('original', 'rotated', 'reflected', 'shifted')",
        "C1_GALLERY_COLUMNS = ('original', 'low-pass', 'high-pass')",
        "C4_GALLERY_COLUMNS = ('original real', 'power-matched real', 'Gaussian-smoothed real', 'generated')",
        "Did orientation/location change predicted Omega_m?",
        "What happens when broad or fine structure is removed?",
        "Does matching the power deficit reproduce the generated-map Omega_m shift?",
        "same cosmology, different realization",
    ):
        assert required in source


def gallery_helper_namespace():
    notebook = json.loads(NOTEBOOK.read_text())
    helper_cell = next(
        cell for cell in notebook["cells"]
        if cell["cell_type"] == "code"
        and "def read_saved_prediction_rows" in "".join(cell["source"])
    )
    root = NOTEBOOK.parents[1]
    namespace = {
        "CODE_ROOT": root,
        "DATA_ROOT": root / "unused-data",
        "ENCODER_PATH": root / "unused-encoder.npz",
        "Path": Path,
        "np": np,
        "pd": pd,
    }
    exec("".join(helper_cell["source"]), namespace)
    return namespace


def test_display_transform_executes_composite_c0_roll():
    namespace = gallery_helper_namespace()
    image = np.arange(16, dtype=np.float32).reshape(1, 1, 4, 4)
    transformed = namespace["apply_saved_transform_for_display"](
        image,
        "dihedral_g0__roll_dx1_dy-1",
    )
    expected = np.roll(image, shift=(-1, 1), axis=(-2, -1))
    np.testing.assert_array_equal(transformed, expected)


def test_saved_transform_records_require_exact_transform_and_matching_truth():
    namespace = gallery_helper_namespace()
    rows = pd.DataFrame(
        {
            "transform": ["identity", "dihedral_g1", "unrelated"],
            "theta_true": [0.31, 0.31, 0.31],
            "theta_pred": [0.30, 0.32, 0.99],
        }
    )
    records = namespace["saved_transform_records"](
        rows,
        ("identity", "dihedral_g1"),
    )
    assert set(records) == {"identity", "dihedral_g1"}
    assert float(records["dihedral_g1"]["theta_pred"]) == pytest.approx(0.32)

    mismatched = rows.copy()
    mismatched.loc[mismatched["transform"] == "dihedral_g1", "theta_true"] = 0.41
    with pytest.raises(ValueError, match="truth values"):
        namespace["saved_transform_records"](
            mismatched,
            ("identity", "dihedral_g1"),
        )


def test_c4_display_records_use_run_specific_transform_and_dataset_keys():
    namespace = gallery_helper_namespace()
    run_name = "example_run"
    dataset_size = 128
    suffix = f"{run_name}__N{dataset_size}"
    rows = pd.DataFrame(
        [
            {"source": "real_original", "run_name": "real_original", "dataset_size": np.nan, "transform": "identity", "z_index": 7, "theta_true": 0.31, "theta_pred": 0.30},
            {"source": "real_measured_transfer", "run_name": run_name, "dataset_size": dataset_size, "transform": f"transfer_Tk__{suffix}", "z_index": 7, "theta_true": 0.31, "theta_pred": 0.29},
            {"source": "real_measured_transfer", "run_name": run_name, "dataset_size": dataset_size, "transform": "decoy_transfer", "z_index": 7, "theta_true": 0.31, "theta_pred": 0.99},
            {"source": "real_gaussian", "run_name": run_name, "dataset_size": dataset_size, "transform": f"gaussian_smoothing__{suffix}", "z_index": 7, "theta_true": 0.31, "theta_pred": 0.28},
            {"source": "generated", "run_name": run_name, "dataset_size": dataset_size, "transform": f"generated__{suffix}", "z_index": 0, "theta_true": 0.31, "theta_pred": 0.27},
        ]
    )
    records = namespace["c4_saved_records_for_display"](
        rows,
        run_name=run_name,
        dataset_size=dataset_size,
        real_z_index=7,
        generated_sample_index=0,
    )
    assert float(records["power-matched real"]["theta_pred"]) == pytest.approx(0.29)
    assert records["power-matched real"]["transform"] == f"transfer_Tk__{suffix}"
    assert float(records["generated"]["theta_true"]) == pytest.approx(0.31)
