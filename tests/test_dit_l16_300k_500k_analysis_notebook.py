import ast
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
BUILDER_PATH = REPO_ROOT / "scripts" / "build_dit_l16_300k_500k_analysis_notebook.py"
NOTEBOOK_PATH = REPO_ROOT / "notebooks" / "nf_generalize_fig2_dit_l16_300k_500k_analysis.ipynb"
EXPECTED_UPDATES = (300, 340, 380, 420, 460, 500)
EXPECTED_TAGS = tuple(f"d2p{power:02d}" for power in range(6, 16))


def load_builder():
    spec = importlib.util.spec_from_file_location("dit_l16_trajectory_builder", BUILDER_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def notebook() -> dict:
    return json.loads(NOTEBOOK_PATH.read_text())


def notebook_source() -> str:
    return "\n".join("".join(cell.get("source", [])) for cell in notebook()["cells"])


def notebook_function(name: str):
    for cell in notebook()["cells"]:
        if cell["cell_type"] != "code":
            continue
        tree = ast.parse("".join(cell["source"]))
        for node in tree.body:
            if isinstance(node, ast.FunctionDef) and node.name == name:
                namespace = {
                    "Path": Path,
                    "pd": pd,
                    "project_path": lambda value: Path(str(value)),
                }
                exec(compile(ast.Module(body=[node], type_ignores=[]), "<notebook>", "exec"), namespace)
                return namespace[name]
    raise AssertionError(f"Notebook function not found: {name}")


def test_builder_and_standalone_notebook_exist():
    assert BUILDER_PATH.is_file()
    assert NOTEBOOK_PATH.is_file()


def test_notebook_is_deterministic_and_unexecuted():
    builder = load_builder()
    first = builder.build_notebook()
    second = builder.build_notebook()
    assert first == second == notebook()
    assert first["nbformat"] == 4
    assert first["nbformat_minor"] == 5
    for cell in first["cells"]:
        if cell["cell_type"] == "code":
            assert cell["execution_count"] is None
            assert cell["outputs"] == []
            compile("".join(cell["source"]), f"cell-{cell['id']}", "exec")


def test_notebook_has_the_approved_analysis_sections_in_order():
    headings = [
        line.strip()
        for cell in notebook()["cells"]
        if cell["cell_type"] == "markdown"
        for line in "".join(cell["source"]).splitlines()
        if line.startswith("## ")
    ]
    required = [
        "## 1. Scope, provenance, and mandatory audit",
        "## 2. Optimization history",
        "## 3. Generated maps across the full sweep",
        "## 4. PCA and SSCD generalization trajectories",
        "## 5. Generalization phase diagrams",
        "## 6. Transition-location summary",
        "## 7. Architecture context",
        "## 8. One-point distributions",
        "## 9. Power-spectrum fidelity",
        "## 10. Power-spectrum uncertainty",
        "## 11. Conservative k=60 outlier sensitivity analysis",
        "## 12. Sampler control",
        "## 13. Patch-boundary diagnostics",
        "## 14. Nearest-training audit",
        "## 15. Joint novelty and physical validity",
        "## 16. Evidence summary",
    ]
    assert headings == required


def test_notebook_enforces_complete_audited_checkpoint_specific_inputs():
    source = notebook_source()
    required = (
        "final_audit.json",
        "status') != 'PASS'",
        "expected_metric_tables",
        "valid_metric_tables",
        "analysis_manifest.json",
        "expected_pairs",
        "expected_sample_label",
        "sample_path",
        "terminal_sigma_is_zero",
        "terminal_sigma_verifiable",
        "physics_summary_rows",
        "selected_bin_rows",
        "patch_boundary_rows",
        "physics_curve_arrays",
    )
    for text in required:
        assert text in source
    for updates in EXPECTED_UPDATES:
        assert str(updates) in source
    for tag in EXPECTED_TAGS:
        assert tag in source


def test_notebook_contains_every_requested_diagnostic():
    source = notebook_source()
    required = (
        "Cycle-averaged denoising loss",
        "generated_maps_300k_380k",
        "generated_maps_420k_500k",
        "PCA q95 novelty",
        "SSCD q95 novelty",
        "generalization_heatmaps",
        "summarize_n50",
        "Historical context only",
        "exact configured training subset",
        "one_point_error_heatmap",
        "power_spectrum_error_heatmap",
        "k-bin 20, 40, and 60",
        "4.5 robust standard deviations",
        "robust_log_ratio_outliers",
        "summarize_filtered_power_ratios",
        "k60_outlier_sample_audit.csv",
        "k60_outlier_group_audit.csv",
        "k60_outlier_distributions.png",
        "k60_flagged_sample_gallery",
        "power_spectrum_selected_bins_outlier_sensitivity.png",
        "power_spectrum_ratios_500k_outlier_sensitivity.png",
        "outlier_excluded_physics_summary.csv",
        "outlier_excluded_novelty_bounds.csv",
        "outlier_excluded_one_point_500k.png",
        "outlier_excluded_power_spectrum_500k.png",
        "feasible interval after exclusion",
        "generated-sample median (diagnostic)",
        "n_kept",
        "DPM-Solver 50",
        "DDPM 500",
        "validate_sampler_endpoint",
        "Patch-boundary",
        "nearest training",
        "Novel but physically inaccurate",
        "evidence_summary.csv",
    )
    for text in required:
        assert text in source


def test_builder_includes_outlier_excluded_analysis():
    builder = load_builder()
    source = "\n".join(
        "".join(cell.get("source", [])) for cell in builder.build_notebook()["cells"]
    )
    assert "how='validate'" not in source
    assert "how='inner'" in source
    assert "outlier_excluded_physics_summary.csv" in source
    assert "outlier_excluded_novelty_bounds.csv" in source


def test_sampler_control_uses_scheduler_specific_endpoint_validation():
    source = notebook_source()
    assert "scheduler_class" in source
    assert "final_timestep" in source
    assert "endpoint_evidence" in source
    assert "'DDPMScheduler'" in source
    assert "Sampler did not terminate at sigma=0" not in source


def test_notebook_stitches_exact_stage_local_loss_histories():
    source = notebook_source()
    assert "checkpoint_metric_candidates" in source
    assert "prepare_stitched_loss_history" in source
    assert "CONT_UPDATES_K[1:]" in source
    assert "previous_expected_checkpoint" in source
    assert "read_latest_metrics" not in source
    assert "target_updates=500_000" not in source


def test_notebook_does_not_reintroduce_legacy_l16_200k_results():
    source = notebook_source()
    forbidden = (
        "DiT-L16 200k",
        "DiT-L16: 200k",
        "dit_l16_d2p06_noaug_200k",
        "nf_generalize_fig2_dit_l16_cont_",
        "dpm50_cont_225k",
        "dpm50_cont_250k",
        "dpm50_cont_275k",
    )
    for text in forbidden:
        assert text not in source


def test_notebook_uses_exact_subset_references_and_full_dataset_range():
    source = notebook_source()
    assert "exact configured training subset" in source
    assert "load_real_reference_from_config" in source
    assert "range(6, 16)" in source
    assert "CONT_TAGS[:5]" not in source


def test_nearest_training_cell_uses_streaming_helper_result_schema():
    source = notebook_source()
    assert "matches['nearest_index'][query_index]" in source
    assert "matches['mse'][query_index]" in source
    assert "matches['cosine_similarity'][query_index]" in source
    assert "matches['nearest_training_index'][query_index]" not in source
    assert "matches['nearest_mse'][query_index]" not in source
    assert "matches['nearest_cosine'][query_index]" not in source


def test_source_config_resolver_falls_back_when_dataframe_value_is_nan():
    resolver = notebook_function("source_config_for_row")
    row = pd.Series({"source_config": np.nan, "config": "configs/source.yaml"})
    assert resolver(row) == Path("configs/source.yaml")
