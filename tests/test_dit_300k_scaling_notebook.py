from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks" / "nf_generalize_fig2_dit_300k_scaling.ipynb"


def load_notebook() -> dict:
    return json.loads(NOTEBOOK.read_text())


def notebook_source() -> str:
    notebook = load_notebook()
    return "\n".join("".join(cell.get("source", [])) for cell in notebook["cells"])


def code_source() -> str:
    notebook = load_notebook()
    return "\n".join(
        "".join(cell.get("source", []))
        for cell in notebook["cells"]
        if cell.get("cell_type") == "code"
    )


def test_notebook_has_reader_focused_section_structure():
    source = notebook_source()
    required = (
        "# DiT Memorization-to-Generalization Scaling at 300k",
        "## TL;DR and interpretation rules",
        "## Input audit",
        "## Generalization transition",
        "## Transition summary",
        "## Fresh L16 optimization across all ten training sizes",
        "## Generated-field stability across all ten training sizes",
        "## Generated samples versus nearest training slices",
        "## One-point distributions across all ten training sizes",
        "## Power spectra across all ten training sizes",
        "## Per-sample outlier distributions",
        "## Sampler audit on the same fresh 300k checkpoints",
        "## Takeaways and limitations",
    )
    for heading in required:
        assert heading in source


def test_notebook_labels_mixed_budgets_and_historical_references():
    source = notebook_source()
    assert "DiT-L8 200k" in source
    assert "DiT-L12 / base 200k" in source
    assert "DiT-L16 fresh 300k" in source
    assert "historical UNet reference" in source
    assert "unequal optimizer-update budgets" in source
    assert "does not establish a universal capacity scaling law" in source


def test_notebook_excludes_legacy_l16_sources_and_fallbacks():
    source = code_source()
    forbidden = (
        "dpm50_cont_",
        "fresh400k",
        "DiT-L16 200k",
        "fallback to legacy",
        "legacy L16 fallback",
    )
    for text in forbidden:
        assert text not in source


def test_notebook_is_committed_unexecuted_with_unique_stable_ids():
    notebook = load_notebook()
    ids = [cell["id"] for cell in notebook["cells"]]
    assert len(ids) == len(set(ids))
    for cell in notebook["cells"]:
        if cell.get("cell_type") == "code":
            assert cell.get("execution_count") is None
            assert cell.get("outputs") == []


def test_notebook_resolves_project_root_and_imports_analysis_helpers():
    source = code_source()
    assert "def resolve_project_root" in source
    assert "scripts.dit_300k_scaling_analysis" in source
    assert "FRESH_SWEEP_NAME" in source
    assert "QUICKCHECK_DIR" in source
    assert "CACHE_DIR" in source


def test_notebook_builds_fresh_mixed_budget_transition_and_n50_outputs():
    source = code_source()
    required = (
        "build_mixed_dit_metric_table",
        "normalize_generalization_table",
        "summarize_n50",
        "nf_generalize_fig2_dit_l16_fresh300k_v2_pca_full_nn_metrics.csv",
        "nf_generalize_fig2_dit_l16_fresh300k_v2_sscd_full_nn_metrics.csv",
        "dit_300k_mixed_budget_transition_full.png",
        "dit_300k_mixed_budget_transition_zoom.png",
        "dit_300k_transition_n50.csv",
        "dit_300k_capacity_n50_diagnostic.png",
        "build_historical_unet_metric_table",
        "fresh independent 300k v2",
    )
    for text in required:
        assert text in source


def test_notebook_audits_and_plots_all_ten_fresh_l16_loss_histories():
    source = code_source()
    required = (
        "prepare_loss_history",
        "target_total_updates",
        "300_000",
        "fresh_loss_audit.csv",
        "dit_l16_fresh300k_loss_all_sizes.png",
        "dit_l16_fresh300k_tail_loss_summary.png",
        "optimizer_updates_recorded",
        "tail_loss_median",
        "tail_loss_q25",
        "tail_loss_q75",
    )
    for text in required:
        assert text in source

    for dataset_tag in ("d2p06", "d2p07", "d2p08", "d2p09", "d2p10",
                        "d2p11", "d2p12", "d2p13", "d2p14", "d2p15"):
        assert dataset_tag in source

    assert "fresh_loss_by_tag" in source
    assert "historical L16" not in source


def test_notebook_audits_fresh_samples_and_exact_training_subsets():
    source = code_source()
    required = (
        "configured_training_reference_info",
        "iter_real_reference_batches_from_config",
        "validate_sample_archive_metadata",
        "FRESH_SAMPLE_LABEL",
        "dpm50_fresh300k_v2",
        "fresh_sample_audit.csv",
        "configured_slices",
        "exact model training subset",
    )
    for text in required:
        assert text in source


def test_notebook_plots_full_sweep_generated_fields_and_nearest_training_audit():
    source = code_source()
    required = (
        "evenly_spaced_indices",
        "streaming_nearest_neighbors",
        "DISPLAY_SAMPLE_COUNT = 4",
        "NEAREST_QUERY_COUNT = 16",
        "dit_l16_fresh300k_generated_all_sizes.png",
        "dit_l16_fresh300k_nearest_examples.png",
        "dit_l16_fresh300k_nearest_distribution.png",
        "dit_l16_fresh300k_nearest_queries.csv",
        "nearest_cosine_similarity",
        "nearest_mse",
    )
    for text in required:
        assert text in source

    for power in range(6, 16):
        assert f"2^{{{power}}}" in source or f"d2p{power:02d}" in source


def test_notebook_streams_exact_reference_physics_and_plots_full_sweep():
    source = code_source()
    required = (
        "aggregate_physical_batches",
        "per_sample_physical_errors",
        "PHYSICAL_HIST_EDGES",
        "PK_NBINS = 30",
        "configured_slices",
        "pixel_coverage",
        "dit_l16_fresh300k_onepoint_all_sizes.png",
        "dit_l16_fresh300k_pk_ratio_all_sizes.png",
        "dit_l16_fresh300k_pk_log2_heatmap.png",
        "dit_l16_fresh300k_physics_per_sample.csv",
        "PHYSICS_METRIC_VERSION",
        "sample_sha256",
        "config_sha256",
        "np.savez_compressed",
        "physics_cache_hit",
    )
    for text in required:
        assert text in source


def test_notebook_retains_physical_error_tails_and_links_novelty_to_physics():
    source = code_source()
    required = (
        "hist_l1",
        "pk_log10_mae",
        "dit_l16_fresh300k_physics_error_tails.png",
        "dit_l16_fresh300k_novelty_vs_physics.png",
        "nearest_cosine_similarity",
        "query_index",
        "SSCD",
        "pixel nearest novelty",
    )
    for text in required:
        assert text in source


def test_notebook_physics_sections_cover_all_ten_sizes_without_old_l16():
    source = code_source()
    for dataset_tag in ("d2p06", "d2p07", "d2p08", "d2p09", "d2p10",
                        "d2p11", "d2p12", "d2p13", "d2p14", "d2p15"):
        assert dataset_tag in source
    assert "dpm50_cont_" not in source


def test_notebook_sampler_audit_is_same_checkpoint_and_four_method_only():
    source = code_source()
    required = (
        "DPM50",
        "DPM100",
        "DPM200",
        "DDPM500",
        "dpm50_fresh300k_v2",
        "dpm100_fresh300k_v2",
        "dpm200_fresh300k_v2",
        "ddpm500_fresh300k_v2",
        "validate_sample_archive_metadata",
        "dit_l16_fresh300k_sampler_archive_audit.csv",
        "dit_l16_fresh300k_sampler_physics_summary.csv",
        "dit_l16_fresh300k_sampler_physics_summary.png",
        "dit_l16_fresh300k_sampler_images_d2p08.png",
        "submit_nf_generalize_fig2_dit_l16_fresh300k_v2_sampler_audit.sh",
    )
    for text in required:
        assert text in source


def test_notebook_sampler_audit_does_not_silently_compare_incomplete_archives():
    source = code_source()
    assert "controlled_comparison_complete" in source
    assert "same_checkpoint" in source
    assert "same_config" in source
    assert "same_seed" in source
    assert "same_sample_count" in source
    assert "No controlled sampler comparison is drawn" in source
