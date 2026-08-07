import json
from pathlib import Path


NOTEBOOK = Path(__file__).resolve().parents[1] / "notebooks" / "nf_generalize_fig2_dit_results.ipynb"


def notebook_source() -> str:
    notebook = json.loads(NOTEBOOK.read_text())
    return "\n".join("".join(cell.get("source", [])) for cell in notebook["cells"])


def test_depth_comparison_is_combined_and_explains_novelty_metric():
    source = notebook_source()

    assert "nf_generalize_fig2_dit_depth_vs_unet_pca_sscd_q95.png" in source
    assert "High score means unlike the training set; it does not guarantee physical fidelity." in source
    assert "PCA embedding" in source
    assert "SSCD embedding" in source


def test_batch_loss_axis_accounts_for_gradient_accumulation():
    source = notebook_source()

    assert "gradient_accumulation_steps" in source
    assert "x = micro_updates / grad_accum" in source
    assert "0.5 * (window - 1) / grad_accum" in source


def test_l16_audit_flags_novel_but_physically_invalid_samples():
    source = notebook_source()

    assert "DiT-L16 validity audit" in source
    assert "novel_but_physically_invalid" in source
    assert "configuration_ok" in source
    assert "max_abs_pk_ratio_minus_1" in source


def test_dit_notebook_plots_generated_samples_against_exact_training_subset():
    source = notebook_source()

    assert "Generated Samples Versus Nearest Training Slices" in source
    assert "nearest_training_matches" in source
    assert "nearest-training search requires the complete configured subset" in source
    assert "'dit_l8,dit_base,dit_l16'" in source
    assert "nf_generalize_fig2_{arch}_generated_vs_nearest_training.png" in source
    assert "nf_generalize_fig2_{arch}_{audit_tag}_nearest_training_audit.png" in source


def test_dit_notebook_tracks_nick_review_requests_and_distribution_distance():
    source = notebook_source()

    assert "Nick review checklist" in source
    assert "answers the requested review questions directly" in source
    assert "not capped for plotting" in source
    assert "reports any plotting cap" not in source
    assert "exact training subset used by that model" in source
    assert "SSCD Fréchet distance" in source
    assert "real-vs-real split baseline" in source
    assert "frechet_feature_distance" in source
    assert "real_split_frechet_baseline" in source
    assert "generated_to_heldout_over_real_split" in source
    assert "literal ImageNet Inception" in source


def test_dit_notebook_covers_all_training_sizes_in_readable_blocks():
    source = notebook_source()

    assert "ALL_DATA_TAGS = [f'd2p{i:02d}' for i in range(6, 16)]" in source
    assert "LOW_DATA_TAGS = ALL_DATA_TAGS[:5]" in source
    assert "HIGH_DATA_TAGS = ALL_DATA_TAGS[5:]" in source
    assert "requested tags are missing" in source
    assert "max_count=5" not in source
    assert "high_data_pk_ratio_zoom" not in source


def test_dit_notebook_summarizes_physical_error_for_all_depths():
    source = notebook_source()

    assert "onepoint_hist_l1" in source
    assert "pk_log_ratio_mae" in source
    assert "pk_low_log_ratio_mae" in source
    assert "pk_mid_log_ratio_mae" in source
    assert "pk_high_log_ratio_mae" in source
    assert "novelty versus physical-statistics error" in source


def test_conditional_appendix_audits_full_parameter_vector_without_claiming_coverage():
    source = notebook_source()

    assert "Conditional Calibration Input Audit" in source
    assert "expected_parameter_count = 6" in source
    assert "theta_norm_repeated" in source
    assert "theta_raw" in source
    assert "heldout_indices_match_manifest" in source
    assert "training_and_heldout_simulations_disjoint" in source
    assert "seed-interval inclusion; not posterior coverage" in source


def test_fresh_300k_l16_diagnostics_use_matching_samples_without_legacy_fallback():
    source = notebook_source()

    assert "Fresh 300k DiT-L16 Samples and Physical Statistics" in source
    assert "FRESH_MANIFEST_PATH" in source
    assert "FRESH_EXPECTED_SAMPLE_LABEL = 'dpm50_fresh300k_v2'" in source
    assert "fresh_300k_sample_audit_pass" in source
    assert "requested_checkpoint_matches_manifest" in source
    assert "resolved_checkpoint_matches_manifest" in source
    assert "stored_config_matches_manifest" in source
    assert "Legacy 200k samples are intentionally not used as a fallback" in source
    assert "DPMSolverMultistepScheduler" in source
    assert "(fresh_300k_sample_audit_df['num_steps'] == 50).all()" in source
    assert "(fresh_300k_sample_audit_df['n_generated'] == 512).all()" in source


def test_fresh_300k_l16_figures_span_all_ten_data_sizes_as_standalone_outputs():
    source = notebook_source()

    assert "nf_generalize_fig2_dit_l16_fresh300k_v2_generated_full_sweep.png" in source
    assert "nf_generalize_fig2_dit_l16_fresh300k_v2_onepoint_full_sweep.png" in source
    assert "nf_generalize_fig2_dit_l16_fresh300k_v2_pk_ratio_full_sweep.png" in source
    assert "nf_generalize_fig2_dit_l16_fresh300k_v2_pk_log2_error.png" in source
    assert "nf_generalize_fig2_dit_l16_fresh300k_v2_nearest_full_sweep.png" in source
    assert "DIT_FRESH_IMAGES_PER_SIZE" in source
    assert "samples_per_size: int = 4" in source
    assert "streaming_nearest_training_match" in source
    assert "FRESH_EXPECTED_TAGS[:5]" in source
    assert "FRESH_EXPECTED_TAGS[5:]" in source


def test_fresh_300k_sampler_audit_requires_same_checkpoint_comparison():
    source = notebook_source()

    assert "Sampler adequacy audit" in source
    assert "No controlled sampler comparison is available yet" in source
    assert "same_expected_checkpoint" in source
    assert "DPM100 or DPM200 and DDPM500" in source
