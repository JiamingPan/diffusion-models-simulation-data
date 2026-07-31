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
    assert source.count("micro_updates / grad_accum") >= 2


def test_l16_audit_flags_novel_but_physically_invalid_samples():
    source = notebook_source()

    assert "DiT-L16 validity audit" in source
    assert "novel_but_physically_invalid" in source
    assert "configuration_ok" in source
    assert "max_abs_pk_ratio_minus_1" in source


def test_continuation_is_not_presented_as_a_depth_scaling_curve():
    source = notebook_source()

    assert "Complete 200k-update depth comparison" in source
    assert "Low-data DiT-L16 checkpoint diagnostic (not a scaling curve)" in source
    assert "DiT depth comparison with continued L16 checkpoints" not in source
    assert "optimizer and learning-rate scheduler state" in source


def test_continuation_image_grid_uses_multiple_samples_per_checkpoint():
    source = notebook_source()

    assert "generated[:4, 0]" in source
    assert "four generated maps per checkpoint" in source


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
    assert "exact training subset used by that model" in source
    assert "SSCD Fréchet distance" in source
    assert "real-vs-real split baseline" in source
    assert "frechet_feature_distance" in source
    assert "real_split_frechet_baseline" in source
    assert "generated_to_heldout_over_real_split" in source
    assert "literal ImageNet Inception" in source
