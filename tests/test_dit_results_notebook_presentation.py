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
