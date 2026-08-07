import json
from pathlib import Path


NOTEBOOKS = [
    Path("notebooks/nf_generalize_fig2_dit_results.ipynb"),
    Path("notebooks/nf_generalize_fig2_dit_results_explained.ipynb"),
]


def notebook_source(path: Path) -> str:
    notebook = json.loads(path.read_text())
    return "\n".join(
        "".join(cell.get("source", [])) for cell in notebook["cells"]
    )


def test_training_plot_uses_verified_fresh_300k_histories():
    for path in NOTEBOOKS:
        source = notebook_source(path)
        assert "FRESH_LOSS_SWEEP_NAME = 'nf_generalize_fig2_dit_l16_fresh300k_v2'" in source
        assert "fresh_loss_by_tag" in source
        assert "recorded_updates >= 0.98 * target_updates" in source
        assert "loss_source = fresh_loss_by_tag[dataset_tag]" in source


def test_training_plot_labels_actual_budgets():
    for path in NOTEBOOKS:
        source = notebook_source(path)
        assert "fresh L16 at 300k" in source
        assert "L16: {l16_budget_k}k" in source
        assert "Fresh 300k v2 loss-source audit" in source
