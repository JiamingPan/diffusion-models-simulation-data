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
