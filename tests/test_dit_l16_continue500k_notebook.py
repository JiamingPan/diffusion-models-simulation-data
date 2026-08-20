import copy
import importlib.util
import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "scripts" / "update_dit_l16_continue500k_notebook.py"
TAG = "dit-l16-continue500k-v2"


def load_updater():
    spec = importlib.util.spec_from_file_location("dit_l16_continue500k_notebook", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def tiny_notebook() -> dict:
    return {
        "cells": [
            {
                "cell_type": "markdown",
                "id": "keep-intro",
                "metadata": {"custom": "preserve"},
                "source": ["# Existing analysis\n"],
            },
            {
                "cell_type": "code",
                "execution_count": 7,
                "id": "keep-code",
                "metadata": {"collapsed": True},
                "outputs": [{"output_type": "stream", "name": "stdout", "text": ["old\n"]}],
                "source": ["existing_value = 7\n"],
            },
            {
                "cell_type": "markdown",
                "id": "rerun-anchor",
                "metadata": {},
                "source": ["## Great Lakes Rerun Command\n"],
            },
        ],
        "metadata": {"kernelspec": {"name": "python3"}, "keep": "yes"},
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def tagged_cells(notebook: dict) -> list[dict]:
    return [
        cell
        for cell in notebook["cells"]
        if TAG in cell.get("metadata", {}).get("tags", [])
    ]


def test_updater_preserves_untagged_cells_and_inserts_before_rerun(tmp_path):
    updater = load_updater()
    notebook = tiny_notebook()
    expected_untagged = copy.deepcopy(notebook["cells"])
    source = tmp_path / "input.ipynb"
    output = tmp_path / "output.ipynb"
    source.write_text(json.dumps(notebook))

    updater.update_notebook(source, output)
    updated = json.loads(output.read_text())

    assert [cell for cell in updated["cells"] if cell not in tagged_cells(updated)] == expected_untagged
    assert updated["metadata"]["keep"] == "yes"
    assert updated["cells"].index(tagged_cells(updated)[0]) < next(
        index
        for index, cell in enumerate(updated["cells"])
        if "## Great Lakes Rerun Command" in "".join(cell.get("source", []))
    )


def test_updater_is_idempotent_and_code_cells_compile(tmp_path):
    updater = load_updater()
    source = tmp_path / "input.ipynb"
    once = tmp_path / "once.ipynb"
    twice = tmp_path / "twice.ipynb"
    source.write_text(json.dumps(tiny_notebook()))

    updater.update_notebook(source, once)
    updater.update_notebook(once, twice)

    first = json.loads(once.read_text())
    second = json.loads(twice.read_text())
    assert first == second
    assert len(tagged_cells(second)) == len(updater.build_cells())
    for cell in tagged_cells(second):
        if cell["cell_type"] == "code":
            compile("".join(cell["source"]), f"cell-{cell['id']}", "exec")


def test_inserted_block_covers_every_required_diagnostic():
    updater = load_updater()
    source = "\n".join("".join(cell["source"]) for cell in updater.build_cells())

    required_text = (
        "final_audit.json",
        "Training loss from 300k to 500k",
        "PCA and SSCD novelty",
        "One-point and power-spectrum trajectories",
        "k-bin 20, 40, and 60",
        "DPM-Solver 50 versus DDPM 500",
        "Patch-boundary diagnostic",
        "nearest training",
        "Do not infer a scaling law",
    )
    for text in required_text:
        assert text in source


def test_physics_figures_expose_large_ratios_and_bootstrap_overlap():
    updater = load_updater()
    source = "\n".join("".join(cell["source"]) for cell in updater.build_cells())

    assert "set_ylim(0, 3.7)" not in source
    assert "set_yscale('log')" in source
    assert "hist_l1_lo" in source
    assert "hist_l1_hi" in source
    assert "pk_log10_mae_lo" in source
    assert "pk_log10_mae_hi" in source
    assert "hist_l1_ci_low" not in source
    assert "pk_log10_mae_ci_low" not in source
    assert "intervals_overlap" in source
    assert "300k/500k 95% CIs" in source
    assert "current['ratio_variance']" not in source
    assert "current['generated_variance']" in source
    assert "current['real_reference_mean'].pow(2)" in source


def test_real_notebook_has_exactly_one_tagged_block_after_update(tmp_path):
    updater = load_updater()
    original_path = REPO_ROOT / "notebooks" / "nf_generalize_fig2_dit_results.ipynb"
    output = tmp_path / "updated.ipynb"

    updater.update_notebook(original_path, output)
    updated = json.loads(output.read_text())
    ids = [cell["id"] for cell in tagged_cells(updated)]

    assert len(ids) == len(set(ids)) == len(updater.build_cells())
    assert all(TAG in cell.get("metadata", {}).get("tags", []) for cell in tagged_cells(updated))
