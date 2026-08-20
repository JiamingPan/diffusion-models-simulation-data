import importlib.util
import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "scripts" / "update_dit_l16_300k_500k_outlier_notebook.py"
TAG = "dit-l16-outlier-excluded-v1"


def load_updater():
    spec = importlib.util.spec_from_file_location("dit_l16_outlier_notebook", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def tiny_notebook() -> dict:
    return {
        "cells": [
            {
                "cell_type": "code",
                "execution_count": 1,
                "id": "outlier-cell",
                "metadata": {},
                "outputs": [],
                "source": ["outlier_analysis = {}\n", "outlier_groups = pd.DataFrame()\n"],
            },
            {
                "cell_type": "code",
                "execution_count": 2,
                "id": "merge-cell",
                "metadata": {},
                "outputs": [{"output_type": "error", "ename": "KeyError", "evalue": "validate", "traceback": []}],
                "source": [
                    "joint = continuation_novelty.merge(continuation_physics, ",
                    "how='validate', validate='many_to_one')\n",
                ],
            },
        ],
        "metadata": {},
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def tagged_cells(notebook: dict) -> list[dict]:
    return [
        cell for cell in notebook["cells"]
        if TAG in cell.get("metadata", {}).get("tags", [])
    ]


def test_updater_fixes_merge_and_inserts_filtered_analysis(tmp_path):
    updater = load_updater()
    source = tmp_path / "source.ipynb"
    output = tmp_path / "output.ipynb"
    source.write_text(json.dumps(tiny_notebook()))

    updater.update_notebook(source, output)
    updated = json.loads(output.read_text())
    notebook_source = "\n".join("".join(cell.get("source", [])) for cell in updated["cells"])

    assert "how='validate'" not in notebook_source
    assert "how='inner'" in notebook_source
    assert "validate='many_to_one'" in notebook_source
    assert len(tagged_cells(updated)) == len(updater.build_cells())
    assert "outlier_excluded_physics_summary.csv" in notebook_source
    assert "outlier_excluded_novelty_bounds.csv" in notebook_source
    assert "exact configured training-subset mean" in notebook_source
    assert "feasible interval" in notebook_source

    merge_cell = next(cell for cell in updated["cells"] if cell["id"] == "merge-cell")
    assert merge_cell["outputs"] == []
    assert merge_cell["execution_count"] is None


def test_updater_is_idempotent_and_generated_code_compiles(tmp_path):
    updater = load_updater()
    source = tmp_path / "source.ipynb"
    once = tmp_path / "once.ipynb"
    twice = tmp_path / "twice.ipynb"
    source.write_text(json.dumps(tiny_notebook()))

    updater.update_notebook(source, once)
    updater.update_notebook(once, twice)

    first = json.loads(once.read_text())
    second = json.loads(twice.read_text())
    assert first == second
    for cell in tagged_cells(second):
        if cell["cell_type"] == "code":
            compile("".join(cell["source"]), cell["id"], "exec")
