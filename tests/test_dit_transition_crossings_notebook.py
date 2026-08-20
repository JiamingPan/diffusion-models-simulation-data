from __future__ import annotations

import json
from pathlib import Path


NOTEBOOK = Path("notebooks/nf_generalize_fig2_dit_results.ipynb")


def notebook_source() -> str:
    payload = json.loads(NOTEBOOK.read_text())
    return "\n".join("".join(cell.get("source", [])) for cell in payload["cells"])


def test_notebook_uses_audited_crossing_helper_and_filters_capacity_fit():
    source = notebook_source()

    assert "interpolate_threshold_crossings" in source
    assert "'crossings'" in source
    assert "n_crossings" in source
    assert "capacity_n50['status'].eq('interpolated')" in source
    assert "capacity_n50_interpolated" in source
