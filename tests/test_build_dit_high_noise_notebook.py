import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/build_dit_high_noise_notebook.py"


def test_notebook_has_all_three_diagnostics(tmp_path):
    spec = importlib.util.spec_from_file_location("builder", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    notebook = module.build_notebook()
    json.dumps(notebook)
    assert notebook["nbformat"] == 4
    source = "\n".join("".join(cell.get("source", [])) for cell in notebook["cells"])
    assert "Direct reconstruction error by timestep" in source
    assert "Skip the earliest high-noise steps" in source
    assert "Pure noise at the terminal timestep" in source
    assert "L8, 200k" in source
    assert "L16, seed456 500k" in source
    assert len({cell["id"] for cell in notebook["cells"]}) == len(notebook["cells"])
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            compile("".join(cell["source"]), cell["id"], "exec")
