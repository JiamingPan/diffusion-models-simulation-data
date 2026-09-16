import importlib.util
import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

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


def test_missing_l12_is_optional_but_partial_or_required_results_fail(tmp_path):
    spec = importlib.util.spec_from_file_location("builder", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    loader = ''.join(module.build_notebook()['cells'][3]['source'])
    names = ['dit_l8_200k', 'dit_l12_200k', 'dit_l16_fresh300k', 'dit_l16_seed456_500k']
    layers = dict(zip(names, [8, 12, 16, 16]))
    for name in names:
        if name == 'dit_l12_200k':
            continue
        (tmp_path / f'{name}.json').write_text(json.dumps(dict(
            status='complete', num_layers=layers[name], patch_size=8,
            prediction_type='v_prediction', weights='raw', code_revision='test',
            checkpoint='fixture', num_reference=6, complete_training_reference_size=256,
            terminal_snr=0, terminal_v_weight=0)))
        np.savez(tmp_path / f'{name}.npz', fixture=np.zeros(1))
    def run():
        scope = dict(EXPECTED=names.copy(), OPTIONAL={'dit_l12_200k'},
                     EXPECTED_LAYERS=layers, LABELS=dict(zip(names, names)),
                     RESULT_DIR=tmp_path, json=json, np=np, pd=pd)
        exec(loader, scope)
        return scope
    assert run()['EXPECTED'] == [names[0], names[2], names[3]]
    (tmp_path / 'dit_l12_200k.json').write_text('{}')
    with pytest.raises(FileNotFoundError):
        run()
    (tmp_path / 'dit_l12_200k.json').unlink()
    (tmp_path / f'{names[0]}.npz').unlink()
    with pytest.raises(FileNotFoundError):
        run()
