from __future__ import annotations

import csv
import importlib
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

prep = importlib.import_module("prepare_nf_conditional_fixedC64_multiplicity_configs")


def _source(tmp_path: Path, n: int = 64) -> np.ndarray:
    rng = np.random.default_rng(0)
    sims = np.sort(rng.choice(np.arange(900), n, replace=False))
    zs = rng.integers(0, 128, n)
    path = (tmp_path / "local" / prep.SOURCE_SWEEP_NAME / "labels"
            / f"{prep.full.run_name(64)}_selected_slices.csv")
    path.parent.mkdir(parents=True)
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["row", "simulation_index", "z_index"])
        for i, (s, z) in enumerate(zip(sims, zs)):
            w.writerow([i, int(s), int(z)])
    return np.column_stack([sims, zs])


def test_run_names_and_sizes():
    assert [prep.dataset_size(m) for m in prep.MULTIPLICITIES] == [64, 128, 256, 512]
    assert prep.run_name(8) == "nf_cond_fixedC64_m08_n512_fresh200k"
    with pytest.raises(ValueError):
        prep.run_name(3)


def test_m1_equals_source_and_higher_m_is_nested_and_unique(tmp_path):
    source = _source(tmp_path)
    loaded = prep.load_source_pairs(tmp_path)
    np.testing.assert_array_equal(loaded, source)
    p1 = prep.multiplicity_pairs(source, 1)
    np.testing.assert_array_equal(p1, source)
    p8 = prep.multiplicity_pairs(source, 8)
    assert p8.shape == (512, 2)
    assert len(np.unique(p8, axis=0)) == 512
    assert set(np.unique(p8[:, 0]).tolist()) == set(source[:, 0].tolist())
    # each simulation contributes 8 evenly spaced slices starting at its source slice
    for k, (sim, z0) in enumerate(source):
        block = p8[8 * k:8 * (k + 1)]
        assert (block[:, 0] == sim).all()
        assert block[0, 1] == z0
        assert sorted(((block[:, 1] - z0) % 128).tolist()) == [0, 16, 32, 48, 64, 80, 96, 112]
    # nesting: every m=2 and m=4 row appears in m=8
    for m in (2, 4):
        rows = {tuple(r) for r in prep.multiplicity_pairs(source, m)}
        assert rows <= {tuple(r) for r in p8}


def test_rejects_bad_source(tmp_path):
    _source(tmp_path, n=63)
    with pytest.raises(ValueError):
        prep.load_source_pairs(tmp_path)


def test_checkpoint_epoch_is_final_epoch():
    for m in prep.MULTIPLICITIES:
        size = prep.dataset_size(m)
        assert prep.base.epochs_for(size, prep.base.TARGET_UPDATES) * prep.base.steps_per_epoch(size) >= prep.base.TARGET_UPDATES
