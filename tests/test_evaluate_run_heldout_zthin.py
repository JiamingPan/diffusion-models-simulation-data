"""evaluate_run.UncondData --heldout-zthin option (proposed for ledger unet128_aug_wave3); default unchanged."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

N_SIMS, NZ, SIZE = 6, 16, 8


def write_config(tmp_path, zthin=8, n_samples=(2, 2)):
    rng = np.random.default_rng(0)
    paths = []
    for k in range(2):
        p = tmp_path / f"src{k}.npy"
        np.save(p, rng.uniform(1.0, 10.0, size=(N_SIMS, NZ, SIZE, SIZE)).astype(np.float32))
        paths.append(str(p))
    cfg = {"data": {"img_path": paths, "img_read_fn": "npy_read_fn", "label_path": None, "reshape": "2d",
                    "zthin": zthin, "n_samples": list(n_samples), "seed": None, "normalization": "tanh",
                    "norm_kwargs": {"center": None, "xmax": None, "alpha": 0.8, "beta": 10.0, "delta": 1.0,
                                    "gamma": 1.0, "sigma": 1.5},
                    "transform": ["log"]}}
    path = tmp_path / "cfg.yaml"
    path.write_text(yaml.safe_dump(cfg))
    return path, paths


def test_default_is_config_zthin(tmp_path):
    from evaluate_run import UncondData

    cfg, _ = write_config(tmp_path, zthin=8)
    a = UncondData(cfg, 3, 512)
    b = UncondData(cfg, 3, 512, heldout_zthin=8)
    assert a.heldout_zthin == 8
    assert a.train.shape == (2 * 2 * 2, SIZE, SIZE) and a.val.shape == (3 * 2 * 2, SIZE, SIZE)
    np.testing.assert_array_equal(a.val, b.val)
    np.testing.assert_array_equal(a.train, b.train)


def test_heldout_zthin_changes_only_heldout(tmp_path):
    from evaluate_run import UncondData

    cfg, paths = write_config(tmp_path, zthin=8)
    a = UncondData(cfg, 3, 512)
    c = UncondData(cfg, 3, 512, heldout_zthin=1)
    # training maps and normalization untouched
    np.testing.assert_array_equal(a.train, c.train)
    np.testing.assert_array_equal(a.train_raw, c.train_raw)
    assert (a.center, a.xmax) == (c.center, c.xmax)
    # held-out: same unused sims (2, 3, 4 of each source), every slice
    expected = np.concatenate([np.load(p)[2:5].reshape(-1, SIZE, SIZE) for p in paths])
    assert c.val.shape == (3 * NZ * 2, SIZE, SIZE)
    np.testing.assert_array_equal(c.val_raw, expected)
    # the default held-out set is the every-8th-slice subset of it
    idx = np.concatenate([b * NZ + np.arange(0, NZ, 8) for b in range(3 * 2)])
    np.testing.assert_array_equal(a.val_raw, c.val_raw[idx])
    np.testing.assert_allclose(a.val, c.val[idx])


def test_heldout_zthin_respects_max_heldout(tmp_path):
    from evaluate_run import UncondData

    cfg, _ = write_config(tmp_path, zthin=8)
    c = UncondData(cfg, 3, 10, heldout_zthin=1)
    assert c.val.shape == (10, SIZE, SIZE)


def test_invalid_heldout_zthin_rejected(tmp_path):
    from evaluate_run import UncondData

    cfg, _ = write_config(tmp_path)
    with pytest.raises(ValueError):
        UncondData(cfg, 3, 512, heldout_zthin=0)


def test_cli_accepts_heldout_zthin(tmp_path, monkeypatch):
    import evaluate_run

    monkeypatch.setattr(sys, "argv", ["evaluate_run.py", "--mode", "unconditional", "--out-dir", str(tmp_path),
                                      "--heldout-zthin", "8"])
    with pytest.raises(SystemExit, match="Refusing"):
        evaluate_run.main()
