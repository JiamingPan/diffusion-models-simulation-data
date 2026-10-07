from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from dit_zero_screen_data import load_native_training_reference


def test_slice_first_reference_selects_and_thins_before_normalizing(tmp_path):
    raw = np.arange(1, 1 + 3 * 4 * 2 * 2, dtype=np.float32).reshape(3, 4, 2, 2)
    path = tmp_path / "source.npy"
    np.save(path, raw)
    config = {
        "global": {"dtype": "float32"},
        "data": {
            "img_path": [str(path)], "img_read_fn": "npy_read_fn",
            "n_samples": [2], "seed": None, "zthin": 2, "reshape": "2d",
            "normalization": "tanh", "norm_kwargs": {
                "center": None, "xmax": None, "alpha": .8, "beta": 10.,
                "gamma": 1., "delta": 1., "sigma": 1.,
            },
            "transform": ["log"],
        },
    }
    reference, metadata = load_native_training_reference(config)
    retained = raw[:2, ::2].reshape(-1, 1, 2, 2)
    assert reference.shape == (4, 1, 2, 2)
    assert metadata["shape"] == [4, 1, 2, 2]
    assert metadata["selections"][0]["volume_indices"] == [0, 1]
    assert metadata["selections"][0]["z_indices"] == [0, 2]
    assert np.isclose(metadata["center"], np.log(retained).mean(dtype=np.float64))
    assert np.isfinite(reference).all()
