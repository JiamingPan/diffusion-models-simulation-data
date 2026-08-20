from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
for path in (ROOT, SCRIPTS):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))


def test_load_heldout_real_slices_reproduces_legacy_pair_order(monkeypatch):
    from simdiff_eval import probe_eval
    from train_nf_conditional_bias_encoder import select_slice_pairs

    heldout = np.array([900, 901, 902], dtype=np.int64)
    expected_pairs = select_slice_pairs(heldout, len(heldout) * 7)
    expected_raw = np.arange(len(expected_pairs) * 4, dtype=np.float32).reshape(
        len(expected_pairs), 1, 2, 2
    ) + 1.0
    params = np.arange(1000 * 6, dtype=np.float32).reshape(1000, 6)
    observed: dict[str, np.ndarray] = {}

    monkeypatch.setattr(probe_eval, "image_path", lambda root: Path(root) / "grid.npy")
    monkeypatch.setattr(probe_eval, "params_path", lambda root: Path(root) / "params.txt")
    monkeypatch.setattr(probe_eval, "load_params", lambda path, count: params)

    def fake_load_raw(path, pairs):
        observed["pairs"] = pairs.copy()
        return expected_raw

    monkeypatch.setattr(probe_eval, "load_raw_slices", fake_load_raw)
    monkeypatch.setattr(
        probe_eval,
        "preprocess_real_slices",
        lambda raw, norm: raw.astype(np.float32) * np.float32(norm["scale"]),
    )

    images, theta_raw, sim_index, z_index = probe_eval.load_heldout_real_slices(
        "/synthetic", heldout, 7, {"scale": 0.5}
    )

    np.testing.assert_array_equal(observed["pairs"], expected_pairs)
    np.testing.assert_array_equal(sim_index, expected_pairs[:, 0])
    np.testing.assert_array_equal(z_index, expected_pairs[:, 1])
    np.testing.assert_array_equal(theta_raw, params[expected_pairs[:, 0]])
    np.testing.assert_array_equal(images, expected_raw * np.float32(0.5))
    assert images.dtype == np.float32
