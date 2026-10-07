import numpy as np
import pytest

torch = pytest.importorskip("torch")
from simdiff_eval.orbit_nn import orbit_max_cosine


def brute_force(q, r):
    def unit(x):
        x = x - x.mean()
        return x / np.linalg.norm(x)
    best = -np.inf
    for g in range(8):
        y = np.rot90(q, g % 4)
        y = np.flip(y, -1) if g >= 4 else y
        for dy in range(q.shape[0]):
            for dx in range(q.shape[1]):
                best = max(best, float(np.sum(unit(np.roll(y, (-dy, -dx), (0, 1))) * unit(r))))
    return best


def test_matches_brute_force_and_detects_transformed_copy():
    rng = np.random.default_rng(3)
    refs = rng.normal(size=(5, 8, 8)).astype(np.float32)
    copy = np.roll(np.flip(np.rot90(refs[2], 3), -1), (3, 5), (0, 1)).copy()
    queries = np.stack([copy, rng.normal(size=(8, 8)).astype(np.float32)])
    out = orbit_max_cosine(queries, refs, device="cpu", query_batch=1, ref_batch=2)
    assert out["ref_index"][0] == 2 and out["max_cosine"][0] == pytest.approx(1.0, abs=1e-5)
    expected = max(brute_force(queries[1], r) for r in refs)
    assert out["max_cosine"][1] == pytest.approx(expected, abs=1e-5)


def test_exclude_same_index_removes_whole_orbit():
    rng = np.random.default_rng(4)
    refs = rng.normal(size=(4, 8, 8)).astype(np.float32)
    out = orbit_max_cosine(refs, refs, exclude_same_index=True, device="cpu", query_batch=3, ref_batch=3)
    assert np.all(out["ref_index"] != np.arange(4)) and np.all(out["max_cosine"] < 0.99)
