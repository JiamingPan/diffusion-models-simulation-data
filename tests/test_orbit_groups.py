"""Restricted orbit groups in simdiff_eval.orbit_nn.orbit_max_cosine (shift-only, D4-only)."""
import argparse

import numpy as np
import pytest

torch = pytest.importorskip("torch")
from simdiff_eval import run_metrics as rm
from simdiff_eval.orbit_nn import orbit_max_cosine


def unit(x):
    x = x - x.mean()
    return x / np.linalg.norm(x)


def d4(x, g):
    y = np.rot90(x, g % 4)
    return np.flip(y, -1) if g >= 4 else y


def brute(q, r, group):
    elements = [0] if group == "shift" else range(8)
    shifts = [(0, 0)] if group == "d4" else [(a, b) for a in range(q.shape[0]) for b in range(q.shape[1])]
    return max(float(np.sum(unit(np.roll(d4(q, g), (-dy, -dx), (0, 1))) * unit(r))) for g in elements for dy, dx in shifts)


@pytest.fixture
def refs():
    return np.random.default_rng(3).normal(size=(5, 8, 8)).astype(np.float32)


@pytest.mark.parametrize("group", ["d4xshift", "shift", "d4"])
def test_matches_brute_force(refs, group):
    q = np.random.default_rng(9).normal(size=(3, 8, 8)).astype(np.float32)
    out = orbit_max_cosine(q, refs, device="cpu", query_batch=2, ref_batch=2, group=group)
    for i in range(len(q)):
        assert out["max_cosine"][i] == pytest.approx(max(brute(q[i], r, group) for r in refs), abs=1e-5)


def test_default_is_d4xshift_unchanged(refs):
    q = np.random.default_rng(1).normal(size=(4, 8, 8)).astype(np.float32)
    a = orbit_max_cosine(q, refs, device="cpu", query_batch=3, ref_batch=2)
    b = orbit_max_cosine(q, refs, device="cpu", query_batch=3, ref_batch=2, group="d4xshift")
    for k in a:
        np.testing.assert_array_equal(a[k], b[k])


def test_shift_group_finds_rolls_not_rotations(refs):
    rolled = np.roll(refs[2], (3, 5), (0, 1)).copy()
    rotated = np.rot90(refs[2], 1).copy()
    out = orbit_max_cosine(np.stack([rolled, rotated]), refs, device="cpu", group="shift")
    assert out["max_cosine"][0] == pytest.approx(1.0, abs=1e-5) and out["ref_index"][0] == 2
    assert out["group_element"][0] == 0 and (out["shift_y"][0], out["shift_x"][0]) == (3, 5)
    assert np.all(out["group_element"] == 0)
    assert out["max_cosine"][1] < 0.99


def test_d4_group_finds_rotations_not_rolls(refs):
    rotflip = np.flip(np.rot90(refs[1], 3), -1).copy()   # element 7
    rolled = np.roll(refs[1], (3, 5), (0, 1)).copy()
    out = orbit_max_cosine(np.stack([rotflip, rolled]), refs, device="cpu", group="d4")
    assert out["max_cosine"][0] == pytest.approx(1.0, abs=1e-5) and out["ref_index"][0] == 1
    assert np.all(out["shift_y"] == 0) and np.all(out["shift_x"] == 0)
    # g applied to the query maps it onto the reference: g = inverse of element 7 = 7 (a reflection)
    assert np.allclose(d4(rotflip, int(out["group_element"][0])), refs[1], atol=1e-6)
    assert out["max_cosine"][1] < 0.99


@pytest.mark.parametrize("group", ["d4xshift", "shift", "d4"])
def test_exclusion_removes_whole_restricted_orbit(refs, group):
    out = orbit_max_cosine(refs, refs, exclude_same_index=True, device="cpu", query_batch=3, ref_batch=3, group=group)
    assert np.all(out["ref_index"] != np.arange(len(refs))) and np.all(out["max_cosine"] < 0.99)


def test_unknown_group_rejected(refs):
    with pytest.raises(ValueError):
        orbit_max_cosine(refs, refs, device="cpu", group="c4")


def test_orbit_copy_stats_respects_group():
    rng = np.random.default_rng(5)
    train = rng.normal(size=(6, 16, 16)).astype(np.float32)
    rolled = np.stack([np.roll(t, (4, 7), (0, 1)) for t in train])
    rotated = np.stack([np.rot90(t, 1) for t in train])
    heldout = rng.normal(size=(4, 16, 16)).astype(np.float32)
    s_roll = rm.orbit_copy_stats(train, rolled, heldout, device="cpu", group="shift")
    d_roll = rm.orbit_copy_stats(train, rolled, heldout, device="cpu", group="d4")
    s_rot = rm.orbit_copy_stats(train, rotated, heldout, device="cpu", group="shift")
    d_rot = rm.orbit_copy_stats(train, rotated, heldout, device="cpu", group="d4")
    full = rm.orbit_copy_stats(train, rotated, heldout, device="cpu")
    # Exact copies under the searched group score 1; outside it they score like unrelated maps.
    # (Copy fractions of non-copies depend on the random-map threshold, so test the cosines.)
    np.testing.assert_allclose(s_roll["orbit_nn"], 1.0, atol=1e-5)
    np.testing.assert_allclose(d_rot["orbit_nn"], 1.0, atol=1e-5)
    np.testing.assert_allclose(full["orbit_nn"], 1.0, atol=1e-5)
    assert np.all(d_roll["orbit_nn"] < 0.9) and np.all(s_rot["orbit_nn"] < 0.9)
    assert s_roll["orbit_copy_fraction"] == 1.0 and d_rot["orbit_copy_fraction"] == 1.0
    assert full["orbit_copy_fraction"] == 1.0
    # thresholds are recalibrated per group: shift-only/zero-shift searches see fewer transforms
    assert d_roll["orbit_threshold"] < s_roll["orbit_threshold"] <= full["orbit_threshold"] + 1e-6
    assert s_roll["orbit_copies_via_transform"] == 1.0 and d_rot["orbit_copies_via_transform"] == 1.0


def test_evaluate_run_extra_groups_and_cli():
    import evaluate_run as er

    assert er.extra_orbit_groups(argparse.Namespace(orbit_group=None)) == []
    assert er.extra_orbit_groups(argparse.Namespace(orbit_group=["shift", "d4xshift", "d4", "shift"])) == ["shift", "d4"]
    assert er.extra_orbit_groups(argparse.Namespace()) == []
