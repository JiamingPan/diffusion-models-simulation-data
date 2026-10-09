"""Copy-fidelity metrics of scripts/evaluate_run.py --mode conditional (pix_std_ratio_median,
copy_fidelity_offset_mean). These tests need the proposed evaluate_run.py patch (helpers maps_sha256,
pix_std_ratio, probe_on_original); until it is applied they are skipped with that reason."""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import evaluate_run  # noqa: E402

if not all(hasattr(evaluate_run, f) for f in ("maps_sha256", "pix_std_ratio", "probe_on_original")):
    pytest.skip("evaluate_run.py copy-fidelity patch not applied yet (pending APPROVE EDIT)", allow_module_level=True)


def smooth_fields(n, size=32, seed=0):
    rng = np.random.default_rng(seed)
    k = np.fft.fftfreq(size)
    kk = np.hypot(k[:, None], k[None, :])
    amp = np.where(kk > 0, kk ** -1.5, 0.0)
    noise = rng.normal(size=(n, size, size)) + 1j * rng.normal(size=(n, size, size))
    return np.fft.ifft2(noise * amp).real.astype(np.float32)


def write_probe_table(tmp_path, train, values, probe_path, key=None):
    csv = tmp_path / "probe_on_train_maps.csv"
    rows = ["model,dataset_size,train_index,maps_sha256,Omega_m_probe"]
    key = key or evaluate_run.maps_sha256(train)
    rows += [f"m,{len(train)},{i},{key},{v!r}" for i, v in enumerate(values)]
    csv.write_text("\n".join(rows) + "\n")
    (tmp_path / "metadata.json").write_text(json.dumps({"vgg_encoder": str(probe_path)}))
    return csv


def test_maps_sha256_matches_probe_train_maps_key_with_or_without_channel_axis():
    train = smooth_fields(4)
    expected = hashlib.sha256(np.ascontiguousarray(train[:, None]).tobytes()).hexdigest()  # probe_train_maps_memorization.py
    assert evaluate_run.maps_sha256(train) == expected
    assert evaluate_run.maps_sha256(train[:, None][:, 0]) == expected


def test_pix_std_ratio_of_scaled_shifted_copy_and_nearest_index():
    from simdiff_eval.conditional_unet_diagnostics import nearest_centered_pixel_cosine

    train = smooth_fields(8, seed=3)
    gen = 2.0 * train[[3, 5]] + 0.7  # centred cosine is scale/offset invariant, so the nearest map is the original
    cos, nn = nearest_centered_pixel_cosine(gen, train)
    assert nn.tolist() == [3, 5] and np.allclose(cos, 1.0, atol=1e-5)
    np.testing.assert_allclose(evaluate_run.pix_std_ratio(gen[:, None], train[nn]), 2.0, rtol=1e-5)


def test_probe_on_original_reads_saved_table_when_maps_and_probe_match(tmp_path):
    train = smooth_fields(6, seed=4)
    probe = tmp_path / "probe.npz"
    probe.write_bytes(b"x")
    csv = write_probe_table(tmp_path, train, [0.1, 0.2, 0.3, 0.4, 0.5, 0.6], probe)

    def never(_):
        raise AssertionError("probe should not run when the saved table matches")

    vals, src = evaluate_run.probe_on_original(train, np.array([5, 0, 5]), csv, probe, never)
    np.testing.assert_allclose(vals, [0.6, 0.1, 0.6])
    assert src == f"csv:{csv}"


@pytest.mark.parametrize("mismatch", ["maps", "probe", "no_table"])
def test_probe_on_original_computes_when_table_does_not_apply(tmp_path, mismatch):
    train = smooth_fields(6, seed=5)
    probe = tmp_path / "probe.npz"
    probe.write_bytes(b"x")
    csv = write_probe_table(tmp_path, train, np.zeros(6), tmp_path / "other_probe.npz" if mismatch == "probe" else probe,
                            key="0" * 64 if mismatch == "maps" else None)
    calls = []

    def predict(maps):
        calls.append(len(maps))
        return maps.reshape(len(maps), -1).mean(1)

    idx = np.array([2, 4, 2])
    vals, src = evaluate_run.probe_on_original(train, idx, None if mismatch == "no_table" else csv, probe, predict)
    assert src == "computed" and calls == [2]  # distinct indices only
    np.testing.assert_allclose(vals, train[idx].reshape(3, -1).mean(1), rtol=1e-6)
