from __future__ import annotations

import numpy as np
import pytest

from simdiff_eval.probe_transforms import get_transform


def image_batch(size: int = 16) -> np.ndarray:
    values = np.linspace(-0.9, 0.9, 2 * size * size, dtype=np.float32)
    return values.reshape(2, 1, size, size)


def apply(name: str, images: np.ndarray) -> np.ndarray:
    output, diagnostics = get_transform(name)(images)
    assert "out_of_range_fraction" in diagnostics
    assert not np.iscomplexobj(output)
    return output


def test_identity_is_bitwise_noop():
    images = image_batch()
    output = apply("identity", images)
    assert output is images
    assert np.array_equal(output, images)


def test_rot90_four_times_returns_original_exactly():
    images = image_batch()
    output = images
    for _ in range(4):
        output = apply("rot90_k1", output)
    assert np.array_equal(output, images)


@pytest.mark.parametrize("name", ["flip_h", "flip_v"])
def test_flips_are_involutions(name):
    images = image_batch()
    assert np.array_equal(apply(name, apply(name, images)), images)


def test_inverse_roll_returns_original_exactly():
    images = image_batch()
    rolled = apply("roll_dx5_dy-3", images)
    restored = apply("roll_dx-5_dy3", rolled)
    assert np.array_equal(restored, images)


def test_dihedral_registry_covers_eight_distinct_square_symmetries():
    images = np.arange(9, dtype=np.float32).reshape(1, 1, 3, 3)
    views = [apply(f"dihedral_g{element}", images) for element in range(8)]
    assert len({view.tobytes() for view in views}) == 8
    assert np.array_equal(views[0], images)


def test_out_of_range_fraction_tracks_transform_output():
    images = image_batch()
    _, inside = get_transform("identity")(images)
    pushed = images * np.float32(2.0)
    _, outside = get_transform("identity")(pushed)
    assert inside["out_of_range_fraction"] == 0.0
    assert outside["out_of_range_fraction"] > 0.0


@pytest.mark.parametrize("window", ["sharp", "hann"])
def test_lowpass_at_nyquist_is_roundtrip_identity(window):
    images = image_batch(16)
    output = apply(f"lowpass_kcut8_{window}", images)
    np.testing.assert_allclose(output, images, rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize("window", ["sharp", "hann"])
def test_lowpass_plus_highpass_reconstructs_original(window):
    images = image_batch(16)
    low = apply(f"lowpass_kcut4_{window}", images)
    high = apply(f"highpass_kcut4_{window}", images)
    np.testing.assert_allclose(low + high, images, rtol=2e-6, atol=2e-6)


@pytest.mark.parametrize(
    "name",
    [
        "lowpass_kcut4_sharp",
        "lowpass_kcut4_hann",
        "highpass_kcut4_hann",
        "fft_roundtrip_null",
    ],
)
def test_fft_transforms_return_strictly_real_float_arrays(name):
    output = apply(name, image_batch(16))
    assert output.dtype == np.float32
    assert not np.iscomplexobj(output)


def test_unit_transfer_is_roundtrip_identity():
    images = image_batch(16)
    k_bins = np.linspace(0.75, 10.5, 12)
    transform = get_transform(
        "transfer_Tk",
        transfer_k=k_bins,
        transfer_values=np.ones_like(k_bins),
    )
    output, diagnostics = transform(images)
    np.testing.assert_allclose(output, images, rtol=1e-6, atol=1e-6)
    assert diagnostics["out_of_range_fraction"] == 0.0


def test_sharp_window_rings_more_than_hann_on_binary_edge():
    images = np.full((1, 1, 32, 32), -1.0, dtype=np.float32)
    images[:, :, 8:24, 8:24] = 1.0
    sharp = apply("lowpass_kcut5_sharp", images)
    hann = apply("lowpass_kcut5_hann", images)
    sharp_overshoot = max(float(sharp.max() - 1.0), float(-1.0 - sharp.min()))
    hann_overshoot = max(float(hann.max() - 1.0), float(-1.0 - hann.min()))
    assert sharp_overshoot > hann_overshoot + 1.0e-3
