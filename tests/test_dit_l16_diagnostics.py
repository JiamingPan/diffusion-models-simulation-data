from __future__ import annotations

import numpy as np
import pytest

from simdiff_eval.dit_diagnostics import (
    bootstrap_histogram_l1_interval,
    bootstrap_mean_interval,
    bootstrap_power_log10_mae_interval,
    one_point_l1_common_bins,
    patch_boundary_statistics,
    selected_power_bin_statistics,
)
from simdiff_eval.metrics import (
    PHYSICAL_HIST_EDGES,
    batch_power_spectra,
    histogram_probability_and_coverage,
    radial_power_spectrum_2d,
)


def test_physical_histogram_edges_are_shared_140_bin_definition():
    expected = np.linspace(-1.0, 1.0, 141, dtype=np.float64)

    assert PHYSICAL_HIST_EDGES.dtype == np.float64
    np.testing.assert_array_equal(PHYSICAL_HIST_EDGES, expected)


def test_power_spectrum_k_max_none_preserves_legacy_binning_exactly():
    rng = np.random.default_rng(4)
    field = rng.normal(size=(128, 128))

    legacy_pk, legacy_k = radial_power_spectrum_2d(field, nbins=91)
    explicit_pk, explicit_k = radial_power_spectrum_2d(
        field, nbins=91, k_max=None
    )

    assert len(legacy_k) == 91
    np.testing.assert_array_equal(explicit_k, legacy_k)
    np.testing.assert_array_equal(explicit_pk, legacy_pk)


def test_power_spectrum_k_max_excludes_bins_above_nyquist():
    rng = np.random.default_rng(5)
    images = rng.normal(size=(3, 1, 128, 128))

    spectra, k_bins = batch_power_spectra(images, nbins=91, k_max=64.0)

    assert spectra.shape == (3, 91)
    assert len(k_bins) == 91
    assert np.all(k_bins <= 64.0)


def test_histogram_coverage_is_one_for_in_range_values():
    probability, coverage = histogram_probability_and_coverage(
        np.asarray([-1.0, -0.5, 0.0, 0.5, 1.0]),
        np.linspace(-1.0, 1.0, 5),
    )

    assert probability.sum() == pytest.approx(1.0)
    assert coverage == pytest.approx(1.0)


def test_histogram_coverage_detects_values_outside_range():
    probability, coverage = histogram_probability_and_coverage(
        np.asarray([-1.1, -0.5, 0.0, 0.5, 1.1]),
        np.linspace(-1.0, 1.0, 5),
    )

    assert probability.sum() == pytest.approx(1.0)
    assert coverage == pytest.approx(3.0 / 5.0)


def test_bootstrap_mean_interval_is_deterministic():
    values = np.linspace(-1.0, 1.0, 41)

    first = bootstrap_mean_interval(values, n_resamples=500, seed=17)
    second = bootstrap_mean_interval(values, n_resamples=500, seed=17)

    assert first == second
    assert first[0] < values.mean() < first[1]


def test_scalar_physics_bootstrap_intervals_are_deterministic():
    rng = np.random.default_rng(31)
    generated = rng.normal(0.15, 0.55, size=(24, 1, 8, 8))
    edges = np.linspace(-1.0, 1.0, 15)
    reference_probability, _ = histogram_probability_and_coverage(
        rng.normal(0.0, 0.5, size=(40, 1, 8, 8)), edges
    )
    generated_probability, _ = histogram_probability_and_coverage(generated, edges)
    hist_point = float(np.abs(generated_probability - reference_probability).sum())

    hist_first = bootstrap_histogram_l1_interval(
        generated,
        reference_probability,
        edges,
        n_resamples=300,
        seed=19,
    )
    hist_second = bootstrap_histogram_l1_interval(
        generated,
        reference_probability,
        edges,
        n_resamples=300,
        seed=19,
    )

    generated_power = rng.lognormal(size=(24, 12))
    real_power = rng.lognormal(size=12)
    ratio = generated_power.mean(axis=0) / real_power
    power_point = float(np.mean(np.abs(np.log10(ratio))))
    power_first = bootstrap_power_log10_mae_interval(
        generated_power,
        real_power,
        n_resamples=300,
        seed=19,
    )
    power_second = bootstrap_power_log10_mae_interval(
        generated_power,
        real_power,
        n_resamples=300,
        seed=19,
    )

    assert hist_first == hist_second
    assert hist_first[0] <= hist_point <= hist_first[1]
    assert power_first == power_second
    assert power_first[0] <= power_point <= power_first[1]


def test_constant_selected_power_bins_have_zero_variance_and_collapsed_interval():
    spectra = np.full((32, 91), 4.5)

    rows = selected_power_bin_statistics(
        spectra, bin_indices=(20, 40, 60), n_resamples=200, seed=9
    )

    assert [row["k_bin"] for row in rows] == [20, 40, 60]
    for row in rows:
        assert row["mean"] == pytest.approx(4.5)
        assert row["variance"] == pytest.approx(0.0)
        assert row["std"] == pytest.approx(0.0)
        assert row["mean_ci_low"] == pytest.approx(4.5)
        assert row["mean_ci_high"] == pytest.approx(4.5)


@pytest.mark.parametrize("bad_bin", [-1, 91])
def test_selected_power_bins_reject_invalid_indices(bad_bin):
    with pytest.raises(ValueError, match="k-bin"):
        selected_power_bin_statistics(np.ones((4, 91)), (bad_bin,))


def test_selected_power_bins_reject_invalid_shape_and_nonfinite_values():
    with pytest.raises(ValueError, match="two-dimensional"):
        selected_power_bin_statistics(np.ones(91), (20,))

    spectra = np.ones((4, 91))
    spectra[0, 20] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        selected_power_bin_statistics(spectra, (20,))


def test_identical_fields_have_zero_common_bin_l1():
    rng = np.random.default_rng(123)
    images = rng.normal(size=(8, 1, 16, 16))

    assert one_point_l1_common_bins(images, images, bins=40) == pytest.approx(0.0)


def test_one_point_l1_rejects_empty_or_nonfinite_fields():
    with pytest.raises(ValueError, match="empty"):
        one_point_l1_common_bins(np.empty((0,)), np.ones((1,)))
    with pytest.raises(ValueError, match="non-finite"):
        one_point_l1_common_bins(np.asarray([np.nan]), np.ones((1,)))


def _smooth_linear_fields(n_images: int = 3, size: int = 32) -> np.ndarray:
    coordinate = np.arange(size, dtype=np.float64)
    field = coordinate[:, None] + coordinate[None, :]
    return np.repeat(field[None, None], n_images, axis=0)


def test_smooth_fields_have_patch_boundary_ratio_near_one():
    stats = patch_boundary_statistics(_smooth_linear_fields(), patch_size=8)

    assert stats["boundary_to_interior_ratio"] == pytest.approx(1.0)
    assert stats["boundary_to_control_ratio"] == pytest.approx(1.0)
    assert stats["boundary_excess_abs_difference"] == pytest.approx(0.0)
    assert stats["horizontal_boundary_to_interior_ratio"] == pytest.approx(1.0)
    assert stats["vertical_boundary_to_interior_ratio"] == pytest.approx(1.0)


def test_eight_pixel_seams_raise_patch_boundary_ratio():
    images = _smooth_linear_fields(n_images=2)
    block_offsets = np.repeat(np.arange(4, dtype=np.float64), 8)
    images = images + 25.0 * (
        block_offsets[None, None, :, None] + block_offsets[None, None, None, :]
    )

    stats = patch_boundary_statistics(images, patch_size=8)

    assert stats["boundary_to_interior_ratio"] > 10.0
    assert stats["boundary_to_control_ratio"] > 10.0
    assert stats["boundary_relative_excess"] > 9.0
    assert stats["horizontal_boundary_to_interior_ratio"] > 10.0
    assert stats["vertical_boundary_to_interior_ratio"] > 10.0


@pytest.mark.parametrize(
    ("images", "patch_size", "message"),
    [
        (np.ones((2, 32, 32)), 8, "N,C,H,W"),
        (np.ones((2, 1, 32, 32)), 1, "patch_size"),
        (np.ones((2, 1, 30, 32)), 8, "divisible"),
    ],
)
def test_patch_boundary_statistics_reject_invalid_inputs(images, patch_size, message):
    with pytest.raises(ValueError, match=message):
        patch_boundary_statistics(images, patch_size=patch_size)
