import numpy as np
import pytest

from simdiff_eval.outlier_sensitivity import (
    filtered_histogram_probability,
    filtered_power_summary,
    novelty_bounds_after_filtering,
)


def test_novelty_bounds_account_for_unknown_removed_copy_labels():
    result = novelty_bounds_after_filtering(
        n_total=100,
        n_removed=10,
        novelty_score=0.8,
    )

    assert result["n_copies_total"] == 20
    assert result["n_kept"] == 90
    assert result["novelty_lower"] == pytest.approx(70 / 90)
    assert result["novelty_upper"] == pytest.approx(80 / 90)


def test_novelty_bounds_reject_invalid_inputs():
    with pytest.raises(ValueError, match="n_removed"):
        novelty_bounds_after_filtering(n_total=10, n_removed=10, novelty_score=0.5)
    with pytest.raises(ValueError, match="novelty_score"):
        novelty_bounds_after_filtering(n_total=10, n_removed=1, novelty_score=1.2)


def test_filtered_histogram_and_power_summary_use_only_retained_samples():
    samples = np.array([[[[0.1, 0.2]]], [[[0.8, 0.9]]], [[[9.0, 9.0]]]])
    keep = np.array([True, True, False])
    histogram = filtered_histogram_probability(samples, bins=np.array([0.0, 0.5, 1.0]), keep_mask=keep)
    assert histogram.tolist() == pytest.approx([0.5, 0.5])

    ratios = np.array([[1.0, 2.0], [3.0, 4.0], [100.0, 100.0]])
    summary = filtered_power_summary(ratios, keep_mask=keep)
    assert summary["mean"].tolist() == pytest.approx([2.0, 3.0])
    assert summary["median"].tolist() == pytest.approx([2.0, 3.0])
    assert summary["variance"].tolist() == pytest.approx([1.0, 1.0])
    assert summary["log10_mae"] == pytest.approx(np.mean(np.abs(np.log10([2.0, 3.0]))))
