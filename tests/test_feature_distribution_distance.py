import unittest

import numpy as np

from simdiff_eval.metrics import (
    frechet_feature_distance,
    real_split_frechet_baseline,
)


class FeatureDistributionDistanceTests(unittest.TestCase):
    def test_identical_feature_sets_have_zero_distance(self):
        features = np.array(
            [
                [0.0, 1.0],
                [1.0, 0.0],
                [2.0, 1.0],
                [1.0, 2.0],
            ],
            dtype=np.float64,
        )

        distance = frechet_feature_distance(features, features.copy())

        self.assertAlmostEqual(distance, 0.0, places=10)

    def test_mean_shift_produces_positive_symmetric_distance(self):
        rng = np.random.default_rng(7)
        first = rng.normal(size=(256, 5))
        second = first + np.array([1.0, -0.5, 0.25, 0.0, 0.0])

        forward = frechet_feature_distance(first, second)
        reverse = frechet_feature_distance(second, first)

        self.assertGreater(forward, 0.0)
        self.assertAlmostEqual(forward, reverse, places=9)
        self.assertAlmostEqual(forward, 1.3125, places=9)

    def test_real_split_baseline_is_deterministic_and_reports_sample_counts(self):
        rng = np.random.default_rng(11)
        features = rng.normal(size=(101, 8))

        first = real_split_frechet_baseline(features, seed=123)
        second = real_split_frechet_baseline(features, seed=123)

        self.assertEqual(first, second)
        self.assertEqual(first["n_first"], 50)
        self.assertEqual(first["n_second"], 50)
        self.assertGreaterEqual(first["distance"], 0.0)

    def test_rejects_feature_dimension_mismatch(self):
        with self.assertRaisesRegex(ValueError, "feature dimensions"):
            frechet_feature_distance(
                np.zeros((4, 2), dtype=np.float64),
                np.zeros((4, 3), dtype=np.float64),
            )


if __name__ == "__main__":
    unittest.main()
