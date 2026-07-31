import unittest

import numpy as np

from simdiff_eval.metrics import nearest_training_matches


class NearestTrainingMatchTests(unittest.TestCase):
    def test_returns_exact_nearest_training_image_and_similarity(self):
        training = np.array(
            [
                [[[0.0, 0.0], [0.0, 0.0]]],
                [[[1.0, 1.0], [1.0, 1.0]]],
                [[[2.0, 0.0], [0.0, 2.0]]],
            ],
            dtype=np.float32,
        )
        generated = np.array(
            [
                [[[0.9, 1.0], [1.0, 1.1]]],
                [[[2.0, 0.0], [0.0, 2.0]]],
            ],
            dtype=np.float32,
        )

        matches = nearest_training_matches(
            generated,
            training,
            max_generated=None,
            max_training=None,
            training_chunk=2,
        )

        np.testing.assert_array_equal(matches["generated_index"], [0, 1])
        np.testing.assert_array_equal(matches["nearest_training_index"], [1, 2])
        self.assertAlmostEqual(float(matches["nearest_mse"][0]), 0.005, places=7)
        self.assertAlmostEqual(float(matches["nearest_mse"][1]), 0.0, places=7)
        self.assertAlmostEqual(float(matches["nearest_cosine"][1]), 1.0, places=6)

    def test_rejects_an_empty_training_reference(self):
        generated = np.zeros((1, 1, 2, 2), dtype=np.float32)
        training = np.zeros((0, 1, 2, 2), dtype=np.float32)

        with self.assertRaisesRegex(ValueError, "training reference is empty"):
            nearest_training_matches(generated, training)


if __name__ == "__main__":
    unittest.main()
