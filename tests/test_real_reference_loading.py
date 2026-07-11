import tempfile
import unittest
from pathlib import Path
import sys
from unittest import mock

import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from simdiff_eval.io import load_real_from_config, load_real_reference_from_config


class RealReferenceLoadingTests(unittest.TestCase):
    def test_reference_is_normalized_with_full_training_set_before_limiting(self):
        cubes = np.array(
            [
                np.full((2, 2, 2), 1.0, dtype=np.float32),
                np.full((2, 2, 2), 2.0, dtype=np.float32),
                np.full((2, 2, 2), 8.0, dtype=np.float32),
                np.full((2, 2, 2), 32.0, dtype=np.float32),
            ]
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            data_path = root / "fields.npy"
            config_path = root / "config.yaml"
            np.save(data_path, cubes)
            config = {
                "global": {"device": "cpu", "dtype": "float32"},
                "data": {
                    "img_path": str(data_path),
                    "img_read_fn": "npy_read_fn",
                    "n_samples": 4,
                    "seed": None,
                    "reshape": "2d",
                    "zthin": 1,
                    "transform": ["log"],
                    "normalization": "tanh",
                    "norm_kwargs": {
                        "center": None,
                        "xmax": None,
                        "alpha": 0.8,
                        "beta": 10.0,
                        "gamma": 1.0,
                        "delta": 1.0,
                        "sigma": 1.5,
                    },
                },
            }
            config_path.write_text(yaml.safe_dump(config))

            full = load_real_from_config(config_path, max_raw_samples=None)
            limited = load_real_reference_from_config(config_path, max_slices=3)

        expected_idx = np.linspace(0, len(full) - 1, 3, dtype=np.int64)
        np.testing.assert_allclose(limited, full[expected_idx])

    def test_zero_or_none_max_slices_returns_complete_reference(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            data_path = root / "fields.npy"
            config_path = root / "config.yaml"
            np.save(data_path, np.ones((3, 2, 2, 2), dtype=np.float32))
            config_path.write_text(
                yaml.safe_dump(
                    {
                        "data": {
                            "img_path": str(data_path),
                            "img_read_fn": "npy_read_fn",
                            "n_samples": 3,
                            "reshape": "2d",
                            "zthin": 1,
                            "normalization": None,
                        }
                    }
                )
            )
            full = load_real_from_config(config_path)
            self.assertEqual(len(load_real_reference_from_config(config_path, max_slices=None)), len(full))
            self.assertEqual(len(load_real_reference_from_config(config_path, max_slices=0)), len(full))

    def test_limited_reference_does_not_materialize_full_normalized_array(self):
        cubes = np.arange(6 * 4 * 3 * 3, dtype=np.float32).reshape(6, 4, 3, 3) + 1.0
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            data_path = root / "fields.npy"
            config_path = root / "config.yaml"
            np.save(data_path, cubes)
            config_path.write_text(
                yaml.safe_dump(
                    {
                        "data": {
                            "img_path": str(data_path),
                            "img_read_fn": "npy_read_fn",
                            "n_samples": 6,
                            "seed": None,
                            "reshape": "2d",
                            "zthin": 2,
                            "transform": ["log"],
                            "normalization": "tanh",
                            "norm_kwargs": {
                                "center": None,
                                "xmax": None,
                                "alpha": 0.8,
                                "beta": 10.0,
                                "gamma": 1.0,
                                "delta": 1.0,
                                "sigma": 1.5,
                            },
                        }
                    }
                )
            )

            with mock.patch(
                "simdiff_eval.io.load_real_from_config",
                side_effect=AssertionError("full normalized materialization is forbidden"),
            ):
                limited = load_real_reference_from_config(config_path, max_slices=5)

        self.assertEqual(limited.shape, (5, 1, 3, 3))
        self.assertTrue(np.isfinite(limited).all())

    def test_streaming_reference_matches_multi_source_loader(self):
        first = np.arange(4 * 4 * 3 * 3, dtype=np.float32).reshape(4, 4, 3, 3) + 1.0
        second = np.arange(5 * 4 * 3 * 3, dtype=np.float32).reshape(5, 4, 3, 3) + 500.0
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            first_path = root / "first.npy"
            second_path = root / "second.npy"
            config_path = root / "config.yaml"
            np.save(first_path, first)
            np.save(second_path, second)
            config_path.write_text(
                yaml.safe_dump(
                    {
                        "data": {
                            "img_path": [str(first_path), str(second_path)],
                            "img_read_fn": "npy_read_fn",
                            "n_samples": [3, 4],
                            "seed": [7, 11],
                            "reshape": "2d",
                            "zthin": 2,
                            "transform": ["log"],
                            "normalization": "tanh",
                            "norm_kwargs": {
                                "center": None,
                                "xmax": None,
                                "alpha": 0.8,
                                "beta": 10.0,
                                "gamma": 1.0,
                                "delta": 1.0,
                                "sigma": 1.5,
                            },
                        }
                    }
                )
            )

            full = load_real_from_config(config_path)
            limited = load_real_reference_from_config(config_path, max_slices=7)

        expected_indices = np.linspace(0, len(full) - 1, 7, dtype=np.int64)
        np.testing.assert_allclose(limited, full[expected_indices], rtol=1e-6, atol=1e-6)


if __name__ == "__main__":
    unittest.main()
