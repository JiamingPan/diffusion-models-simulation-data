import tempfile
import unittest
from pathlib import Path
import sys

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


if __name__ == "__main__":
    unittest.main()
