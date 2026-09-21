import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from evaluate_dit_adaln_zero_screen_results import (
    boundary_ratio,
    max_cosine_torch,
    mean_radial_power,
)


def test_exact_cosine_and_chunked_power():
    rng = np.random.default_rng(5)
    reference = rng.normal(size=(7, 1, 8, 8)).astype(np.float32)
    samples = reference[[3, 5]]
    cosine = max_cosine_torch(samples, reference, "cpu", chunk=1)
    np.testing.assert_allclose(cosine, 1.0, rtol=2e-6, atol=2e-6)
    one, k1 = mean_radial_power(reference, batch_size=1)
    all_at_once, k2 = mean_radial_power(reference, batch_size=len(reference))
    np.testing.assert_allclose(one, all_at_once, rtol=1e-12, atol=1e-12)
    np.testing.assert_array_equal(k1, k2)


def test_flat_patch_boundary_ratio_is_one_for_uniform_ramp():
    ramp = np.arange(128, dtype=np.float32)[None, None, None, :]
    image = np.broadcast_to(ramp, (1, 1, 128, 128)).copy()
    np.testing.assert_allclose(boundary_ratio(image), 1.0)
