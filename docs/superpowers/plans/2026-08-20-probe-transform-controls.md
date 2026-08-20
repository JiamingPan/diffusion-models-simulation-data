# Frozen VGG Probe Transform Controls Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build tested C0 symmetry, C1 scale-cut, and C4 degraded-real input controls for the existing frozen VGG cosmology probe without running real-data controls.

**Architecture:** A shared loader supplies the exact held-out CAMELS slice order, pure NumPy transforms modify only probe inputs, and a dependency-light analysis module builds tidy predictions, bootstrap reports, and manifests. Two thin scripts connect those units to the existing frozen VGG loader: one for C0/C1 transform controls and one for C4 generated-power-matched degraded reals.

**Tech Stack:** Python 3, NumPy FFT, pandas, scikit-learn version metadata, pytest, existing Torch/TorchVision VGG runtime only at command-line execution time.

**Spec:** `docs/superpowers/specs/2026-08-20-probe-transform-controls-design.md`

## Global Constraints

- Do not retrain or fine-tune VGG16, its regression head, or any diffusion model.
- Do not generate samples, submit Slurm jobs, or execute the controls against CAMELS in this implementation pass.
- Real evaluation is restricted to simulations 900 through 931 inclusive, with 128 z-slices per simulation and 4096 slices total.
- Every control invocation writes `manifest.json` with Git revision and dirty state, encoder `.npz` path and SHA-256, head pickle path and SHA-256 when available, installed scikit-learn version, transforms, and all RNG seeds.
- `simdiff_eval/probe_transforms.py` contains pure NumPy only and performs no I/O.
- Every run includes identity exactly once.
- Fourier transforms use `rfft2` and `irfft2`; no code silently takes `.real` from a complex inverse transform.
- The transform prediction artifact is one long-format CSV; aggregated reports use JSON.
- Tests use small synthetic arrays and fake encoders only; they require no CAMELS data, downloaded weights, or GPU.
- Write and witness each failing test before adding the production behavior it specifies.

---

### Task 1: Extract the exact held-out real-map loader

**Files:**
- Create: `simdiff_eval/probe_eval.py`
- Modify: `scripts/train_nf_conditional_bias_vgg_encoder.py:23-33,236-274`
- Create: `tests/test_probe_controls.py`

**Interfaces:**
- Consumes: existing `select_slice_pairs`, `load_raw_slices`, `preprocess_real_slices`, `image_path`, `params_path`, `load_params`, and `N_TRAIN_SIMS`.
- Produces: `load_heldout_real_slices(data_root, heldout_indices, slices_per_sim, norm) -> tuple[images, theta_raw, sim_index, z_index]`.

- [ ] **Step 1: Write the regression test for the legacy pair order**

Create `tests/test_probe_controls.py` with repository/script path setup and this test. Patch the loader primitives so the test is independent of CAMELS files while still asserting the exact old selection expression:

```python
from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
for path in (ROOT, SCRIPTS):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))


def test_load_heldout_real_slices_reproduces_legacy_pair_order(monkeypatch):
    from simdiff_eval import probe_eval
    from train_nf_conditional_bias_encoder import select_slice_pairs

    heldout = np.array([900, 901, 902], dtype=np.int64)
    expected_pairs = select_slice_pairs(heldout, len(heldout) * 7)
    expected_raw = np.arange(len(expected_pairs) * 4, dtype=np.float32).reshape(
        len(expected_pairs), 1, 2, 2
    ) + 1.0
    params = np.arange(1000 * 6, dtype=np.float32).reshape(1000, 6)
    observed = {}

    monkeypatch.setattr(probe_eval, "image_path", lambda root: Path(root) / "grid.npy")
    monkeypatch.setattr(probe_eval, "params_path", lambda root: Path(root) / "params.txt")
    monkeypatch.setattr(probe_eval, "load_params", lambda path, count: params)

    def fake_load_raw(path, pairs):
        observed["pairs"] = pairs.copy()
        return expected_raw

    monkeypatch.setattr(probe_eval, "load_raw_slices", fake_load_raw)
    monkeypatch.setattr(
        probe_eval,
        "preprocess_real_slices",
        lambda raw, norm: raw.astype(np.float32) * np.float32(norm["scale"]),
    )

    images, theta_raw, sim_index, z_index = probe_eval.load_heldout_real_slices(
        "/synthetic", heldout, 7, {"scale": 0.5}
    )

    np.testing.assert_array_equal(observed["pairs"], expected_pairs)
    np.testing.assert_array_equal(sim_index, expected_pairs[:, 0])
    np.testing.assert_array_equal(z_index, expected_pairs[:, 1])
    np.testing.assert_array_equal(theta_raw, params[expected_pairs[:, 0]])
    np.testing.assert_array_equal(images, expected_raw * np.float32(0.5))
    assert images.dtype == np.float32
```

- [ ] **Step 2: Run the regression test and verify the missing-module failure**

Run: `python -m pytest tests/test_probe_controls.py::test_load_heldout_real_slices_reproduces_legacy_pair_order -v`

Expected: FAIL because `simdiff_eval.probe_eval` does not exist.

- [ ] **Step 3: Implement the shared loader**

Create `simdiff_eval/probe_eval.py`. Import the existing script helpers through the repository's established scripts path and validate the public inputs before composing them:

```python
"""Shared loading for frozen-probe evaluation on held-out real maps."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from prepare_nf_conditional_u128_config import (
    N_TRAIN_SIMS,
    image_path,
    load_params,
    params_path,
)
from train_nf_conditional_bias_encoder import (
    load_raw_slices,
    preprocess_real_slices,
    select_slice_pairs,
)


def load_heldout_real_slices(
    data_root: str | Path,
    heldout_indices: np.ndarray,
    slices_per_sim: int,
    norm: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    heldout = np.asarray(heldout_indices, dtype=np.int64).reshape(-1)
    slices_per_sim = int(slices_per_sim)
    if heldout.size == 0:
        raise ValueError("heldout_indices is empty")
    if slices_per_sim < 1 or slices_per_sim > 128:
        raise ValueError("slices_per_sim must lie in 1..128")
    pairs = select_slice_pairs(heldout, len(heldout) * slices_per_sim)
    raw = load_raw_slices(image_path(data_root), pairs)
    images = preprocess_real_slices(raw, norm).astype(np.float32, copy=False)
    params = load_params(params_path(data_root), N_TRAIN_SIMS)
    sim_index = pairs[:, 0].astype(np.int64, copy=False)
    z_index = pairs[:, 1].astype(np.int64, copy=False)
    theta_raw = params[sim_index].astype(np.float32, copy=False)
    return images, theta_raw, sim_index, z_index
```

- [ ] **Step 4: Refactor only the trainer's held-out branch**

Import `load_heldout_real_slices` in `scripts/train_nf_conditional_bias_vgg_encoder.py`. Keep training and validation in `embed_pairs`, but replace the old test-pair call and test `embed_pairs` call with:

```python
    test_images, y_test_raw, test_sim_index, test_z_index = load_heldout_real_slices(
        args.data_root,
        heldout,
        args.test_slices_per_sim,
        norm,
    )
    test_pairs = np.stack([test_sim_index, test_z_index], axis=1)
    x_test = vgg_embed(
        test_images,
        vgg,
        device=device,
        batch_size=args.embedding_batch_size,
        image_size=args.image_size,
        value_min=args.value_min,
        value_max=args.value_max,
        pool=args.pool,
    )
    y_test_norm = params_norm_all[test_sim_index].astype(np.float32, copy=False)
```

Delete the old `test_pairs = select_slice_pairs(...)` and `x_test, ... = embed_pairs(test_pairs)` statements. Do not change training, validation, model fitting, or metric logic.

- [ ] **Step 5: Verify the loader regression and trainer import**

Run: `python -m pytest tests/test_probe_controls.py::test_load_heldout_real_slices_reproduces_legacy_pair_order -v`

Run: `python -c "import scripts.train_nf_conditional_bias_vgg_encoder; import simdiff_eval.probe_eval"`

Expected: PASS and both imports exit 0.

- [ ] **Step 6: Commit Task 1**

```bash
git add simdiff_eval/probe_eval.py scripts/train_nf_conditional_bias_vgg_encoder.py tests/test_probe_controls.py
git commit -m "refactor: share heldout probe loading"
```

---

### Task 2: Add the pure NumPy transform registry

**Files:**
- Create: `simdiff_eval/probe_transforms.py`
- Create: `tests/test_probe_transforms.py`

**Interfaces:**
- Consumes: float32 arrays shaped `(N,1,H,W)` and optional per-bin transfer arrays.
- Produces: `Transform`, geometric/spectral factories including `transfer_transform(k_bins, transfer_values)`, `compose_transforms`, and `get_transform(name, transfer_k=None, transfer_values=None)`.

- [ ] **Step 1: Write failing geometric-transform and diagnostic tests**

Create `tests/test_probe_transforms.py` with deterministic synthetic input and the mandatory exact properties:

```python
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


def test_out_of_range_fraction_tracks_transform_output():
    images = image_batch()
    _, inside = get_transform("identity")(images)
    pushed = images * np.float32(2.0)
    _, outside = get_transform("identity")(pushed)
    assert inside["out_of_range_fraction"] == 0.0
    assert outside["out_of_range_fraction"] > 0.0
```

- [ ] **Step 2: Run geometric tests and verify the missing-module failure**

Run: `python -m pytest tests/test_probe_transforms.py -v`

Expected: collection FAIL because `simdiff_eval.probe_transforms` does not exist.

- [ ] **Step 3: Implement validation, diagnostics, and exact index transforms**

Create `simdiff_eval/probe_transforms.py` with:

```python
"""Pure NumPy input transforms for frozen cosmology probes."""

from __future__ import annotations

import re
from collections.abc import Callable

import numpy as np

Transform = Callable[[np.ndarray], tuple[np.ndarray, dict[str, float]]]


def _images(images: np.ndarray) -> np.ndarray:
    array = np.asarray(images)
    if array.ndim != 4 or array.shape[1] != 1:
        raise ValueError(f"Expected (N,1,H,W), got {array.shape}")
    if array.shape[-2] != array.shape[-1]:
        raise ValueError("Probe transforms require square images")
    if not np.issubdtype(array.dtype, np.floating):
        raise TypeError("Probe transforms require floating-point images")
    if not np.isfinite(array).all():
        raise ValueError("Probe transform input contains non-finite values")
    return array


def _finish(output: np.ndarray) -> tuple[np.ndarray, dict[str, float]]:
    array = np.ascontiguousarray(output)
    fraction = float(np.mean((array < -1.0) | (array > 1.0)))
    return array, {"out_of_range_fraction": fraction}


def identity_transform() -> Transform:
    def transform(images: np.ndarray):
        array = _images(images)
        return array, {"out_of_range_fraction": float(np.mean((array < -1.0) | (array > 1.0)))}
    return transform


def rot90_transform(k: int) -> Transform:
    def transform(images: np.ndarray):
        return _finish(np.rot90(_images(images), int(k) % 4, axes=(-2, -1)))
    return transform


def flip_transform(axis: int) -> Transform:
    def transform(images: np.ndarray):
        return _finish(np.flip(_images(images), axis=axis))
    return transform


def dihedral_transform(element: int) -> Transform:
    element = int(element)
    if element not in range(8):
        raise ValueError("dihedral element must lie in 0..7")
    rotation = element % 4
    def transform(images: np.ndarray):
        output = np.rot90(_images(images), rotation, axes=(-2, -1))
        if element >= 4:
            output = np.flip(output, axis=-1)
        return _finish(output)
    return transform


def roll_transform(dx: int, dy: int) -> Transform:
    def transform(images: np.ndarray):
        return _finish(np.roll(_images(images), shift=(int(dy), int(dx)), axis=(-2, -1)))
    return transform
```

Add `compose_transforms` so C0 can apply a dihedral element then a roll while reporting diagnostics from the final output.

- [ ] **Step 4: Implement the geometric name resolver and pass its tests**

Parse the mandatory names with anchored regular expressions, mapping horizontal flips to axis `-1`, vertical flips to `-2`, `dx` to width, and `dy` to height. Unknown names raise `KeyError` containing the name.

Run: `python -m pytest tests/test_probe_transforms.py -k "identity or rot90 or flip or roll or out_of_range" -v`

Expected: PASS.

- [ ] **Step 5: Add failing Fourier identity, complement, real-output, and transfer tests**

Append:

```python
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
    ["lowpass_kcut4_sharp", "lowpass_kcut4_hann", "highpass_kcut4_hann", "fft_roundtrip_null"],
)
def test_fft_transforms_return_strictly_real_float_arrays(name):
    output = apply(name, image_batch(16))
    assert output.dtype == np.float32
    assert not np.iscomplexobj(output)


def test_unit_transfer_is_roundtrip_identity():
    images = image_batch(16)
    k_bins = np.linspace(0.75, 10.5, 12)
    transform = get_transform(
        "transfer_Tk", transfer_k=k_bins, transfer_values=np.ones_like(k_bins)
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
```

- [ ] **Step 6: Run Fourier tests and verify missing-name failures**

Run: `python -m pytest tests/test_probe_transforms.py -k "nyquist or reconstructs or fft or transfer or rings" -v`

Expected: FAIL because Fourier names are not implemented.

- [ ] **Step 7: Implement complementary Fourier responses with `rfft2/irfft2`**

Add `_frequency_grid`, `_lowpass_response`, and `_spectral_transform`:

```python
def _frequency_grid(height: int, width: int) -> np.ndarray:
    ky = np.fft.fftfreq(height) * height
    kx = np.fft.rfftfreq(width) * width
    kkx, kky = np.meshgrid(kx, ky)
    return np.sqrt(kkx**2 + kky**2)


def _lowpass_response(k_radius: np.ndarray, k_cut: float, window: str, k_nyquist: float) -> np.ndarray:
    if k_cut >= k_nyquist:
        return np.ones_like(k_radius, dtype=np.float64)
    if window == "sharp":
        return (k_radius <= k_cut).astype(np.float64)
    if window != "hann":
        raise ValueError(f"Unknown Fourier window: {window}")
    lower = max(0.0, k_cut - 2.0)
    upper = k_cut + 2.0
    response = np.ones_like(k_radius, dtype=np.float64)
    response[k_radius >= upper] = 0.0
    transition = (k_radius > lower) & (k_radius < upper)
    phase = (k_radius[transition] - lower) / max(upper - lower, 1.0e-12)
    response[transition] = 0.5 * (1.0 + np.cos(np.pi * phase))
    return response


def _apply_response(images: np.ndarray, response: np.ndarray) -> np.ndarray:
    array = _images(images).astype(np.float32, copy=False)
    spectrum = np.fft.rfft2(array, axes=(-2, -1))
    output = np.fft.irfft2(
        spectrum * response[None, None],
        s=array.shape[-2:],
        axes=(-2, -1),
    )
    return output.astype(np.float32, copy=False)
```

Low-pass uses the response; high-pass uses `1.0 - response`. `fft_roundtrip_null` uses an all-one response. Return through `_finish` and include numeric `k_cut` and `k_cut_over_knyq` diagnostics for spectral filters.

- [ ] **Step 8: Implement bin-constant `transfer_Tk` and complete the resolver**

Implement `transfer_transform(k_bins: np.ndarray, transfer_values: np.ndarray) -> Transform`. Validate one-dimensional, equal-length, finite, strictly increasing `k_bins` and non-negative finite `transfer_values`. Infer uniform bin edges from adjacent centers, assign modes with `np.digitize`, set the DC response to one, clip bin indices to the nearest endpoint for corners, and call `_apply_response`. `get_transform("transfer_Tk", ...)` delegates to this public factory. Parse spectral cutoff names with `r"^(lowpass|highpass)_kcut([0-9]+(?:\\.[0-9]+)?)_(sharp|hann)$"` so both integer defaults and explicit floating-point CLI values resolve consistently.

Run: `python -m pytest tests/test_probe_transforms.py -v`

Expected: all transform tests PASS.

- [ ] **Step 9: Commit Task 2**

```bash
git add simdiff_eval/probe_transforms.py tests/test_probe_transforms.py
git commit -m "feat: add probe input transform registry"
```

---

### Task 3: Build shared prediction, aggregation, manifest, and CLI foundations

**Files:**
- Create: `simdiff_eval/probe_controls.py`
- Create: `scripts/evaluate_probe_transform_controls.py`
- Modify: `tests/test_probe_controls.py`

**Interfaces:**
- Consumes: Task 1 loader, Task 2 transforms, existing `load_vgg_encoder`, `PARAM_NAMES`, and encoder protocol `predict_norm`/`norm_to_raw`.
- Produces: `TransformSpec`, `evaluate_transform_specs`, `aggregate_prediction_table`, `build_run_manifest`, and the C0/C1 command-line harness.

- [ ] **Step 1: Write failing tidy-row, identity, and two-grain aggregation tests**

Append a fake encoder and tests:

```python
class FakeEncoder:
    model_path = Path("fake-head.pkl")

    def predict_norm(self, images, batch_size=512):
        means = np.asarray(images, dtype=np.float32).mean(axis=(1, 2, 3))
        return np.stack([means + offset for offset in range(6)], axis=1)

    def norm_to_raw(self, theta_norm):
        return np.asarray(theta_norm, dtype=np.float32)


def synthetic_probe_inputs():
    images = np.arange(8 * 8 * 8, dtype=np.float32).reshape(8, 1, 8, 8)
    images = images / images.max()
    sim_index = np.repeat(np.array([900, 901]), 4)
    z_index = np.tile(np.arange(4), 2)
    theta_raw = np.stack(
        [np.linspace(0.1 + j, 0.2 + j, 8, dtype=np.float32) for j in range(6)],
        axis=1,
    )
    return images, theta_raw, sim_index, z_index


def test_transform_evaluation_has_required_long_columns_and_one_identity():
    from simdiff_eval.probe_controls import TransformSpec, evaluate_transform_specs
    from simdiff_eval.probe_transforms import get_transform

    images, theta_raw, sim_index, z_index = synthetic_probe_inputs()
    specs = [
        TransformSpec("identity", "identity", get_transform("identity")),
        TransformSpec("flip_h", "dihedral", get_transform("flip_h")),
    ]
    table = evaluate_transform_specs(
        images, theta_raw, sim_index, z_index, FakeEncoder(), specs, batch_size=4
    )
    required = {
        "transform", "transform_family", "k_cut", "k_cut_over_knyq", "window",
        "sim_index", "z_index", "parameter", "theta_true", "theta_pred",
        "out_of_range_fraction",
    }
    assert required.issubset(table.columns)
    assert table[table["transform"] == "identity"].shape[0] == 8 * 6
    assert table.shape[0] == 2 * 8 * 6


def test_aggregation_reports_per_slice_and_per_cosmology_with_bootstrap():
    from simdiff_eval.probe_controls import (
        TransformSpec,
        aggregate_prediction_table,
        evaluate_transform_specs,
    )
    from simdiff_eval.probe_transforms import get_transform

    images, theta_raw, sim_index, z_index = synthetic_probe_inputs()
    table = evaluate_transform_specs(
        images,
        theta_raw,
        sim_index,
        z_index,
        FakeEncoder(),
        [TransformSpec("identity", "identity", get_transform("identity"))],
        batch_size=4,
    )
    report = aggregate_prediction_table(table, n_boot=50, seed=17)
    grains = {row["grain"] for row in report["metrics"]}
    assert grains == {"per_slice", "per_cosmology"}
    omega_rows = [row for row in report["metrics"] if row["parameter"] == "Omega_m"]
    assert {row["n"] for row in omega_rows} == {2, 8}
    assert all("rmse_ci_low" in row and "slope_ci_high" in row for row in omega_rows)
```

- [ ] **Step 2: Run the tests and verify missing shared-control failures**

Run: `python -m pytest tests/test_probe_controls.py -k "long_columns or aggregation" -v`

Expected: FAIL because `simdiff_eval.probe_controls` does not exist.

- [ ] **Step 3: Implement `TransformSpec` and long-row evaluation**

Create `simdiff_eval/probe_controls.py` with a frozen dataclass containing `name`, `family`, `transform`, optional `k_cut`, `window`, `dihedral_g`, `roll_dx`, and `roll_dy`. In `evaluate_transform_specs`:

```python
PARAM_NAMES = ("Omega_m", "sigma_8", "A_SN1", "A_AGN1", "A_SN2", "A_AGN2")

for spec in specs:
    transformed, diagnostics = spec.transform(images)
    prediction = encoder.norm_to_raw(
        encoder.predict_norm(transformed, batch_size=int(batch_size))
    )
    for image_index in range(len(images)):
        for parameter_index, parameter in enumerate(PARAM_NAMES):
            rows.append({
                "transform": spec.name,
                "transform_family": spec.family,
                "k_cut": spec.k_cut,
                "k_cut_over_knyq": None if spec.k_cut is None else spec.k_cut / (images.shape[-1] / 2.0),
                "window": spec.window,
                "dihedral_g": spec.dihedral_g,
                "roll_dx": spec.roll_dx,
                "roll_dy": spec.roll_dy,
                "sim_index": int(sim_index[image_index]),
                "z_index": int(z_index[image_index]),
                "parameter": parameter,
                "theta_true": float(theta_raw[image_index, parameter_index]),
                "theta_pred": float(prediction[image_index, parameter_index]),
                "out_of_range_fraction": float(diagnostics["out_of_range_fraction"]),
            })
```

Reject empty specs, duplicate transform names, any suite without exactly one identity, shape mismatches, and non-finite predictions.

- [ ] **Step 4: Implement deterministic bootstrap metrics for both grains**

Build per-cosmology data by grouping descriptor columns plus `sim_index` and `parameter`, taking the first true value and median predicted value. For each grain/group compute:

```python
error = theta_pred - theta_true
rmse = float(np.sqrt(np.mean(error**2)))
bias = float(np.mean(error))
slope = float(np.polyfit(theta_true, theta_pred, 1)[0]) if np.var(theta_true) > 1.0e-30 else float("nan")
```

Use a recorded NumPy generator to resample observation rows with replacement `n_boot` times and calculate percentile 16/84 intervals for all three metrics. Return JSON-safe `{"metrics": rows, "bootstrap": {"n_resamples": ..., "seed": ...}}`.

Run: `python -m pytest tests/test_probe_controls.py -k "long_columns or aggregation" -v`

Expected: PASS.

- [ ] **Step 5: Write failing manifest provenance test**

Append a test using temporary encoder/head files and monkeypatched distribution/Git metadata:

```python
def test_manifest_records_frozen_encoder_environment_and_seeds(tmp_path, monkeypatch):
    from simdiff_eval import probe_controls

    encoder_path = tmp_path / "encoder.npz"
    head_path = tmp_path / "head.pkl"
    encoder_path.write_bytes(b"encoder artifact")
    head_path.write_bytes(b"head artifact")
    monkeypatch.setattr(probe_controls, "installed_sklearn_version", lambda: "9.9.9")
    monkeypatch.setattr(
        probe_controls,
        "git_state",
        lambda project_dir: {"revision": "abc123", "dirty": True},
    )
    manifest = probe_controls.build_run_manifest(
        project_dir=tmp_path,
        encoder_path=encoder_path,
        head_path=head_path,
        heldout_indices=np.arange(900, 932),
        slices_per_sim=128,
        transforms=[{"name": "identity", "family": "identity"}],
        seeds={"bootstrap": 17, "roll": 23},
        arguments={"device": "cpu"},
    )
    assert manifest["git"] == {"revision": "abc123", "dirty": True}
    assert manifest["encoder"]["path"] == str(encoder_path.resolve())
    assert len(manifest["encoder"]["sha256"]) == 64
    assert manifest["head"]["path"] == str(head_path.resolve())
    assert manifest["scikit_learn_version"] == "9.9.9"
    assert manifest["heldout_indices"] == list(range(900, 932))
    assert manifest["seeds"] == {"bootstrap": 17, "roll": 23}
```

- [ ] **Step 6: Run the manifest test and verify missing-function failure**

Run: `python -m pytest tests/test_probe_controls.py::test_manifest_records_frozen_encoder_environment_and_seeds -v`

Expected: FAIL because `build_run_manifest` is missing.

- [ ] **Step 7: Implement hashing, Git state, scikit-learn metadata, and manifest construction**

Use `hashlib.sha256`, `importlib.metadata.version("scikit-learn")`, and read-only `subprocess.run` calls for `git rev-parse HEAD` and `git status --porcelain`. Resolve and require the encoder path. Include schema version `1`, UTC timestamp, transforms, seeds, CLI arguments, held-out list, and slices per simulation. Include the head block only when a path is supplied.

Run: `python -m pytest tests/test_probe_controls.py::test_manifest_records_frozen_encoder_environment_and_seeds -v`

Expected: PASS.

- [ ] **Step 8: Implement the shared C0/C1 CLI foundation**

Create `scripts/evaluate_probe_transform_controls.py` with arguments for project directory, data root, VGG encoder, device, embedding batch size, output directory, bootstrap count, bootstrap seed, roll seed, and repeatable `--k-cut`. The CLI must:

1. Resolve the encoder and read its `normalization` and `heldout_indices` fields.
2. Assert held-out indices equal `np.arange(900, 932)`.
3. Call `load_heldout_real_slices(..., slices_per_sim=128, ...)`.
4. Call the existing `load_vgg_encoder` without fitting anything.
5. Build the requested transform specs, de-duplicate identity, and call `evaluate_transform_specs`.
6. Write `probe_transform_predictions.csv` atomically, then JSON metrics and `manifest.json`.

At this task, implement `identity_specs()` and an identity-only run. Do not expose incomplete C0 or C1 command choices yet. Task 4 adds `--control c0`; Task 6 adds `--control c1`. This keeps every intermediate commit runnable without deferred branches.

- [ ] **Step 9: Verify CLI import and focused control tests**

Run: `python scripts/evaluate_probe_transform_controls.py --help`

Run: `python -m pytest tests/test_probe_controls.py -k "load_heldout or long_columns or aggregation or manifest" -v`

Expected: PASS.

- [ ] **Step 10: Commit Task 3**

```bash
git add simdiff_eval/probe_controls.py scripts/evaluate_probe_transform_controls.py tests/test_probe_controls.py
git commit -m "feat: add shared probe control harness"
```

---

### Task 4: Implement C0 symmetry views and baseline-normalized reports

**Files:**
- Modify: `simdiff_eval/probe_controls.py`
- Modify: `scripts/evaluate_probe_transform_controls.py`
- Modify: `tests/test_probe_controls.py`

**Interfaces:**
- Consumes: `TransformSpec`, `compose_transforms`, geometric transform factories, and long predictions.
- Produces: `build_c0_specs(seed) -> tuple[list[TransformSpec], list[tuple[int,int]]]` and `c0_symmetry_summary(predictions, n_boot, seed) -> dict`.

- [ ] **Step 1: Write the failing 40-view deterministic-spec test**

Append:

```python
def test_c0_builds_40_deterministic_views_with_recorded_nonzero_rolls():
    from simdiff_eval.probe_controls import build_c0_specs

    first, first_offsets = build_c0_specs(seed=23)
    second, second_offsets = build_c0_specs(seed=23)
    assert len(first) == 40
    assert [spec.name for spec in first] == [spec.name for spec in second]
    assert first_offsets == second_offsets
    assert len(first_offsets) == 4
    assert len(set(first_offsets)) == 4
    assert all((dx, dy) != (0, 0) for dx, dy in first_offsets)
    assert sum(spec.name == "identity" for spec in first) == 1
    assert {spec.dihedral_g for spec in first} == set(range(8))
```

- [ ] **Step 2: Run it and verify the missing-builder failure**

Run: `python -m pytest tests/test_probe_controls.py::test_c0_builds_40_deterministic_views_with_recorded_nonzero_rolls -v`

Expected: FAIL because `build_c0_specs` is missing.

- [ ] **Step 3: Implement deterministic C0 spec construction**

Use `np.random.default_rng(seed)` to draw unique integer offsets from `[-63, 63]` until four non-zero pairs exist. For every `g in range(8)` and state `(0,0) + offsets`, compose `dihedral_transform(g)` then `roll_transform(dx,dy)`. Name `g=0, dx=dy=0` exactly `identity`; name other no-roll views `dihedral_g{g}` and rolled views `dihedral_g{g}__roll_dx{dx}_dy{dy}`. Set no-roll families to `identity` for g0 and `dihedral` otherwise; set rolled families to `roll`.

Run the Step 2 command again. Expected: PASS.

- [ ] **Step 4: Write the failing separate-family and baseline-ratio test**

Construct a synthetic Omega-m prediction table with two simulations, four z-slices, eight dihedral views, and five roll states. Make roll variation larger than dihedral variation and identity z-slice variation non-zero. Assert:

```python
report = c0_symmetry_summary(table, n_boot=50, seed=31)
assert {row["family"] for row in report["per_slice"]} == {"dihedral", "roll"}
assert all(np.isfinite(row["std_over_within_sim_std"]) for row in report["per_slice"])
assert report["family_summary"]["roll"]["median_std_ratio"] > report["family_summary"]["dihedral"]["median_std_ratio"]
assert report["baseline"]["definition"] == "identity Omega_m spread across z-slices within each simulation"
```

- [ ] **Step 5: Run it and verify the missing-summary failure**

Run: `python -m pytest tests/test_probe_controls.py -k "separate_family and baseline" -v`

Expected: FAIL because `c0_symmetry_summary` is missing.

- [ ] **Step 6: Implement C0 spread and bootstrap summaries**

Filter to `parameter == "Omega_m"`. Compute identity within-simulation standard deviation (`ddof=0`) and max-minus-min over z. For dihedral, group the eight no-roll views by `(sim_index,z_index)`; for roll, group five roll states by `(sim_index,z_index,dihedral_g)`, calculate spreads, then take their median across `dihedral_g` for each slice. Divide each transform spread by its simulation baseline, returning `NaN` when the baseline is zero. Aggregate per-slice ratios to per-simulation medians and bootstrap those 32 simulation values for each family without pooling families.

- [ ] **Step 7: Wire C0 into the CLI and manifest**

Add repeatable `--control` with choices `identity` and `c0`, using `build_c0_specs(args.roll_seed)` for C0. Write `c0_symmetry_summary.json`, record the four offsets and roll/bootstrap seeds in `manifest.json`, and keep identity de-duplication stable when another control is requested.

- [ ] **Step 8: Verify C0 tests and CLI help**

Run: `python -m pytest tests/test_probe_controls.py -k "c0 or separate_family or baseline" -v`

Run: `python scripts/evaluate_probe_transform_controls.py --help`

Expected: PASS.

- [ ] **Step 9: Commit Task 4**

```bash
git add simdiff_eval/probe_controls.py scripts/evaluate_probe_transform_controls.py tests/test_probe_controls.py
git commit -m "feat: add C0 symmetry probe control"
```

---

### Task 5: Implement C4 measured-transfer and Gaussian degraded-real controls

**Files:**
- Modify: `simdiff_eval/probe_controls.py`
- Create: `scripts/evaluate_probe_degradation_control.py`
- Modify: `tests/test_probe_controls.py`

**Interfaces:**
- Consumes: held-out loader, `batch_power_spectra`, `field_histogram`, `transfer_transform`, existing generated-NPZ path helpers, frozen encoder, row evaluator, aggregator, and manifest builder.
- Produces: deterministic split, measured transfer, Gaussian fit, generated subset helper, C4 CLI, long predictions, power/PDF/summary JSON, and limitation metadata.

- [ ] **Step 1: Write failing deterministic split and Gaussian-fit tests**

Append:

```python
def test_c4_split_is_deterministic_disjoint_and_complete():
    from simdiff_eval.probe_controls import deterministic_cosmology_split

    heldout = np.arange(900, 932)
    derive_a, evaluate_a = deterministic_cosmology_split(heldout, seed=41)
    derive_b, evaluate_b = deterministic_cosmology_split(heldout, seed=41)
    np.testing.assert_array_equal(derive_a, derive_b)
    np.testing.assert_array_equal(evaluate_a, evaluate_b)
    assert len(derive_a) == len(evaluate_a) == 16
    assert set(derive_a).isdisjoint(evaluate_a)
    assert set(derive_a) | set(evaluate_a) == set(heldout)


def test_gaussian_fit_recovers_known_smoothing_scale():
    from simdiff_eval.probe_controls import fit_gaussian_smoothing

    k = np.linspace(1.0, 20.0, 25)
    expected_sigma = 0.08
    ratio = np.exp(-(expected_sigma * k) ** 2)
    fitted = fit_gaussian_smoothing(k, ratio)
    assert fitted == pytest.approx(expected_sigma, rel=1e-3)
```

- [ ] **Step 2: Run them and verify missing-function failures**

Run: `python -m pytest tests/test_probe_controls.py -k "c4_split or gaussian_fit" -v`

Expected: FAIL because both helpers are missing.

- [ ] **Step 3: Implement split, ratio, Gaussian fit, and subset helpers**

`deterministic_cosmology_split` validates unique one-dimensional indices, requires an even count, permutes with the recorded seed, and returns sorted halves. `fit_gaussian_smoothing` uses finite positive ratios and non-zero k bins:

```python
log_ratio = np.log(np.clip(ratio, 1.0e-30, None))
k2 = k**2
sigma_squared = max(0.0, -float(np.dot(k2, log_ratio)) / float(np.dot(k2, k2)))
return float(np.sqrt(sigma_squared))
```

Add `power_ratio_transfer(real_images, generated_images, nbins)` using the existing `batch_power_spectra` and returning centers, mean spectra, ratio, and `sqrt(clip(ratio,0,inf))`. Add `subset_generated_cosmologies` that validates NPZ ordering (`len(samples) == len(heldout_indices) * samples_per_cosmology`) and returns samples/targets/index arrays for a requested simulation set.

Run the Step 2 command again. Expected: PASS.

- [ ] **Step 4: Write failing measured-transfer and limitation tests**

Append tests that build random synthetic images, derive `k, ratio, transfer = power_ratio_transfer(real, generated, nbins=8)`, apply the public Task 2 transfer factory, and assert finite real float32 output:

```python
def test_measured_transfer_builds_finite_real_degraded_maps():
    from simdiff_eval.probe_controls import power_ratio_transfer
    from simdiff_eval.probe_transforms import transfer_transform

    rng = np.random.default_rng(51)
    real = rng.normal(size=(6, 1, 16, 16)).astype(np.float32)
    generated = (0.7 * real + 0.1 * rng.normal(size=real.shape)).astype(np.float32)
    k, real_mean, generated_mean, ratio, transfer = power_ratio_transfer(
        real, generated, nbins=8
    )
    degraded, diagnostics = transfer_transform(k, transfer)(real)
    assert degraded.shape == real.shape
    assert degraded.dtype == np.float32
    assert not np.iscomplexobj(degraded)
    assert np.isfinite(degraded).all()
    assert np.isfinite(ratio).all()
    assert diagnostics["out_of_range_fraction"] >= 0.0
```

Also assert:

```python
from simdiff_eval.probe_controls import C4_LIMITATION
assert "two-point" in C4_LIMITATION
assert "one-point PDF" in C4_LIMITATION
assert "higher-order" in C4_LIMITATION
assert "only" in C4_LIMITATION
```

- [ ] **Step 5: Run those tests and verify the missing constant/helper behavior**

Run: `python -m pytest tests/test_probe_controls.py -k "measured_transfer or limitation" -v`

Expected: FAIL until the helper return contract and limitation are complete.

- [ ] **Step 6: Implement the C4 CLI without executing it on artifacts**

Create `scripts/evaluate_probe_degradation_control.py` with arguments matching the existing generated evaluator where practical: project directory, run manifest, repeatable run name, VGG encoder/device, samples per cosmology, embedding batch size, power bins, bootstrap count/seed, split seed, sample seed, and output directory.

For each selected run:

1. Resolve the existing NPZ through `output_path_for` and load `samples`, `theta_raw`, `heldout_indices`, and `samples_per_cosmology`.
2. Restrict derivation spectra to derivation-half real and generated maps.
3. Build measured and Gaussian transfer transforms.
4. Evaluate identity real, measured-transfer real, Gaussian real, and generated evaluation-half inputs with the frozen encoder.
5. Add `dataset_size`, `source`, and `run_name` columns while preserving all shared required columns.
6. Write `probe_degradation_predictions.csv`, `probe_degradation_metrics.json`, `power_transfer_curves.json`, `field_histograms.json`, and `manifest.json`.

The power JSON contains centers, real/generated means, ratio, measured transfer, fitted sigma, and Gaussian transfer. The histogram JSON calls existing `field_histogram` for original evaluation reals, both degraded-real variants, and generated evaluation maps. Both the summary and manifest include `C4_LIMITATION` unchanged. Record derivation/evaluation simulation lists and every seed.

- [ ] **Step 7: Add an import/help smoke test and verify all C4 helpers**

Add a subprocess test that runs `python scripts/evaluate_probe_degradation_control.py --help` and asserts return code zero.

Run: `python -m pytest tests/test_probe_controls.py -k "c4 or gaussian or measured_transfer or limitation or degradation" -v`

Expected: PASS without loading CAMELS or generated NPZs.

- [ ] **Step 8: Commit Task 5**

```bash
git add simdiff_eval/probe_controls.py scripts/evaluate_probe_degradation_control.py tests/test_probe_controls.py
git commit -m "feat: add C4 degraded real probe control"
```

---

### Task 6: Implement C1 scale-cut suites and three-curve reports

**Files:**
- Modify: `simdiff_eval/probe_controls.py`
- Modify: `scripts/evaluate_probe_transform_controls.py`
- Modify: `tests/test_probe_controls.py`

**Interfaces:**
- Consumes: spectral transforms, long prediction table, and shared aggregation.
- Produces: `DEFAULT_K_CUTS`, `build_c1_specs`, and `c1_scale_cut_summary` wired into the shared CLI.

- [ ] **Step 1: Write the failing C1 suite-completeness test**

Append:

```python
def test_c1_suite_has_all_required_arms_windows_and_cutoffs():
    from simdiff_eval.probe_controls import DEFAULT_K_CUTS, build_c1_specs

    assert DEFAULT_K_CUTS == (4.0, 6.0, 8.0, 12.0, 16.0, 24.0, 32.0, 40.0, 52.0, 64.0)
    specs = build_c1_specs(DEFAULT_K_CUTS)
    names = [spec.name for spec in specs]
    assert names.count("identity") == 1
    assert names.count("fft_roundtrip_null") == 1
    for k_cut in DEFAULT_K_CUTS:
        label = f"{k_cut:g}"
        for arm in ("lowpass", "highpass"):
            for window in ("sharp", "hann"):
                assert f"{arm}_kcut{label}_{window}" in names
    assert len(specs) == 42
```

- [ ] **Step 2: Run it and verify the missing-builder failure**

Run: `python -m pytest tests/test_probe_controls.py::test_c1_suite_has_all_required_arms_windows_and_cutoffs -v`

Expected: FAIL because `build_c1_specs` is missing.

- [ ] **Step 3: Implement C1 specs with explicit descriptor metadata**

Create identity and round-trip-null specs, then both arms and windows for all ten cutoffs. Set families to `identity`, `fft_roundtrip_null`, `lowpass`, or `highpass`; record `k_cut` and `window`; ensure name formatting matches the test and transform resolver.

- [ ] **Step 4: Write the failing three-curve/two-grain summary test**

Build a small prediction table using identity, one low-pass, and one high-pass transform. Call `c1_scale_cut_summary(table, n_boot=50, seed=47)` and assert every row contains `rmse`, `bias`, `slope`, all six CI endpoints, `grain`, `transform_family`, `k_cut`, `k_cut_over_knyq`, `window`, and `out_of_range_fraction`; assert both grains and all three families are present.

- [ ] **Step 5: Run it and verify the missing-summary failure**

Run: `python -m pytest tests/test_probe_controls.py -k "three_curve and two_grain" -v`

Expected: FAIL because `c1_scale_cut_summary` is missing.

- [ ] **Step 6: Implement C1 summary and wire it into the CLI**

Reuse `aggregate_prediction_table`, retain only C1 descriptor groups, and attach mean `out_of_range_fraction` from the prediction rows to each curve row. Expand repeatable `--control` choices to `identity`, `c0`, and `c1`, using `build_c1_specs(args.k_cut or DEFAULT_K_CUTS)` for C1. Write `c1_scale_cut_summary.json`, including the explicit note that filters act after log/tanh normalization and before the VGG clamp.

- [ ] **Step 7: Verify C1 and all focused probe tests**

Run: `python -m pytest tests/test_probe_transforms.py tests/test_probe_controls.py -v`

Expected: PASS.

- [ ] **Step 8: Commit Task 6**

```bash
git add simdiff_eval/probe_controls.py scripts/evaluate_probe_transform_controls.py tests/test_probe_controls.py
git commit -m "feat: add C1 scale cut probe control"
```

---

### Task 7: Final contract checks and complete verification

**Files:**
- Modify if a failing contract requires correction: `simdiff_eval/probe_eval.py`
- Modify if a failing contract requires correction: `simdiff_eval/probe_transforms.py`
- Modify if a failing contract requires correction: `simdiff_eval/probe_controls.py`
- Modify if a failing contract requires correction: `scripts/evaluate_probe_transform_controls.py`
- Modify if a failing contract requires correction: `scripts/evaluate_probe_degradation_control.py`
- Modify if a failing contract requires correction: `tests/test_probe_transforms.py`
- Modify if a failing contract requires correction: `tests/test_probe_controls.py`

**Interfaces:**
- Consumes: all completed tasks.
- Produces: a clean, locally verified branch ready for the user to run later on the real data system.

- [ ] **Step 1: Run static source guards for hard constraints**

Run:

```bash
python - <<'PY'
from pathlib import Path

transform_source = Path("simdiff_eval/probe_transforms.py").read_text()
assert "import torch" not in transform_source
assert "torch." not in transform_source
assert "irfft2" in transform_source
assert ".real" not in transform_source

script_source = "\n".join(
    Path(path).read_text()
    for path in (
        "scripts/evaluate_probe_transform_controls.py",
        "scripts/evaluate_probe_degradation_control.py",
    )
)
for forbidden in ("sbatch", ".fit(", "vgg16("):
    assert forbidden not in script_source
print("hard-constraint source guards passed")
PY
```

Expected: prints `hard-constraint source guards passed`.

- [ ] **Step 2: Run focused tests with warnings treated as errors**

Run: `python -m pytest tests/test_probe_transforms.py tests/test_probe_controls.py -W error -v`

Expected: all new tests PASS with no warnings.

- [ ] **Step 3: Run both CLI help commands**

Run: `python scripts/evaluate_probe_transform_controls.py --help`

Run: `python scripts/evaluate_probe_degradation_control.py --help`

Expected: both commands exit 0 without loading data, VGG weights, or pickles.

- [ ] **Step 4: Run the complete repository test suite**

Run: `python -m pytest -q`

Expected: all baseline 105 tests plus the new probe tests PASS. The two pre-existing Torch `FutureWarning` messages may remain; no new warning is acceptable.

- [ ] **Step 5: Check formatting, diff scope, and repository status**

Run: `git diff --check`

Run: `git status --short`

Run: `git log --oneline --decorate -8`

Expected: no whitespace errors; changes are limited to the approved design/plan, two new `simdiff_eval` modules plus loader/controls, two new evaluation scripts, the one trainer refactor, and two new test files.

- [ ] **Step 6: Commit any final test-driven corrections**

If Step 1 through Step 5 required a correction, first add a regression assertion that fails, then make the minimum correction and commit only those files:

```bash
git add simdiff_eval scripts/evaluate_probe_transform_controls.py scripts/evaluate_probe_degradation_control.py scripts/train_nf_conditional_bias_vgg_encoder.py tests/test_probe_transforms.py tests/test_probe_controls.py
git commit -m "test: finalize probe control contracts"
```

If no correction was required, do not create an empty commit.
