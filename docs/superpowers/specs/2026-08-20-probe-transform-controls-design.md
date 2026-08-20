# Frozen VGG Probe Transform Controls Design

**Date:** 2026-08-20

## Goal

Build tested, reusable input-only validation controls for the frozen VGG16 +
MLP cosmology probe. The controls establish whether recovered cosmological
parameters depend on large-scale structure, small-scale texture, or measured
generated-map power deficits. They must not retrain the probe, retrain a
diffusion model, generate samples, or submit compute jobs.

## Scope and sequencing

Implementation follows this order:

1. Extract held-out real-map loading and refactor the VGG trainer to use it.
2. Add the pure NumPy transform registry.
3. Add the shared evaluation harness and manifest/output contracts.
4. Add C0 symmetry reporting.
5. Add C4 degraded-real evaluation.
6. Add C1 scale-cut reporting.

Only simulations 900 through 931 inclusive are valid for real-map evaluation.
With 128 z-slices per simulation, the complete real evaluation set contains
4096 slices. The implementation stops after local code and synthetic tests are
green; it does not run the controls against CAMELS or generated artifacts.

## Architecture

### Held-out real-map loading

`simdiff_eval/probe_eval.py` owns the reusable loader:

```python
def load_heldout_real_slices(
    data_root: str | Path,
    heldout_indices: np.ndarray,
    slices_per_sim: int,
    norm: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    ...
```

It composes the existing `select_slice_pairs`, `load_raw_slices`,
`preprocess_real_slices`, `image_path`, `params_path`, and `load_params`
functions. It returns normalized float32 images, raw six-parameter targets,
simulation indices, and z-indices in exactly the selection order currently
created in `train_nf_conditional_bias_vgg_encoder.py`.

The VGG trainer continues using its existing training and validation loading.
Only its held-out test branch changes: it calls the shared loader, embeds the
returned images, and reconstructs the same `(sim_index, z_index)` pair table for
the existing summarizer. A regression test compares the extracted pair order to
the old `select_slice_pairs(heldout, len(heldout) * slices_per_sim)` expression.

### Transform registry

`simdiff_eval/probe_transforms.py` contains no Torch imports and performs no
I/O. A transform is:

```python
Transform = Callable[[np.ndarray], tuple[np.ndarray, dict[str, float]]]
```

The input contract is float32 `(N, 1, H, W)`. Production controls use
`H = W = 128`; arbitrary square synthetic sizes remain supported for fast unit
tests. Each transform reports `out_of_range_fraction`, measured on the output
before the VGG probe clamps to `[-1, 1]`.

The public API contains small factories plus a named resolver:

```python
def get_transform(
    name: str,
    *,
    transfer_k: np.ndarray | None = None,
    transfer_values: np.ndarray | None = None,
) -> Transform:
    ...
```

Supported names are `identity`, `rot90_k1`, `rot90_k2`, `rot90_k3`,
`flip_h`, `flip_v`, `dihedral_g0` through `dihedral_g7`, arbitrary integer
`roll_dx{dx}_dy{dy}`, `lowpass_kcut{k}_{sharp|hann}`,
`highpass_kcut{k}_{sharp|hann}`, `fft_roundtrip_null`, and `transfer_Tk`.
Combined C0 views are built by function composition and receive stable names
that retain their dihedral element and roll offsets.

`dx` shifts the width axis and `dy` shifts the height axis. Dihedral elements
0 through 3 are rotations by 0, 90, 180, and 270 degrees; elements 4 through 7
apply a horizontal flip after the corresponding rotation. Transformed arrays
are contiguous so negative-stride NumPy views remain safe inputs to Torch.

### Fourier filters

Filters operate on the normalized probe input. They use `rfft2` and `irfft2`,
which preserve a real-valued output by construction. The radial grid uses
`fftfreq * N`, matching `simdiff_eval.metrics.radial_power_spectrum_2d`; the
half-spectrum width axis uses the corresponding `rfftfreq * N` coordinates.

For `k_cut < k_Nyquist`, the sharp low-pass mask is the radial top-hat and the
sharp high-pass mask is its exact complement. The Hann low-pass response tapers
from one to zero across four radial grid units centered on `k_cut`; its
high-pass response is `1 - lowpass`. Therefore low-pass plus high-pass
reconstructs the original within floating-point tolerance for both windows.

`k_Nyquist = min(H, W) / 2`. Although radial corner modes extend above the
per-axis Nyquist, the explicit null-control contract requires `k_cut = 64` to
be an all-pass on 128-pixel maps. Consequently any low-pass cutoff at or above
the per-axis Nyquist uses an all-one response, its complementary high-pass is
all-zero, and `fft_roundtrip_null` performs exactly one FFT/iFFT all-pass.

The `transfer_Tk` transform receives the radial-bin centers and transfer values.
It reconstructs the uniform edges used by `radial_power_spectrum_2d`, assigns a
constant transfer to every mode in each bin, preserves the DC mode with a
factor of one, and extends the nearest transfer value to unbinned radial corner
modes. A unit transfer is an FFT round-trip identity within tolerance.

### Shared control analysis

`simdiff_eval/probe_controls.py` provides pure or dependency-light helpers for:

- transform descriptors and C0 view construction;
- converting encoder predictions to required tidy rows;
- per-slice and per-cosmology aggregation;
- deterministic bootstrap intervals for RMSE, signed bias, and recovered-vs-
  true slope;
- C0 symmetry spread and baseline ratios;
- deterministic derive/evaluate cosmology splits;
- measured transfer and Gaussian-transfer fitting;
- manifest construction and SHA-256 calculation.

Encoder execution stays in scripts because it requires Torch/TorchVision and a
pickled scikit-learn model. Unit tests use a fake encoder and never need a GPU,
VGG weights, CAMELS files, or generated samples.

## Shared control harness

`scripts/evaluate_probe_transform_controls.py` loads the fixed held-out set,
loads the frozen VGG encoder through the existing `load_vgg_encoder`, evaluates
the requested C0/C1 transform suite, and writes one long-format prediction CSV.

Required columns are:

```text
transform, transform_family, k_cut, k_cut_over_knyq, window,
sim_index, z_index, parameter, theta_true, theta_pred,
out_of_range_fraction
```

Optional descriptor columns such as `dihedral_g`, `roll_dx`, and `roll_dy` are
included so downstream grouping never has to parse display names. Identity is
always present exactly once in every invocation.

All reports are derived from a groupby over this CSV. To preserve the one-CSV
contract, report tables are serialized as JSON rather than additional CSVs:

- `probe_transform_predictions.csv`: the sole long prediction table;
- `probe_transform_metrics.json`: per-slice and per-cosmology RMSE, signed
  bias, slope, and deterministic bootstrap intervals;
- `c0_symmetry_summary.json` when C0 is requested;
- `c1_scale_cut_summary.json` when C1 is requested;
- `manifest.json`.

The per-slice grain contains 4096 observations for a full transform. The
per-cosmology grain first takes the median prediction over 128 z-slices and
contains 32 observations. Both grains receive bootstrap intervals rather than
bare point estimates.

## C0 symmetry control

A fixed recorded seed draws four non-zero periodic roll offsets once. The view
set is the Cartesian product of eight dihedral elements and five roll states:
no roll plus the four recorded offsets, producing 40 views per slice.

The report keeps two effects separate:

- `dihedral`: standard deviation and max-minus-min over the eight no-roll
  dihedral views for each `(sim_index, z_index)`;
- `roll`: for each dihedral element, standard deviation and max-minus-min over
  the no-roll and four rolled views, followed by an explicit aggregation over
  dihedral elements.

The mandatory baseline is the identity prediction's within-simulation
slice-to-slice standard deviation and max-minus-min over 128 z-slices.
Transform spreads are divided by the corresponding simulation baseline.
Zero baseline spreads produce `NaN` ratios rather than infinities. The JSON
report retains separate dihedral and roll families and bootstrap summaries over
the 32 simulations.

## C1 scale-cut control

The default sweep uses ten fixed cutoffs spanning 4 through 64 grid units:

```text
4, 6, 8, 12, 16, 24, 32, 40, 52, 64
```

The suite includes identity, `fft_roundtrip_null`, low-pass and high-pass arms,
and sharp and Hann windows at every cutoff. Each descriptor records `k_cut`,
`k_cut / k_Nyquist`, and window. The summary reports RMSE, signed bias, and
slope curves with bootstrap intervals for both aggregation grains. Rising
`out_of_range_fraction` remains visible per cutoff so ringing/clipping can be
distinguished from information removal.

## C4 degraded-real control

`scripts/evaluate_probe_degradation_control.py` uses the same held-out loader,
encoder interface, tidy-row builder, aggregation helpers, and manifest format.
It accepts generated NPZ runs already present on disk; it never generates new
samples.

A recorded RNG seed permutes held-out simulations. The first 16 simulations
form the derivation half and the remaining 16 form the probe-evaluation half.
For each generated dataset size:

1. Compute mean real and generated spectra on the derivation cosmologies with
   the existing `batch_power_spectra`.
2. Form `R(k) = P_gen(k) / P_real(k)` and
   `T(k) = sqrt(clip(R(k), 0, inf))`.
3. Apply the piecewise-bin transfer to evaluation-half real maps.
4. Fit a non-negative Gaussian smoothing scale from
   `log R(k) ~= -sigma^2 k^2`, apply its amplitude transfer
   `exp(-0.5 * sigma^2 k^2)`, and retain both results even when they disagree.
5. Evaluate the frozen probe on measured-transfer reals, Gaussian-degraded
   reals, original evaluation reals, and evaluation-half generated maps.
6. Compute one-point summaries with the existing `field_histogram` for all
   sources.

Outputs are a long prediction CSV, JSON power/transfer curves, JSON field
histograms, JSON bootstrap summaries, and `manifest.json`. The manifest and
summary state this limitation verbatim in substance: matching a two-point
power deficit does not match the one-point PDF or higher-order structure, so a
negative result rules out only the measured two-point deficit as the
explanation.

## Reproducibility manifest

Every control invocation writes `manifest.json` containing at least:

- Git commit revision and dirty-state flag;
- absolute encoder `.npz` path and SHA-256;
- VGG head pickle path and SHA-256 when available;
- installed scikit-learn version;
- held-out indices and slices per simulation;
- transform names and complete numeric parameters;
- all RNG seeds, including roll, bootstrap, and split seeds;
- deterministic derive/evaluate split lists for C4;
- CLI arguments and output schema version.

The manifest is written before result handoff and fails loudly if the encoder
artifact cannot be hashed or the scikit-learn version cannot be determined.

## Validation strategy

Tests are written before production changes and each is observed failing for
the intended missing behavior. `tests/test_probe_transforms.py` covers:

- bitwise identity;
- four rotations returning the original exactly;
- horizontal and vertical flip involutions;
- inverse periodic rolls;
- the Nyquist all-pass null;
- low-pass plus high-pass reconstruction for both windows;
- strictly real FFT outputs;
- unit transfer identity;
- greater sharp-window ringing than Hann-window ringing on a synthetic edge;
- correct out-of-range diagnostics.

`tests/test_probe_controls.py` covers:

- exact legacy held-out pair ordering and target association;
- required tidy columns and identity inclusion;
- both aggregation grains and deterministic bootstrap output;
- 40 deterministic C0 views with recorded non-zero roll offsets;
- separate dihedral and roll reports plus within-simulation baseline ratios;
- deterministic, disjoint C4 halves;
- measured transfer construction and Gaussian fit behavior;
- complete manifest provenance, including scikit-learn and hashes;
- the explicit C4 interpretation limitation.

After focused red/green cycles, the complete existing and new test suite is
run. No test accesses CAMELS, downloads VGG weights, requires Torch, or submits
external work.

## Non-goals

- No VGG or MLP head fitting or fine-tuning.
- No diffusion training, sampling, checkpoint modification, or Slurm work.
- No phase-scramble, summary-statistic ladder, or cross-suite controls.
- No interpretation of real control results in this implementation pass.
