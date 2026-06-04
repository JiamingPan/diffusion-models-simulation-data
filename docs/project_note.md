# Project Note: CAMELS Diffusion Model Evaluation

This note summarizes the current diffusion-model project details in one place.
More specialized notes live in:

- `docs/continuous_cosmology_bias_probe_notes.md`
- `docs/sscd_generalizability_note.md`
- `docs/nick_meeting_notes.md`

## Project Goal

We train diffusion models on CAMELS cosmological simulation fields and ask two
questions:

1. When does the model memorize training examples instead of generating new,
   statistically valid fields?
2. For conditional models, does the generated field actually reflect the
   requested input cosmology?

The important evaluation principle is that good-looking samples are not enough.
We also need statistics, nearest-neighbor checks, and conditional calibration.

## Data Representation

CAMELS grid files are 3D simulation volumes. A typical file is:

```text
Grids_HI_IllustrisTNG_LH_128_z=0.0.npy
```

The raw array has shape like:

```text
(N_sim, 128, 128, 128)
```

The training config converts 3D volumes into 2D slices:

```yaml
data:
  reshape: 2d
  zthin: 8
```

With `zthin=8`, each loaded 3D simulation contributes:

```text
128 / 8 = 16 2D training slices
```

So in these runs:

```text
number of 2D training images = number of loaded simulations * 16
```

Important distinction:

- `data.n_samples` in the YAML = number of 3D simulations/volumes loaded.
- `dataset_size` or plotted `N` = number of materialized 2D training images.

For the Fig. 2-style sweep:

```text
N = 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768
```

These are numbers of 2D slices, not numbers of CAMELS simulations.

## Log Transform And Normalization

CAMELS fields are highly skewed: most pixels are low density, while a small
number of pixels are very bright. The raw field values therefore have a large
dynamic range.

The current Fig. 2 and conditional HI runs use:

```yaml
data:
  transform:
    - log
  normalization: tanh
  norm_kwargs:
    alpha: 0.8
    beta: 10.0
    delta: 1.0
    gamma: 1.0
    sigma: 1.5
```

Conceptually:

```text
raw HI field -> log transform -> tanh normalization -> diffusion model
```

The log transform makes the pixel-value distribution smoother and easier for
the diffusion model to learn. Without it, the model can spend too much capacity
on rare extreme pixels.

## Fig. 2-Style Training Sweep

The Fig. 2 sweep trains separate unconditional HI diffusion models at different
training-set sizes and different UNet widths:

```text
u64, u128, u256
```

The config generator is:

```text
scripts/prepare_nf_generalize_fig2_configs.py
```

The main run pattern is:

```text
nf_fig2_<arch>_<dataset_tag>_noaug_200k
```

Example:

```text
nf_fig2_u128_d2p14_noaug_200k
```

This means:

- architecture: `u128`
- training size: `d2p14 = 2^14 = 16,384` 2D slices
- no data augmentation
- about 200k optimizer updates in the initial run

The u256 continuation runs add more optimizer updates to test whether the
apparent u256 transition is limited by undertraining.

## Reproducibility And Generalization Metrics

Let \( \phi(x) \) be a feature embedding of image \(x\), either PCA or SSCD.

Similarity:

\[
s(x,y) = \cos(\phi(x), \phi(y)).
\]

The adaptive threshold is calibrated from real training fields:

\[
\tau_{95}
= Q_{0.95}\left[
\max_{i \ne j} s(y_i, y_j)
\right].
\]

This is intentionally not chosen from generated samples, because that would
hide memorization.

### Reproducibility

Reproducibility compares generated sets from independently trained models. For
two architectures \(A\) and \(B\):

\[
\mathrm{RP}(A,B)
=
\frac{1}{|G_A|}
\sum_{g \in G_A}
\mathbf{1}
\left[
\max_{h \in G_B} s(g,h) > \tau_{95}
\right].
\]

In the plots, this corresponds to comparisons like:

```text
UNet-64 generated samples  vs  UNet-128 generated samples
UNet-64 generated samples  vs  UNet-256 generated samples
UNet-128 generated samples vs  UNet-256 generated samples
```

High reproducibility means independently trained models generate the same kinds
of samples.

### Generalization

Generalization compares generated samples to the training set:

\[
\mathrm{GL}(A)
=
1 -
\frac{1}{|G_A|}
\sum_{g \in G_A}
\mathbf{1}
\left[
\max_{y \in \mathrm{Train}} s(g,y) > \tau_{95}
\right].
\]

Interpretation:

- low GL: many generated samples are too close to training examples;
- high GL: generated samples are not near-copies of training examples.

## Pixel Nearest-Training Visual Check

The quickcheck notebook also plots generated samples next to their closest
training slice in pixel space. This is a sanity check, not the final metric.

For each generated field \(g\), it finds:

\[
y^\star = \arg\min_{y \in \mathrm{Train}} \mathrm{MSE}(g,y).
\]

The figure shows:

```text
generated sample
closest training sample
absolute difference
```

The cosine value in that plot is the pixel-space cosine similarity between the
generated image and the nearest training image.

## One-Point PDF And Power Spectrum Checks

The one-point PDF checks whether the generated pixel-value distribution matches
real fields.

The lower P(k) panels in the quickcheck figures are power-spectrum ratios.
For one image:

\[
P(k) =
\left\langle
|\mathrm{FFT}(x)|^2
\right\rangle_{\mathrm{modes\ in\ bin}\ k}.
\]

The plotted ratio is:

\[
\frac{\langle P_{\mathrm{generated}}(k) \rangle}
     {\langle P_{\mathrm{real}}(k) \rangle}.
\]

Here the average is over all generated samples from that one model/run and all
loaded real reference samples. It is not averaged over different UNet
architectures.

Interpretation:

- ratio \(= 1\): generated and real fields match at that spatial scale;
- ratio \(< 1\): generated fields are underpowered / too smooth at that scale;
- ratio \(> 1\): generated fields have excess structure at that scale.

The x-axis `k bin` is a radial Fourier-frequency bin on the 2D image grid:

- small `k` bin: large-scale structure;
- large `k` bin: fine-scale structure.

## Conditional Cosmology Bias Probe

The continuous-conditioning experiment asks whether an HI diffusion model follows
the requested CAMELS cosmology.

Main setup:

```text
input cosmology theta
-> conditional diffusion model
-> generated HI field
-> frozen PCA + Ridge encoder
-> recovered theta
```

Main no-CFG comparison:

| Regime | Dataset size | Meaning |
|---|---:|---|
| Memorization | `N=128` | 128 materialized 2D HI fields |
| Generalization | `N=16,384` | 16,384 materialized 2D HI fields |

The model is `UNet2DConditionModel` with cross-attention. The six CAMELS
parameters are normalized and passed as:

```text
theta_norm: shape (B, 6)
encoder_hidden_states: shape (B, 1, 6)
```

The held-out cosmologies are simulations `900-931`, 32 simulations total. They
are excluded from:

- diffusion-model training,
- PCA fitting,
- Ridge-head training,
- encoder validation/training split.

For each held-out cosmology, we generate `K=64` samples with different diffusion
noise seeds and the same input cosmology. Each generated field is encoded back
to parameters. The plotted calibration point is:

\[
\mathrm{median}(\theta_{\mathrm{rec}})
\]

and the vertical error bar is:

\[
[Q_{16}(\theta_{\mathrm{rec}}), Q_{84}(\theta_{\mathrm{rec}})].
\]

So the error bar measures generated-sample spread at fixed input cosmology. It
is not the encoder validation error.

The detailed exact setup is recorded in:

```text
docs/continuous_cosmology_bias_probe_notes.md
```

## Current Resume Bullets

Suggested AI/ML-facing version:

- Built a PyTorch diffusion training pipeline for CAMELS 3D simulation fields,
  converting raw volumes into normalized 2D slices with configurable log
  transforms, U-Net sizes, noise schedules, EMA, and SLURM workflows.
- Implemented DPM-Solver multistep inference, reducing 512-sample generation
  wall time by roughly `8x` versus a 500-step DDPM baseline while preserving
  one-point statistics and power-spectrum diagnostics.
- Developed memorization/generalization diagnostics using PCA/SSCD embeddings,
  nearest-neighbor similarity, reproducibility tests across independently
  trained models, and generated-vs-training copy checks.
- Added continuous conditional generation on cosmology parameters and evaluated
  calibration with a frozen real-data PCA + Ridge probe, testing whether
  generated HI fields recover the requested input parameters in low- vs
  high-data regimes.

Shorter three-bullet version:

- Built a PyTorch diffusion pipeline for CAMELS 3D scientific simulation data,
  including volume-to-slice preprocessing, log/tanh normalization, configurable
  U-Net architectures, noise schedules, EMA, and SLURM workflows.
- Implemented DPM-Solver multistep sampling, cutting 512-sample inference time
  by about `8x` relative to a 500-step DDPM baseline while preserving one-point
  and power-spectrum diagnostics.
- Developed PCA/SSCD-based evaluation tools for memorization, reproducibility,
  generalization, and conditional calibration, including a frozen real-data
  PCA + Ridge probe for recovering requested cosmology parameters.
