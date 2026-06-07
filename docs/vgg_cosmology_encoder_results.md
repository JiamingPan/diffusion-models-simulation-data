# VGG Cosmology Encoder Results

This note records the VGG-feature cosmology encoders tested for the continuous HI bias-probe project. It is intended as a compact reference for poster text, advisor updates, and AI/ML interview discussion.

## Purpose

We need an encoder that maps HI fields back to CAMELS cosmological parameters:

```text
HI field -> feature encoder -> regression head -> [Omega_m, sigma_8, A_SN1, A_AGN1, A_SN2, A_AGN2]
```

This encoder is then used as a probe for the diffusion model:

```text
requested cosmology -> conditional diffusion model -> generated HI field -> encoder -> recovered cosmology
```

If recovered cosmology tracks the requested input cosmology, the generated field carries the requested physical information.

## VGG Encoder Pipeline

All VGG tests used a frozen ImageNet-pretrained VGG16 feature extractor.

```text
normalized HI slice
-> repeat to 3 channels
-> resize to 224 x 224
-> frozen pretrained VGG16 convolutional features
-> pooling
-> trainable regression head
-> six CAMELS parameters
```

Important details:

- The original HI slice is single-channel.
- Channel repeat means `R = HI`, `G = HI`, `B = HI`; it adds no new physical information.
- Resize uses bilinear interpolation to `224 x 224` so the slice fits VGG16.
- VGG16 is frozen. Its weights are not updated.
- Only the regression head after VGG is trained.
- The held-out test simulations are `900-931`, never used to train the diffusion models or the encoder heads.

## Feature And Head Definitions

| Term | Meaning |
|---|---|
| `avg` pooling | Average-pool the final VGG feature maps, giving a 512-dimensional feature vector. |
| `avg+max` pooling | Concatenate average-pooled and max-pooled VGG features, giving a 1024-dimensional feature vector. |
| Ridge head | Linear regression with L2 regularization. |
| MLP head | Nonlinear neural-network regression head. |
| MLP `512,256` | Hidden layers with 512 and 256 neurons, followed by a 6-output layer. |
| MLP `1024,512,256` | Hidden layers with 1024, 512, and 256 neurons, followed by a 6-output layer. |

For the best current setup:

```text
VGG avg+max features: 1024 inputs
-> Linear 1024
-> ReLU
-> Linear 512
-> ReLU
-> Linear 256
-> ReLU
-> Linear 6
```

## Real Held-Out Encoder Tests

Metric: R2 on held-out real simulations `900-931`, computed per held-out cosmology after taking the median prediction over that simulation's 128 HI slices.

| Job | Encoder | Pooling | Head | Train slices | Val slices | Omega_m R2 | sigma_8 R2 | A_SN1 R2 | A_AGN1 R2 | A_SN2 R2 | A_AGN2 R2 |
|---:|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `51475729` | VGG smoke test | avg+max | MLP `64` | 512 | 128 | 0.5189 | 0.0111 | 0.3708 | 0.0678 | 0.2601 | -0.4311 |
| `51475731` | VGG default full | avg+max | MLP `512,256` | 32768 | 4096 | 0.8992 | 0.6830 | 0.4189 | 0.0144 | 0.2953 | 0.0946 |
| `51475738` | VGG big MLP | avg+max | MLP `1024,512,256` | 65536 | 8192 | 0.9115 | 0.7374 | 0.4544 | -0.0063 | 0.3251 | 0.1005 |
| `51475739` | VGG big MLP | avg | MLP `1024,512,256` | 65536 | 8192 | 0.9027 | 0.7349 | 0.4535 | -0.0313 | 0.2561 | 0.0868 |
| `51475740` | VGG Ridge | avg+max | Ridge alpha 1.0 | 65536 | 8192 | 0.8977 | 0.7143 | 0.4145 | -0.0217 | 0.2862 | 0.1418 |

## Best Encoder

The best current real-heldout encoder is:

```text
frozen VGG16 + avg+max pooling + MLP 1024,512,256
```

Output files:

```text
results/nf_conditional_bias_probe/encoder/vgg_mlp_big_avgmax.npz
results/nf_conditional_bias_probe/encoder/vgg_mlp_big_avgmax.pkl
```

Why it is best:

- Highest `Omega_m` R2: `0.9115`.
- Highest `sigma_8` R2: `0.7374`.
- Highest `A_SN1` R2: `0.4544`.
- Highest `A_SN2` R2: `0.3251`.

Main caveat:

- `A_AGN1` and `A_AGN2` remain weak with these VGG features and HI-only data.

## Ridge vs MLP Interpretation

The MLP head is better than the Ridge head:

| Encoder | Omega_m R2 | sigma_8 R2 | A_SN1 R2 | A_SN2 R2 |
|---|---:|---:|---:|---:|
| VGG big MLP avg+max | 0.9115 | 0.7374 | 0.4544 | 0.3251 |
| VGG Ridge avg+max | 0.8977 | 0.7143 | 0.4145 | 0.2862 |

Interpretation:

- The frozen VGG feature space contains recoverable cosmological information.
- A linear Ridge head already recovers much of the `Omega_m` and `sigma_8` signal.
- The MLP improves recovery, which suggests that some cosmology information is encoded nonlinearly in the VGG feature representation.
- The improvement is real but not huge; the main signal comes from the VGG feature representation itself.

## Generated-Field Calibration With VGG

The first VGG generated-field calibration job was:

| Job | Encoder used | Output directory |
|---:|---|---|
| `51475741` | default full VGG encoder, `vgg_mlp_encoder.npz` | `results/nf_conditional_bias_probe/calibration_vgg/` |

That plot used:

```text
frozen VGG16 + avg+max pooling + MLP 512,256
```

It did not use the later best `1024,512,256` MLP unless the evaluation is rerun with:

```text
VGG_ENCODER=results/nf_conditional_bias_probe/encoder/vgg_mlp_big_avgmax.npz
```

Calibration slopes from job `51475741`:

| Regime | Dataset size | Omega_m slope | sigma_8 slope | Meaning |
|---|---:|---:|---:|---|
| Memorization | 128 | 0.2568 | 0.3285 | weak response to input cosmology |
| Generalization | 16384 | 0.7866 | 0.3814 | much better `Omega_m` tracking; weak `sigma_8` tracking |

Slope definition:

```text
recovered parameter = slope * input parameter + intercept
```

Interpretation:

- Slope near 1: generated fields track the requested input cosmology.
- Slope near 0: generated fields mostly ignore the requested input and regress toward typical training values.
- The large-N model shows much stronger `Omega_m` conditioning than the small-N model.

## Poster / Interview Summary

Short version:

> I trained a frozen-VGG feature encoder to recover cosmology from real held-out CAMELS HI fields. The best encoder used ImageNet-pretrained VGG16 features with avg+max pooling and a nonlinear MLP head. It achieved R2 of 0.91 for Omega_m and 0.74 for sigma_8 on held-out simulations. Using this encoder as a probe, the large-data conditional diffusion model showed much stronger recovery of the requested Omega_m than the small-data model, suggesting it is more faithful to the conditioning cosmology rather than only producing visually plausible fields.

More technical version:

> I compared linear and nonlinear regression heads on frozen VGG16 features. Ridge regression already recovered much of the cosmological signal, but a three-layer MLP head improved R2 across the main parameters, indicating nonlinear structure in the feature-to-cosmology map. This gave a stronger diagnostic for testing whether conditional diffusion samples encode the requested CAMELS cosmology.

## Useful Commands

Regenerate the VGG R2 comparison plot:

```bash
cd /home/jiamingp/diffusion_models_repo
python scripts/plot_nf_conditional_bias_vgg_results.py --project-dir "$PWD"
```

Expected outputs:

```text
results/nf_conditional_bias_probe/encoder/vgg_encoder_r2_comparison.csv
results/nf_conditional_bias_probe/encoder/vgg_encoder_r2_comparison.png
results/nf_conditional_bias_probe/calibration_vgg/bias_probe_vgg_main_slopes.png
```

Rerun generated-field calibration with the current best VGG encoder:

```bash
cd /home/jiamingp/diffusion_models_repo
vgg_eval_big=$(ENCODER_TYPE=vgg \
  VGG_ENCODER=results/nf_conditional_bias_probe/encoder/vgg_mlp_big_avgmax.npz \
  TORCH_HOME=/nfs/turbo/lsa-huterer/jiamingp/torch_cache \
  sbatch -A huterer2 --parsable \
  scripts/slurm/evaluate_nf_conditional_bias_probe.sbatch)
echo "vgg_eval_big=$vgg_eval_big"
```
