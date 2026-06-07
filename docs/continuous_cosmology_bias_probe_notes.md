# Continuous HI Cosmology Bias Probe Notes

This note records the exact setup used for the recent continuous-cosmology HI bias-probe runs.

## One-Sentence Summary

We trained two continuous-conditioning HI diffusion models, one with `N=128` 2D training fields and one with `N=16,384` 2D training fields, then tested whether generated HI fields at held-out cosmologies encode back to the requested input cosmology. The first calibration used a real-only frozen PCA + Ridge encoder; later checks added PCA + MLP and VGG16-feature + MLP encoders, with VGG16 features giving the strongest real-heldout recovery so far.

## Main Question

Does the continuous-conditioning diffusion model recover the input cosmology in the generalization regime, but regress toward the training distribution in the memorization regime?

Operational test:

1. Pick a held-out cosmology `theta_in`.
2. Generate `K=64` HI fields using the same `theta_in` but different diffusion noise seeds.
3. Encode each generated field back to cosmology parameters using a real-only frozen feature encoder plus a regression head.
4. Plot median recovered parameter vs input parameter.
5. Fit a line; slope near 1 means the generated fields track the input cosmology, while slope near 0 means poor conditional calibration / regression toward a typical training value.

## Exact Main Runs

The main no-CFG runs were:

| Regime | Run name | Dataset size | What `N` means | SLURM job |
|---|---:|---:|---|---:|
| Memorization | `nf_cond_bias_hi_u128_d2p07_n128_200k` | `128` | 128 materialized 2D HI fields | `51322618` |
| Generalization | `nf_cond_bias_hi_u128_d2p14_n16384_200k` | `16,384` | 16,384 materialized 2D HI fields | `51322619` |

Important: here `N` is the number of 2D slices/fields, not the number of CAMELS simulations.

Completed downstream jobs:

| Step | Job name | SLURM job | Status seen |
|---|---|---:|---|
| Encoder | `nf_bias_encoder` | `51322880` | completed |
| Sampling | `nf_bias_sample` | `51322881` | completed |
| Evaluation | `nf_bias_eval` | `51322882` | completed |

## Data

| Item | Value |
|---|---|
| Simulation suite | CAMELS CMD |
| Hydro model | IllustrisTNG |
| Split | LH |
| Field | HI only |
| Resolution | `128 x 128` 2D fields from `128^3` grids |
| Redshift | `z=0.0` |
| Raw HI grid path | `/scratch/huterer_root/huterer0/CAMELS/CMD/3d_grids/IllustrisTNG/Grids_HI_IllustrisTNG_LH_128_z=0.0.npy` |
| Raw parameter path | `/scratch/huterer_root/huterer0/CAMELS/CMD/3d_grids/IllustrisTNG/params_LH_IllustrisTNG.txt` |

Parameter order:

1. `Omega_m` = Omega matter
2. `sigma_8`
3. `A_SN1`
4. `A_AGN1`
5. `A_SN2`
6. `A_AGN2`

Expected CAMELS parameter ranges used for sanity checks:

| Parameter | Expected range |
|---|---:|
| `Omega_m` | `[0.1, 0.5]` |
| `sigma_8` | `[0.6, 1.0]` |
| `A_SN1` | `[0.25, 4.0]` |
| `A_AGN1` | `[0.25, 4.0]` |
| `A_SN2` | `[0.5, 2.0]` |
| `A_AGN2` | `[0.5, 2.0]` |

## Held-Out Cosmologies

Exact held-out simulations:

```text
900, 901, 902, 903, 904, 905, 906, 907,
908, 909, 910, 911, 912, 913, 914, 915,
916, 917, 918, 919, 920, 921, 922, 923,
924, 925, 926, 927, 928, 929, 930, 931
```

Exact values:

| Item | Value |
|---|---:|
| Held-out start index | `900` |
| Held-out count | `32` |
| Held-out simulation indices | `900-931` inclusive |
| Non-held-out simulations | `968` |
| Held-out cosmologies used in calibration | `32` |
| Generated samples per held-out cosmology | `K=64` |
| Generated samples per model for calibration | `32 x 64 = 2048` |
| Generated samples for both main models | `4096` |

The held-out cosmologies were excluded from:

- diffusion-model training,
- PCA fitting,
- Ridge-head training,
- encoder validation/training split.

## Image And Parameter Normalization

Image preprocessing:

1. Raw HI field.
2. Log transform with floor behavior equivalent to `log(max(x, 1e-30))` in the encoder path.
3. Tanh normalization.

Tanh normalization hyperparameters:

| Key | Value |
|---|---:|
| `alpha` | `0.8` |
| `beta` | `10.0` |
| `delta` | `1.0` |
| `gamma` | `1.0` |
| `sigma` | `1.5` |

The image-normalization `center` and `xmax` were fit once from `4096` non-held-out real HI slices.

Parameter normalization:

- Mean/std were fit on the `968` non-held-out simulations.
- The six-dimensional normalized parameter vector is used for continuous conditioning.
- Raw parameters are saved too, so plots are in physical/raw parameter values.

Great Lakes files with exact fitted normalization values:

```text
/home/jiamingp/diffusion_models_repo/local/nf_conditional_bias_probe/heldout/shared_image_norm_stats.json
/home/jiamingp/diffusion_models_repo/local/nf_conditional_bias_probe/heldout/param_norm_stats.json
```

To print them:

```bash
cd /home/jiamingp/diffusion_models_repo
python - <<'PY'
import json
from pathlib import Path
root = Path("local/nf_conditional_bias_probe/heldout")
print("image normalization")
print(json.dumps(json.loads((root / "shared_image_norm_stats.json").read_text()), indent=2))
print("parameter normalization")
print(json.dumps(json.loads((root / "param_norm_stats.json").read_text()), indent=2))
PY
```

## How The Exact 2D Training Fields Were Selected

The training arrays were materialized before training. The selected 2D fields are exact `(simulation_index, z_index)` pairs.

Selection rule:

- remove held-out simulations `900-931`,
- flatten the remaining `(sim, z)` pairs with `z_size=128`,
- use evenly spaced indices over the flattened list via `np.linspace`,
- materialize exactly `N` 2D fields.

Selected-pair CSVs:

```text
local/nf_conditional_bias_probe/labels/nf_cond_bias_hi_u128_d2p07_n128_200k_selected_slices.csv
local/nf_conditional_bias_probe/labels/nf_cond_bias_hi_u128_d2p14_n16384_200k_selected_slices.csv
```

Prepared image arrays on scratch:

```text
/scratch/huterer_root/huterer0/jiamingp/saved_runs/nf_conditional_bias_probe/prepared_data/nf_cond_bias_hi_u128_d2p07_n128_200k_train_images.npy
/scratch/huterer_root/huterer0/jiamingp/saved_runs/nf_conditional_bias_probe/prepared_data/nf_cond_bias_hi_u128_d2p14_n16384_200k_train_images.npy
```

## Diffusion Model Architecture

Both main models used the same architecture:

| Item | Value |
|---|---|
| Model class | `UNet2DConditionModel` |
| Conditioning type | continuous |
| Input field | HI |
| `sample_size` | `128` |
| `in_channels` | `1` |
| `out_channels` | `1` |
| `layers_per_block` | `2` |
| `block_out_channels` | `[32, 64, 128]` |
| Down blocks | `DownBlock2D`, `DownBlock2D`, `CrossAttnDownBlock2D` |
| Up blocks | `CrossAttnUpBlock2D`, `UpBlock2D`, `UpBlock2D` |
| `norm_num_groups` | `32` |
| `cross_attention_dim` | `32` |
| `encoder_hid_dim` | `6` |

How conditioning enters:

```text
theta_norm:          shape (B, 6)
encoder_hidden_states: shape (B, 1, 6)
```

So this is not class conditioning. The six CAMELS parameters are passed through the Hugging Face conditional UNet cross-attention pathway.

## Training Objective And Scheduler

Training noise scheduler:

| Item | Value |
|---|---|
| Scheduler class | `DDPMScheduler` |
| Training timesteps | `500` |
| Beta schedule | `squaredcos_cap_v2` |
| `rescale_betas_zero_snr` | `True` |
| Prediction type | `v_prediction` |
| `clip_sample` | `False` |
| `thresholding` | `False` |
| `sample_max_value` | `2.0` |

Loss:

- The model is trained with a denoising MSE target in normalized image space.
- Because `prediction_type = v_prediction`, the target is the diffusion velocity target, not raw noise epsilon.
- Conceptually, for timestep `t`:

```text
x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * epsilon
v_t = sqrt(alpha_bar_t) * epsilon - sqrt(1 - alpha_bar_t) * x_0
loss = weighted_MSE(model(x_t, t, theta), v_t)
```

Loss-related config:

| Item | Value |
|---|---:|
| `min_snr_gamma` | `5.0` |
| `sigma_log_normal` | `None` |
| `max_grad_norm` | `1.0` |
| Mixed precision | `fp16` |
| Gradient accumulation | `1` |

## Optimizer And Training Length

Optimizer:

| Item | Value |
|---|---:|
| Optimizer | `AdamW` |
| Learning rate | `1.0e-4` |
| Weight decay | `1.0e-2` |

LR scheduler:

| Item | Value |
|---|---:|
| Scheduler | `CosineAnnealingWarmRestarts` |
| `T_0` | `4000` |
| `eta_min` | `1.0e-7` |

EMA:

| Item | Value |
|---|---|
| `ema_sigma_rels` | `[0.02, 0.10]` |
| `ema_update_every` | `1` |
| `ema_burn_in` | `1000` |

Batch/training schedule:

| Run | Batch size | Steps/epoch | Epochs | Target updates | Actual updates | Checkpoint every |
|---|---:|---:|---:|---:|---:|---:|
| `N=128` | `32` | `4` | `50,000` | `200,000` | `200,000` | `5,000` epochs = `20,000` updates |
| `N=16,384` | `32` | `512` | `391` | `200,000` | `200,192` | `39` epochs ≈ `19,968` updates |

SLURM training resources:

| Item | Value |
|---|---|
| Account | `huterer2` |
| Partition | `spgpu` |
| GPUs | `1` |
| CPUs/task | `4` |
| Memory | `80gb` |
| Walltime | `24:00:00` |

## Sampling Setup

Main no-CFG sampling:

| Item | Value |
|---|---|
| Sampler | `DPMSolverMultistepScheduler` |
| Sampler shorthand | DPM50 |
| Sampling steps | `50` |
| Seed | `123` |
| Device | `cuda` |
| Sampling batch size | `8` |
| Samples per held-out cosmology | `64` |
| Held-out cosmologies | `32` |
| Samples per model | `2048` |
| CFG guidance | off |
| `guidance_scale` | `None` |
| `s_churn`, `s_tmin`, `s_tmax`, `s_noise` | `None` |

Main sample files:

```text
results/nf_conditional_bias_probe/samples/nf_cond_bias_hi_u128_d2p07_n128_200k_seed123_dpm50_heldout_k64.npz
results/nf_conditional_bias_probe/samples/nf_cond_bias_hi_u128_d2p14_n16384_200k_seed123_dpm50_heldout_k64.npz
```

Each sample file is annotated with:

- `run_name`,
- `regime`,
- `dataset_size`,
- `cfg_dropout`,
- `guidance_scale`,
- `guidance_label`,
- `seed`,
- `samples_per_cosmology`,
- `heldout_indices`,
- `theta_norm_repeated`,
- `theta_raw`.

## PCA + Ridge Encoder

Encoder purpose:

```text
generated HI field -> frozen PCA features -> Ridge head -> recovered cosmology
```

Encoder config:

| Item | Value |
|---|---:|
| PCA components cap | `8192` |
| PCA target explained variance | `0.98` |
| PCA fit slices | `16,384` |
| Ridge head train slices | `16,384` |
| Ridge head validation slices | `4096` |
| Validation simulations | `64` |
| Ridge alpha | `10.0` |
| Encoder seed | `123` |
| PCA/Ridge embedding batch size | `512` |

Encoder exclusions:

- PCA sees real HI fields only.
- Ridge sees real HI fields only.
- Held-out cosmologies `900-931` are excluded.
- Encoder validation simulations are excluded from PCA fitting.
- Generated fields are never used to train PCA or Ridge.

Encoder output files:

```text
results/nf_conditional_bias_probe/encoder/pca_basis_98.npz
results/nf_conditional_bias_probe/encoder/frozen_pca_ridge_encoder.npz
results/nf_conditional_bias_probe/encoder/encoder_val_metrics.csv
results/nf_conditional_bias_probe/encoder/encoder_real_split.json
results/nf_conditional_bias_probe/encoder/encoder_val_metrics.png
```

To print the exact PCA rank, explained variance, hash, and validation metrics:

```bash
cd /home/jiamingp/diffusion_models_repo
python - <<'PY'
import json
import pandas as pd
from pathlib import Path
root = Path("results/nf_conditional_bias_probe/encoder")
split = json.loads((root / "encoder_real_split.json").read_text())
print("PCA basis:", split["pca_basis_path"])
print("PCA sha256:", split["pca_basis_sha256"])
print("PCA rank:", split["pca_rank"])
print("PCA explained variance:", split["pca_explained_variance_sum"])
print("heldout indices:", split["heldout_indices"])
print("val sims count:", len(split["val_sims"]))
print("train sims count:", len(split["train_sims"]))
print(pd.read_csv(root / "encoder_val_metrics.csv").round(4).to_string(index=False))
PY
```

## Real-Only PCA + MLP Encoder Test

Nick requested a direct sanity check of the PCA encoder on real held-out data, without using the diffusion model. The test is:

```text
real held-out HI slice -> frozen PCA features -> MLP regression head -> recovered cosmology
```

This is a regression test, not a classifier test, because the six CAMELS targets are continuous parameters.

Important details:

| Item | Value |
|---|---:|
| Diffusion samples used? | No |
| PCA basis | same leakage-safe 98% real-only basis |
| PCA rank in current run | `4451` |
| MLP default hidden layers | `(256, 128)` |
| MLP default L2 penalty | `1e-4` |
| MLP default learning rate | `1e-3` |
| MLP max iterations | `700` |
| Training slices | `32,768` non-held-out real slices |
| Validation slices | `4,096` non-held-out real slices |
| Test simulations | `32` held-out simulations, indices `900-931` |
| Test slices per held-out simulation | `128` |

The main plotted point is one point per held-out simulation. For each held-out simulation, the MLP predicts cosmology from each of its 128 real HI slices; the plotted value is the median predicted cosmology over those slices. The vertical bar is the 16th-to-84th percentile spread over slices.

Run on Great Lakes:

```bash
cd /home/jiamingp/diffusion_models_repo
sbatch -A huterer2 scripts/slurm/test_nf_conditional_bias_pca_mlp_encoder.sbatch
```

Expected outputs:

```text
results/nf_conditional_bias_probe/encoder/pca_mlp_encoder.pkl
results/nf_conditional_bias_probe/encoder/pca_mlp_encoder.npz
results/nf_conditional_bias_probe/encoder/pca_mlp_real_test_1to1.png
results/nf_conditional_bias_probe/encoder/pca_mlp_real_test_metrics.csv
results/nf_conditional_bias_probe/encoder/pca_mlp_real_test_per_cosmology_predictions.csv
results/nf_conditional_bias_probe/encoder/pca_mlp_real_test_per_slice_predictions.csv
results/nf_conditional_bias_probe/encoder/pca_mlp_real_test_metadata.json
```

After this MLP encoder exists, rerun the generated-sample cosmology calibration with the MLP head:

```bash
cd /home/jiamingp/diffusion_models_repo
ENCODER_TYPE=mlp sbatch -A huterer2 scripts/slurm/evaluate_nf_conditional_bias_probe.sbatch
```

This writes separate MLP-based calibration outputs so they do not overwrite the old PCA+Ridge baseline:

```text
results/nf_conditional_bias_probe/calibration_mlp/bias_probe_calibration_recovered_vs_input.png
results/nf_conditional_bias_probe/calibration_mlp/bias_probe_regime_slopes.csv
results/nf_conditional_bias_probe/calibration_mlp/bias_probe_per_cosmology_points.csv
results/nf_conditional_bias_probe/calibration_mlp/bias_probe_per_sample_predictions.csv
```

The old `results/nf_conditional_bias_probe/calibration/` outputs are PCA + Ridge. The corrected nonlinear recovery version should be read from `calibration_mlp/`.

## Real-Only VGG Feature Encoder Test

A second nonlinear encoder path uses frozen ImageNet-pretrained VGG16 convolutional features instead of PCA:

```text
real HI slice -> repeat to 3 channels -> resize to 224 -> frozen VGG16 features -> MLP regression head -> recovered cosmology
```

Equivalently:

```text
HI slice
-> repeat to 3 channels
-> resize to 224 x 224
-> frozen pretrained VGG16 feature extractor
-> trainable MLP/Ridge regression head
-> cosmology parameters
```

The VGG feature extractor is frozen and is not fine-tuned. The VGG weights are not updated. Only the regression head after VGG is trained on real non-held-out CAMELS HI slices.

Precise tensor conversion:

```text
normalized HI slice:  (128, 128) or (1, 128, 128)
add channel if needed: (1, 128, 128)
batch form:            (B, 1, 128, 128)
repeat channel:        (B, 3, 128, 128)
bilinear resize:       (B, 3, 224, 224)
ImageNet normalize:    channelwise mean/std normalization
frozen VGG16:          feature map
pool:                  average pooling or average+max pooling
regression head:       MLP or Ridge -> six cosmology parameters
```

The channel repeat is a shape adaptation for VGG. It does not add new information:

```text
R channel = HI
G channel = HI
B channel = HI
```

The resize is bilinear interpolation to `224 x 224`. This differs slightly from standard ImageNet preprocessing, which commonly resizes and center-crops natural images. For the HI maps we do not crop because cropping would discard cosmological structure; the goal is to feed the whole 2D field into the frozen feature extractor.

Terminology:

| Term | Meaning |
|---|---|
| Frozen VGG16 feature extractor | ImageNet-pretrained VGG16 convolutional layers. Parameters are not updated. |
| VGG features | The vector produced by pooling frozen VGG feature maps. |
| MLP head | Trainable nonlinear regression model after VGG features. |
| Ridge head | Trainable linear regression model with L2 regularization after VGG features. |
| Larger MLP head | More hidden layers/units in the trainable regression part, not a deeper VGG. |

Example MLP heads:

```text
default: VGG features -> Linear(512) -> activation -> Linear(256) -> activation -> Linear(6)
large:   VGG features -> Linear(1024) -> activation -> Linear(512) -> activation -> Linear(256) -> activation -> Linear(6)
```

Slice-count meaning:

| Setting | Meaning |
|---|---|
| `HEAD_TRAIN_SLICES=65536` | 65,536 real 2D HI slices for training the VGG regression head. |
| `HEAD_VAL_SLICES=8192` | 8,192 real 2D HI slices for validation. |
| `TEST_SLICES_PER_SIM=128` | All 128 z-slices from each held-out simulation for real-heldout testing. |
| Test set size | `32 held-out sims x 128 slices = 4096` real held-out slices. |

These are counts of 2D images/slices, not counts of CAMELS simulations.

Environment note:

- VGG uses `torchvision`, so the working Great Lakes environment is `/home/jiamingp/venvs/cosmodiff_nf_class`.
- `torch==2.1.2+cu118` and `torchvision==0.16.2+cu118` were verified to import after disabling the stale path-injection file:

```text
/home/jiamingp/venvs/cosmodiff_nf_class/lib/python3.10/site-packages/00-cosmodiff-base-venv.pth
```

That file injected `/home/jiamingp/venvs/cosmodiff_nf/lib/python3.10/site-packages` and caused incompatible package mixing.

Run a tiny smoke test first:

```bash
cd /home/jiamingp/diffusion_models_repo
vgg_smoke=$(HEAD_TRAIN_SLICES=512 \
  HEAD_VAL_SLICES=128 \
  TEST_SLICES_PER_SIM=4 \
  MLP_HIDDEN_LAYERS=64 \
  MLP_MAX_ITER=30 \
  ENCODER_OUT=results/nf_conditional_bias_probe/encoder_vgg_smoketest/vgg_smoketest.npz \
  MODEL_OUT=results/nf_conditional_bias_probe/encoder_vgg_smoketest/vgg_smoketest.pkl \
  sbatch -A huterer2 --parsable \
  scripts/slurm/train_nf_conditional_bias_vgg_encoder.sbatch)
echo "vgg_smoke=$vgg_smoke"
```

Smoke-test result from job `51475729`:

| Parameter | R2 | MAE | RMSE | Note |
|---|---:|---:|---:|---|
| `Omega_m` | `0.5189` | `0.0579` | `0.0749` | already much better than PCA+MLP smoke |
| `sigma_8` | `0.0111` | `0.0947` | `0.1211` | weak |
| `A_SN1` | `0.3708` | `0.4481` | `0.5914` | useful signal |
| `A_AGN1` | `0.0678` | `0.7544` | `0.9459` | weak |
| `A_SN2` | `0.2601` | `0.3205` | `0.3788` | useful signal |
| `A_AGN2` | `-0.4311` | `0.4118` | `0.5287` | bad |

The smoke test used only `512` train slices and `30` MLP iterations, so it is only a package/runtime and rough-signal check.

Then run the full VGG encoder if the smoke test passes:

```bash
cd /home/jiamingp/diffusion_models_repo
vgg_enc=$(sbatch -A huterer2 --parsable \
  scripts/slurm/train_nf_conditional_bias_vgg_encoder.sbatch)
echo "vgg_enc=$vgg_enc"
```

Full VGG encoder result from job `51475731`:

| Parameter | R2 | MAE | RMSE | Bias |
|---|---:|---:|---:|---:|
| `Omega_m` | `0.8992` | `0.0253` | `0.0343` | `-0.0033` |
| `sigma_8` | `0.6830` | `0.0598` | `0.0686` | `0.0110` |
| `A_SN1` | `0.4189` | `0.4848` | `0.5684` | `0.1053` |
| `A_AGN1` | `0.0144` | `0.7851` | `0.9726` | `-0.1170` |
| `A_SN2` | `0.2953` | `0.3149` | `0.3697` | `0.0073` |
| `A_AGN2` | `0.0946` | `0.3459` | `0.4205` | `-0.0010` |

Interpretation:

- VGG16 features + MLP are now credible for `Omega_m` and `sigma_8`.
- `A_SN1` and `A_SN2` have partial signal.
- `A_AGN1` and `A_AGN2` remain weak from HI with this simple encoder.
- For presentation, lead with `Omega_m` and `sigma_8`; treat feedback parameters as harder / weakly constrained.

Current VGG outputs:

```text
results/nf_conditional_bias_probe/encoder/vgg_mlp_encoder.npz
results/nf_conditional_bias_probe/encoder/vgg_mlp_encoder.pkl
results/nf_conditional_bias_probe/encoder/vgg_real_test_1to1.png
results/nf_conditional_bias_probe/encoder/vgg_real_test_metrics.csv
results/nf_conditional_bias_probe/encoder/vgg_real_test_per_cosmology_predictions.csv
results/nf_conditional_bias_probe/encoder/vgg_real_test_per_slice_predictions.csv
results/nf_conditional_bias_probe/encoder/vgg_real_test_metadata.json
```

VGG ablations submitted after the full encoder:

| Job | Purpose | Key settings |
|---:|---|---|
| `51475738` | Larger MLP, avg+max pooling | `HEAD_TRAIN_SLICES=65536`, `HEAD_VAL_SLICES=8192`, `MLP_HIDDEN_LAYERS=1024,512,256`, `MLP_MAX_ITER=1200`, `VGG_POOL=avgmax` |
| `51475739` | Larger MLP, average pooling | same as above but `VGG_POOL=avg` |
| `51475740` | Linear Ridge head sanity check | `HEAD_TYPE=ridge`, `RIDGE_ALPHA=1.0`, `VGG_POOL=avgmax` |

VGG ablation real-heldout results:

| Encoder | `Omega_m` R2 | `sigma_8` R2 | `A_SN1` R2 | `A_AGN1` R2 | `A_SN2` R2 | `A_AGN2` R2 |
|---|---:|---:|---:|---:|---:|---:|
| default MLP avg+max (`51475731`) | `0.8992` | `0.6830` | `0.4189` | `0.0144` | `0.2953` | `0.0946` |
| big MLP avg+max (`51475738`) | `0.9115` | `0.7374` | `0.4544` | `-0.0063` | `0.3251` | `0.1005` |
| big MLP avg (`51475739`) | `0.9027` | `0.7349` | `0.4535` | `-0.0313` | `0.2561` | `0.0868` |
| Ridge avg+max (`51475740`) | `0.8977` | `0.7143` | `0.4145` | `-0.0217` | `0.2862` | `0.1418` |

Best current encoder for presentation: `big MLP avg+max` (`51475738`). It gives the best `Omega_m`, `sigma_8`, `A_SN1`, and `A_SN2` recovery. `A_AGN1` remains basically unrecovered, and `A_AGN2` is weak.

If the real held-out VGG encoder has acceptable `R^2`, run the generated-sample calibration:

```bash
cd /home/jiamingp/diffusion_models_repo
VGG_DEVICE=auto ENCODER_TYPE=vgg sbatch -A huterer2 \
  scripts/slurm/evaluate_nf_conditional_bias_probe.sbatch
```

VGG calibration outputs go to:

```text
results/nf_conditional_bias_probe/calibration_vgg/
```

Generated-sample VGG evaluation job:

| Job | Purpose |
|---:|---|
| `51475741` | Evaluate generated samples using the current default VGG encoder at `results/nf_conditional_bias_probe/encoder/vgg_mlp_encoder.npz`. |

This job is independent of the ablations above. It uses the already-finished default full VGG encoder unless a different `--vgg-encoder` path is passed.

VGG generated-sample calibration result from job `51475741`:

| Regime | Dataset size | `Omega_m` slope | `sigma_8` slope | Interpretation |
|---|---:|---:|---:|---|
| Memorization | `128` | `0.2568` | `0.3285` | weak dependence on requested cosmology |
| Generalization | `16,384` | `0.7866` | `0.3814` | `Omega_m` tracks input much better; `sigma_8` still weak |

Useful plotting helper:

```bash
cd /home/jiamingp/diffusion_models_repo
python scripts/plot_nf_conditional_bias_vgg_results.py --project-dir "$PWD"
```

This writes:

```text
results/nf_conditional_bias_probe/encoder/vgg_encoder_r2_comparison.csv
results/nf_conditional_bias_probe/encoder/vgg_encoder_r2_comparison.png
results/nf_conditional_bias_probe/calibration_vgg/bias_probe_vgg_main_slopes.png
```

## Calibration Evaluation

Main evaluation files:

```text
results/nf_conditional_bias_probe/calibration/bias_probe_per_sample_predictions.csv
results/nf_conditional_bias_probe/calibration/bias_probe_per_cosmology_points.csv
results/nf_conditional_bias_probe/calibration/bias_probe_regime_slopes.csv
results/nf_conditional_bias_probe/calibration/bias_probe_eval_metadata.json
results/nf_conditional_bias_probe/calibration/bias_probe_calibration_recovered_vs_input.png
```

Evaluation config:

| Item | Value |
|---|---:|
| Samples per held-out cosmology | `64` |
| Held-out cosmologies | `32` |
| Encoder batch size | `512` |
| Bootstrap resamples for slope CI | `1000` |
| Bootstrap seed | `123` |

Exact error-bar calculation:

For each model/regime, each held-out cosmology, and each parameter:

```python
vals = recovered_parameter_values_for_64_generated_samples
q16, med, q84 = np.quantile(vals, [0.16, 0.50, 0.84])

plotted_y = med
lower_error = med - q16
upper_error = q84 - med
```

Important interpretation:

- The error bar is the generated-sample spread at fixed input cosmology.
- It is not the difference between recovered and true cosmology.
- It is not the PCA/Ridge encoder validation error.
- Smaller memorization bars can mean lower generated diversity or collapse; it does not automatically mean better calibration.

Exact slope calculation:

- One point per held-out cosmology.
- `x = theta_in`
- `y = median(theta_rec)` over the 64 generated samples.
- Fit `y = slope * x + intercept` using `np.polyfit`.
- Bootstrap CI: resample the 32 held-out cosmology points with replacement `1000` times, then report 16/84 percentiles of bootstrap slopes.

To print exact final slopes:

```bash
cd /home/jiamingp/diffusion_models_repo
python - <<'PY'
import pandas as pd
path = "results/nf_conditional_bias_probe/calibration/bias_probe_regime_slopes.csv"
df = pd.read_csv(path)
print(df.round(4).to_string(index=False))
PY
```

To print exact per-parameter average error-bar widths:

```bash
cd /home/jiamingp/diffusion_models_repo
python - <<'PY'
import pandas as pd
path = "results/nf_conditional_bias_probe/calibration/bias_probe_per_cosmology_points.csv"
df = pd.read_csv(path)
df["half_width_16_84"] = 0.5 * (df["theta_rec_q84"] - df["theta_rec_q16"])
summary = (
    df.groupby(["regime", "dataset_size", "parameter"], as_index=False)
      .agg(mean_half_width=("half_width_16_84", "mean"),
           median_half_width=("half_width_16_84", "median"))
)
print(summary.round(4).to_string(index=False))
PY
```

## CFG / Guidance Ablation

This ablation is separate from the main no-CFG run.

Actual reduced ablation submitted:

| Item | Value |
|---|---|
| Dataset size | `N=16,384` only |
| CFG dropout during training | `0.1` |
| Guidance scales for sampling | `None`, `1.0`, `1.5`, `2.0` |
| Train job | `51341391_0` |
| Sample job | `51341392_0` |
| Eval job | `51341393` |

Run name:

```text
nf_cond_bias_hi_u128_d2p14_n16384_200k_cfgdrop0p1
```

Important:

- `cfg_dropout` is a training-time setting.
- `guidance_scale` is a sampling-time setting.
- One CFG-trained checkpoint can be sampled multiple times with different guidance scales.
- The no-guidance sample from the CFG-trained model is not the same as the main v1 no-CFG model, because the model was trained with conditional dropout.

CFG sweep namespace:

```text
nf_conditional_bias_probe_cfg_sweep
```

CFG sample paths have guidance labels:

```text
results/nf_conditional_bias_probe_cfg_sweep/samples/<run>_seed123_dpm50_heldout_k64_noguidance.npz
results/nf_conditional_bias_probe_cfg_sweep/samples/<run>_seed123_dpm50_heldout_k64_g1.npz
results/nf_conditional_bias_probe_cfg_sweep/samples/<run>_seed123_dpm50_heldout_k64_g1p5.npz
results/nf_conditional_bias_probe_cfg_sweep/samples/<run>_seed123_dpm50_heldout_k64_g2.npz
```

## Notebook For Inspection

Notebook:

```text
notebooks/nf_conditional_bias_probe_check.ipynb
```

It reads the completed CSV/JSON outputs and regenerates:

```text
results/nf_conditional_bias_probe/calibration/bias_probe_calibration_recovered_vs_input_clean.png
results/nf_conditional_bias_probe/calibration/bias_probe_errorbar_spread_summary.png
```

## Slide-Friendly Explanation Of Error Bars

For each held-out cosmology, the input `theta` is fixed. We sample 64 generated HI fields by changing only the diffusion noise seed. Each generated field is passed through the same chosen encoder: PCA + Ridge, PCA + MLP, or VGG16 features + MLP. The marker is the median recovered parameter, and the vertical error bar is the 16th-to-84th percentile spread of the 64 recovered values.

Therefore, the bar measures stochastic generated-sample variation at fixed input cosmology. It is not the model bias itself. Bias is read mainly from the median location and fitted slope relative to the diagonal.
