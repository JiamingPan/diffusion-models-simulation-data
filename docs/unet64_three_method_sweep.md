# UNet-64 three-method data-size experiment

Status: local preparation only. No upload, training, sampling, or Slurm submission.
The inventory is explicitly NOT RUNNABLE; it is not a training config bundle.

## Fixed scope

Unconditional UNet width 64, block widths [16,32,64]. Fresh training at each
N=64,128,256,512,1024,2048,4096,8192,16384,32768. Three new curves:

1. Uniform D4 orientation followed by a uniform periodic integer translation.
2. One smooth-warp recipe: composition of sinusoidal periodic shears, amplitude
   2 pixels, four wavelengths across the 128-pixel image, random phases and
   directions per fetch, bilinear interpolation. This is a proposed fixed
   distortion control, not cosmology-preserving augmentation or a strength sweep.
3. A native scalar-feature D4-equivariant UNet, implemented locally with D4-tied
   convolution kernels and stride-1 convolution plus symmetric average-pool
   downsampling. Channels remain [16,32,64]; scalar channels are a stronger
   restriction than orientation-carrying group convolutions. Effective free
   kernel degrees of freedom decrease even though stored parameter counts do
   not. This is not inference-time ensembling or eight-pass prediction averaging.
   PyTorch numerical tests have NOT run locally (PyTorch unavailable).

Use no input augmentation in branch 3 so architectural symmetry is separable
from the augmentation treatment. The D4 architecture does not enforce all
periodic translations; its symmetry set differs from the combined first arm.

## Baseline audit

Local configs for all ten original UNet-64 runs exist. They specify batch size
32, scaled tanh normalization, and approximately 200k optimizer updates. Exact
epoch-rounded budgets total 2,001,152 updates per curve: 6,003,456 new updates
over all three curves. Do not multiply the conditional u128 runtime by 30 to
estimate cost; these are different models and the equivariant network may have
different cost per update.

Preserve each config's data fields verbatim: the selected LH/CV redshift grids
and number of volumes differ across N. This is NOT the conditional z=0 sweep.
Before reuse, verify each remote checkpoint, saved config, real training-array
digest, actual training augmentation code path, sampler and EMA/raw-weight
selection. Local YAML and noaug filenames alone do not prove an unaugmented
historical baseline. Training RNG seeds are not yet verified.

## Evaluation

Use a shared training-fitted PCA representation and the same per-N PCA95
threshold for every method, including the baseline. Evaluate the same 512
sampling seeds, scheduler and checkpoint-weight convention. Also check copying
against the orbit of stored training maps (D4 and periodic translations), using
the same procedure in all branches. Do not call transformed copies novel.

Report PDFs, P(k), boundary diagnostics, examples, and nearest matches against
the ORIGINAL unwarped reference. A secondary warped-reference comparison may
explain degradation but must not replace the physical reference. P(k) in
normalized model space is not physical-unit HI power.

Warp-specific copying is not ruled out by the D4/shift search; retain that
limitation. Single-seed curves are exploratory, not a replicated scaling law.

## Implemented locally

- `simdiff_eval/unet64_sweep_transforms.py`: exact pixel permutations and a
  periodic smooth-warp reference with serializable private RNGs.
- `scripts/prepare_unet64_three_method_plan.py`: checks baseline config shape,
  records source hashes, computes update counts, emits a NEW draft inventory and
  preserves copies of original configs. Refuses an existing output directory.
- `tests/test_unet64_sweep_transforms.py`: four local tests, NumPy path only.
- `simdiff_eval/d4_unet64.py` and `tests/test_d4_unet64.py`: model plus mandatory
  equivariance, gradient, checkpoint round-trip and Torch transform tests.
- `scripts/run_unet64_three_method.py`: preflight by default; explicit training
  requires matching preflight/code/config hashes and a new scratch destination.
  Uses the original runtime training loop, optimizer, scheduler and EMA settings.

## Required before an upload/run preview

1. Validate the new native D4 model with measured equivariance error, round-trip checkpoint save/
   load, and declared width/parameter-count policy. Verify optimizer,
   EMA and sampling integration. No claim of parity based solely on width names.
2. Validate the fresh-training runner against the remote runtime and saved
   baseline artifacts. The scalar-feature architecture, changed downsampling,
   and effective parameter restriction are part of the architectural treatment.
3. Torch-path tests in the existing environment, without venv edits/downloads.
4. Read-only remote baseline and runtime audit; a separately approved benchmark
   if necessary for GPU memory/time and storage estimates.
5. Exact upload command and separate APPROVE PUSH; exact Slurm commands, runtime
   budget, output paths, checkpoint retention and separate APPROVE RUN.

All new large outputs must go to new scratch directories. Never overwrite the
baseline. No older approval applies to this expanded three-method sweep.
