# Nick's action items — 18 September 2026

## Status and order

| Action | Prepared locally | Still needed |
|---|---|---|
| Nearest-training-parameter sanity check for the biased red points | CPU evaluator, tests, read-only comparison notebook | Real conditional manifests, prepared fields, frozen probe and saved predictions on Great Lakes |
| Replot recovery/calibration with zero-init models | The conditional/unconditional distinction has been audited | A **continuously parameter-conditioned** zero-init model and matched evaluation; current constant-label DiTs are not usable here |
| Finish fresh DiT depth/data-size sweep and replace plots | 30 draft configs: L8/L12/L16 × ten N values; **300k target updates per model, chosen by the user** | Verified initialization runtime, per-N data audit, GPU smoke, approved launch |
| Connect paper to existing ML work; add DiT figures | Primary-source connection notes and figure checklist below | Incorporate after choosing the authoritative manuscript and reading full-sweep results |

After the user's separate `APPROVE PUSH` and `APPROVE SEND`, the previewed zero-init
commit was pushed and [cosmodiff PR #9](https://github.com/nkern/cosmo_diffusion/pull/9)
was opened. The new sweep/control files have not been pushed; no new Great Lakes jobs
or person-to-person messages have been submitted by this work.
Existing jobs/checkpoints have not been altered. Job completion cannot be checked live from this local session.

## Critical distinction: which model can answer which question?

`prepare_nf_generalize_fig2_dit_configs.py` uses `data.constant_label: 0`, no parameter-label file,
and discrete conditioning with one label. The A40 zero-init experiment preserves that setup.
It tests unconditional image generation, memorization and grid artifacts.

`prepare_nf_conditional_bias_probe_configs.py` and the fresh full sweep instead use
`UNet2DConditionModel`, six-dimensional **continuous** CAMELS conditioning, and held-out
cosmologies. These are the sources of the red/blue parameter-recovery plots.

Changing the initialization of an unconditional checkpoint does not make it conditional.
The native class-label DiT API cannot simply be handed a six-dimensional vector as class IDs.
A separate, validated continuous-conditioning path is required before a new DiT can answer Nick's first item.
Do not apply zero initialization to trained checkpoint weights.

The saved 128-draw L16 patch-8 zero-init result had 128 near-copies, compared with 50 near-copies,
2 intermediate and 76 low-similarity draws for the old fresh L16. This is encouraging evidence
for removing the severe artifact mode in that single-N/single-seed test. It is **not** evidence
of generalization or of unbiased conditional inference. Source: the user's exported
`dit_l16_a40_early_fix_results.md` and accompanying images, not a fresh cluster read.

## Nearest-parameter control

New implementation:

- `scripts/evaluate_parameter_neighbor_control.py`
- `simdiff_eval/parameter_neighbor_control.py`
- `notebooks/conditional_parameter_neighbor_control.ipynb`

For request `theta_h`, choose a represented training cosmology using

`distance(h,j) = sqrt(sum_p ((theta_h[p] - theta_j[p]) / frozen_train_std[p])**2)`.

All six physical parameters are used; the scales come from the existing non-held-out parameter
normalization metadata. This is **not** nearest-image selection. Duplicate slices do not give
a cosmology extra weight in this distance. Ties are recorded, then resolved by simulation ID.

Compare four series at identical held-out IDs:

1. Saved generated-field probe predictions.
2. True parameters of the nearest represented training cosmology: no image estimator involved.
3. Probe predictions on **all exact selected training fields** from that cosmology.
4. Probe predictions on 64 deterministically selected real slices at the requested held-out cosmology.

The evaluator verifies label normalization, disjoint simulations, selected `(simulation,z)` row
mapping, config/manifest image-path agreement, and agreement of the encoded training fields with
their source grid. An optional prepared-image manifest hash is checked. It rejects reselection,
reshaping/thinning, unexpected normalization, duplicate generator conditions, and mismatched
requested parameters. It uses the existing frozen encoder; no refit or generator sampling occurs.
Pretrained VGG weights must already be cached; no implicit download is allowed.

The generated evaluation metadata must identify the same probe path/type. Legacy metadata did
not hash historical probe contents, so unchanged historical weights still need an operational check.
The descriptor's parameter order/scales, normalization and train/validation exclusions must match
the control. New metadata records input hashes, probe-head/basis/cache hashes and selected encoded rows. The large source grid
is identified by path plus selected-slice checks, not claimed to have a whole-file hash.

### Existing-allocation analysis command template

The files first need to be transferred through a separately approved push; this is not yet staged
on Great Lakes. The following is a CPU analysis invocation, **not a job submission**. Run only
inside an existing approved allocation using the working review environment. Input/output paths
must be verified before execution; choose a new scratch output directory.

```bash
/home/jiamingp/venvs/cosmodiff_nf_class/bin/python \
  /home/jiamingp/diffusion_models_repo/scripts/evaluate_parameter_neighbor_control.py \
  --project-dir /home/jiamingp/diffusion_models_repo \
  --manifest /home/jiamingp/diffusion_models_repo/local/nf_conditional_bias_probe/manifest.json \
  --param-stats /home/jiamingp/diffusion_models_repo/local/nf_conditional_bias_probe/heldout/param_norm_stats.json \
  --params-table /scratch/huterer_root/huterer0/CAMELS/CMD/3d_grids/IllustrisTNG/params_LH_IllustrisTNG.txt \
  --grid-path /scratch/huterer_root/huterer0/CAMELS/CMD/3d_grids/IllustrisTNG/Grids_HI_IllustrisTNG_LH_128_z=0.0.npy \
  --encoder results/nf_conditional_bias_probe/encoder/vgg_mlp_encoder.npz \
  --encoder-type vgg --real-k 64 --batch-size 64 \
  --generated-predictions /home/jiamingp/diffusion_models_repo/results/nf_conditional_bias_probe/calibration_vgg/bias_probe_per_sample_predictions.csv \
  --generated-metadata /home/jiamingp/diffusion_models_repo/results/nf_conditional_bias_probe/calibration_vgg/bias_probe_eval_metadata.json \
  --out-dir /scratch/huterer_root/huterer0/jiamingp/parameter_neighbor_control_v1
```

For the ten-size fresh conditional sweep, select its manifest and its matching saved prediction
CSV/metadata instead; do not pool old and fresh runs. The notebook reads the new output folder via
`NEIGHBOR_CONTROL_DIR` or its setup cell. It does not launch the evaluator itself.

### Reading the result

- True nearest parameters biased: finite six-dimensional training coverage is a plausible contributor.
- Probe adds offset on nearest real training fields: the recovery estimator also contributes.
- Held-out real probe biased: the diagonal is not a clean generator-only benchmark.
- Generator differs from both real controls: inspect learned conditioning, morphology and normalization.

The residual decomposition is an algebraic accounting identity, not a causal proof. Similar curves
do not establish that the generator implements nearest-neighbor retrieval. A six-dimensional
nearest neighbor can differ appreciably in Omega_m even when an Omega_m-only neighbor exists;
the other five requested parameters cannot be ignored silently.

## Recovery spread is not a Bayesian posterior

The current evaluator encodes repeated fields generated at fixed requested parameters and takes
quantiles of the resulting regression estimates. Those values are a **recovery distribution**.
Their central-interval inclusion fraction is not automatically credible-interval coverage of
`p(theta | observed field)`.

For true posterior calibration, first identify the actual posterior-sampling inference pipeline,
its prior, observed-field selection and estimator. Then run identical real-vs-surrogate inference
and a suitable calibration procedure. Talts et al.'s SBC specifically validates algorithms
that generate posterior samples; it is not interchangeable with the present forward-generation
check. [Talts et al., *Validating Bayesian Inference Algorithms with Simulation-Based Calibration*](https://arxiv.org/abs/1804.06788).

## Fresh DiT refresh

`scripts/prepare_dit_adaln_zero_refresh.py` drafts new configs without training/submitting anything.
It retains the original width, heads, pixel patch size 8, effective batch 8, data source allocation,
optimizer, schedule, weighting, EMA and loss. It changes the fresh initialization policy and
uses new checkpoint/sample destinations. **The user selected 300k target updates for every new
model.** The old depth/data-size sweep used 200k, so the refresh changes both initialization
and budget relative to that sweep; it is not a budget-matched causal initialization comparison.

Local draft: `local/nf_generalize_fig2_dit_adaln_zero_300k_draft/plan.json` and its 30 configs.
Status is deliberately `draft_not_launch_ready`. Its local config paths must be regenerated
in the intended Great Lakes staging location, not mistaken for an already frozen launch plan.

Before launch, reconcile the opt-in factory with the immutable Great Lakes runtime. The prepared
cosmodiff fix is commit `816d693a629ed45052db08dd3108be757846ab74`, now published in
[PR #9](https://github.com/nkern/cosmo_diffusion/pull/9), not yet merged or staged on Great Lakes.
The existing seed-restart pin does not become patched merely because a YAML key is added.
Verify native CPU initialization/gradient behavior, exact slice-first subsets and normalization
at every N, initialization and data hashes, seed handling, and A40 memory/throughput. Report
actual successful updates and AMP skips. New long-run jobs require a complete exact launch/cost
preview and `APPROVE RUN`; publication requires its own approvals.

Do not cancel or duplicate the existing A40 ablation automatically. The last available record
had the combined patch-4/zero-init arm running and the full sampling array waiting on training.
Use cluster accounting and logs to establish its present state before any new submission.

### Figure replacement checklist

Keep old native-init results as labeled controls, not members of the new depth curve.
For every fresh depth and N, regenerate: exact-subset novelty/nearest matches, generalization
score, one-point PDFs, power-spectrum comparisons and boundary diagnostics. Check that the
feature encoder, thresholds, training subsets, raw/EMA policy and sample counts agree across
the comparison. Do not exclude low-similarity generated fields from main distributional metrics;
show stratified diagnostics in addition to whole-population results.

## ML connections and paper figures

Primary sources checked on 18 September 2026:

1. [Peebles & Xie, *Scalable Diffusion Models with Transformers*](https://arxiv.org/html/2212.09748v2).
   Connect the initialization intervention to the reference adaLN-Zero residual identity and
   zero-initialized prediction head, and put depth/patch comparisons beside their scaling work.
   Our implementation ports the modulation/output zeroing, not the complete reference recipe.
   Their ImageNet comparison does not prove the cause of our cosmological L16 artifacts.

2. [Zhang et al., *The Emergence of Reproducibility and Consistency in Diffusion Models*, ICML 2024](https://proceedings.mlr.press/v235/zhang24cn.html).
   Connect the finite-data memorization/generalization regimes and paired-noise comparisons.
   Our contribution should test scientific-statistic and recovery reliability within these regimes,
   rather than treating novelty alone as scientific validity.

3. [Niedoba et al., *Towards a Mechanistic Explanation of Diffusion Model Generalization*, ICML 2025](https://arxiv.org/abs/2411.19339).
   Their local empirical denoisers provide a relevant comparison for patch-local inductive biases.
   A low patch-majority score alone does not establish that our artifacts are stitched training
   patches: noise can also have low majority. Present our patch diagnostics as evidence requiring
   patch-match quality and controls, not as confirmation of their mechanism.

4. [An et al., *On Inductive Biases That Enable Generalization in Diffusion Transformers*, NeurIPS 2025](https://papers.neurips.cc/paper_files/paper/2025/hash/2dc52e27a26afabb0bf2123bceaf9208-Abstract-Conference.html).
   Their association of early attention locality with generalization and attention-window
   interventions motivates comparing convolutional and transformer inductive biases.
   Our zero-init intervention is different: it does not impose local attention windows.

Suggested additions to the paper:

- **DiT depth/data-size result:** L8/L12/L16 novelty and statistical accuracy under one fresh
  initialization/budget protocol; pending retrained sweep, not extrapolated from N=256.
- **Initialization ablation:** paired-noise maps, outcome fractions, calibrated boundary ratios
  and P(k); state the single seed and training-budget/batch differences from old controls.
- **Biased recovery sanity check:** generated vs parameter-nearest true labels, encoded neighbor
  fields and encoded held-out real fields, all using the same requested IDs and frozen probe.

The manuscript copies are currently divergent and some are dirty. Keep these notes separate
until the authoritative paper file is chosen; do not overwrite the user's existing edits.

## Local validation and remaining gaps

All 30 control/refresh and related full-sweep regression tests pass. The reader notebook executes top to bottom with
explicit pending inputs on this machine. Actual CAMELS grids, conditional manifests, frozen
encoder files and generated predictions are not available locally, so the scientific control
has **not** been executed on real data. A plotted synthetic fixture is only a rendering test.
The six-panel recovery/residual plots were visually inspected with that explicitly labeled
fixture. The saved notebook validates and has no execution errors; its native notebook-viewer
presentation has not been visually inspected. No new bias diagnosis or posterior-calibration result is claimed.
