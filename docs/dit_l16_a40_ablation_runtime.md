# Runtime and validation

## Interpretation

The sampler matrix showed partial, not complete, improvement under stochastic
sampling. It does not demonstrate spurious fixed points, exclude every sampling
method, or prove an initialization defect. These ablations test hypotheses. One
successful seed supports an intervention; it does not establish a universal
depth-dependent cause. The missing exact L12 checkpoint remains omitted.

The init helper zeros each block's `norm1.linear` weight/bias and the two output
projections, after fresh construction and before training/EMA creation. For L16
this is 18 Linear modules / 36 tensors. It preserves other native initialization,
per-block time embedders and the rest of the model. This is NOT a reproduction
of the complete [reference DiT initialization](https://github.com/facebookresearch/DiT/blob/main/models.py).
The [diffusers 0.35.1 DiT API](https://github.com/huggingface/diffusers/blob/v0.35.1/src/diffusers/models/transformers/dit_transformer_2d.py)
is explicitly checked. Zeroing a restored model is forbidden.

Patch4 uses 1,024 tokens, versus 256 at patch8; dense attention score storage is
16x, although actual memory and speed depend on the attention implementation.
Patch4 does not make patch-local memorization impossible. Its intervention also
changes tokenization and positional granularity.

## Isolation / resources

Great Lakes spgpu has [A40 48 GB GPUs](https://documentation.its.umich.edu/node/4976).
Use one process, one node, one GPU per task, 4 CPU / 80 GB host memory. Arrays
must be `0-2%2`: at most two concurrent training GPUs, not multi-node training.
Each training arm has a five-day wall-clock cap (up to 360 aggregate GPU-hours),
not a promise that it finishes in a day. Three dependent sample tasks have one
hour each (up to 3 GPU-hours). Test A uses a CPU-only standard job with 4 CPU,
16 GB host memory and a four-hour cap (up to 16 CPU-hours). Monetary charge is
account-specific and not known here. Any extra/recovery job needs approval.
The [partition policy](https://documentation.its.umich.edu/arc-hpc/greatlakes/policies)
allows longer caps, but we do not increase this one automatically.

Original runtime checkout remains `iaifi_poster_code_555f350`, immutable pin
`cosmodiff_seed_restart_pin_58c77eb_555f350`, Python `cosmodiff_nf_class` venv.
Preserve the pin's `PYTHONPATH` order and original runtime manifest/code identity.
No installs or patching in shared environments. Require diffusers 0.35.1; version
disagreement fails rather than silently changing the recipe. Preflight runs
small native CPU models through scheduler, v/min-SNR backward and AdamW update,
checks the actual trainer signature, and does not load real data or use a GPU.
The full A40 run must separately confirm first-update finite loss/head weights,
GPU model, peak VRAM and a 100-step rough throughput estimate in stdout.

All new outputs live under the dedicated scratch experiment
`/scratch/huterer_root/huterer0/jiamingp/dit_l16_a40_init_patch_v1`.
Do not use /home for large artifacts; home is already >95% full. Scratch is not
archival storage; review and approve a durable copy separately after completion.
Neither old 300k/500k checkpoints nor C4 are written. A prior launch, even a
partial directory, makes fresh training refuse; no implicit rerun/resume.

Retain six checkpoints per arm at ~50k nominal-update intervals and at 300k,
with no deletion/pruning. This cadence is an operational difference from the
old 5k cadence, not an optimization change. Budget conservatively 120 GiB per
arm / 360 GiB total plus samples/logs and free-space headroom; actual footprint
depends on optimizer/EMA/metrics serialization. Check free space before launch;
quota limits must also be checked, since `df` alone is not a quota check. A40
OOM must stop, not silently change batch, accumulation or precision.

## Frozen inputs / reports

Baseline is the exact matrix fresh300k YAML. Frozen matrix plan SHA256:
`8404ae7b4f607923251d430da2b78afe07871846269927c0a9e2b6e3d8c0e333`.
Preparation stores its config SHA, all three new config SHAs and exact saved
matrix initial-noise bytes. Require the reviewed new plan SHA and code revision
on every invocation. Verify N=256 / 32 updates per epoch / 9,375 epochs / final
epoch 9,374. Changes to the plan/config/reference fail closed.

Native training arrays must agree elementwise/orderwise with the frozen IO
reference to absolute 2e-6 (zero relative tolerance), allowing only Torch/NumPy
log/tanh rounding. Record both SHA256s and maximum delta. Never replace actual
training tensors with the evaluation reader's array. Labels must be long[256]
and all zero; the existing three-path constant-label adapter logs the injection,
native no-op or refusal. Genuine differing labels still raise.

Training completion is written by ONE owner after the native trainer returns,
after the microbatch count matches the existing nominal epoch budget and
successful/AMP-skipped optimizer updates are recorded, and after tensor-loading
the full model/optimizer/scheduler/scaler/exact EMA snapshots. Never extend the
epoch budget silently to compensate for AMP overflow; report any skipped-update
count as a qualification when comparing arms. Hash final artifacts and recheck
them before sampling. Started provenance
is not terminal success. Distinct exclusive atomic output directories avoid
shared mutable JSON writers; interrupted data is retained without PASS. Files
are on shared scratch, not node-local. Exclusive creation/rename helps publication
but does not promise cross-client durability or make mutable multi-writer updates
safe. Append-only event reporting is a future alternative, not this patch.

CPU Test A scans same-location and any-location 8x8 patches with <=1 MiB
similarity chunks, versus the supplied script's ~1 GiB chunk. Its bank has 65,536
candidate patches, so it is real CPU work, not zero-compute. Calibrate against
self-matches, known synthetic patchworks, corrupted single maps, white noise
and Fourier-phase-randomized maps. Report raw-amplitude/offset residuals too.
Self-match only verifies implementation; nearest-bank cosine is biased by bank
size. Low majority/high cosine alone does not prove copying. No automatic
`bad_fraction` is invented without an independently validated invalidity rule.

New sampling uses 128 matrix noises, raw weights, class0, 50 DPM++ order2 steps,
and distinct step seed124. Save every image and each image's power/boundary/cosine
diagnostics. Sampling microbatch2 is conservative for patch4; old matrix used8,
so numerical repeatability is a qualification, not an unreported change. Measure
both patch4 and patch8 grids so moving the artifact grid cannot look like a fix.
Report low-similarity, intermediate and near-copy populations separately. Power
spectra here are normalized-field checks, not physical-density power spectra.

## Execution order

1. Local native 0.35.1 tests, relevant regression suite, syntax and diff checks.
2. Preview exact branch push; STOP for APPROVE PUSH.
3. On Great Lakes, prepare isolated plan, run CPU preflight, review plan/hash,
   free space and quota. Staging/preparation must be part of an approved preview.
4. Preview frozen train array, dependent sampling array and CPU Test A commands
   with cost/resource caps; STOP for APPROVE RUN. Do not submit on generic “run”.
5. Read actual states and artifacts through final comparison. Do not infer
   successful training merely from jobs disappearing from the queue. Display
   saved maps/population diagnostics before accepting an arm as the paper fix.

This checkout has no authenticated Great Lakes connection. User password/MFA
must remain user-entered; do not claim to have started or inspected remote jobs.
