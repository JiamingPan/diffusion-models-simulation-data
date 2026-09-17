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
The [diffusers 0.38.0 DiT API](https://github.com/huggingface/diffusers/blob/v0.38.0/src/diffusers/models/transformers/dit_transformer_2d.py)
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
No installs or patching in shared environments. Freeze Python/Torch/NumPy/
diffusers/huggingface-hub versions from the original reviewed matrix plan,
recheck that source plan's SHA256, and require exact equality at every preflight,
train and sample startup. Stdout prints expected and actual versions before any
version failure. The actual successful matrix used Python 3.10.9, Torch
2.1.2+cu118, NumPy 1.26.4, diffusers 0.38.0 and huggingface-hub 0.36.2; ablation
API tests now exercise native diffusers 0.38.0. An untested matrix diffusers
version or missing runtime contract fails closed. Preflight runs
small native CPU models through scheduler, v/min-SNR backward and AdamW update,
checks the actual trainer signature, and does not load real data or use a GPU.
That API check alone is NOT a real-data preflight. A separate CPU-only
`ABLATION_MODE=data-preflight` allocation must load each arm's actual native
dataset, verify every retained pixel/label/normalization statistic, rescore the
saved fresh300k DPM50 baseline, and publish the complete receipt before any
training arm is eligible. Train refuses a missing/stale receipt and rechecks
native tensor/source hashes on its own startup.
The full A40 run must separately confirm first-update finite loss/head weights,
GPU model, peak VRAM and a 100-step rough throughput estimate in stdout.

All new outputs live under the dedicated scratch experiment
`/scratch/huterer_root/huterer0/jiamingp/dit_l16_a40_init_patch_v2_datafix`.
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

Native training arrays must agree elementwise/orderwise with an independent
slice-first NumPy reference to absolute 2e-6 (zero relative tolerance), allowing
only Torch/NumPy log/tanh rounding. Both native and independent normalizers fit
on the retained slices, matching pinned cosmodiff's select/reshape/log/norm order.
Freeze the retained raw-slice hash, volume/z order, normalization statistics and
reference hash at preparation. Keep the old IO reference and its hash unchanged
as legacy provenance, NOT as the native training equality target. Changes to
either reference or the retained data still fail closed. Record both native and
independent SHA256s, maximum rounding delta, and the legacy-reference delta.
Never replace actual training tensors with a NumPy reference. Labels must be long[256]
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
New sampling evaluates against the retained-slice reference. The real-data CPU
preflight rescales no saved images: it rescores existing baseline sample bytes
against that same matched reference, producing population counts, boundary
medians and conditional P(k) ratios. Old matrix spectra fitted with the legacy
reference are not directly comparable until rescored. No causal verdict follows
from the normalization mismatch or this rescore.

## Execution order

1. Local native 0.38.0 tests, relevant regression suite, syntax and diff checks.
2. Preview exact branch push; STOP for APPROVE PUSH.
3. On Great Lakes, prepare isolated plan, run CPU API AND real-data preflight, review plan/hash,
   free space and quota. Staging/preparation must be part of an approved preview.
4. Preview frozen train array, dependent sampling array and CPU Test A commands
   with cost/resource caps; STOP for APPROVE RUN. Do not submit on generic “run”.
5. Read actual states and artifacts through final comparison. Do not infer
   successful training merely from jobs disappearing from the queue. Display
   saved maps/population diagnostics before accepting an arm as the paper fix.

This checkout has no authenticated Great Lakes connection. User password/MFA
must remain user-entered; do not claim to have started or inspected remote jobs.

## Recovery from the version-check failure

CPU preflight 61288400 failed before native model smoke checks or training:
adapter c87aea9 incorrectly required the locally tested diffusers 0.35.1 instead
of the working matrix's 0.38.0. Do not downgrade the shared venv. Preserve the
original experiment directory, its plan hash
`0bae872d5ff3a15eaa9763b93440c4575e9c9b3f01574bb7410a7c74df45b3e6`
and failed-job logs. Reprepare the corrected immutable code in the new scratch
root above; never edit the old plan's code revision or bypass its hash checks.
The fresh plan records both the corrected code revision and the source matrix's
runtime contract, with scientifically unchanged arm recipes, inputs and budgets.
Local native tests use the host's CPU Torch, not Great Lakes Torch 2.1.2/CUDA;
repeat the small checks inside the approved CPU-only allocation in the real pin
before submitting the expensive A40 train/sample arrays. Those extra submissions
remain separate protected actions, not part of the branch push.

## Recovery from the retained-slice equality-check failure

All three training tasks 61290230_0/1/2 failed at the adapter's dataset check,
before native model construction/initialization and before `optim.train`.
Dependent sample array 61290231 was cancelled; CPU Test A 61290232 completed.
Do not repeat the full launcher, delete partial arm directories, or repeat Test A.
Preserve root `dit_l16_a40_init_patch_v1_runtime038`, frozen plan
`c75df5176aa49d1cb58b1de1907ca44384c0d903922f782181e189c967fbfc7d`,
all logs and the completed Test A output.

The adapter incorrectly required equality to legacy evaluation IO. That IO
fits center/max over unthinned configured volumes and then takes retained
z-slices. Native cosmodiff thins/reshapes first and fits on the retained training
slices. An excluded extreme therefore changes legacy IO normalization but not
native training normalization. This is a population/operation-order difference,
not merely float rounding; increasing tolerance or changing training tensors
would hide it. Local regression tests reproduce that exact mismatch, and the
corrected audit still rejects changed data, permutations, dtype/shape differences,
nonfinite pixels, altered normalization and conflicting labels. The mmap warning
is not the raised exception; the adapter does not modify native tensor storage.

Reprepare corrected code in the new root above. Stage and run ONE CPU-only
API+real-data validation job first (4 CPU,16GB,20-minute cap; <=1.34 CPU-hours,
no GPU training/sampling). Require a fresh approval for that staging/job preview.
Only after all real-data checks and baseline rescore finish successfully should
the expensive train/sample recovery be previewed separately. Source checkpoints,
main, C4, the immutable original runtime and shared packages remain untouched.
