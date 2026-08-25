# DiT-L16 Seed-Restart Parse-to-First-Step Audit

## Outcome

The two stage-1 array tasks failed before model construction or any optimizer
update. Both reached the pinned runtime and loaded the dataset, then stopped in
the in-process constant-label adapter. The fixed adapter removes that blocker
without allowing genuine conditional labels to be overwritten.

No further definite incompatibility was found between `parse_config_data` and
the first `optimizer.step()` for the frozen `58c77eb` path. A Great Lakes GPU
run is still required to demonstrate the first real optimizer update because
the previous jobs never reached it.

## Root-cause correction

The base revision `58c77eb45de6e4d135697ba83ffee93ae54d918c` is
not natively constant-label aware. Its `parse_config_data` constructs
`ArrayDataset(..., labels=out['labels'])` and never reads
`data.constant_label`. The immutable pin builder applies
`patch_cosmodiff_constant_label.py`, which inserts the constant-label block.

Therefore the actual sequence was:

1. The pin patch created the all-zero label tensor.
2. The wrapper called the patched parser.
3. The wrapper saw non-`None` labels and treated them as genuine labels.
4. Both jobs raised before training.

Schema-3 pin manifests now record both `native_in_base_revision` and
`effective_in_published_pin`, plus whether support came from the base revision
or the declared patch.

## Ordered path audit

### 1. Frozen runtime verification

- **Assumption:** the venv Torch, sklearn isolation, Torch backend shim,
  Diffusers, Hugging Face Hub, and source-tree Cosmodiff imports coexist.
- **Evidence:** the no-submit preflight and stage-1 precheck passed; the failed
  jobs printed `PASS: immutable cosmodiff seed-restart pin`.
- **Disposition:** guarded by the runtime auditor and immutable inventory.

### 2. Dataset parsing and constant labels

- **Older-API assumption:** the wrapper assumed the external parser ignored
  `data.constant_label` and would return `labels=None`.
- **Actual pin:** the declared compatibility patch had already created labels.
- **Fix:** distinguish legacy injection, an already-correct no-op, and a real
  conflict. Existing labels are accepted only when their length and element
  count match the images and every element equals the requested constant.
- **Evidence:** separate tests cover all three required paths and verify the
  startup provenance line.

### 3. Read-only NumPy input

- **Observed:** both failed jobs emitted PyTorch's non-writable NumPy-array
  warning from `cosmodiff/utils.py`.
- **Cause:** with configured `n_samples`, the frozen first-`n` slice can remain
  read-only before `torch.as_tensor`.
- **Current effect:** not the exception; parsing completed. The current
  transform/normalization path does not intentionally write into the backing
  tensor.
- **Risk:** undefined behavior would be possible if a future augmentation or
  transform mutates it in place. The base source also contains
  `np.asarray(images, copy=True)`, which is incompatible with NumPy 1.26 but is
  not reached for this fixed-`n_samples` configuration. This remains a
  documented configuration-sensitive risk, not a blocker fixed in this patch.

### 4. Exact checkpoint discovery

- **Older-API assumption:** the external entry point chooses the latest
  directory and could select a half-written or out-of-stage checkpoint.
- **Guard:** `install_exact_checkpoint_finder` requires the configured output
  directory and returns the wrapper-validated exact checkpoint.
- **Evidence:** unit tests cover malformed, behind-stage, beyond-target, and
  already-complete targets.

### 5. Epoch semantics

- **Older-API assumption:** Cosmodiff versions disagree on whether
  `num_epochs` is an absolute endpoint or an additional duration.
- **Actual base:** `58c77eb` uses
  `range(start_epoch, start_epoch + num_epochs)` (additional duration).
- **Guard:** the AST-based adapter detects the semantics, validates the exact
  resume epoch, and computes the argument required for the exact target.
- **Evidence:** focused tests cover direct and indirect additional-duration
  loops and absolute-endpoint loops.

### 6. Model class reconstruction

- **Older-API assumption:** base `utils.load_checkpoint` uses
  `AutoModel.from_pretrained`, which is not a sufficiently strict DiT class
  contract.
- **Guard:** the pin applies the class-loader patch and the wrapper replaces
  `utils.load_checkpoint` with `load_checkpoint_preserving_class`, reads
  `_class_name`, resolves it from Diffusers, and rejects meta parameters.
- **Evidence:** the stage-1 precheck reconstructed
  `DiTTransformer2DModel` with 183,966,016 parameters for both datasets.

### 7. Optimizer, scheduler, scaler, and RNG restore

- **Older-API assumption:** checkpoint layouts may use legacy pickles or native
  Accelerate files.
- **Guard:** the loader accepts only coherent legacy or native pairs and the
  precheck requires a nonempty optimizer state, bound scheduler, scaler, and
  saved RNG state.
- **Evidence:** stage-1 precheck reported restored optimizer moments and
  scheduler progress for both 300k checkpoints.

### 8. New RNG trajectory

- **Assumption:** `Accelerator.load_state` restores the old RNG after model and
  optimizer setup.
- **Guard:** the class hook reseeds Python, NumPy, Torch CPU, and Torch CUDA to
  456 immediately after that restore and only at the copied 300k origin.
- **Evidence:** subprocess/unit coverage verifies ordering and later-stage
  checkpoint-RNG continuation.

### 9. Post-hoc EMA

- **Older-API assumption:** base `58c77eb` creates a fresh `PostHocEMA` after
  `load_state` and does not restore the two historical profiles.
- **Guard:** the wrapper replaces the EMA factory, loads both exact snapshots,
  validates their common step and sigma profiles, and sets effective burn-in to
  zero while retaining the original burn-in in checkpoint metadata.
- **Evidence:** profile restoration and absolute-step tests pass; the stage-1
  precheck validated the exact 1,199,000 and 1,199,128 source EMA steps.

### 10. Labels at model forward

- **Older-API assumption:** older training loops called the model without
  `class_labels`.
- **Actual base:** `58c77eb` already reads `batch['labels']` and passes
  `class_labels=labels` for discrete conditioning. The declared DiT-label
  patch is therefore idempotent for this source.
- **Guard:** the adapter guarantees `torch.long`, one label per image for the
  legacy path, and refuses conflicting preexisting labels.
- **Risk:** the exact CUDA DiT forward has not yet executed in this restart
  chain because the parser failed first.

### 11. Backward and first optimizer step

- **Assumption:** no gradient may be computed before full checkpoint restore.
- **Guard:** the Accelerator hook raises if `backward` occurs before
  `load_state`, then records the first resumed loss and its absolute optimizer
  and microbatch indices.
- **Base behavior:** Accelerate wraps `optimizer.step`; during gradient
  accumulation, non-sync calls are no-ops and the real update occurs at the
  configured accumulation boundary.
- **Evidence:** hook-order and first-loss tests pass. A live Great Lakes GPU
  run is the remaining end-to-end evidence.

## Other compatibility assumptions reviewed

- **Package metadata:** the pin's declared source-metadata patch handles
  source-tree imports and is checked by the immutable manifest.
- **Torch/Diffusers backend attributes:** the canonical explicit shim and
  subprocess regression tests cover missing XPU/MPS/NPU-style namespaces; this
  was confirmed working by the failed jobs reaching data parsing.
- **Hugging Face Hub/Diffusers versions:** import versions and locations are in
  the runtime audit. No later Hub call occurs on the local `from_pretrained`
  checkpoint path.
- **Checkpoint save completeness:** the declared checkpoint-state patch adds
  strict optimizer, scheduler, and noise-scheduler artifacts for newly saved
  stages; post-save validation refuses incomplete targets.
- **Existing failed-job artifacts:** failure happened before target checkpoint
  creation and before the Accelerate resume hook wrote a first-loss audit. The
  copied 300k source checkpoints were not modified.

## Remaining risks before declaring the chain end-to-end proven

1. Run the no-submit preflight against a newly built schema-3 pin.
2. Submit a new stage-1 chain only after the protected-action approval gate.
3. Confirm each `.out` contains
   `path=existing_constant_noop dtype=torch.int64 ... unique=[0]`.
4. Confirm each resume audit contains `first_resumed_loss` and
   `first_resumed_optimizer_step`.
5. Reassess the read-only NumPy warning separately if any in-place data
   augmentation is enabled.
