# DiT seed-restart runtime compatibility verification

Date: 2026-08-25

Implementation head before this report: `94cf031b` (`codex/probe-c4-umap`)

Base revision: `3d78f91da7ac035e8e7faaa87342ea948cfe403b`

Scope: fix and test the different-seed DiT-L16 300k-to-500k continuation
runtime. C4-v3 code and results are explicitly out of scope.

## Root cause

The immediate `torch.xpu` failure was caused by candidates **(c) and (d)**,
with (d) as the direct import-time condition:

- At the base revision, the precheck created
  `results/cache/python_stubs/seed_restart_precheck_${SLURM_JOB_ID}` and put it
  first in the parent `PYTHONPATH`.
- `build_cosmodiff_seed_restart_pin.py::imported_modules` copied the parent
  environment and then replaced `PYTHONPATH` with only the staging pin root:
  `env["PYTHONPATH"] = str(root)`. This discarded the parent runtime path
  (candidate c) and left the generated shim directory absent from the child
  path (candidate d).
- The child command was `[python_bin, "-c", program]`, so neither `-S` nor
  `-E` was present. Candidate (b) is false.
- A cluster/base `sitecustomize.py` may still exist, but it was not the
  necessary cause. The explicit shim subprocess test succeeds even when a
  decoy `sitecustomize.py` is first. Candidate (a) is therefore not the root
  cause.

The fix makes compatibility an explicit import contract instead of depending
on whichever `sitecustomize.py` Python happens to select.

## Implemented design

- `simdiff_eval/torch_compat.py` is the only handwritten Torch compatibility
  implementation. It is idempotent, records all entry points, marks unavailable
  `xpu`, `mps`, and `npu` backends unavailable, and fails closed if Diffusers was
  imported first.
- `simdiff_eval/seed_restart_runtime.py` builds one deterministic runtime root
  containing exactly the generated adapter and narrow sklearn stub.
- `scripts/check_cosmodiff_seed_restart_imports.py` is the one child import
  auditor. It verifies and reports the selected sitecustomize, canonical shim,
  Python executable, Torch, sklearn, NumPy, Diffusers, Hugging Face Hub, and all
  required Cosmodiff modules.
- The immutable pin is schema 2. It records SHA-256 hashes for the canonical
  shim, generated adapter, sklearn stub files, and every applied patch script,
  plus the normalized child audit.
- The precheck, train array, and submit chain all consume the runtime frozen
  inside the pin. They no longer construct per-job compatibility files.
- `scripts/preflight_nf_generalize_fig2_dit_l16_seed_restart500k_no_submit.sh`
  builds or verifies the pin and validates the two source rows without calling
  any Slurm command.
- A repository policy test rejects copied shims and any editable
  Diffusers/Cosmodiff import that precedes the canonical installer, including
  Python heredocs and generated direct-UNet patch source.

## Test-first evidence

Observed RED failures included:

- an unshimmed real subprocess fixture failed with
  `AttributeError: module 'torch' has no attribute 'xpu'`;
- canonical-shim tests initially failed with `ModuleNotFoundError`;
- schema-2 provenance tests initially failed because schema 1 had no runtime
  compatibility record;
- canonical-shim tampering initially produced `DID NOT RAISE` until the hash
  verification was restored;
- the import policy initially reported six copied shim implementations and 59
  late import sites;
- the generated direct-UNet patch test initially reported two late Diffusers
  imports.

Observed GREEN evidence after implementation:

- canonical shim checkpoint: `5 passed`;
- runtime assets and shim: `11 passed`;
- runtime auditor checkpoint: `15 passed`;
- immutable pin focused checkpoint: `9 passed, 9 deselected`;
- full seed-restart tests: `19 passed`;
- final affected regression group: `120 passed, 2 warnings`;
- final focused suite: `70 passed in 12.50s`;
- final full suite: `275 passed, 3 warnings in 26.56s`.

The three final warnings are unrelated: two existing `torch.load` future
warnings and one joblib physical-core detection warning.

`python -m compileall -q simdiff_eval scripts tests` exited 0.
`git diff --check 3d78f91..HEAD` exited 0. Every changed `.sh` and `.sbatch`
file passed `bash -n` (27 files).

## Import and provenance assertions

The child audit requires:

- `sitecustomize.__file__ == <PIN_ROOT>/seed_restart_runtime/sitecustomize.py`;
- `torch.__file__` under `/home/jiamingp/venvs/cosmodiff_nf_class`;
- `sklearn.__file__` under the pin runtime with
  `RUNTIME_KIND == "simdiff-seed-restart-stub"`;
- NumPy outside both incompatible Great Lakes site-package roots;
- Diffusers exports `DDPMScheduler` and `DiTTransformer2DModel`;
- Hugging Face Hub exports `hf_hub_download` and `snapshot_download`;
- `cosmodiff`, `optim`, `utils`, `augment`, and `transform` all resolve under
  the immutable pin;
- the canonical shim and auditor resolve under the frozen code root;
- `PYTHONPATH` begins with runtime root, code root, and pin root;
- the child command contains neither `-S` nor `-E`.

Exact Great Lakes paths and package versions are intentionally not claimed from
the macOS test environment. The no-job Great Lakes preflight runs this auditor
with the real class venv and records the exact values in
`seed_restart_pin_manifest.json` before any job may be submitted.

## Next-failure audit

- Other Torch/Diffusers version skew: `xpu`, `mps`, `npu`, compiler methods,
  dtype aliases, pytree registration, device mesh, and functional collectives
  are covered by the canonical installer and subprocess tests. Unknown future
  symbols fail during the no-job import audit.
- Cosmodiff package metadata: `patch_cosmodiff_package_metadata.py` remains the
  mandatory first patch; the auditor checks the resulting source-tree version.
- NumPy ABI skew: the auditor imports NumPy, prints its exact file/version, and
  rejects both known incompatible Great Lakes roots. The current pin records
  `not_required_after_import_audit`; no unnecessary NumPy patch is applied.
- Hugging Face Hub drift: both symbols used by this path are imported explicitly
  and recorded; a missing-symbol subprocess regression test fails closed.
- sklearn unpickling: the DiT continuation does not load a frozen sklearn probe
  head. The pin supplies a narrow import-only stub whose estimator-facing
  `roc_curve` raises an actionable error, preventing accidental estimator use.

## Scope and data-safety evidence

- The diff from the base revision is empty for
  `scripts/evaluate_probe_c4_umap.py`, `simdiff_eval/probe_c4_umap.py`,
  `scripts/slurm/probe_c4_frozen_vgg_umap.sbatch`, and
  `tests/test_probe_c4_umap.py`.
- No C4-v3 result file is changed.
- No checkpoint or result artifact is tracked in this change.
- The no-job preflight only inventories the original d2p08/d2p10 300k
  checkpoints. The submission workflow seeds separate restart directories with
  byte-identical copies and refuses an already-populated target.
- The five continuation targets remain 340k, 380k, 420k, 460k, and 500k, with
  resume seed 456 for both datasets.

## Great Lakes-only verification still required

Run the no-job preflight from an exact frozen checkout before any Slurm
submission. It may create one new immutable scratch pin and its manifest. It
does not request a GPU, submit a job, or modify the original 300k checkpoints.
Only after its printed audit and checkpoint inventories are reviewed should the
separate protected Slurm submission be approved.
