# DiT-L16 Runtime Compatibility and Pin Audit Design

## Purpose

Make the different-seed DiT-L16 continuation from the existing 300k
checkpoints to 500k start reliably on Great Lakes without changing the model,
the source checkpoints, or the frozen scientific configuration.

The immediate failure is an import-time compatibility error:

```text
cosmodiff -> diffusers -> diffusers/utils/torch_utils.py
"xpu": torch.xpu.empty_cache
AttributeError: module 'torch' has no attribute 'xpu'
```

The Great Lakes runtime uses PyTorch `2.1.2+cu118`, while the installed
Diffusers code probes newer backend namespaces at import time. The repository
already provides compatibility shims in several entry points, but the
immutable cosmodiff pin builder and verifier launch child Python processes
whose environment omits the generated shim directory. This design replaces
that fragile, duplicated behavior with one explicit and auditable runtime
contract.

C4-v3 is outside this change. Its code, completed results, runtime, and output
directories must remain untouched.

## Verified root cause

The current pin builder and verifier construct a new child environment and
assign:

```python
env["PYTHONNOUSERSITE"] = "1"
env["PYTHONPATH"] = str(root)
```

That assignment overwrites the parent `PYTHONPATH`. Consequently:

1. the generated `sitecustomize.py` directory is absent from the child path;
2. the compatibility shim does not run before the child imports cosmodiff;
3. cosmodiff imports Diffusers;
4. Diffusers dereferences `torch.xpu` while constructing its device dispatch
   table;
5. PyTorch 2.1.2 raises the observed `AttributeError`.

The observed cause is therefore both candidate **(c)**, a freshly constructed
subprocess environment that drops the required path, and candidate **(d)**,
the stub directory being absent from the child's `PYTHONPATH`.

Candidate **(b)** is ruled out by the command itself: the child is started with
`-c`, not `-S` or `-E`. Candidate **(a)**, a Great Lakes Anaconda
`sitecustomize.py` shadowing the repository-generated one, is not the immediate
cause because the intended stub directory is never supplied to this child.
The Great Lakes no-job preflight will nevertheless print the selected
`sitecustomize.__file__` so future shadowing is visible.

## Design principles

1. **Explicit import is the contract.** Correctness must not depend only on
   Python's implicit `sitecustomize` selection.
2. **One canonical implementation.** Backend compatibility behavior lives in
   one Python module and all editable entry points call it before importing
   Diffusers or cosmodiff.
3. **Fail closed.** A late or missing shim produces a short actionable error
   before Diffusers reaches an unrelated attribute failure.
4. **The child runtime is audited as a unit.** Python executable, ordered
   import roots, backend shim, sklearn isolation, package metadata, and file
   hashes are verified together.
5. **No scientific drift.** The continuation manifest, resume seed, schedule,
   checkpoint lineage, and original 300k checkpoint bytes remain unchanged.

## 1. Canonical Torch compatibility module

Create `simdiff_eval/torch_compat.py`. Its public entry point installs the
backend namespace compatibility needed by the repository's pinned runtime:

```python
install_torch_backend_compat(entry_point=__name__)
```

The installer will:

- import PyTorch without importing Diffusers;
- add only missing backend namespaces and methods required by the installed
  Diffusers import path;
- cover the audited `xpu`, `mps`, and `npu` probes, including harmless
  `empty_cache`, availability, and device-count behavior where required;
- leave real backends untouched;
- be idempotent;
- set an internal process marker containing the module path, caller-supplied
  entry-point name, and compatibility schema version;
- return a structured report that can be printed and tested.

The compatibility objects are import guards, not simulated accelerators. They
must report unavailable and must never select a non-CUDA device.

### Import-order guard

Before installing anything, the module checks `sys.modules`. If Diffusers is
already present and the canonical compatibility marker is absent, it raises a
dedicated error such as:

```text
Torch compatibility was installed too late in <entry point>: diffusers is
already imported. Import simdiff_eval.torch_compat and call
install_torch_backend_compat(...) before importing diffusers or cosmodiff.
```

Editable entry points that import Diffusers or cosmodiff must call the
installer before those imports. A repository policy test will find those
imports and verify the canonical call appears first. The eight existing
copies of the xpu/mps shim will be removed or reduced to thin adapters.

## 2. `sitecustomize` becomes a convenience adapter

Keep `scripts/write_diffusers_runtime_sitecustomize.py` for subprocesses and
third-party entry points that cannot be edited. It will generate a small
adapter that loads the canonical module from the pinned code root and invokes
the same installer. It will no longer contain a second implementation of the
backend logic.

Every runtime that generates this adapter records:

- the absolute canonical code root;
- the generated adapter SHA256;
- the canonical `torch_compat.py` SHA256;
- the compatibility schema version.

The adapter remains useful as early startup protection, but every editable
repository entry point also calls the canonical installer explicitly. Thus a
decoy or shadowed `sitecustomize.py` cannot silently break the supported path.

## 3. Audited child-process runtime

The immutable pin builder and runtime verifier will share one helper for
constructing child environments. The ordered `PYTHONPATH` is:

1. an audited temporary runtime root containing the generated
   `sitecustomize.py` and sklearn stub;
2. the exact repository code root containing `simdiff_eval/torch_compat.py`;
3. the immutable cosmodiff pin root;
4. only explicitly approved residual paths.

The helper preserves required parent variables without blindly inheriting a
contaminated `PYTHONPATH`, sets `PYTHONNOUSERSITE=1`, and rejects known
incompatible roots. Child commands use neither `-S` nor `-E`.

The child verification program explicitly calls
`install_torch_backend_compat()` before importing cosmodiff. It then imports
the required modules and prints:

- `sys.executable`;
- the loaded `sitecustomize.__file__`, or a clear `NOT_LOADED` value;
- `torch.__file__` and version;
- `sklearn.__file__` and runtime kind;
- `cosmodiff.__file__` and source-distribution version;
- the canonical compatibility report;
- the complete ordered child `sys.path`.

Verification fails if any printed path or version differs from the manifest
contract.

## 4. sklearn isolation

Great Lakes' system Anaconda sklearn 1.2.1 cannot safely enter this runtime;
its compiled stack has already produced a `GLIBCXX_3.4.29` failure. The class
virtual environment does not contain sklearn, so asserting that
`sklearn.__file__` starts with the virtual-environment prefix would contradict
the current intentional design.

For the DiT continuation, the correct fail-closed contract is:

- `torch.__file__` must be inside
  `/home/jiamingp/venvs/cosmodiff_nf_class`;
- `sklearn.__file__` must be inside the audited runtime root created for the
  job and must identify the repository's narrow sklearn stub;
- neither module may resolve inside
  `/sw/pkgs/arc/python3.10-anaconda/2023.03` or the incompatible
  `/home/jiamingp/venvs/cosmodiff_nf` site-packages root;
- both resolved paths are printed before a failure.

The stub exports only the API used by this training path. It is not permitted
for unpickling frozen probe heads or for C4 analysis.

## 5. Immutable-pin provenance

Preserve the existing atomic pin publication and exact base-revision checks.
Extend the pin manifest with a runtime-compatibility section containing:

- compatibility schema version;
- canonical module repository-relative path and SHA256;
- generated `sitecustomize.py` SHA256;
- sklearn-stub file inventory and SHA256 values;
- ordered child `PYTHONPATH` roles;
- Python executable and resolved executable path;
- applied cosmodiff patch scripts and SHA256 values;
- package-metadata distribution name/version;
- imported module paths and versions from successful child verification.

The verifier recomputes all hashes and rejects missing, modified, or extra
files before importing cosmodiff. A failed build never publishes the staging
tree. Existing published pins are not mutated in place; the repaired runtime
creates a new immutable pin identity.

The original 300k checkpoints remain read-only inputs. Seeding the dedicated
restart tree retains the existing byte-identity verification and never writes
to the source checkpoint directories.

## 6. Entry-point migration

Audit every editable entry point that imports Diffusers or cosmodiff. The
known duplicated compatibility locations include:

- `scripts/sample_cosmodiff.py`;
- `scripts/check_nf_class_conditional_u128_runtime.py`;
- `scripts/check_nf_conditional_bias_probe_runtime.py`;
- `scripts/setup_nf_class_conditional_env.sh`;
- `scripts/preflight_nf_class_conditional_u128.sh`;
- `scripts/slurm/sample_nf_class_conditional_u128.sbatch`;
- `scripts/slurm/train_nf_class_conditional_u128.sbatch`;
- `scripts/write_diffusers_runtime_sitecustomize.py`.

Python entry points will explicitly invoke the canonical module. Shell and
Slurm entry points will generate or select the audited runtime root and call a
small Python preflight that invokes the canonical module; they will not embed
their own backend namespace implementation.

The seed-restart builder, verifier, precheck, and training job are included in
the same policy audit. C4-v3-specific execution and result paths are excluded
from behavioral changes.

## 7. Subprocess regression tests

Tests use real child Python processes rather than only in-process monkeypatches.
They cover:

1. a controlled fake Torch module without `xpu`, where the unshimmed
   Diffusers-style import reproduces the observed `AttributeError`;
2. the same child with the canonical explicit installer, which succeeds;
3. a decoy `sitecustomize.py` earlier on `sys.path`, where the explicitly
   shimmed entry point still succeeds;
4. a child whose environment tries to introduce an incompatible Anaconda or
   alternate-venv path, which fails with both resolved module paths printed;
5. the pin builder's child retaining the runtime root first in `PYTHONPATH`;
6. no use of `-S` or `-E`;
7. a late call after Diffusers appears in `sys.modules`, which fails with the
   actionable import-order message;
8. idempotent repeated installation;
9. pin-manifest rejection after canonical shim, generated adapter, or sklearn
   stub tampering;
10. atomic cleanup when child import verification fails.

The subprocess fixtures remain local and deterministic; they do not download
models, import C4 results, or submit Slurm jobs.

## 8. Next-failure audit

The repaired preflight must audit the next likely version-skew failures before
submission:

- **Other Torch backends:** inspect the installed Diffusers import path for
  import-time `torch.mps`, `torch.npu`, `torch.backends.*`, and related probes.
  Add only the unavailable namespace behavior actually required for import;
  fail if a probe could select a fake accelerator.
- **Source-tree package metadata:** confirm
  `patch_cosmodiff_package_metadata.py` is applied to the new seed-restart pin
  and that `importlib.metadata.version("cosmodiff")` returns the recorded
  source version.
- **NumPy ABI skew:** confirm the cosmodiff NumPy compatibility patch state and
  import the modules used by training under the exact child environment.
  Compiled modules must resolve consistently with the class virtual
  environment.
- **Hugging Face Hub drift:** import the exact Diffusers and hub symbols used by
  cosmodiff. If the installed APIs are incompatible, either apply a narrow
  recorded source patch or fail preflight with versions and missing symbols;
  do not defer the error to the GPU job.
- **sklearn mismatch:** verify that the DiT path uses only the audited stub and
  does not unpickle a frozen sklearn estimator. Any path that requires such an
  estimator is out of scope for the stub and must fail with an explicit
  message.

The final implementation report will state for each candidate whether it can
occur on this continuation path, what evidence was collected, and whether it
was fixed or documented as a residual risk.

## 9. Great Lakes no-job preflight and submission separation

Delivery provides two separate commands.

The first command is a read-only Great Lakes preflight. It fetches the exact
approved revision, creates or verifies a detached clean worktree, builds and
verifies the new immutable cosmodiff pin, audits runtime paths and package
versions, checks the exact 300k source checkpoints, and prints the intended
continuation rows. It contains no `sbatch`, `srun`, or scheduler mutation.

The second command is the final Slurm submission payload. It is shown only
after local verification passes and the preflight command is available. Its
preview will name all jobs, partition/account, resource requests, dependency
chain, checkpoint/result targets, and expected quota cost. Submitting it
requires a fresh `APPROVE RUN` for that exact payload.

Pushing the implementation likewise requires a separate preview and fresh
`APPROVE PUSH`. Local commits do not authorize either action.

## Verification and acceptance criteria

Implementation is accepted only when all of the following are true:

- subprocess tests reproduce the unshimmed failure and prove all shimmed
  paths succeed, including the decoy-sitecustomize case;
- the pin builder and verifier preserve the audited runtime root and print the
  selected sitecustomize path;
- Torch resolves from the class virtual environment and sklearn resolves from
  the audited stub, with incompatible roots absent;
- the pin manifest records and verifies every compatibility artifact hash;
- editable Diffusers/cosmodiff entry points use the canonical installer before
  those imports;
- package metadata, NumPy, Hugging Face Hub, and backend-probe audits complete
  or fail early with actionable evidence;
- original 300k checkpoints remain byte-identical and unmodified;
- C4-v3 files and results have no diff;
- focused tests, the full test suite, `python -m compileall`, `bash -n` for
  changed shell/Slurm files, and `git diff --check` all pass;
- no Slurm job is submitted during implementation or local verification.
