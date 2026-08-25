# DiT-L16 Runtime Compatibility and Pin Audit Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the Great Lakes different-seed DiT-L16 continuation import and verify its exact PyTorch/Diffusers/cosmodiff runtime before submitting the existing 300k-to-500k training chain.

**Architecture:** Put all PyTorch version-compatibility behavior in `simdiff_eval/torch_compat.py`, and make explicit installation before Diffusers/cosmodiff imports the supported contract. Build a hashed runtime directory containing a thin `sitecustomize` adapter and narrow sklearn stub into each immutable cosmodiff pin, then use one child-import auditor from both the builder and verifier.

**Tech Stack:** Python 3.10, PyTorch 2.1.2+cu118 on Great Lakes, Diffusers, pathlib, subprocess, JSON/SHA256 manifests, Bash/Slurm, pytest.

**Spec:** `docs/superpowers/specs/2026-08-25-dit-runtime-compat-design.md`

## Global Constraints

- Do not modify C4-v3 code, runtime, or completed results.
- Do not push, merge, or submit/cancel Slurm jobs while implementing or testing.
- Preserve branch `codex/iaifi-poster-copy` lineage from `3d78f91da7ac035e8e7faaa87342ea948cfe403b`.
- Preserve the original 300k DiT-L16 checkpoints byte-for-byte; only the dedicated seed-restart tree may receive continuation outputs.
- Preserve resume seed `456` and the exact staged targets `340k`, `380k`, `420k`, `460k`, and `500k` for `d2p08` and `d2p10`.
- Preserve `/home/jiamingp/venvs/cosmodiff_nf_class/bin/python` without resolving its symlink.
- Torch must resolve inside `/home/jiamingp/venvs/cosmodiff_nf_class`; sklearn must resolve inside the audited narrow runtime stub, not Great Lakes Anaconda.
- Treat `/home/jiamingp/venvs/cosmodiff_nf/lib/python3.10/site-packages` and `/sw/pkgs/arc/python3.10-anaconda/2023.03` as incompatible runtime roots.
- Keep immutable-pin publication atomic and fail closed on missing, modified, or extra recorded files.
- Read `superpowers:test-driven-development/references/writing-good-tests.md` before the first test edit, and observe every required RED failure before production implementation.

---

### Task 1: Canonical Torch compatibility module

**Files:**
- Create: `simdiff_eval/torch_compat.py`
- Create: `tests/test_torch_compat.py`

**Interfaces:**
- Produces: `COMPAT_SCHEMA_VERSION: int = 1`
- Produces: `TorchCompatibilityError(RuntimeError)`
- Produces: `install_torch_backend_compat(*, entry_point: str, torch_module: ModuleType | None = None) -> dict[str, Any]`
- Produces: `get_torch_compat_report(torch_module: ModuleType | None = None) -> dict[str, Any]`
- Contract: the marker `_simdiff_eval_torch_compat` lives on the selected Torch module and contains `schema_version`, `module_file`, `entry_points`, and `installed_attributes`.

- [ ] **Step 1: Read the test-quality reference before editing tests**

```bash
sed -n '1,260p' /Users/apple/.codex/plugins/cache/openai-curated-remote/superpowers/6.3.0/skills/test-driven-development/references/writing-good-tests.md
```

- [ ] **Step 2: Write the baseline subprocess reproduction**

Create a fixture root with a minimal `torch.py` lacking `xpu` and a minimal
`diffusers/__init__.py` containing:

```python
import torch
DEVICE_EMPTY_CACHE = {"xpu": torch.xpu.empty_cache}
```

Run a child with only the fixture root on `PYTHONPATH` and assert:

```python
assert completed.returncode != 0
assert "AttributeError" in completed.stderr
assert "has no attribute 'xpu'" in completed.stderr
```

- [ ] **Step 3: Run the reproduction test and record the baseline**

```bash
pytest -q tests/test_torch_compat.py::test_unshimmed_child_reproduces_diffusers_xpu_failure
```

Expected: PASS because the child faithfully reproduces the original failure.

- [ ] **Step 4: Add failing canonical-shim tests**

Add subprocess and focused unit tests covering:

```python
report = install_torch_backend_compat(
    entry_point="tests.shimmed_child",
    torch_module=fake_torch,
)
assert fake_torch.xpu.is_available() is False
assert fake_torch.xpu.device_count() == 0
assert fake_torch.xpu.empty_cache() is None
assert fake_torch.mps.is_available() is False
assert fake_torch.npu.is_available() is False
assert report["schema_version"] == 1
second = install_torch_backend_compat(
    entry_point="tests.second_call",
    torch_module=fake_torch,
)
assert second["installed_attributes"] == report["installed_attributes"]
```

Insert a fake `diffusers` module into `sys.modules` before installation and assert `TorchCompatibilityError` names the supplied entry point and says installation occurred too late.

- [ ] **Step 5: Run the canonical tests and confirm RED**

```bash
pytest -q tests/test_torch_compat.py -k 'shimmed or idempotent or late' -v
```

Expected: FAIL because `simdiff_eval.torch_compat` does not exist.

- [ ] **Step 6: Implement the minimal canonical installer**

Implement an unavailable backend object and idempotent installer:

```python
class _UnavailableBackend:
    def is_available(self) -> bool: return False
    def device_count(self) -> int: return 0
    def empty_cache(self) -> None: return None
    def _is_compiled(self) -> bool: return False
    def is_built(self) -> bool: return False
    def current_device(self) -> int: return 0
    def set_device(self, *args, **kwargs) -> None: return None
    def synchronize(self, *args, **kwargs) -> None: return None
    def manual_seed(self, *args, **kwargs) -> None: return None
    def manual_seed_all(self, *args, **kwargs) -> None: return None
```

Install missing members for `xpu`, `mps`, and `npu` without replacing a complete real backend. Move the existing float8/uint aliases, `torch.compiler`, pytree registration, device-mesh, and functional-collectives compatibility into this module. Do not catch a blanket `Exception`; every failure must raise `TorchCompatibilityError` with the entry-point name.

- [ ] **Step 7: Prove fake accelerators remain unavailable**

Assert all installed backend stubs report unavailable and zero devices. Assert a pre-existing complete backend object retains object identity.

- [ ] **Step 8: Run the complete focused test**

```bash
pytest -q tests/test_torch_compat.py
```

Expected: PASS, including unshimmed failure reproduction and shimmed success.

- [ ] **Step 9: Commit the canonical module**

```bash
git add simdiff_eval/torch_compat.py tests/test_torch_compat.py
git commit -m "fix: centralize torch diffusers compatibility"
```

### Task 2: Audited runtime assets and child environment

**Files:**
- Create: `simdiff_eval/seed_restart_runtime.py`
- Create: `tests/test_seed_restart_runtime.py`
- Modify: `scripts/write_diffusers_runtime_sitecustomize.py`

**Interfaces:**
- Consumes: `install_torch_backend_compat(...)` from Task 1.
- Produces: `RUNTIME_SCHEMA_VERSION: int = 1`
- Produces: `RUNTIME_DIR_NAME: str = "seed_restart_runtime"`
- Produces: `write_runtime_assets(runtime_root: Path, *, code_root: Path, entry_point: str) -> dict[str, Any]`
- Produces: `build_child_env(base_env: Mapping[str, str], *, runtime_root: Path, code_root: Path, pin_root: Path, incompatible_paths: Sequence[Path] = (), approved_residual_paths: Sequence[Path] = ()) -> dict[str, str]`
- Produces: `runtime_file_inventory(runtime_root: Path) -> dict[str, dict[str, Any]]`
- Produces: `write_sitecustomize(path: Path, *, code_root: Path, entry_point: str) -> Path`

- [ ] **Step 1: Write failing runtime-layout tests**

Test that `write_runtime_assets` creates exactly:

```text
seed_restart_runtime/
  sitecustomize.py
  sklearn/__init__.py
  sklearn/metrics/__init__.py
```

Require `sklearn.__version__ == "0+simdiff-seed-restart-stub"`,
`sklearn.RUNTIME_KIND == "simdiff-seed-restart-stub"`, and `roc_curve`
to raise a message saying the stub is unsuitable for estimator loading.

Test exact path order:

```python
env = build_child_env(
    {"PATH": "/bin", "PYTHONPATH": f"{bad}:{residual}"},
    runtime_root=runtime_root,
    code_root=code_root,
    pin_root=pin_root,
    incompatible_paths=(bad,),
    approved_residual_paths=(residual,),
)
assert env["PYTHONPATH"].split(os.pathsep) == [
    str(runtime_root.resolve()),
    str(code_root.resolve()),
    str(pin_root.resolve()),
    str(residual.resolve()),
]
assert env["PYTHONNOUSERSITE"] == "1"
```

- [ ] **Step 2: Run runtime-layout tests and confirm RED**

```bash
pytest -q tests/test_seed_restart_runtime.py -k 'assets or child_env' -v
```

Expected: FAIL because the runtime-layout module does not exist.

- [ ] **Step 3: Implement deterministic runtime assets**

Generate a thin `sitecustomize.py` containing only:

```python
from pathlib import Path
from simdiff_eval import torch_compat as _torch_compat

_EXPECTED = Path(<absolute-code-root>) / "simdiff_eval/torch_compat.py"
if Path(_torch_compat.__file__).resolve() != _EXPECTED.resolve():
    raise RuntimeError("canonical torch compatibility module resolved outside the pinned code root")
_torch_compat.install_torch_backend_compat(entry_point=<entry-point>)
```

Write files with stable bytes and trailing newlines. Reject an existing runtime root containing different bytes; identical regeneration may succeed.

- [ ] **Step 4: Convert the writer into a thin adapter CLI**

Keep its positional output path and add:

```text
--code-root PATH
--entry-point NAME
```

Default `--code-root` to the repository root containing the invoked script. Remove the embedded backend implementation and delegate to `write_sitecustomize`.

- [ ] **Step 5: Add the decoy-sitecustomize subprocess test**

Create a decoy `sitecustomize.py` that sets `DECOY_LOADED=1` without patching Torch. Let Python load that decoy, then explicitly call the canonical installer before importing the fake Diffusers package. Assert:

```python
assert completed.returncode == 0
assert "DECOY_LOADED=1" in completed.stdout
assert "SHIMMED_IMPORT_PASSED" in completed.stdout
```

- [ ] **Step 6: Run focused tests**

```bash
pytest -q tests/test_seed_restart_runtime.py tests/test_torch_compat.py
```

Expected: PASS.

- [ ] **Step 7: Commit runtime construction**

```bash
git add simdiff_eval/seed_restart_runtime.py scripts/write_diffusers_runtime_sitecustomize.py tests/test_seed_restart_runtime.py
git commit -m "feat: build audited seed restart runtime"
```

### Task 3: One child-import auditor with next-failure checks

**Files:**
- Create: `scripts/check_cosmodiff_seed_restart_imports.py`
- Modify: `simdiff_eval/seed_restart_runtime.py`
- Modify: `tests/test_seed_restart_runtime.py`

**Interfaces:**
- Consumes: `build_child_env(...)` and the canonical installer.
- Produces: `collect_runtime_report(*, pin_root: Path, runtime_root: Path, code_root: Path, expected_torch_prefix: Path, incompatible_paths: Sequence[Path]) -> dict[str, Any]`
- Produces: `run_runtime_audit(python_bin: Path, *, pin_root: Path, runtime_root: Path, code_root: Path, expected_torch_prefix: Path, incompatible_paths: Sequence[Path]) -> dict[str, Any]`
- Output contract: the CLI's final stdout line starts with `SEED_RESTART_RUNTIME_JSON=` followed by strict JSON.

- [ ] **Step 1: Write failing subprocess audit tests**

Build fixture packages for `torch`, `diffusers`, `huggingface_hub`, and
`cosmodiff`. The cosmodiff fixture exposes `optim`, `utils`, `augment`, and
`transform`; Diffusers exposes `DDPMScheduler` and `DiTTransformer2DModel`;
Hub exposes `hf_hub_download` and `snapshot_download`.

Assert:

```python
assert report["python"]["executable"]
assert report["sitecustomize"]["file"].endswith("seed_restart_runtime/sitecustomize.py")
assert report["torch"]["compat"]["schema_version"] == 1
assert report["sklearn"]["runtime_kind"] == "simdiff-seed-restart-stub"
assert report["diffusers"]["symbols"] == ["DDPMScheduler", "DiTTransformer2DModel"]
assert report["huggingface_hub"]["symbols"] == ["hf_hub_download", "snapshot_download"]
assert set(report["cosmodiff"]["modules"]) == {
    "cosmodiff", "cosmodiff.optim", "cosmodiff.utils",
    "cosmodiff.augment", "cosmodiff.transform",
}
```

- [ ] **Step 2: Run audit tests and confirm RED**

```bash
pytest -q tests/test_seed_restart_runtime.py -k runtime_audit -v
```

Expected: FAIL because the audit CLI and runner do not exist.

- [ ] **Step 3: Implement explicit import order in the auditor**

The first project import and call must be:

```python
from simdiff_eval.torch_compat import install_torch_backend_compat

TORCH_COMPAT_REPORT = install_torch_backend_compat(
    entry_point="scripts.check_cosmodiff_seed_restart_imports"
)
```

Only then import Diffusers or cosmodiff. Record module paths and distribution versions. Import the two Hub symbols and two Diffusers classes used by this path so API drift fails in preflight.

- [ ] **Step 4: Implement fail-closed path validation**

Require:

```python
Path(torch.__file__).resolve().is_relative_to(expected_torch_prefix.resolve())
Path(sklearn.__file__).resolve().is_relative_to(runtime_root.resolve())
```

Require all cosmodiff module files under `pin_root`; require the canonical module and auditor under `code_root`; reject loaded modules or `sys.path` entries under an incompatible root. On failure, print Torch, sklearn, sitecustomize, Python executable, and complete `sys.path` values before raising.

- [ ] **Step 5: Add negative audit tests**

Cover wrong Torch prefix, sklearn from a fake Anaconda path, cosmodiff outside the pin, generated sitecustomize not selected, a missing Hub symbol, missing source-tree package metadata, and NumPy under an incompatible root. Assert each error names the offending path or symbol.

- [ ] **Step 6: Add production subprocess argument guards**

Invoke exactly `[str(python_bin), str(auditor), ...]`. Capture the vector in a test and assert `-S` and `-E` are absent. Assert runtime root remains first in `PYTHONPATH`.

- [ ] **Step 7: Run focused tests**

```bash
pytest -q tests/test_seed_restart_runtime.py
```

Expected: PASS.

- [ ] **Step 8: Commit the runtime auditor**

```bash
git add scripts/check_cosmodiff_seed_restart_imports.py simdiff_eval/seed_restart_runtime.py tests/test_seed_restart_runtime.py
git commit -m "feat: audit seed restart child imports"
```

### Task 4: Bind runtime hashes and audit output into the immutable pin

**Files:**
- Modify: `scripts/build_cosmodiff_seed_restart_pin.py`
- Modify: `scripts/verify_cosmodiff_seed_restart_runtime.py`
- Modify: `tests/test_nf_fig2_dit_l16_seed_restart500k.py`

**Interfaces:**
- Consumes: `write_runtime_assets(...)`, `runtime_file_inventory(...)`, and `run_runtime_audit(...)`.
- Changes: `build_pin(..., code_root: Path, expected_torch_prefix: Path, incompatible_paths: Sequence[Path] = ()) -> dict[str, Any]`
- Changes: `verify_pin(..., code_root: Path, expected_torch_prefix: Path, incompatible_paths: Sequence[Path] = (), check_source_contract: bool = True) -> dict[str, Any]`
- Manifest: increment `pin_schema_version` to `2` and add `runtime_compatibility`.

- [ ] **Step 1: Update fixtures and write failing provenance tests**

Pass `code_root=REPO_ROOT`, `expected_torch_prefix=Path(sys.prefix)`, and an
empty incompatible-path tuple in local fixture builds. Assert:

```python
runtime = manifest["runtime_compatibility"]
assert runtime["schema_version"] == 1
assert runtime["runtime_root"] == "seed_restart_runtime"
assert runtime["canonical_shim"]["path"] == "simdiff_eval/torch_compat.py"
assert len(runtime["canonical_shim"]["sha256"]) == 64
assert runtime["sitecustomize"]["path"] == "seed_restart_runtime/sitecustomize.py"
assert len(runtime["sitecustomize"]["sha256"]) == 64
assert runtime["sklearn_stub"]["files"]
assert runtime["python_executable"] == str(python_bin.absolute())
assert runtime["runtime_audit"]["sklearn"]["runtime_kind"] == "simdiff-seed-restart-stub"
```

- [ ] **Step 2: Run provenance tests and confirm RED**

```bash
pytest -q tests/test_nf_fig2_dit_l16_seed_restart500k.py -k 'pin and runtime' -v
```

Expected: FAIL because schema 1 has no runtime-compatibility record.

- [ ] **Step 3: Generate and audit runtime assets before publication**

Inside the existing staging tree, apply the exact four patch scripts, write `staging/seed_restart_runtime`, run the shared auditor with runtime root first, record hashes and normalized module paths, compute inventory, write schema 2, recheck inventory, and publish with the existing single `os.replace`.

- [ ] **Step 4: Verify runtime artifacts before importing the pin**

Reject schema 1, recompute the canonical module hash from `code_root`, recompute every runtime file hash, validate patch hashes and full inventory, and only then call `run_runtime_audit`. Compare the normalized audit result with the manifest.

- [ ] **Step 5: Add tamper and cleanup tests**

Independently modify a copied canonical shim, generated sitecustomize, sklearn stub, patch script, and cosmodiff source file. Assert verification fails before imports. Force child audit failure during build and assert destination absence and staging cleanup.

- [ ] **Step 6: Record package metadata and NumPy disposition**

Assert `patch_cosmodiff_package_metadata.py` remains first. Record cosmodiff source version, NumPy version/path, Diffusers version/path, and Hub version/path. If the frozen source imports without the precedent NumPy patch, record `numpy_compatibility.status = "not_required_after_import_audit"`; do not add an unnecessary fifth patch.

- [ ] **Step 7: Update both CLIs**

Add:

```text
--code-root PATH
--expected-torch-prefix PATH
--incompatible-python-path PATH
```

The incompatible-path option is repeatable.

- [ ] **Step 8: Run pin tests**

```bash
pytest -q tests/test_nf_fig2_dit_l16_seed_restart500k.py -k 'pin or immutable'
```

Expected: PASS.

- [ ] **Step 9: Commit immutable-pin integration**

```bash
git add scripts/build_cosmodiff_seed_restart_pin.py scripts/verify_cosmodiff_seed_restart_runtime.py tests/test_nf_fig2_dit_l16_seed_restart500k.py
git commit -m "fix: bind seed restart runtime to immutable pin"
```

### Task 5: Integrate the runtime into the no-job preflight and Slurm chain

**Files:**
- Create: `scripts/preflight_nf_generalize_fig2_dit_l16_seed_restart500k_no_submit.sh`
- Modify: `scripts/slurm/precheck_nf_generalize_fig2_dit_l16_seed_restart500k.sbatch`
- Modify: `scripts/slurm/train_nf_generalize_fig2_dit_l16_seed_restart500k_array.sbatch`
- Modify: `scripts/slurm/submit_nf_generalize_fig2_dit_l16_seed_restart500k.sh`
- Modify: `tests/test_nf_fig2_dit_l16_seed_restart500k.py`

**Interfaces:**
- Consumes: schema-2 pin and its `seed_restart_runtime` directory.
- Produces: a no-`sbatch` Great Lakes preflight.
- Preserves: exact precheck-report producer ID and the 5-stage dependency chain.

- [ ] **Step 1: Write failing wrapper tests**

Assert all runtime consumers define:

```bash
RUNTIME_ROOT=${COSMODIFF_PIN_ROOT}/seed_restart_runtime
export PYTHONNOUSERSITE=1
export PYTHONPATH="${RUNTIME_ROOT}:${CODE_ROOT}:${COSMODIFF_PIN_ROOT}"
```

Assert they pass code root, expected Torch prefix, and both incompatible paths to the verifier. Assert they no longer write ad hoc sklearn or sitecustomize files.

For the no-job preflight, assert no `sbatch`, `srun`, `scancel`, or scheduler command appears and that it prints both dataset tags, source checkpoint paths, source hashes, all stage targets, and seed 456.

- [ ] **Step 2: Run wrapper tests and confirm RED**

```bash
pytest -q tests/test_nf_fig2_dit_l16_seed_restart500k.py -k 'wrapper or no_submit or slurm'
```

Expected: FAIL because wrappers still construct per-job stubs and the no-job script is absent.

- [ ] **Step 3: Use the pinned runtime in all three consumers**

Stop writing sklearn files and generating sitecustomize. Use the exact three-entry `PYTHONPATH` after the venv sanitizer. Pass `${VENV_PATH}` as expected Torch prefix and both incompatible roots as repeated verifier options.

- [ ] **Step 4: Preserve checkpoint and report safety**

Keep terminal-report lifecycle commands, producer ID matching, source byte validation, EMA-step checks, clean code-root checks, and dedicated output checks. Require the schema-2 runtime audit before starting the `INCOMPLETE` report.

- [ ] **Step 5: Implement the no-job Great Lakes preflight**

Accept `PROJECT_DIR`, `CODE_ROOT`, `EXPECTED_COMMIT`, `PYTHON_BIN`, `EXPECTED_TORCH_PREFIX`, `BASE_COSMODIFF_DIR`, `EXPECTED_COSMODIFF_BASE_REVISION`, and `COSMODIFF_PIN_ROOT`. Verify clean roots; build a new pin only if absent, otherwise verify it; run manifest check-only validation; validate but never seed/modify original 300k checkpoints; print two rows and five targets; print `NO JOB SUBMITTED` last.

- [ ] **Step 6: Run wrapper tests and syntax checks**

```bash
pytest -q tests/test_nf_fig2_dit_l16_seed_restart500k.py -k 'wrapper or no_submit or slurm'
bash -n scripts/preflight_nf_generalize_fig2_dit_l16_seed_restart500k_no_submit.sh
bash -n scripts/slurm/precheck_nf_generalize_fig2_dit_l16_seed_restart500k.sbatch
bash -n scripts/slurm/train_nf_generalize_fig2_dit_l16_seed_restart500k_array.sbatch
bash -n scripts/slurm/submit_nf_generalize_fig2_dit_l16_seed_restart500k.sh
```

Expected: all PASS.

- [ ] **Step 7: Commit wrapper integration**

```bash
git add scripts/preflight_nf_generalize_fig2_dit_l16_seed_restart500k_no_submit.sh scripts/slurm/precheck_nf_generalize_fig2_dit_l16_seed_restart500k.sbatch scripts/slurm/train_nf_generalize_fig2_dit_l16_seed_restart500k_array.sbatch scripts/slurm/submit_nf_generalize_fig2_dit_l16_seed_restart500k.sh tests/test_nf_fig2_dit_l16_seed_restart500k.py
git commit -m "fix: use pinned runtime in DiT restart jobs"
```

### Task 6: Remove duplicated shims and enforce import order

**Files:**
- Create: `tests/test_torch_compat_policy.py`
- Modify: `scripts/sample_cosmodiff.py`
- Modify: `scripts/check_nf_class_conditional_u128_runtime.py`
- Modify: `scripts/check_nf_conditional_bias_probe_runtime.py`
- Modify: `scripts/setup_nf_class_conditional_env.sh`
- Modify: `scripts/preflight_nf_class_conditional_u128.sh`
- Modify: `scripts/slurm/train_nf_class_conditional_u128.sbatch`
- Modify: `scripts/slurm/sample_nf_class_conditional_u128.sbatch`
- Modify: `scripts/check_nf_generalize_fig2_dit_resume.py`
- Modify: `scripts/check_nf_generalize_fig2_dit_runtime.py`
- Modify: `scripts/estimate_nf_generalize_scaling.py`
- Modify: `scripts/print_model_param_count.py`
- Modify: `scripts/run_cosmodiff_train_with_dit_resume.py`
- Modify: `simdiff_eval/io.py`
- Modify: `scripts/patch_cosmodiff_direct_unet_checkpoint.py`
- Modify embedded Python in: `scripts/setup_cosmodiff_main.sh`, `scripts/setup_cosmodiff_normalization_fixes.sh`, `scripts/slurm/sample_nf_sweep_array.sbatch`, `scripts/slurm/sample_repro_u64_u128_array.sbatch`, `scripts/slurm/train_nf_conditional_u128.sbatch`, `scripts/slurm/train_nf_conditional_u128_continue400k_array.sbatch`, `scripts/slurm/train_nf_conditional_u128_continue800k_array.sbatch`, `scripts/slurm/train_nf_generalize_fig2_array.sbatch`, `scripts/slurm/train_nf_generalize_fig2_continue400k.sbatch`, `scripts/slurm/train_nf_generalize_fig2_dit_array.sbatch`, `scripts/slurm/train_nf_generalize_fig2_u128_d2p15_continue24h.sbatch`, `scripts/slurm/train_nf_generalize_fig2_u256_continue_array.sbatch`, `scripts/slurm/train_nf_generalize_nick_data_array.sbatch`, `scripts/slurm/train_nf_sweep_array.sbatch`, `scripts/slurm/train_nf_sweep_aug_array.sbatch`, `scripts/slurm/train_nf_sweep_ema_sigma_array.sbatch`, `scripts/slurm/train_nf_sweep_small_ema_array.sbatch`, `scripts/slurm/train_nf_sweep_v2_array.sbatch`, and `scripts/slurm/train_repro_u64_u128_array.sbatch`.

**Interfaces:**
- Consumes: canonical installer from Task 1.
- Produces: policy tests rejecting handwritten backend shims and late imports.
- Preserves: third-party cosmodiff entry points use the adapter where external source cannot be edited.

- [ ] **Step 1: Write the failing duplicate-shim policy test**

Scan production files and reject these fragments outside the canonical module:

```python
"class _OptionalDeviceStub"
"class TorchOptionalDeviceStub"
"for _backend in (\"xpu\", \"mps\")"
"for backend in (\"xpu\", \"mps\")"
```

- [ ] **Step 2: Write the failing import-order policy test**

For Python files, parse AST and require `install_torch_backend_compat` before the first executable Diffusers/cosmodiff import. For shell/Slurm, extract each `<<'PY' ... PY` block and apply the same rule. Exempt patch-generator string literals, but verify their generated source separately.

Accepted prefix for ordinary modules:

```python
from simdiff_eval.torch_compat import install_torch_backend_compat
install_torch_backend_compat(entry_point=__name__)
```

The policy also accepts a non-empty literal entry-point name, as used by the
dedicated runtime auditor.

- [ ] **Step 3: Run policy tests and confirm RED**

```bash
pytest -q tests/test_torch_compat_policy.py -v
```

Expected: FAIL and list current duplicated or late-import files.

- [ ] **Step 4: Replace the remaining handwritten implementations**

Remove backend classes/loops from the sampler, runtime checker, class-environment setup, class preflight, and two class Slurm jobs. Add the canonical prefix before Diffusers/cosmodiff imports. Keep diagnostics and tiny forward checks.

- [ ] **Step 5: Migrate remaining editable entry points**

Add the same prefix to every Python file and heredoc listed above. In `simdiff_eval/io.py`, call immediately before its lazy cosmodiff import. Make the direct-UNet patch generator insert the canonical call before generated Diffusers imports.

- [ ] **Step 6: Preserve frozen-root diagnostics**

Extend existing runtime tests to assert the compatibility report names the entry point while existing cosmodiff and Torch path assertions remain unchanged.

- [ ] **Step 7: Run policy and affected regression tests**

```bash
pytest -q tests/test_torch_compat_policy.py tests/test_torch_compat.py tests/test_nf_fig2_continuation_guards.py tests/test_nf_fig2_dit_sampling_audit.py tests/test_nf_fig2_dit_l16_seed_restart500k.py
```

Expected: PASS.

- [ ] **Step 8: Syntax-check changed shell/Slurm files**

Run `bash -n` once per changed `.sh` and `.sbatch` path listed in this task. Expected: exit 0 for every file.

- [ ] **Step 9: Commit migration**

```bash
git add \
  simdiff_eval/io.py \
  scripts/sample_cosmodiff.py \
  scripts/check_nf_class_conditional_u128_runtime.py \
  scripts/check_nf_conditional_bias_probe_runtime.py \
  scripts/setup_nf_class_conditional_env.sh \
  scripts/preflight_nf_class_conditional_u128.sh \
  scripts/check_nf_generalize_fig2_dit_resume.py \
  scripts/check_nf_generalize_fig2_dit_runtime.py \
  scripts/estimate_nf_generalize_scaling.py \
  scripts/print_model_param_count.py \
  scripts/run_cosmodiff_train_with_dit_resume.py \
  scripts/patch_cosmodiff_direct_unet_checkpoint.py \
  scripts/setup_cosmodiff_main.sh \
  scripts/setup_cosmodiff_normalization_fixes.sh \
  scripts/slurm/sample_nf_class_conditional_u128.sbatch \
  scripts/slurm/sample_nf_sweep_array.sbatch \
  scripts/slurm/sample_repro_u64_u128_array.sbatch \
  scripts/slurm/train_nf_class_conditional_u128.sbatch \
  scripts/slurm/train_nf_conditional_u128.sbatch \
  scripts/slurm/train_nf_conditional_u128_continue400k_array.sbatch \
  scripts/slurm/train_nf_conditional_u128_continue800k_array.sbatch \
  scripts/slurm/train_nf_generalize_fig2_array.sbatch \
  scripts/slurm/train_nf_generalize_fig2_continue400k.sbatch \
  scripts/slurm/train_nf_generalize_fig2_dit_array.sbatch \
  scripts/slurm/train_nf_generalize_fig2_u128_d2p15_continue24h.sbatch \
  scripts/slurm/train_nf_generalize_fig2_u256_continue_array.sbatch \
  scripts/slurm/train_nf_generalize_nick_data_array.sbatch \
  scripts/slurm/train_nf_sweep_array.sbatch \
  scripts/slurm/train_nf_sweep_aug_array.sbatch \
  scripts/slurm/train_nf_sweep_ema_sigma_array.sbatch \
  scripts/slurm/train_nf_sweep_small_ema_array.sbatch \
  scripts/slurm/train_nf_sweep_v2_array.sbatch \
  scripts/slurm/train_repro_u64_u128_array.sbatch \
  tests/test_torch_compat_policy.py
git commit -m "refactor: enforce canonical torch compatibility"
```

### Task 7: Full verification, provenance audit, and delivery previews

**Files:**
- Modify only if evidence requires correction: files changed in Tasks 1-6.
- Create: `docs/superpowers/reports/2026-08-25-dit-runtime-compat-verification.md`

**Interfaces:**
- Consumes: all prior tasks.
- Produces: test evidence, root-cause evidence, next-risk audit, final local SHA, no-job preflight command, and separate protected push/submission previews.

- [ ] **Step 1: Compile Python**

```bash
python -m compileall -q simdiff_eval scripts tests
```

Expected: exit 0.

- [ ] **Step 2: Syntax-check all changed shell files**

List changed `.sh` and `.sbatch` files with Git and run `bash -n` on each. Record every path and exit status.

- [ ] **Step 3: Run focused suites**

```bash
pytest -q tests/test_torch_compat.py tests/test_seed_restart_runtime.py tests/test_torch_compat_policy.py tests/test_nf_fig2_dit_l16_seed_restart500k.py tests/test_dit_checkpoint_resume.py
```

Expected: PASS.

- [ ] **Step 4: Run the full suite**

```bash
python -m pytest -q
```

Expected: PASS; copy exact count and duration into the report.

- [ ] **Step 5: Prove C4-v3 is untouched**

```bash
git diff 3d78f91da7ac035e8e7faaa87342ea948cfe403b -- \
  scripts/evaluate_probe_c4_umap.py \
  simdiff_eval/probe_c4_umap.py \
  scripts/slurm/probe_c4_frozen_vgg_umap.sbatch \
  tests/test_probe_c4_umap.py
```

Expected: no output.

- [ ] **Step 6: Run hygiene checks**

```bash
git diff --check
git status --short
```

Expected: no whitespace errors; only intended verification-report changes before its commit.

- [ ] **Step 7: Write the evidence report**

Record root cause candidates (c) and (d) with evidence; files changed; exact RED and GREEN output; loaded sitecustomize/Torch/sklearn/Diffusers/Hub/NumPy/cosmodiff paths and versions; Great Lakes-only checks; each next-failure disposition; proof checkpoints and C4 were untouched; final local SHA.

- [ ] **Step 8: Commit the report**

```bash
git add docs/superpowers/reports/2026-08-25-dit-runtime-compat-verification.md
git commit -m "docs: record DiT runtime compatibility verification"
```

- [ ] **Step 9: Prepare but do not execute the no-job Great Lakes command**

Provide the complete command that checks out the exact final commit and invokes the no-submit preflight. State that it writes only a new immutable scratch pin and diagnostics, does not modify source checkpoints, requests no GPU, and submits no job.

- [ ] **Step 10: Prepare but do not execute protected actions**

Provide separately:

1. exact push target and command, plus remote branch side effect, then stop for fresh `APPROVE PUSH`;
2. exact Slurm environment and command, including account, partition, five prechecks, ten 48-hour array tasks, dependency chain, targets, and worst-case requested GPU-hours, then stop for fresh `APPROVE RUN`.

Do not combine approval gates or treat spec/plan approval as authorization.
