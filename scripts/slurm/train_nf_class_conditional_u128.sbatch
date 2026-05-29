#!/bin/bash
#SBATCH --job-name=nf_cls_u128
#SBATCH --partition=spgpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=80gb
#SBATCH --time=24:00:00
#SBATCH -o /home/jiamingp/diffusion_models_repo/logs/nf_class_conditional_u128/train_%j.out
#SBATCH -e /home/jiamingp/diffusion_models_repo/logs/nf_class_conditional_u128/train_%j.err
#SBATCH -A huterer0

set -euo pipefail

PROJECT_DIR=${PROJECT_DIR:-/home/jiamingp/diffusion_models_repo}
INHERITED_COSMODIFF_DIR=${COSMODIFF_DIR:-}
PREFERRED_COSMODIFF_DIR=/home/jiamingp/Diffusion_model/cosmo_diffusion_main
if [[ -n "${COSMODIFF_DIR_OVERRIDE:-}" ]]; then
  COSMODIFF_DIR=${COSMODIFF_DIR_OVERRIDE}
else
  COSMODIFF_DIR=${PREFERRED_COSMODIFF_DIR}
fi
if [[ -n "${INHERITED_COSMODIFF_DIR}" && -z "${COSMODIFF_DIR_OVERRIDE:-}" && "${INHERITED_COSMODIFF_DIR}" != "${COSMODIFF_DIR}" ]]; then
  echo "Ignoring inherited COSMODIFF_DIR=${INHERITED_COSMODIFF_DIR}; using ${COSMODIFF_DIR}" >&2
fi
VENV_PATH=${VENV_PATH:-/home/jiamingp/venvs/cosmodiff_nf_class}
BASE_VENV_PATH=${BASE_VENV_PATH:-/home/jiamingp/venvs/cosmodiff_nf}
PYTHON_BIN=${PYTHON_BIN:-python}
REQUIRED_COSMODIFF_BRANCH=${REQUIRED_COSMODIFF_BRANCH:-main}

cd "${PROJECT_DIR}"
source "${PROJECT_DIR}/scripts/nf_class_conditional_pythonpath.sh"
mkdir -p logs/nf_class_conditional_u128
if [[ -f "${VENV_PATH}/bin/activate" ]]; then
  source "${VENV_PATH}/bin/activate"
else
  echo "Missing class-conditional env: ${VENV_PATH}" >&2
  echo "Run first: bash scripts/setup_nf_class_conditional_env.sh" >&2
  exit 1
fi

set_nf_class_conditional_pythonpath "${PYTHON_BIN}" "${BASE_VENV_PATH}"

RUN_NAME=$("${PYTHON_BIN}" scripts/prepare_nf_class_conditional_u128_config.py --project-dir "${PROJECT_DIR}" --print-runs)
CONFIG_PATH="${PROJECT_DIR}/local/nf_class_conditional_u128/configs/${RUN_NAME}.yaml"
export CONFIG_PATH
"${PYTHON_BIN}" scripts/prepare_nf_class_conditional_u128_config.py --project-dir "${PROJECT_DIR}"

STUB_ROOT="${PROJECT_DIR}/results/cache/python_stubs"
mkdir -p "${STUB_ROOT}/sklearn/metrics"
printf 'from . import metrics\n' > "${STUB_ROOT}/sklearn/__init__.py"
printf "def roc_curve(*args, **kwargs):\n    raise RuntimeError('sklearn.metrics.roc_curve is stubbed for cosmodiff training')\n" > "${STUB_ROOT}/sklearn/metrics/__init__.py"
cat > "${STUB_ROOT}/sitecustomize.py" <<'PY'
try:
    from contextlib import nullcontext
    import sys
    from types import ModuleType
    import torch
    class _XPUStub:
        def is_available(self): return False
        def device_count(self): return 0
        def empty_cache(self): return None
        def _is_compiled(self): return False
        def current_device(self): return 0
        def set_device(self, *args, **kwargs): return None
        def synchronize(self, *args, **kwargs): return None
        def manual_seed(self, *args, **kwargs): return None
        def manual_seed_all(self, *args, **kwargs): return None
        def seed(self, *args, **kwargs): return 0
        def initial_seed(self, *args, **kwargs): return 0
        def get_rng_state(self, *args, **kwargs): return None
        def set_rng_state(self, *args, **kwargs): return None
        def is_built(self, *args, **kwargs): return False
        def current_stream(self, *args, **kwargs): return None
        def stream(self, *args, **kwargs): return nullcontext()
        def device(self, *args, **kwargs): return nullcontext()
        def memory_allocated(self, *args, **kwargs): return 0
        def max_memory_allocated(self, *args, **kwargs): return 0
        def reset_peak_memory_stats(self, *args, **kwargs): return None
        def get_device_name(self, *args, **kwargs): return "optional-device-unavailable"
        def get_device_properties(self, *args, **kwargs): return None
        def __getattr__(self, name):
            def missing(*args, **kwargs):
                if name.startswith("is_"):
                    return False
                return None
            return missing
    class _CompilerStub:
        def disable(self, fn=None, recursive=True):
            if fn is None:
                return lambda inner: inner
            return fn
        def is_compiling(self): return False
        def is_exporting(self): return False
    _stub = _XPUStub()
    _required = ("empty_cache", "is_available", "device_count", "manual_seed")
    for _backend in ("xpu", "mps"):
        _existing = getattr(torch, _backend, None)
        if _existing is None or any(not hasattr(_existing, _name) for _name in _required):
            setattr(torch, _backend, _stub)
            continue
        for _name in dir(_stub):
            if _name.startswith("__"):
                continue
            if not hasattr(_existing, _name):
                setattr(_existing, _name, getattr(_stub, _name))
    for _name in (
        "float8_e4m3fn",
        "float8_e4m3fnuz",
        "float8_e5m2",
        "float8_e5m2fnuz",
        "float8_e8m0fnu",
        "float4_e2m1fn_x2",
    ):
        if not hasattr(torch, _name):
            setattr(torch, _name, torch.float16)
    for _bits in range(1, 8):
        _name = f"uint{_bits}"
        if not hasattr(torch, _name):
            setattr(torch, _name, torch.uint8)
    _compiler_stub = _CompilerStub()
    _compiler = getattr(torch, "compiler", None)
    if _compiler is None:
        torch.compiler = _compiler_stub
    else:
        for _name in ("disable", "is_compiling", "is_exporting"):
            if not hasattr(_compiler, _name):
                setattr(_compiler, _name, getattr(_compiler_stub, _name))
    try:
        from torch.utils import _pytree
    except Exception:
        _pytree = None
    if _pytree is not None and not hasattr(_pytree, "register_pytree_node"):
        _private_register = getattr(_pytree, "_register_pytree_node", None)
        if _private_register is not None:
            def _register_pytree_node(cls, flatten_fn, unflatten_fn, *args, **kwargs):
                try:
                    return _private_register(cls, flatten_fn, unflatten_fn, *args, **kwargs)
                except TypeError:
                    _supported = {
                        _key: kwargs[_key]
                        for _key in ("to_dumpable_context", "from_dumpable_context")
                        if _key in kwargs
                    }
                    try:
                        return _private_register(cls, flatten_fn, unflatten_fn, *args, **_supported)
                    except TypeError:
                        return _private_register(cls, flatten_fn, unflatten_fn)
            _pytree.register_pytree_node = _register_pytree_node
    if hasattr(torch, "distributed") and not hasattr(torch.distributed, "device_mesh"):
        from types import SimpleNamespace
        torch.distributed.device_mesh = SimpleNamespace(DeviceMesh=object)
    if hasattr(torch, "distributed") and "torch.distributed._functional_collectives" not in sys.modules:
        _funcol = ModuleType("torch.distributed._functional_collectives")
        class _AsyncCollectiveTensor:
            pass
        def _identity_collective(tensor, *args, **kwargs):
            return tensor
        _funcol.AsyncCollectiveTensor = _AsyncCollectiveTensor
        _funcol.all_to_all_single = _identity_collective
        _funcol.all_gather_tensor = _identity_collective
        _funcol.permute_tensor = _identity_collective
        sys.modules["torch.distributed._functional_collectives"] = _funcol
        torch.distributed._functional_collectives = _funcol
except Exception:
    pass
PY
export COSMODIFF_STUB_SKLEARN=1
export TORCHDYNAMO_DISABLE=1
export PYTHONPATH="${STUB_ROOT}:${COSMODIFF_DIR}:${PROJECT_DIR}:${PYTHONPATH:-}"
export COSMODIFF_DIR

if [[ ! -f "${CONFIG_PATH}" ]]; then
  echo "Missing config: ${CONFIG_PATH}" >&2
  echo "Run first: python scripts/prepare_nf_class_conditional_u128_config.py --project-dir ${PROJECT_DIR}" >&2
  exit 1
fi

if [[ ! -f "${COSMODIFF_DIR}/scripts/cosmodiff_train.py" ]]; then
  echo "Missing main cosmo_diffusion checkout: ${COSMODIFF_DIR}" >&2
  exit 1
fi

PATCH_LOCK="${COSMODIFF_DIR}/.codex_conditional_labels_patch.lock"
while ! mkdir "${PATCH_LOCK}" 2>/dev/null; do
  sleep 1
done
trap 'rmdir "${PATCH_LOCK}" 2>/dev/null || true' EXIT
"${PYTHON_BIN}" "${PROJECT_DIR}/scripts/patch_cosmodiff_package_metadata.py" "${COSMODIFF_DIR}"
"${PYTHON_BIN}" "${PROJECT_DIR}/scripts/patch_cosmodiff_safe_log.py" "${COSMODIFF_DIR}"
"${PYTHON_BIN}" "${PROJECT_DIR}/scripts/patch_cosmodiff_continuous_labels.py" "${COSMODIFF_DIR}"
"${PYTHON_BIN}" "${PROJECT_DIR}/scripts/patch_cosmodiff_multinorm_labels.py" "${COSMODIFF_DIR}"
"${PYTHON_BIN}" "${PROJECT_DIR}/scripts/patch_cosmodiff_nonfinite_loss_guard.py" "${COSMODIFF_DIR}"
rmdir "${PATCH_LOCK}" 2>/dev/null || true
trap - EXIT

COSMODIFF_BRANCH=$(git -C "${COSMODIFF_DIR}" rev-parse --abbrev-ref HEAD)
COSMODIFF_COMMIT=$(git -C "${COSMODIFF_DIR}" rev-parse --short HEAD)
echo "cosmodiff git branch: ${COSMODIFF_BRANCH}"
echo "cosmodiff git commit: ${COSMODIFF_COMMIT}"
if [[ "${COSMODIFF_BRANCH}" != "${REQUIRED_COSMODIFF_BRANCH}" ]]; then
  echo "Wrong cosmo_diffusion branch: expected ${REQUIRED_COSMODIFF_BRANCH}, got ${COSMODIFF_BRANCH}" >&2
  exit 1
fi

"${PYTHON_BIN}" scripts/check_nf_class_conditional_u128_runtime.py \
  --project-dir "${PROJECT_DIR}" \
  --cosmodiff-dir "${COSMODIFF_DIR}" \
  --config "${CONFIG_PATH}"

"${PYTHON_BIN}" scripts/prepare_nf_class_conditional_u128_config.py \
  --project-dir "${PROJECT_DIR}" \
  --check-only

echo "Training discrete class-conditional ${RUN_NAME}"
echo "  config:    ${CONFIG_PATH}"
echo "  cosmodiff: ${COSMODIFF_DIR}"
echo "  venv:      ${VENV_PATH}"
echo "  python:    ${PYTHON_BIN}"

"${PYTHON_BIN}" scripts/train_cosmodiff.py \
  --config "${CONFIG_PATH}" \
  --cosmodiff-train "${COSMODIFF_DIR}/scripts/cosmodiff_train.py"
