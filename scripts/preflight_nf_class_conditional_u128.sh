#!/bin/bash
set -euo pipefail

PROJECT_DIR=${PROJECT_DIR:-/home/jiamingp/diffusion_models_repo}
COSMODIFF_DIR=${COSMODIFF_DIR:-/home/jiamingp/Diffusion_model/cosmo_diffusion_main}
VENV_PATH=${VENV_PATH:-/home/jiamingp/venvs/cosmodiff_nf}
PYTHON_BIN=${PYTHON_BIN:-python}

cd "${PROJECT_DIR}"
if [[ -f "${VENV_PATH}/bin/activate" ]]; then
  source "${VENV_PATH}/bin/activate"
fi

RUN_NAME=$("${PYTHON_BIN}" scripts/prepare_nf_class_conditional_u128_config.py --project-dir "${PROJECT_DIR}" --print-runs)
CONFIG_PATH="${PROJECT_DIR}/local/nf_class_conditional_u128/configs/${RUN_NAME}.yaml"

STUB_ROOT="${PROJECT_DIR}/results/cache/python_stubs"
mkdir -p "${STUB_ROOT}/sklearn/metrics"
printf 'from . import metrics\n' > "${STUB_ROOT}/sklearn/__init__.py"
printf "def roc_curve(*args, **kwargs):\n    raise RuntimeError('sklearn.metrics.roc_curve is stubbed for cosmodiff preflight')\n" > "${STUB_ROOT}/sklearn/metrics/__init__.py"
cat > "${STUB_ROOT}/sitecustomize.py" <<'PY'
try:
    from contextlib import nullcontext
    import torch
    class _OptionalDeviceStub:
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
    _stub = _OptionalDeviceStub()
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
except Exception:
    pass
PY
export COSMODIFF_STUB_SKLEARN=1
export TORCHDYNAMO_DISABLE=1
export PYTHONPATH="${STUB_ROOT}:${COSMODIFF_DIR}:${PROJECT_DIR}:${PYTHONPATH:-}"
export COSMODIFF_DIR

if [[ ! -f "${CONFIG_PATH}" ]]; then
  echo "Missing config: ${CONFIG_PATH}" >&2
  echo "Run: python scripts/prepare_nf_class_conditional_u128_config.py --project-dir ${PROJECT_DIR}" >&2
  exit 1
fi

if [[ ! -f "${COSMODIFF_DIR}/scripts/cosmodiff_train.py" ]]; then
  echo "Missing cosmo_diffusion checkout: ${COSMODIFF_DIR}" >&2
  exit 1
fi

"${PYTHON_BIN}" "${PROJECT_DIR}/scripts/patch_cosmodiff_package_metadata.py" "${COSMODIFF_DIR}"
"${PYTHON_BIN}" "${PROJECT_DIR}/scripts/patch_cosmodiff_continuous_labels.py" "${COSMODIFF_DIR}"

echo "project:    ${PROJECT_DIR}"
echo "cosmodiff:  ${COSMODIFF_DIR}"
echo "config:     ${CONFIG_PATH}"
echo "python:     $(${PYTHON_BIN} -c 'import sys; print(sys.executable)')"
echo "repo head:  $(git -C "${PROJECT_DIR}" rev-parse --short HEAD)"
echo "cosmodiff:  $(git -C "${COSMODIFF_DIR}" rev-parse --abbrev-ref HEAD) $(git -C "${COSMODIFF_DIR}" rev-parse --short HEAD)"

"${PYTHON_BIN}" scripts/check_nf_class_conditional_u128_runtime.py \
  --project-dir "${PROJECT_DIR}" \
  --cosmodiff-dir "${COSMODIFF_DIR}" \
  --config "${CONFIG_PATH}"
