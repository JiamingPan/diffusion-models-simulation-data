#!/bin/bash

write_nf_class_conditional_python_stubs() {
  local project_dir=$1
  local reason=${2:-runtime}
  NF_CLASS_CONDITIONAL_STUB_ROOT="${project_dir}/results/cache/python_stubs"
  mkdir -p "${NF_CLASS_CONDITIONAL_STUB_ROOT}/sklearn/metrics"
  printf 'from . import metrics\n' > "${NF_CLASS_CONDITIONAL_STUB_ROOT}/sklearn/__init__.py"
  printf "def roc_curve(*args, **kwargs):\n    raise RuntimeError('sklearn.metrics.roc_curve is stubbed for cosmodiff ${reason}')\n" > "${NF_CLASS_CONDITIONAL_STUB_ROOT}/sklearn/metrics/__init__.py"
  cat > "${NF_CLASS_CONDITIONAL_STUB_ROOT}/sitecustomize.py" <<'PY'
try:
    from contextlib import nullcontext
    import sys
    from types import ModuleType
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
    class _CompilerStub:
        def disable(self, fn=None, recursive=True):
            if fn is None:
                return lambda inner: inner
            return fn
        def is_compiling(self): return False
        def is_exporting(self): return False
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
    _compiler_stub = _CompilerStub()
    _compiler = getattr(torch, "compiler", None)
    if _compiler is None:
        torch.compiler = _compiler_stub
    else:
        for _name in ("disable", "is_compiling", "is_exporting"):
            if not hasattr(_compiler, _name):
                setattr(_compiler, _name, getattr(_compiler_stub, _name))
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
}
