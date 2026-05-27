#!/usr/bin/env python
"""Write Great Lakes import shims for recent diffusers on mixed Python envs."""

from __future__ import annotations

import argparse
from pathlib import Path


SHIM = r'''
try:
    from contextlib import nullcontext
    import sys
    from types import ModuleType, SimpleNamespace
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

    stub = _OptionalDeviceStub()
    required = ("empty_cache", "is_available", "device_count", "manual_seed")
    for backend in ("xpu", "mps"):
        existing = getattr(torch, backend, None)
        if existing is None or any(not hasattr(existing, name) for name in required):
            setattr(torch, backend, stub)
            continue
        for name in dir(stub):
            if name.startswith("__"):
                continue
            if not hasattr(existing, name):
                setattr(existing, name, getattr(stub, name))

    for name in (
        "float8_e4m3fn",
        "float8_e4m3fnuz",
        "float8_e5m2",
        "float8_e5m2fnuz",
        "float8_e8m0fnu",
        "float4_e2m1fn_x2",
    ):
        if not hasattr(torch, name):
            setattr(torch, name, torch.float16)
    for bits in range(1, 8):
        name = f"uint{bits}"
        if not hasattr(torch, name):
            setattr(torch, name, torch.uint8)

    class _CompilerStub:
        def disable(self, fn=None, recursive=True):
            if fn is None:
                return lambda inner: inner
            return fn
        def is_compiling(self): return False
        def is_exporting(self): return False

    compiler_stub = _CompilerStub()
    compiler = getattr(torch, "compiler", None)
    if compiler is None:
        torch.compiler = compiler_stub
    else:
        for name in ("disable", "is_compiling", "is_exporting"):
            if not hasattr(compiler, name):
                setattr(compiler, name, getattr(compiler_stub, name))

    try:
        from torch.utils import _pytree
    except Exception:
        _pytree = None
    if _pytree is not None and not hasattr(_pytree, "register_pytree_node"):
        private_register = getattr(_pytree, "_register_pytree_node", None)
        if private_register is not None:
            def register_pytree_node(cls, flatten_fn, unflatten_fn, *args, **kwargs):
                try:
                    return private_register(cls, flatten_fn, unflatten_fn, *args, **kwargs)
                except TypeError:
                    supported = {
                        key: kwargs[key]
                        for key in ("to_dumpable_context", "from_dumpable_context")
                        if key in kwargs
                    }
                    try:
                        return private_register(cls, flatten_fn, unflatten_fn, *args, **supported)
                    except TypeError:
                        return private_register(cls, flatten_fn, unflatten_fn)
            _pytree.register_pytree_node = register_pytree_node

    if hasattr(torch, "distributed") and not hasattr(torch.distributed, "device_mesh"):
        torch.distributed.device_mesh = SimpleNamespace(DeviceMesh=object)
    if hasattr(torch, "distributed") and "torch.distributed._functional_collectives" not in sys.modules:
        funcol = ModuleType("torch.distributed._functional_collectives")

        class AsyncCollectiveTensor:
            pass

        def identity_collective(tensor, *args, **kwargs):
            return tensor

        funcol.AsyncCollectiveTensor = AsyncCollectiveTensor
        funcol.all_to_all_single = identity_collective
        funcol.all_gather_tensor = identity_collective
        funcol.permute_tensor = identity_collective
        sys.modules["torch.distributed._functional_collectives"] = funcol
        torch.distributed._functional_collectives = funcol
except Exception:
    pass
'''.lstrip()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", help="Path to write, usually results/cache/python_stubs/sitecustomize.py.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    path = Path(args.path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(SHIM)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
