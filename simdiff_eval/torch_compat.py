"""Canonical import compatibility for newer Diffusers on older PyTorch.

The compatibility objects in this module keep import-time optional-backend
probes from crashing.  They never advertise or select an unavailable device.
"""

from __future__ import annotations

from contextlib import nullcontext
import importlib
from pathlib import Path
import sys
from types import ModuleType
from typing import Any


COMPAT_SCHEMA_VERSION = 1
_MARKER_ATTRIBUTE = "_simdiff_eval_torch_compat"


class TorchCompatibilityError(RuntimeError):
    """Raised when the compatibility contract cannot be installed safely."""


class _UnavailableBackend:
    def is_available(self) -> bool:
        return False

    def device_count(self) -> int:
        return 0

    def empty_cache(self) -> None:
        return None

    def _is_compiled(self) -> bool:
        return False

    def is_built(self, *args: Any, **kwargs: Any) -> bool:
        return False

    def current_device(self) -> int:
        return 0

    def set_device(self, *args: Any, **kwargs: Any) -> None:
        return None

    def synchronize(self, *args: Any, **kwargs: Any) -> None:
        return None

    def manual_seed(self, *args: Any, **kwargs: Any) -> None:
        return None

    def manual_seed_all(self, *args: Any, **kwargs: Any) -> None:
        return None

    def seed(self, *args: Any, **kwargs: Any) -> int:
        return 0

    def initial_seed(self, *args: Any, **kwargs: Any) -> int:
        return 0

    def get_rng_state(self, *args: Any, **kwargs: Any) -> None:
        return None

    def set_rng_state(self, *args: Any, **kwargs: Any) -> None:
        return None

    def current_stream(self, *args: Any, **kwargs: Any) -> None:
        return None

    def stream(self, *args: Any, **kwargs: Any):
        return nullcontext()

    def device(self, *args: Any, **kwargs: Any):
        return nullcontext()

    def memory_allocated(self, *args: Any, **kwargs: Any) -> int:
        return 0

    def max_memory_allocated(self, *args: Any, **kwargs: Any) -> int:
        return 0

    def reset_peak_memory_stats(self, *args: Any, **kwargs: Any) -> None:
        return None

    def get_device_name(self, *args: Any, **kwargs: Any) -> str:
        return "optional-device-unavailable"

    def get_device_properties(self, *args: Any, **kwargs: Any) -> None:
        return None

    def __getattr__(self, name: str):
        def missing(*args: Any, **kwargs: Any):
            if name.startswith("is_"):
                return False
            return None

        return missing


class _CompilerStub:
    def disable(self, fn=None, recursive: bool = True):
        if fn is None:
            return lambda inner: inner
        return fn

    def is_compiling(self) -> bool:
        return False

    def is_exporting(self) -> bool:
        return False


def _diffusers_is_loaded() -> bool:
    return any(
        name == "diffusers" or name.startswith("diffusers.")
        for name in sys.modules
    )


def _copy_report(marker: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": int(marker["schema_version"]),
        "module_file": str(marker["module_file"]),
        "entry_points": list(marker["entry_points"]),
        "installed_attributes": list(marker["installed_attributes"]),
    }


def _record(installed: list[str], name: str) -> None:
    if name not in installed:
        installed.append(name)


def _install_backends(torch: ModuleType, installed: list[str]) -> None:
    stub = _UnavailableBackend()
    required = (
        "is_available",
        "device_count",
        "empty_cache",
        "manual_seed",
    )
    members = tuple(
        name for name in dir(_UnavailableBackend) if not name.startswith("__")
    )
    for backend_name in ("xpu", "mps", "npu"):
        backend = getattr(torch, backend_name, None)
        if backend is None:
            setattr(torch, backend_name, _UnavailableBackend())
            _record(installed, backend_name)
            continue
        for member_name in members:
            if hasattr(backend, member_name):
                continue
            setattr(backend, member_name, getattr(stub, member_name))
            _record(installed, f"{backend_name}.{member_name}")
        missing = [name for name in required if not hasattr(backend, name)]
        if missing:
            raise TorchCompatibilityError(
                f"optional backend {backend_name} still lacks {missing}"
            )


def _install_dtype_aliases(torch: ModuleType, installed: list[str]) -> None:
    float16 = getattr(torch, "float16", None)
    uint8 = getattr(torch, "uint8", None)
    if float16 is None or uint8 is None:
        raise TorchCompatibilityError(
            "Torch lacks float16 or uint8, so Diffusers compatibility aliases "
            "cannot be installed"
        )
    for name in (
        "float8_e4m3fn",
        "float8_e4m3fnuz",
        "float8_e5m2",
        "float8_e5m2fnuz",
        "float8_e8m0fnu",
        "float4_e2m1fn_x2",
    ):
        if not hasattr(torch, name):
            setattr(torch, name, float16)
            _record(installed, name)
    for bits in range(1, 8):
        name = f"uint{bits}"
        if not hasattr(torch, name):
            setattr(torch, name, uint8)
            _record(installed, name)


def _install_compiler(torch: ModuleType, installed: list[str]) -> None:
    stub = _CompilerStub()
    compiler = getattr(torch, "compiler", None)
    if compiler is None:
        torch.compiler = stub
        _record(installed, "compiler")
        return
    for name in ("disable", "is_compiling", "is_exporting"):
        if not hasattr(compiler, name):
            setattr(compiler, name, getattr(stub, name))
            _record(installed, f"compiler.{name}")


def _install_pytree(torch: ModuleType, installed: list[str]) -> None:
    utils = getattr(torch, "utils", None)
    pytree = getattr(utils, "_pytree", None)
    if pytree is None or hasattr(pytree, "register_pytree_node"):
        return
    private_register = getattr(pytree, "_register_pytree_node", None)
    if private_register is None:
        return

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
                return private_register(
                    cls,
                    flatten_fn,
                    unflatten_fn,
                    *args,
                    **supported,
                )
            except TypeError:
                return private_register(cls, flatten_fn, unflatten_fn)

    pytree.register_pytree_node = register_pytree_node
    _record(installed, "utils._pytree.register_pytree_node")


def _install_distributed(torch: ModuleType, installed: list[str]) -> None:
    distributed = getattr(torch, "distributed", None)
    if distributed is None:
        return
    if not hasattr(distributed, "device_mesh"):
        device_mesh = ModuleType("torch.distributed.device_mesh")
        device_mesh.DeviceMesh = object
        distributed.device_mesh = device_mesh
        _record(installed, "distributed.device_mesh")

    module_name = "torch.distributed._functional_collectives"
    if hasattr(distributed, "_functional_collectives"):
        return
    funcol = ModuleType(module_name)

    class AsyncCollectiveTensor:
        pass

    def identity_collective(tensor, *args, **kwargs):
        return tensor

    funcol.AsyncCollectiveTensor = AsyncCollectiveTensor
    funcol.all_to_all_single = identity_collective
    funcol.all_gather_tensor = identity_collective
    funcol.permute_tensor = identity_collective
    distributed._functional_collectives = funcol
    if sys.modules.get("torch") is torch:
        sys.modules[module_name] = funcol
    _record(installed, "distributed._functional_collectives")


def install_torch_backend_compat(
    *,
    entry_point: str,
    torch_module: ModuleType | None = None,
) -> dict[str, Any]:
    """Install the audited import compatibility before Diffusers is imported."""

    if not entry_point or not entry_point.strip():
        raise TorchCompatibilityError("entry_point must be a non-empty name")
    torch = torch_module or importlib.import_module("torch")
    marker = getattr(torch, _MARKER_ATTRIBUTE, None)
    if marker is not None:
        if marker.get("schema_version") != COMPAT_SCHEMA_VERSION:
            raise TorchCompatibilityError("Torch compatibility marker schema mismatch")
        if entry_point not in marker["entry_points"]:
            marker["entry_points"].append(entry_point)
        return _copy_report(marker)

    if _diffusers_is_loaded():
        raise TorchCompatibilityError(
            f"Torch compatibility was installed too late in {entry_point}: "
            "diffusers is already imported. Import simdiff_eval.torch_compat "
            "and call install_torch_backend_compat(...) before importing "
            "diffusers or cosmodiff."
        )

    installed: list[str] = []
    try:
        _install_backends(torch, installed)
        _install_dtype_aliases(torch, installed)
        _install_compiler(torch, installed)
        _install_pytree(torch, installed)
        _install_distributed(torch, installed)
    except TorchCompatibilityError:
        raise
    except (AttributeError, TypeError, ValueError) as exc:
        raise TorchCompatibilityError(
            f"Torch compatibility installation failed in {entry_point}: {exc}"
        ) from exc

    marker = {
        "schema_version": COMPAT_SCHEMA_VERSION,
        "module_file": str(Path(__file__).resolve()),
        "entry_points": [entry_point],
        "installed_attributes": sorted(installed),
    }
    setattr(torch, _MARKER_ATTRIBUTE, marker)
    return _copy_report(marker)


def get_torch_compat_report(
    torch_module: ModuleType | None = None,
) -> dict[str, Any]:
    """Return the installed compatibility report or fail if it never ran."""

    torch = torch_module or importlib.import_module("torch")
    marker = getattr(torch, _MARKER_ATTRIBUTE, None)
    if marker is None:
        raise TorchCompatibilityError("Torch compatibility has not been installed")
    return _copy_report(marker)
