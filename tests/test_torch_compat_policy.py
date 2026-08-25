from __future__ import annotations

import ast
from pathlib import Path
import re
import runpy

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
PRODUCTION_ROOTS = (REPO_ROOT / "scripts", REPO_ROOT / "simdiff_eval")
TARGET_MODULES = {"diffusers", "cosmodiff"}
DUPLICATE_FRAGMENTS = (
    "class _OptionalDeviceStub",
    "class TorchOptionalDeviceStub",
    'for _backend in ("xpu", "mps")',
    'for backend in ("xpu", "mps")',
)
HEREDOC_START = re.compile(
    r"<<-?['\"]?(?P<tag>[A-Za-z_][A-Za-z0-9_]*)['\"]?[^\n]*\n"
)


def _production_files():
    for root in PRODUCTION_ROOTS:
        for suffix in ("*.py", "*.sh", "*.sbatch"):
            yield from sorted(root.rglob(suffix))


def _target_imports(tree: ast.AST) -> list[ast.AST]:
    imports = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules = {alias.name.split(".", 1)[0] for alias in node.names}
        elif isinstance(node, ast.ImportFrom):
            modules = {(node.module or "").split(".", 1)[0]}
        else:
            continue
        if modules & TARGET_MODULES:
            imports.append(node)
    return imports


def _installer_calls(tree: ast.AST) -> list[ast.Call]:
    calls = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        function = node.func
        name = function.id if isinstance(function, ast.Name) else (
            function.attr if isinstance(function, ast.Attribute) else ""
        )
        if name == "install_torch_backend_compat":
            calls.append(node)
    return calls


def _late_imports(source: str, label: str) -> list[str]:
    tree = ast.parse(source, filename=label)
    calls = _installer_calls(tree)
    failures = []
    for node in _target_imports(tree):
        if not any(call.lineno < node.lineno for call in calls):
            failures.append(f"{label}:{node.lineno}")
    return failures


def _python_heredocs(source: str):
    position = 0
    while True:
        match = HEREDOC_START.search(source, position)
        if match is None:
            return
        tag = match.group("tag")
        end = re.search(rf"(?m)^{re.escape(tag)}[ \t]*$", source[match.end():])
        if end is None:
            return
        body_start = match.end()
        body_end = body_start + end.start()
        yield body_start, source[body_start:body_end]
        position = body_start + end.end()


def test_no_handwritten_torch_backend_shims_remain_outside_canonical_module():
    canonical = REPO_ROOT / "simdiff_eval/torch_compat.py"
    failures = []
    for path in _production_files():
        if path == canonical:
            continue
        source = path.read_text()
        for fragment in DUPLICATE_FRAGMENTS:
            if fragment in source:
                failures.append(f"{path.relative_to(REPO_ROOT)}: {fragment}")

    assert failures == [], "duplicate Torch compatibility:\n" + "\n".join(failures)


def test_editable_python_entry_points_install_compat_before_runtime_imports():
    failures = []
    for path in _production_files():
        relative = path.relative_to(REPO_ROOT).as_posix()
        source = path.read_text()
        if path.suffix == ".py":
            failures.extend(_late_imports(source, relative))
            continue
        for index, (_, body) in enumerate(_python_heredocs(source), start=1):
            try:
                failures.extend(_late_imports(body, f"{relative}#heredoc{index}"))
            except SyntaxError as exc:
                if re.search(r"\b(?:diffusers|cosmodiff)\b", body):
                    pytest.fail(
                        f"cannot audit runtime-import heredoc {relative}#{index}: {exc}"
                    )

    assert failures == [], "compatibility installed after runtime import:\n" + "\n".join(failures)


def test_direct_unet_patch_installs_compat_before_generated_diffusers_imports(tmp_path):
    utils_path = tmp_path / "utils.py"
    utils_path.write_text(
        "import os\n"
        "from diffusers import AutoModel\n"
        "def load_checkpoint(ckpt_path):\n"
        "    model = AutoModel.from_pretrained(ckpt_path)\n"
        "    return model\n"
    )
    patcher = runpy.run_path(
        str(REPO_ROOT / "scripts/patch_cosmodiff_direct_unet_checkpoint.py")
    )

    assert patcher["patch_utils"](utils_path) is True
    generated = utils_path.read_text()
    function = ast.parse(generated).body[2]
    assert isinstance(function, ast.FunctionDef)
    function_source = ast.unparse(function)
    assert _late_imports(function_source, "generated cosmodiff.utils") == []
