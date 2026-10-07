from __future__ import annotations

import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ENTRY_POINTS = (
    ROOT / "scripts/preflight_dit_adaln_zero_data.py",
    ROOT / "scripts/smoke_dit_adaln_zero_screen.py",
    ROOT / "scripts/train_cosmodiff_seeded.py",
)


def _top_level_position(tree: ast.Module, predicate) -> int:
    for index, node in enumerate(tree.body):
        if predicate(node):
            return index
    raise AssertionError("required top-level statement is missing")


def test_zero_init_entrypoints_install_torch_compat_before_cosmodiff_can_load():
    """Every process that can import cosmodiff must install the shim first."""
    for path in ENTRY_POINTS:
        tree = ast.parse(path.read_text(), filename=str(path))
        install_position = _top_level_position(
            tree,
            lambda node: (
                isinstance(node, ast.Assign)
                and any(
                    isinstance(target, ast.Name) and target.id == "TORCH_COMPAT_REPORT"
                    for target in node.targets
                )
                and isinstance(node.value, ast.Call)
                and isinstance(node.value.func, ast.Name)
                and node.value.func.id == "install_torch_backend_compat"
            ),
        )
        main_position = _top_level_position(
            tree,
            lambda node: isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name == "main",
        )
        assert install_position < main_position, path
