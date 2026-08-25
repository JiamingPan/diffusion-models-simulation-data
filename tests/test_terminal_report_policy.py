from __future__ import annotations

import ast
from pathlib import Path
import re


REPO_ROOT = Path(__file__).resolve().parents[1]
PYTHON_PRODUCERS = (
    REPO_ROOT / "scripts/check_nf_generalize_fig2_dit_l16_seed_restart500k.py",
    REPO_ROOT / "scripts/evaluate_probe_c4_umap.py",
    REPO_ROOT / "scripts/audit_nf_generalize_fig2_dit_l16_continue500k_v2_results.py",
)
SLURM_PRODUCERS = (
    REPO_ROOT / "scripts/slurm/precheck_nf_generalize_fig2_dit_l16_seed_restart500k.sbatch",
    REPO_ROOT / "scripts/slurm/probe_c4_frozen_vgg_umap.sbatch",
    REPO_ROOT / "scripts/slurm/audit_nf_generalize_fig2_dit_l16_continue500k_v2.sbatch",
)


def terminal_pass_assignments(tree: ast.AST) -> list[int]:
    violations = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Dict):
            for key, value in zip(node.keys, node.values):
                if (
                    isinstance(key, ast.Constant)
                    and key.value == "status"
                    and isinstance(value, ast.Constant)
                    and value.value == "PASS"
                ):
                    violations.append(node.lineno)
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            value = node.value
            if not (isinstance(value, ast.Constant) and value.value == "PASS"):
                continue
            for target in targets:
                if (
                    isinstance(target, ast.Subscript)
                    and isinstance(target.slice, ast.Constant)
                    and target.slice.value == "status"
                ):
                    violations.append(node.lineno)
    return violations


def embedded_python_blocks(shell_source: str) -> list[str]:
    return re.findall(
        r"<<'PY'\s*\n(.*?)\nPY(?:\n|$)", shell_source, flags=re.DOTALL
    )


def test_terminal_report_python_producers_never_self_assign_pass():
    violations = {}
    for path in PYTHON_PRODUCERS:
        lines = terminal_pass_assignments(ast.parse(path.read_text()))
        if lines:
            violations[str(path.relative_to(REPO_ROOT))] = lines
    assert violations == {}


def test_slurm_terminal_pass_is_owned_by_shared_finalizer_not_embedded_python():
    for path in SLURM_PRODUCERS:
        source = path.read_text()
        assert "finalize_terminal_report.py" in source
        assert "--status PASS" in source
        for block in embedded_python_blocks(source):
            assert terminal_pass_assignments(ast.parse(block)) == []
