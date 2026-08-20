from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_every_nbconvert_sbatch_uses_node_local_secure_runtime_directory():
    sbatch_paths = sorted((ROOT / "scripts" / "slurm").glob("*.sbatch"))
    nbconvert_paths = [
        path for path in sbatch_paths if "nbconvert" in path.read_text()
    ]

    assert nbconvert_paths, "expected at least one tracked nbconvert sbatch"
    for path in nbconvert_paths:
        source = path.read_text()
        assert 'export JUPYTER_RUNTIME_DIR="${SLURM_TMPDIR:-/tmp}/' in source, path
        assert 'export TMPDIR="$JUPYTER_RUNTIME_DIR"' in source, path
        assert 'chmod 700 "$JUPYTER_RUNTIME_DIR"' in source, path
        assert 'chmod 600 "$RUNTIME_PROBE"' in source, path
        assert 'stat -c "%a" "$RUNTIME_PROBE"' in source, path
        assert "secure runtime preflight failed" in source, path
