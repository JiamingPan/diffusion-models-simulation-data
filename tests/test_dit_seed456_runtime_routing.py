"""Exercise the real shell wrapper with stand-in git/runtime verification."""
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/slurm/sample_dit_seed456_review.sbatch"


class RuntimeRoutingTests(unittest.TestCase):
    def test_preflight_uses_original_root_not_adapter_root(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            runtime = root / "runtime-code"
            adapter = root / "adapter-code"
            runtime.mkdir()
            adapter.mkdir()
            git = root / "git"
            git.write_text('#!/bin/bash\nif [[ "$3" == rev-parse ]]; then\n'
                           'if [[ "$2" == "$RUNTIME_CODE_ROOT" ]]; then\n'
                           'echo 555f350f82c913ff06150c96e66709719856cd83\n'
                           'else echo adapter; fi; fi\n')
            git.chmod(0o755)
            python = root / "fake-python"
            python.write_text('#!/bin/bash\nset -eu\n'
                              'test "$PWD" = "$RUNTIME_CODE_ROOT"\n'
                              'test "$PYTHONPATH" = "$COSMODIFF_PIN_ROOT/seed_restart_runtime:$RUNTIME_CODE_ROOT:$COSMODIFF_PIN_ROOT"\n'
                              '[[ "$*" == *"--code-root $RUNTIME_CODE_ROOT"* ]]\n'
                              '[[ "$*" != *"$CODE_ROOT"* ]]\n'
                              'echo VERIFIED_ORIGINAL_ROOT\n')
            python.chmod(0o755)
            env = dict(os.environ, PATH=f"{root}:{os.environ['PATH']}",
                       RUNTIME_CODE_ROOT=str(runtime), CODE_ROOT=str(adapter),
                       EXPECTED_COMMIT="adapter", PROJECT_DIR=str(root),
                       COSMODIFF_PIN_ROOT=str(root / "pin"), COSMODIFF_PIN_MANIFEST="manifest",
                       EXPECTED_COSMODIFF_BASE_REVISION="base", PYTHON_BIN=str(python),
                       PREFLIGHT_ONLY="1")
            result = subprocess.run(["bash", str(SCRIPT)], env=env, text=True, capture_output=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("VERIFIED_ORIGINAL_ROOT", result.stdout)
            self.assertIn("NO SAMPLING", result.stdout)


if __name__ == "__main__":
    unittest.main()
