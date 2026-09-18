"""Pure local fixtures; no model execution, cluster access, or compute submission."""
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import sample_dit_a40_early as early


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def fixture(root):
    root = Path(root)
    (root / "plan").mkdir()
    (root / "plan/plan.json").write_text("synthetic plan")
    arms = []
    for name in early.ARMS:
        folder = root / "checkpoints" / name
        final = folder / "checkpoint-epoch-9374"
        final.mkdir(parents=True)
        (final / "synthetic_weights").write_text(name)
        row = {"name": name, "checkpoint_dir": str(folder), "expected_checkpoint": str(final)}
        arms.append(row)
        receipt = folder / "complete_record/complete.json"
        receipt.parent.mkdir()
        receipt.write_text(json.dumps({"status": "complete", "arm": row,
             "plan_sha256": sha(root / "plan/plan.json"), "nominal_optimizer_steps": 300000,
             "successful_optimizer_steps": 299999, "amp_skipped_optimizer_steps": 1,
             "checkpoint_validation": {"fixture_only": True},
             "checkpoint_files_sha256": {"synthetic_weights": sha(final / "synthetic_weights")}}))
    return SimpleNamespace(load_plan=lambda _: ({"arms": arms}, None),
        require_data_preflight=lambda *_: None, file_hash=sha), arms


class EarlyTests(unittest.TestCase):
    def test_both_complete_and_checkpoint_tamper(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            adapter, arms = fixture(root)
            early.preflight(adapter, root)
            self.assertFalse((root / early.EARLY_FOLDER).exists())
            Path(arms[1]["expected_checkpoint"], "synthetic_weights").write_text("changed")
            with self.assertRaisesRegex(ValueError, "checkpoint changed"):
                early.preflight(adapter, root)

    def test_missing_receipt_existing_output_and_partial(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            adapter, arms = fixture(root)
            destination = root / early.EARLY_FOLDER / early.ARMS[0]
            destination.mkdir(parents=True)
            with self.assertRaises(FileExistsError):
                early.preflight(adapter, root)
            destination.rmdir()
            pending = destination.with_name(".p8_zero.pending.fixture")
            pending.mkdir()
            with self.assertRaises(FileExistsError):
                early.preflight(adapter, root)
            pending.rmdir()
            Path(arms[1]["checkpoint_dir"], "complete_record/complete.json").unlink()
            with self.assertRaises(FileNotFoundError):
                early.preflight(adapter, root)

    def test_redirect_only_expected_outputs_and_restore(self):
        seen = []
        @contextmanager
        def original(path):
            seen.append(path)
            yield path
        trace = SimpleNamespace(atomic_output=original)
        root = Path("/private/tmp/synthetic-early-only")
        with early.redirect_outputs(trace, root):
            for name in early.ARMS:
                with trace.atomic_output(root / "samples" / name) as pending:
                    self.assertEqual(pending, root / early.EARLY_FOLDER / name)
            with self.assertRaises(ValueError):
                with trace.atomic_output(root / "samples/p4_zero"):
                    pass
            with self.assertRaises(ValueError):
                with trace.atomic_output(root / "checkpoints/complete_record"):
                    pass
        self.assertIs(trace.atomic_output, original)
        self.assertEqual(seen, [root / early.EARLY_FOLDER / name for name in early.ARMS])
        with self.assertRaises(RuntimeError):
            with early.redirect_outputs(trace, root):
                raise RuntimeError("fixture failure")
        self.assertIs(trace.atomic_output, original)

    def test_run_sequential_only_0_1(self):
        seen = []
        @contextmanager
        def original(path):
            yield path
        trace = SimpleNamespace(__file__="/private/tmp/frozen/scripts/trace_trajectories.py", atomic_output=original)
        adapter = SimpleNamespace(__file__="/private/tmp/frozen/scripts/dit_a40_ablation.py",
                                  sample=lambda root, index: seen.append(index))
        torch = SimpleNamespace(cuda=SimpleNamespace(empty_cache=lambda: None))
        adapter.load_plan = lambda _: ({"runtime_versions": {}}, None)
        adapter.runtime = lambda _: (torch, None)
        adapter.single_a40 = lambda _: None
        with patch.object(early.importlib, "import_module", return_value=trace), patch.dict(sys.modules, torch=torch):
            early.run(adapter, Path("/private/tmp/synthetic-early-only"))
        self.assertEqual(seen, [0, 1])
        self.assertIs(trace.atomic_output, original)


if __name__ == "__main__":
    unittest.main()
