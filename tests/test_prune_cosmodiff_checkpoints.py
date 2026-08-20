import importlib.util
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
PRUNE_PATH = REPO_ROOT / "scripts" / "prune_cosmodiff_checkpoints.py"


def load_module():
    spec = importlib.util.spec_from_file_location("prune_checkpoints_for_test", PRUNE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def make_complete(path: Path) -> Path:
    path.mkdir()
    for name in (
        "config.json",
        "diffusion_pytorch_model.safetensors",
        "optimizer.pkl",
        "noise_scheduler.pkl",
        "lr_scheduler.pkl",
        "random_states_0.pkl",
    ):
        (path / name).write_bytes(b"test")
    return path


class CheckpointPruneTests(unittest.TestCase):
    def test_prune_keeps_two_newest_complete_and_never_deletes_partial(self):
        module = load_module()
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            old = make_complete(root / "checkpoint-epoch-0001")
            keep_a = make_complete(root / "checkpoint-epoch-0002")
            keep_b = make_complete(root / "checkpoint-epoch-0003")
            partial = root / "checkpoint-epoch-0004"
            partial.mkdir()
            (partial / "config.json").write_text("{}")

            removed = module.prune(root, keep=2)

            self.assertEqual([path.resolve() for path in removed], [old.resolve()])
            self.assertFalse(old.exists())
            self.assertTrue(keep_a.exists())
            self.assertTrue(keep_b.exists())
            self.assertTrue(partial.exists())

    def test_quarantine_moves_only_incomplete_checkpoints(self):
        module = load_module()
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            complete = make_complete(root / "checkpoint-epoch-0001")
            partial = root / "checkpoint-epoch-0002"
            partial.mkdir()
            (partial / "config.json").write_text("{}")

            moved = module.quarantine_incomplete(root)

            self.assertTrue(complete.exists())
            self.assertFalse(partial.exists())
            self.assertEqual(len(moved), 1)
            self.assertTrue(moved[0].exists())
            self.assertEqual(moved[0].parent.name, "_incomplete_checkpoints")


if __name__ == "__main__":
    unittest.main()
