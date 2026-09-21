import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from dit_checkpoint_path import resolve_checkpoint


class CheckpointPathTests(unittest.TestCase):
    def test_padded_and_unpadded(self):
        for name, epoch in [('0073', 73), ('0292', 292), ('1171', 1171), ('73', 73)]:
            with tempfile.TemporaryDirectory() as tmp:
                expected = Path(tmp) / f'checkpoint-epoch-{name}'
                expected.mkdir()
                self.assertEqual(resolve_checkpoint(tmp, epoch), expected)

    def test_missing_and_ambiguous_fail(self):
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / 'checkpoint-epoch-0072').mkdir()
            with self.assertRaises(ValueError):
                resolve_checkpoint(tmp, 73)
            for name in ['73', '0073']:
                (Path(tmp) / f'checkpoint-epoch-{name}').mkdir()
            with self.assertRaises(ValueError):
                resolve_checkpoint(tmp, 73)
