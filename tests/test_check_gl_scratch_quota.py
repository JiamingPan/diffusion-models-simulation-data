from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from check_gl_scratch_quota import bytes_from_size


def test_quota_sizes_accept_zero_and_fractional_megabytes():
    assert bytes_from_size("0") == 0
    assert bytes_from_size("194.3M") == 194.3 * 1024**2
    assert bytes_from_size("3.923T") == 3.923 * 1024**4
