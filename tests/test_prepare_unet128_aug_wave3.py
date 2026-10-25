"""Plan checks for scripts/prepare_unet128_aug_wave3.py (ledger unet128_aug_wave3)."""
import math
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import prepare_unet128_aug_wave3 as p3  # noqa: E402

EXPECTED_PLAN = [("jitter4", 16), ("stride32", 16), ("shiftonly", 4), ("d4shift", 4), ("jitter4", 32),
                 ("jitter4", 64), ("jitter4", 128), ("jitter4", 256),
                 ("shiftonly", 8), ("shiftonly", 16), ("shiftonly", 32),
                 ("d4shift", 8), ("d4shift", 16), ("d4shift", 32),
                 ("d4only", 16), ("d4only", 32), ("stride32", 64), ("stride32", 256)]


@pytest.mark.parametrize("n, batch, epochs, ckpt", [(4, 4, 200_000, 20_000), (8, 8, 200_000, 20_000),
                                                    (16, 16, 200_000, 20_000), (32, 32, 200_000, 20_000),
                                                    (64, 32, 100_000, 10_000), (128, 32, 50_000, 5_000),
                                                    (256, 32, 25_000, 2_500)])
def test_budget_is_200k_updates_without_drop_last(n, batch, epochs, ckpt):
    b = p3.budget(n, 32)
    assert b["effective_batch"] == batch and b["num_epochs"] == epochs and b["checkpoint_every_n_epochs"] == ckpt
    assert b["num_epochs"] * math.ceil(n / 32) == 200_000 == b["total_updates"]


def test_budget_rejects_fractional_epochs():
    with pytest.raises(ValueError):
        p3.budget(96, 32)  # 3 updates per epoch does not divide 200000


def test_plan_order_names_labels_and_zthin():
    assert p3.PLAN == EXPECTED_PLAN and len(set(p3.PLAN)) == 18
    per_arm = {}
    for a, n in p3.PLAN:
        per_arm.setdefault(a, []).append(n)
    assert {a: sorted(v) for a, v in per_arm.items()} == {
        "jitter4": [16, 32, 64, 128, 256], "shiftonly": [4, 8, 16, 32], "d4shift": [4, 8, 16, 32],
        "d4only": [16, 32], "stride32": [16, 64, 256]}
    names = [p3.run_name(a, n) for a, n in p3.PLAN]
    assert len(set(names)) == 18
    assert names[:6] == ["nf_fig2_u128_d2p04_jitter4_200k", "nf_fig2_u128_d2p04_stride32_200k",
                         "nf_fig2_u128_d2p02_shiftonly_200k", "nf_fig2_u128_d2p02_d4shift_200k",
                         "nf_fig2_u128_d2p05_jitter4_200k", "nf_fig2_u128_d2p06_jitter4_200k"]
    assert {p3.sample_label(a) for a, _ in p3.PLAN} == {f"dpm50_w3_{a}" for a in p3.ARMS}
    # N <= 64: slices per box = 128 / zthin, 4 boxes; N = 128 / 256 keep the baseline zthin 8
    assert all(4 * (128 // p3.ZTHIN[n]) == n for n in p3.ZTHIN if n <= 64)
    assert p3.ZTHIN[128] == p3.ZTHIN[256] == 8


def test_arm_names_match_transform_registry():
    from simdiff_eval.aug_wave3_transforms import ARMS, VIEWS_128
    assert set(p3.ARMS.values()) <= set(ARMS)
    assert all(p3.VIEWS[t] == VIEWS_128[a] for t, a in p3.ARMS.items())


def test_base_selection():
    assert p3.base_for(4)[0].endswith("d2p06_noaug_200k.yaml") and p3.base_for(64)[0].endswith("d2p06_noaug_200k.yaml")
    assert p3.base_for(128)[0].endswith("d2p07_noaug_200k.yaml") and p3.base_for(256)[0].endswith("d2p08_noaug_200k.yaml")
    assert p3.allowed_changes(128) == p3.allowed_changes(256) == {("io", "output_dir")}


def test_config_diff_reports_nested_changes():
    a = {"io": {"output_dir": "x"}, "train": {"num_epochs": 1, "lr": 2}}
    b = {"io": {"output_dir": "y"}, "train": {"num_epochs": 1, "lr": 3}, "new": 1}
    assert p3.config_diff(a, b) == {("io", "output_dir"), ("train", "lr"), ("new",)}


def test_built_configs_only_change_allowed_keys():
    if not all((ROOT / b[0]).is_file() for b in p3.BASES.values()):
        pytest.skip("base configs are git-ignored; present on Great Lakes only")
    import yaml
    for (arm, n), (name, cfg) in zip(p3.PLAN, p3.build_configs(ROOT)):
        base = yaml.safe_load((ROOT / p3.base_for(n)[0]).read_text())
        assert p3.config_diff(base, cfg) <= p3.allowed_changes(n)
        assert cfg["data"]["zthin"] == p3.ZTHIN[n]
        assert cfg["train"]["num_epochs"] * math.ceil(n / cfg["train"]["batch_size"]) == 200_000
        assert cfg["io"]["output_dir"] == f"{p3.CHECKPOINT_ROOT}/{name}_checkpoints"
        if n in (128, 256):
            assert p3.config_diff(base, cfg) == {("io", "output_dir")}
