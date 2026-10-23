"""Tests for ledger nf_cond_omsigc_ft_controls_300k (prepare / preflight helpers; no checkpoints, no GPU)."""
import importlib
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
prep = importlib.import_module("prepare_nf_cond_omsigc_ft_controls_300k")


@pytest.mark.parametrize("src, n_ep, every, want", [
    (6249, 1563, 7031, [7030, 7812]),   # N = 1024 seg1
    (7812, 1562, 7031, [9374]),         # N = 1024 seg2 / C
    (1562, 391, 1758, [1757, 1953]),    # N = 4096 seg1
    (1953, 390, 1758, [2343]),          # N = 4096 seg2 / C
])
def test_save_rule_lands_on_planned_epochs(src, n_ep, every, want):
    assert prep.saved_epochs(src, n_ep, every) == want


def test_planned_update_counts_match_ledger():
    for n, (_, _, src0, spe, *_rest) in prep.BASES.items():
        saves = {**prep.BASES[n][7], **prep.BASES[n][8]}
        for tag, ep in saves.items():
            assert (ep - src0) * spe == prep.PLANNED_UPDATES[(n, tag)]


def test_wrapper_flags_per_arm_segment():
    assert prep.flags_for("noaug", 1) == prep.flags_for("noaug", 2) == ["--allow-no-augmentations"]
    assert prep.flags_for("idaug", 1) == []
    assert prep.flags_for("idaug", 2) == prep.flags_for("d4shift", 2) == ["--allow-checkpoint-augmentations"]


def _augment():
    try:
        return importlib.import_module("cosmodiff.augment")
    except Exception:  # noqa: BLE001
        pytest.skip("cosmodiff not importable here")


def test_identity_pipeline_is_bitwise_identity_and_draws_rng():
    torch = pytest.importorskip("torch")
    aug = _augment()
    pipe = aug.config_augmentations(prep.IDAUG)
    torch.manual_seed(0)
    for _ in range(50):
        x = torch.randn(1, 16, 16)
        st = torch.get_rng_state()
        y = pipe(x)
        assert torch.equal(x, y)
        assert not torch.equal(st, torch.get_rng_state())  # p = 0 / size 1 still consume CPU RNG


def test_pipeline_descriptions_match_preflight_expectations():
    pytest.importorskip("torch")
    aug = _augment()
    wrapper = importlib.import_module("finetune_cosmodiff_resume")
    if not hasattr(wrapper, "pipeline_description"):
        pytest.skip("finetune_cosmodiff_resume opt-in diff not applied")
    pf = importlib.import_module("preflight_nf_cond_omsigc_ft_controls_300k")
    for arm in ("idaug", "d4shift"):
        desc = wrapper.pipeline_description(aug.config_augmentations(prep.AUGS[arm]))
        assert json.loads(json.dumps(desc)) == pf.EXPECTED_DESC[arm]
    assert wrapper.pipeline_description(None) is None
    assert wrapper.pipeline_description(aug.config_augmentations(prep.IDAUG)) != \
        wrapper.pipeline_description(aug.config_augmentations(prep.D4SHIFT))


def test_manifest_if_prepared():
    m = ROOT / "local/nf_cond_omsigc_ft_controls_300k/manifest.json"
    if not m.exists():
        pytest.skip("local manifest not prepared on this machine")
    rows = json.loads(m.read_text())
    assert len(rows) == 11 and len({r["run_name"] for r in rows}) == 11
    assert all(r["sample_path"].startswith("results/nf_cond_omsigc_ft_controls_300k/samples/") for r in rows)
    for r in rows:
        assert r["requested_checkpoint"].endswith(f"checkpoint-epoch-{r['checkpoint_epoch']}")
        assert r["finetune_flags"] == prep.flags_for(r["arm"], r["segment"])
