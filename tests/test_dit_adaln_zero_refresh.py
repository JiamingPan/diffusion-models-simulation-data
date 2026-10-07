from __future__ import annotations
import json
import sys
from pathlib import Path

import pytest
import yaml

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"scripts"))
import prepare_dit_adaln_zero_refresh as refresh


def test_refresh_preserves_the_recipe_except_init_budget_and_new_outputs(tmp_path):
    plan, configs=refresh.draft_runs(tmp_path/"draft",Path("/scratch/experiment/fresh_zero"),300000)
    assert plan["status"]=="draft_not_launch_ready"
    assert len(plan["runs"])==30
    assert {r["num_layers"] for r in plan["runs"]}=={8,12,16}
    for new,old in zip(plan["runs"],refresh.base.iter_runs()):
        text=configs[Path(new["config"])]
        cfg=yaml.safe_load(text)
        expected=refresh.base.build_config(old["run_name"],old["arch"],old["source_counts"],old["dataset_size"])
        assert cfg["data"]==expected["data"]
        assert cfg["model"]["initialization"]=="adaln_zero"
        assert cfg["model"]["kwargs"]["norm_type"]=="ada_norm_zero"
        assert new["nominal_updates"]>=300000
        assert new["effective_batch_size"]==8
        cfg["io"]=expected["io"]
        cfg["model"].pop("initialization")
        cfg["model"]["kwargs"].pop("norm_type")
        cfg["train"]["num_epochs"]=expected["train"]["num_epochs"]
        assert cfg==expected


def test_old_output_root_cannot_be_reused(tmp_path):
    with pytest.raises(ValueError,match="old sweep"):
        refresh.draft_runs(tmp_path,Path(refresh.base.CHECKPOINT_ROOT),200000)


def test_cli_can_freeze_the_four_l16_high_n_screen(tmp_path, monkeypatch):
    output = tmp_path / "screen"
    monkeypatch.setattr(sys, "argv", [
        "prepare_dit_adaln_zero_refresh.py",
        "--out-dir", str(output),
        "--checkpoint-root", "/scratch/experiment/l16_highn",
        "--target-updates", "300000",
        "--depth", "16",
        "--dataset-tag", "d2p09",
        "--dataset-tag", "d2p11",
        "--dataset-tag", "d2p13",
        "--dataset-tag", "d2p15",
    ])
    refresh.main()
    plan = json.loads((output / "plan.json").read_text())
    assert plan["selection"] == {
        "depths": [16], "dataset_tags": ["d2p09", "d2p11", "d2p13", "d2p15"]}
    assert [(row["dataset_tag"], row["dataset_size"]) for row in plan["runs"]] == [
        ("d2p09", 512), ("d2p11", 2048), ("d2p13", 8192), ("d2p15", 32768)]
    assert all(row["target_updates"] == 300000 and row["num_layers"] == 16
               and row["patch_size"] == 8 and row["training_seed"] == 123
               for row in plan["runs"])
    assert len(list((output / "configs").glob("*.yaml"))) == 4
