from __future__ import annotations

import importlib
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

prep = importlib.import_module("prepare_nf_conditional_label_sweep_configs")


def test_class_ids_cover_36_bins_and_edges():
    corners = np.array([[0.1, 0.6], [0.5, 1.0], [0.31, 0.81], [0.1 + 1e-9, 0.6 + 1e-9]])
    ids = prep.class_ids(corners)
    assert ids.tolist() == [0, 35, 21, 0]
    centers = prep.class_centers()
    assert centers.shape == (36, 2)
    assert prep.class_ids(centers).tolist() == list(range(36))
    with pytest.raises(ValueError):
        prep.class_ids(np.array([[0.05, 0.8]]))


def test_run_names_and_variant_config_shapes():
    assert prep.run_name("omsig_continuous", 64).startswith("nf_cond_omsigc_hi_u128_d2p06_n64")
    assert prep.run_name("omsig_class36", 32768).startswith("nf_cond_omsig36_hi_u128_d2p15_n32768")
    with pytest.raises(ValueError):
        prep.run_name("omsig_class36", 100)
    base_cfg = {"io": {"output_dir": "x"}, "data": {"label_path": "old"},
                "model": {"class": "UNet2DConditionModel", "kwargs": {"encoder_hid_dim": 6}},
                "train": {"conditioning": "continuous", "cfg_dropout": 0.0},
                "generate": {"conditioning": "continuous", "continuous_labels": "old", "labels": None}}
    c = prep.variant_config("omsig_continuous", base_cfg, Path("lab.npy"), Path("held.npy"), Path("out"))
    assert c["model"]["kwargs"]["encoder_hid_dim"] == 2 and c["generate"]["continuous_labels"] == "held.npy"
    d = prep.variant_config("omsig_class36", base_cfg, Path("lab.npy"), Path("held.npy"), Path("out"))
    assert d["model"]["class"] == "UNet2DModel" and d["model"]["kwargs"]["num_class_embeds"] == 37
    assert d["train"]["conditioning"] == "discrete" and d["generate"]["labels"] == "held.npy"
    assert d["generate"]["continuous_labels"] is None and d["train"]["cfg_dropout"] == 0.0
    assert base_cfg["model"]["class"] == "UNet2DConditionModel"   # input not mutated
