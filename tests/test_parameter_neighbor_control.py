from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from simdiff_eval.parameter_neighbor_control import (
    PARAM_NAMES, ensemble_points, nearest_parameter_matches, residual_decomposition, summarize_points,
)

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from evaluate_parameter_neighbor_control import audit_probe, evaluate_manifest, generated_points


def test_nearest_parameters_are_standardized_and_not_image_nearest():
    train = np.array([[1,0,0,0,0,0], [0,2,0,0,0,0]], float)
    result = nearest_parameter_matches(train, [3,4], np.zeros((1,6)), [9], [1,10,1,1,1,1])
    assert result.nearest_sim.tolist() == [4]
    assert result.parameter_distance.iloc[0] == pytest.approx(.2)


def test_duplicate_slices_do_not_change_cosmology_distance_and_ties_are_visible():
    train = np.array([[-1]*6, [1]*6, [-1]*6],float)
    result = nearest_parameter_matches(train, [3,2,3], np.zeros((1,6)), [9], np.ones(6))
    assert result.nearest_sim.iloc[0] == 2
    assert result.tied_cosmologies.iloc[0] == 2
    assert result.selected_training_fields.iloc[0] == 1


def test_training_and_heldout_must_be_disjoint():
    with pytest.raises(ValueError, match="overlap"):
        nearest_parameter_matches(np.ones((1,6)), [9], np.zeros((1,6)), [9], np.ones(6))


@pytest.mark.parametrize("scale", [np.zeros(6), np.full(6,np.nan), np.ones(5)])
def test_invalid_scale_is_rejected(scale):
    with pytest.raises(ValueError, match="scale"):
        nearest_parameter_matches(np.ones((1,6)), [1], np.zeros((1,6)), [9], scale)


def test_inconsistent_slice_labels_are_rejected():
    with pytest.raises(ValueError, match="disagree within"):
        nearest_parameter_matches(np.array([[0]*6,[1]*6]), [1,1], np.zeros((1,6)), [9], np.ones(6))


def test_ensemble_quantiles_and_residual_accounting_are_exact():
    requested = np.array([[2]*6,[4]*6],float)
    pieces = []
    for kind, offset in [("nearest_theta",1), ("nearest_training_field",.5), ("generated",-.25)]:
        pieces.append(ensemble_points([x[None]+offset for x in requested], requested, [8,9], kind))
    points = pd.concat(pieces, ignore_index=True)
    points["run_name"] = "test"
    parts = residual_decomposition(points)
    np.testing.assert_allclose(parts.training_parameter_offset,1)
    np.testing.assert_allclose(parts.probe_on_neighbor_offset,-.5)
    np.testing.assert_allclose(parts.generated_vs_neighbor_offset,-.75)
    np.testing.assert_allclose(parts.generated_total_residual,-.25)
    summary = summarize_points(points)
    assert np.allclose(summary.slope,1)
    assert summary[summary.kind == "generated"].mean_residual.iloc[0] == pytest.approx(-.25)


def _generated_table():
    return pd.DataFrame([{"run_name":"test", "heldout_sim":9, "seed_index":s,
                          "parameter":p, "theta_in":2., "theta_rec":2.+s,
                          "guidance_label":"noguidance", "cfg_dropout":0.}
                         for s in range(3) for p in PARAM_NAMES])


def test_generated_comparison_is_paired_by_id_not_row_order():
    table = _generated_table().sample(frac=1,random_state=2)
    points = generated_points(table,"test",np.full((1,6),2.),np.array([9]))
    assert points.theta_rec_median.tolist() == [3.]*6


def test_duplicate_sampling_conditions_cannot_be_pooled():
    table = _generated_table()
    with pytest.raises(ValueError,match="duplicate"):
        generated_points(pd.concat([table,table]),"test",np.full((1,6),2.),np.array([9]))


class IdentityProbe:
    def predict_norm(self, images, batch_size):
        assert batch_size > 0
        return images[:,0,0,:6]

    def norm_to_raw(self, prediction):
        return prediction


@pytest.fixture
def inputs(tmp_path):
    theta = np.array([[1]*6,[3]*6,[1.2]*6],float)
    grid = np.zeros((3,4,8,8), np.float32)
    for sim in range(3):
        grid[sim] = np.exp(theta[sim,0]/10)
    np.save(tmp_path/"grid.npy", grid)
    np.savetxt(tmp_path/"params.txt",theta)
    train = grid[[0,0,1],[0,1,2]][:,None]
    np.save(tmp_path/"train.npy",train)
    np.save(tmp_path/"raw.npy",theta[[0,0,1]])
    np.save(tmp_path/"norm.npy",theta[[0,0,1]])
    np.savetxt(tmp_path/"heldout.txt",[2],fmt="%d")
    pd.DataFrame({"row":[0,1,2],"simulation_index":[0,0,1],"z_index":[0,1,2]}).to_csv(tmp_path/"pairs.csv",index=False)
    stats = {"param_names":list(PARAM_NAMES), "mean":[0]*6,"std":[1]*6}
    (tmp_path/"stats.json").write_text(json.dumps(stats))
    cfg = {"train":{"conditioning":"continuous"},
           "data":{"img_path":"train.npy","label_path":"norm.npy","transform":["log"],
                   "normalization":"tanh","reshape":None,"zthin":1,"n_samples":None,
                   "norm_kwargs":{"center":0,"xmax":1}}}
    (tmp_path/"config.yaml").write_text(yaml.safe_dump(cfg))
    row = {"run_name":"test","dataset_size":3,"param_names":list(PARAM_NAMES),"config":"config.yaml",
           "selected_pairs_path":"pairs.csv","train_raw_params_path":"raw.npy",
           "prepared_image_path":"train.npy","heldout_indices_path":"heldout.txt"}
    (tmp_path/"manifest.json").write_text(json.dumps([row]))
    return tmp_path


def _evaluate(root):
    return evaluate_manifest(root, root/"manifest.json",root/"stats.json",root/"params.txt",
                             root/"grid.npy",IdentityProbe(),real_k=3,batch_size=2)


def test_manifest_control_uses_only_exact_selected_fields(inputs):
    points, matches, provenance = _evaluate(inputs)
    assert matches.nearest_sim.iloc[0] == 0
    assert matches.selected_training_fields.iloc[0] == 2
    assert provenance[0]["encoded_training_rows"] == [0,1]
    assert set(points.kind) == {"nearest_theta","nearest_training_field","heldout_real"}
    assert points[points.kind == "nearest_training_field"].n_fields.iloc[0] == 2
    assert points[points.kind == "heldout_real"].n_fields.iloc[0] == 3


def test_prepared_source_mismatch_stops_analysis(inputs):
    bad = np.load(inputs/"train.npy")
    bad[0,0,0,0] += 1
    np.save(inputs/"train.npy",bad)
    with pytest.raises(ValueError, match="declared source"):
        _evaluate(inputs)


def test_constant_label_dit_is_not_parameter_conditioned(inputs):
    path=inputs/"config.yaml"
    cfg=yaml.safe_load(path.read_text())
    cfg["train"]["conditioning"]="discrete"
    path.write_text(yaml.safe_dump(cfg))
    with pytest.raises(ValueError,match="continuous"):
        _evaluate(inputs)


def _probe_descriptor():
    return {"param_names": np.array(PARAM_NAMES), "param_mean": np.zeros(6),
            "param_std": np.ones(6), "heldout_indices": np.array([2]),
            "train_sims": np.array([0]), "val_sims": np.array([1]),
            "normalization": np.array({"transform": ["log"], "method": "tanh",
                                       "center": 0., "xmax": 1.}, dtype=object)}


def test_probe_descriptor_matches_the_manifest(inputs):
    audit_probe(_probe_descriptor(), json.loads((inputs/"stats.json").read_text()),
                yaml.safe_load((inputs/"config.yaml").read_text()), np.array([2]))


@pytest.mark.parametrize("key,value,match", [
    ("param_std", np.full(6, 2.), "parameter normalization"),
    ("heldout_indices", np.array([5]), "not excluded"),
    ("train_sims", np.array([0,2]), "leaked"),
    ("val_sims", np.array([1,2]), "leaked"),
    ("normalization", np.array({"transform": ["log"], "method": "tanh",
                                "center": .2, "xmax": 1.}, dtype=object), "image normalization"),
])
def test_mismatched_probe_is_rejected(inputs, key, value, match):
    saved = _probe_descriptor()
    saved[key] = value
    with pytest.raises(ValueError, match=match):
        audit_probe(saved, json.loads((inputs/"stats.json").read_text()),
                    yaml.safe_load((inputs/"config.yaml").read_text()), np.array([2]))
