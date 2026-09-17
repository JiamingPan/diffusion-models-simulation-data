from copy import deepcopy
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import yaml

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
import dit_a40_ablation as ab
from dit_ablation_data import CONTRACT, array_hash, audit_native_dataset, load_native_training_reference


def data_config(tmp_path):
    rng = np.random.default_rng(7)
    raw = np.exp(rng.normal(size=(3, 4, 8, 8))).astype(np.float32)
    # Deliberate extremum in an excluded z-plane: must NOT fit native scaling.
    raw[:, 1] = np.exp(15)
    path = tmp_path / "raw.npy"
    np.save(path, raw)
    config = {"global": {"dtype": "float32", "device": "cuda"}, "data": {
        "img_path": [str(path)], "img_read_fn": "npy_read_fn", "n_samples": [2],
        "seed": None, "reshape": "2d", "zthin": 2, "keep_on_cpu": True,
        "normalization": "tanh", "transform": ["log"], "constant_label": 0,
        "norm_kwargs": {"center": None, "xmax": None, "alpha": .8, "beta": 10.,
                        "gamma": 1., "delta": 1., "sigma": 1.5}}}
    return config, raw


def native_parsed(config):
    """Torch formulation of the pinned load_data -> Normalization order."""
    data = config["data"]
    chunks = []
    for path, count in zip(data["img_path"], data["n_samples"]):
        raw = torch.tensor(np.load(path), dtype=torch.float32)
        raw = raw[:count, ::data["zthin"]].reshape(-1, 1, *raw.shape[-2:])
        chunks.append(raw)
    transformed = torch.cat(chunks).log()
    kw = data["norm_kwargs"]
    center = float(transformed.mean()) if kw.get("center") is None else kw["center"]
    shifted = transformed - center
    xmax = float(shifted.abs().max()) if kw.get("xmax") is None else kw["xmax"]
    shifted = shifted / xmax - kw.get("mu", 0.)
    positive = kw["alpha"] * torch.tanh(kw["gamma"] * shifted / kw["alpha"])
    negative = kw["beta"] * torch.tanh(kw["delta"] * shifted / kw["beta"])
    images = torch.where(shifted >= 0, positive, negative) * kw["sigma"]
    return {"data": SimpleNamespace(arrays=images, labels=torch.zeros(len(images), dtype=torch.long)),
            "norm": SimpleNamespace(kwargs={"center": center, "xmax": xmax})}


def test_native_retained_slice_reference_reproduces_failure_and_corrects_it(tmp_path):
    from evaluate_late_start import load_reference
    config, raw = data_config(tmp_path)
    path = tmp_path / "run.yaml"
    path.write_text(yaml.safe_dump(config))
    legacy = load_reference(path, None)
    reference, metadata = load_native_training_reference(config)
    native = native_parsed(config)
    actual = native["data"].arrays.numpy()
    assert reference.shape == (4, 1, 8, 8)
    assert not np.allclose(actual, legacy, rtol=0, atol=2e-6)
    assert np.max(np.abs(actual - legacy)) > .1
    audit = audit_native_dataset(native, reference, metadata, torch)
    assert audit["reference_max_abs_delta"] < 2e-6
    assert metadata["selected_raw_sha256"] == array_hash(raw[:2, ::2].reshape(4, 1, 8, 8))
    assert metadata["contract"] == CONTRACT
    assert metadata["xmax"] < 5  # excluded log( exp(15) ) was not fitted
    assert audit["training_tensor_sha256"] == array_hash(actual)


def test_excluded_slices_cannot_change_training_reference(tmp_path):
    config, raw = data_config(tmp_path)
    original, metadata = load_native_training_reference(config)
    raw[:, 1::2] *= 10
    raw[2] *= 5
    np.save(config["data"]["img_path"][0], raw)
    changed, changed_metadata = load_native_training_reference(config)
    assert np.array_equal(original, changed)
    assert metadata == changed_metadata


def test_retained_slices_changes_are_detected(tmp_path):
    config, raw = data_config(tmp_path)
    original, metadata = load_native_training_reference(config)
    raw[0, 0, 1, 2] *= 3
    np.save(config["data"]["img_path"][0], raw)
    changed, changed_metadata = load_native_training_reference(config)
    assert changed_metadata["selected_raw_sha256"] != metadata["selected_raw_sha256"]
    assert not np.array_equal(original, changed)


def test_data_audit_preserves_native_storage_and_values(tmp_path):
    config, _ = data_config(tmp_path)
    reference, metadata = load_native_training_reference(config)
    parsed = native_parsed(config)
    before = parsed["data"].arrays.clone()
    pointer = parsed["data"].arrays.data_ptr()
    audit_native_dataset(parsed, reference, metadata, torch)
    assert parsed["data"].arrays.data_ptr() == pointer
    assert torch.equal(before, parsed["data"].arrays)


@pytest.mark.parametrize("corruption", ["permuted", "wrong_dtype", "wrong_shape", "nan", "scale", "offset", "labels", "missing_labels", "norm"])
def test_genuine_data_mismatch_still_fails_closed(tmp_path, corruption):
    config, _ = data_config(tmp_path)
    reference, metadata = load_native_training_reference(config)
    parsed = native_parsed(config)
    dataset = parsed["data"]
    if corruption == "permuted":
        dataset.arrays = dataset.arrays.flip(0)
    elif corruption == "wrong_dtype":
        dataset.arrays = dataset.arrays.half()
    elif corruption == "wrong_shape":
        dataset.arrays = dataset.arrays[:-1]
    elif corruption == "nan":
        dataset.arrays[0, 0, 0, 0] = float("nan")
    elif corruption == "scale":
        dataset.arrays *= 1.1
    elif corruption == "offset":
        dataset.arrays += .01
    elif corruption == "labels":
        dataset.labels[0] = 1
    elif corruption == "missing_labels":
        dataset.labels = None
    else:
        parsed["norm"].kwargs["xmax"] *= 2
    with pytest.raises(ValueError):
        audit_native_dataset(parsed, reference, metadata, torch)


def test_configured_source_volume_and_z_order(tmp_path):
    config, raw = data_config(tmp_path)
    second = raw.copy() * 2
    path = tmp_path / "second.npy"
    np.save(path, second)
    config["data"]["img_path"].append(str(path))
    config["data"]["n_samples"].append(1)
    _, metadata = load_native_training_reference(config)
    expected = np.concatenate([raw[:2, ::2].reshape(-1, 1, 8, 8), second[:1, ::2].reshape(-1, 1, 8, 8)])
    assert metadata["selected_raw_sha256"] == array_hash(expected)
    assert metadata["selections"][1]["volume_indices"] == [0]
    assert metadata["selections"][1]["z_indices"] == [0, 2]


def test_seeded_selection_order_matches_native(tmp_path):
    config, raw = data_config(tmp_path)
    config["data"]["seed"] = 123
    _, metadata = load_native_training_reference(config)
    selection = np.random.default_rng(123).choice(len(raw), size=2, replace=False)
    assert metadata["selections"][0]["volume_indices"] == selection.tolist()
    assert metadata["selected_raw_sha256"] == array_hash(raw[selection, ::2].reshape(-1, 1, 8, 8))


def test_fixed_normalization_parameters_preserved(tmp_path):
    config, _ = data_config(tmp_path)
    config["data"]["norm_kwargs"].update(center=.5, xmax=6.)
    reference, metadata = load_native_training_reference(config)
    assert metadata["center"] == .5 and metadata["xmax"] == 6.
    audit_native_dataset(native_parsed(config), reference, metadata, torch)


@pytest.mark.parametrize("key,value", [("reshape", "3d"), ("transform", ["log", "fft2"]),
    ("normalization", ["tanh"]), ("zthin", 0), ("n_samples", [99]), ("n_samples", [1, 2]),
    ("img_read_fn", "unsupported"), ("log", True)])
def test_unsupported_or_changed_data_recipe_refused(tmp_path, key, value):
    config, _ = data_config(tmp_path)
    config["data"][key] = value
    with pytest.raises(ValueError):
        load_native_training_reference(config)


def test_training_requires_real_data_receipt_before_any_gpu_or_output(tmp_path, monkeypatch):
    row = {"checkpoint_dir": str(tmp_path / "arm")}
    monkeypatch.setattr(ab, "load_plan", lambda _: ({"arms": [row]}, None))
    monkeypatch.setattr(ab, "runtime", lambda _: pytest.fail("GPU/runtime must not start"))
    with pytest.raises(ValueError, match="real-data CPU preflight"):
        ab.train(tmp_path, 0)
    assert not (tmp_path / "arm").exists()


def test_cpu_data_preflight_rank_guard_before_publication(tmp_path, monkeypatch):
    monkeypatch.setenv("RANK", "1")
    with pytest.raises(RuntimeError, match="nonzero rank"):
        ab.data_preflight(tmp_path)
    assert not list(tmp_path.iterdir())


def test_real_data_receipt_rejects_plan_runtime_and_reference_drift(tmp_path):
    plan_path = tmp_path / "plan/plan.json"
    plan_path.parent.mkdir()
    plan_path.write_text('{}')
    plan = {"code_revision": "test", "runtime_versions": {"torch": "test"},
            "training_reference_sha256": "native", "reference_sha256": "legacy"}
    report = {"status": "complete", "code_revision": "test", "runtime_versions": plan["runtime_versions"],
        "training_reference_sha256": "native", "legacy_reference_sha256": "legacy",
        "plan_sha256": ab.file_hash(plan_path), "arms": [{"arm": r[0]} for r in ab.ARMS]}
    record = tmp_path / "data_preflight_record/data_preflight.json"
    record.parent.mkdir()
    record.write_text(json.dumps(report))
    assert ab.require_data_preflight(tmp_path, plan) == report
    for key in ("status", "code_revision", "runtime_versions", "training_reference_sha256", "legacy_reference_sha256", "plan_sha256", "arms"):
        changed = deepcopy(report)
        changed[key] = "wrong"
        record.write_text(json.dumps(changed))
        with pytest.raises(ValueError):
            ab.require_data_preflight(tmp_path, plan)


def saved_baseline_fixture(tmp_path, reference):
    root = tmp_path / "matrix"
    directory = root / "tasks/dit_l16_fresh300k__dpm50"
    directory.mkdir(parents=True)
    samples = np.tile(reference, (128 // len(reference), 1, 1, 1))
    path = directory / "samples.npz"
    np.savez_compressed(path, samples=samples, initial_noise_batch_sha256="paired-noise")
    report = {"status": "complete", "plan_sha256": ab.MATRIX_SHA,
        "task": {"model": "dit_l16_fresh300k", "condition": "dpm50"},
        "reference_sha256": "legacy",
        "artifacts_sha256": {"samples.npz": ab.file_hash(path)}}
    receipt = directory / "complete.json"
    receipt.write_text(json.dumps(report))
    plan = {"matrix_plan_path": str(root / "plan/plan.json"), "reference_sha256": "legacy",
            "noise_batch_sha256": "paired-noise"}
    return plan, path, receipt


def test_saved_baseline_rescore_is_read_only_and_uses_matched_reference(tmp_path):
    reference = np.random.default_rng(123).normal(size=(4, 1, 128, 128)).astype(np.float32)
    plan, path, report = saved_baseline_fixture(tmp_path, reference)
    before = (ab.file_hash(path), ab.file_hash(report))
    summary = ab.rescore_saved_baseline(plan, reference)
    assert summary["reference_sha256"] == array_hash(reference)
    assert summary["data_reference_contract"] == CONTRACT
    assert summary["populations"]["near_copy"]["n"] == 128
    assert summary["populations"]["low_similarity"]["n"] == 0
    assert summary["populations"]["near_copy"]["pk_ratio_hi"] == pytest.approx(1.)
    assert before == (ab.file_hash(path), ab.file_hash(report))


@pytest.mark.parametrize("corruption", ["sample_bytes", "noise_pairing", "task", "status"])
def test_baseline_rescore_rejects_stale_or_unpaired_artifacts(tmp_path, corruption):
    reference = np.random.default_rng(123).normal(size=(4, 1, 128, 128)).astype(np.float32)
    plan, path, receipt = saved_baseline_fixture(tmp_path, reference)
    report = json.loads(receipt.read_text())
    if corruption == "sample_bytes":
        with path.open("ab") as handle:
            handle.write(b"changed")
    elif corruption == "noise_pairing":
        plan["noise_batch_sha256"] = "different"
    elif corruption == "task":
        report["task"]["condition"] = "ddpm500"
    else:
        report["status"] = "started"
    receipt.write_text(json.dumps(report))
    with pytest.raises(ValueError):
        ab.rescore_saved_baseline(plan, reference)


@pytest.mark.parametrize("failed_arm", [None, "p4_native", "rescore"])
def test_cpu_data_preflight_publishes_only_after_every_arm_and_rescore(tmp_path, monkeypatch, failed_arm):
    from types import ModuleType
    pin = tmp_path / "pin"
    pin.mkdir()
    utils_path = pin / "utils.py"
    utils_path.write_text('# Native formulation fixture; no production pin changes.\n')
    utils = ModuleType("cosmodiff.utils")
    utils.__file__ = str(utils_path)
    config, _ = data_config(tmp_path)
    reference, metadata = load_native_training_reference(config)
    rows = []
    for name, _, _ in ab.ARMS:
        path = tmp_path / f"{name}.yaml"
        path.write_text(yaml.safe_dump(config))
        rows.append({"name": name, "config": str(path)})
    plan_dir = tmp_path / "plan"
    plan_dir.mkdir()
    (plan_dir / "plan.json").write_text('{}')
    plan = {"arms": rows, "training_data_reference": metadata, "code_revision": "test",
        "runtime_versions": {"torch": "test"}, "training_reference_sha256": array_hash(reference),
        "reference_sha256": "legacy"}
    calls = []

    def parse(actual_config):
        calls.append("data")
        parsed = native_parsed(actual_config)
        if failed_arm == "p4_native" and len(calls) == 2:
            parsed["data"].arrays += .01
        return parsed

    utils.parse_config_data = parse
    package = ModuleType("cosmodiff")
    package.utils = utils
    monkeypatch.setitem(sys.modules, "cosmodiff", package)
    monkeypatch.setenv("COSMODIFF_PIN_ROOT", str(pin))
    monkeypatch.setattr(ab, "load_plan", lambda _: (plan, None))
    monkeypatch.setattr(ab, "runtime", lambda _: (torch, None))
    monkeypatch.setattr(ab, "reference_for", lambda *_: (reference, reference + .2))

    def rescore(*_):
        calls.append("rescore")
        if failed_arm == "rescore":
            raise ValueError("baseline rescore failed")
        return {"reference_sha256": array_hash(reference)}

    monkeypatch.setattr(ab, "rescore_saved_baseline", rescore)
    if failed_arm:
        with pytest.raises(ValueError, match="slice-first" if failed_arm == "p4_native" else "rescore"):
            ab.data_preflight(tmp_path)
        assert calls == (["data", "data"] if failed_arm == "p4_native" else ["data", "data", "data", "rescore"])
        assert not (tmp_path / "data_preflight_record").exists()
    else:
        ab.data_preflight(tmp_path)
        report = ab.require_data_preflight(tmp_path, plan)
        assert calls == ["data", "data", "data", "rescore"]
        assert report["status"] == "complete"
        assert [r["arm"] for r in report["arms"]] == [r[0] for r in ab.ARMS]
        assert len({r["training_tensor_sha256"] for r in report["arms"]}) == 1
    assert not (tmp_path / "checkpoints").exists()


def test_cpu_data_preflight_does_not_publish_success_if_rescore_fails(tmp_path, monkeypatch):
    # The rescore must occur before terminal publication, not just before print.
    source = Path(ab.__file__).read_text()
    begin = source.index("def data_preflight(")
    end = source.index("def train(", begin)
    body = source[begin:end]
    assert body.index("baseline = rescore_saved_baseline(") < body.index("publish_json(")
    wrapper = (SCRIPTS / "slurm/dit_a40_ablation.sbatch").read_text()
    assert "train|sample|test-a|data-preflight" in wrapper
    assert '--data-preflight --out-dir "$ABLATION_OUT_DIR"' in wrapper
