#!/usr/bin/env python
"""CPU pre-submission checks for the width-128 augmentation sweep. No training.

1. Runner preflight (pytest for D4 model + transforms) -> receipt.
2. Per N: rebuild the baseline dataset through the runner's own path
   (utils.parse_config_data on the unchanged nf_fig2_u128_*_noaug_200k config),
   record per-map hashes, the fitted normalization and the dataset hash the
   runner will require; cross-check against the PCA scorer's training loader.
3. Per arm: instantiate the exact width-128 model, check block_out_channels /
   norm_num_groups / class, count stored and (for D4) independent parameters.
4. D4 equivariance for all 8 elements at the real width-128 configuration.
5. Save/reload round trip through scripts/sample_cosmodiff._load_for_sampling.
6. scripts/sample_cosmodiff.py entry point on the saved timing checkpoints:
   --preflight-only, plus a 1-sample, 2-step CPU draw; and a negative check
   that a D4 checkpoint with the plain-UNet config is refused.
Writes local/unet128_aug_sweep/{data_reference.json,verification_report.json}.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
TIMING = Path("/scratch/huterer_root/huterer0/jiamingp/unet128-timing.xmFNro")
SCORER_PYTHON = Path("/home/jiamingp/venvs/cosmodiff_nf/bin/python")
SCORER_SNIPPET = """
import json, sys
from pathlib import Path
import numpy as np, yaml
root, row, out = Path(sys.argv[1]), json.loads(sys.argv[2]), sys.argv[3]
sys.path.insert(0, str(root / 'scripts'))
import compute_nf_generalize_pca_full_nn as pca
config = yaml.safe_load((root / row['config']).read_text())
train, _, norm = pca.load_train_and_validation(row, config, val_raw_per_source=0, max_val_slices=0)
np.save(out + '.npy', train)
Path(out + '.json').write_text(json.dumps({k: float(v) for k, v in norm.items()}))
"""
ARM_DIRS = {"d4_periodic_shift": "d4_periodic_shift", "smooth_periodic_warp": "smooth_periodic_warp",
            "d4_equivariant_unet": "d4_equivariant_unet"}


def free_parameters(model, torch) -> int:
    """Independent values. D4 kernel averaging ties each k x k kernel to one value per
    D4 orbit of kernel positions (3 for 3x3: centre, edges, corners; 1 for 1x1)."""
    from torch.nn.utils import parametrize
    from simdiff_eval.d4_unet64 import d4
    total = sum(p.numel() for p in model.parameters())
    for module in model.modules():
        if parametrize.is_parametrized(module, "weight"):
            w = module.parametrizations.weight.original
            k = w.shape[-1]
            ids = torch.arange(k * k).reshape(k, k)
            orbits = {frozenset(int(d4(ids, g)[i, j]) for g in range(8)) for i in range(k) for j in range(k)}
            total += w[..., 0, 0].numel() * len(orbits) - w.numel()
    return total


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=ROOT / "local/unet128_aug_sweep")
    args = parser.parse_args()
    out_dir = args.out_dir
    report_path, reference_path = out_dir / "verification_report.json", out_dir / "data_reference.json"
    receipt = out_dir / "preflight_receipt.json"
    for p in (report_path,):
        if p.exists():
            raise FileExistsError(f"{p} exists; move it aside before re-verifying")

    # 1. runner preflight (the runner re-checks the receipt's code/runtime hashes before training)
    if receipt.exists():
        assert json.loads(receipt.read_text())["status"] == "implementation_tests_passed"
        print("Reusing preflight receipt", receipt)
    else:
        subprocess.run([sys.executable, str(ROOT / "scripts/run_unet64_three_method.py"),
                        "--runtime-root", str(args.runtime_root), "--receipt", str(receipt)], check=True)

    from simdiff_eval.torch_compat import install_torch_backend_compat
    install_torch_backend_compat(entry_point="verify_unet128_aug_sweep")
    import numpy as np
    import torch
    import yaml
    from simdiff_eval.d4_unet64 import register_model, d4
    register_model()
    sys.path.insert(0, str(args.runtime_root))
    from cosmodiff import utils
    from sample_cosmodiff import _load_for_sampling
    torch.set_num_threads(int(os.environ.get("SLURM_CPUS_PER_TASK", "4")))

    runs = json.loads((out_dir / "runs.json").read_text())
    fig2 = {r["run_name"]: r for r in json.loads((ROOT / "local/nf_generalize_fig2/manifest.json").read_text())}
    report = {"preflight_receipt": str(receipt), "data": {}, "models": {}, "sampler_entry_point": {}}

    # 2. dataset identity per N (identical config for all three arms)
    reference = {}
    for base_name in sorted({r["baseline_run_name"] for r in runs}, key=lambda s: fig2[s]["dataset_size"]):
        row = fig2[base_name]
        cfg_path = ROOT / row["config"]
        config = yaml.safe_load(cfg_path.read_text())
        assert config["model"]["kwargs"]["block_out_channels"] == [32, 64, 128]
        assert config["model"]["kwargs"]["norm_num_groups"] == 32
        assert not config.get("augmentations") and config["data"]["seed"] is None
        loaded = utils.parse_config_data(config)
        ds = loaded["data"]
        assert ds.augmentations is None, "baseline dataset path applies augmentation"
        h, maps = hashlib.sha256(), []
        arr = np.stack([ds[i]["images"].cpu().numpy() for i in range(len(ds))])
        for img in arr:
            h.update(np.ascontiguousarray(img).tobytes())
            maps.append(hashlib.sha256(np.ascontiguousarray(img).tobytes()).hexdigest())
        norm = loaded["norm"]
        # Independent loader used by the PCA95 score for the same row.
        # The PCA scorer imports matplotlib, which fails in this venv (GLIBCXX); run its loader in the
        # cosmodiff_nf venv used by the PCA95 job and compare arrays here.
        with tempfile.TemporaryDirectory() as tmp:
            subprocess.run([str(SCORER_PYTHON), "-c", SCORER_SNIPPET, str(ROOT), json.dumps(row), str(Path(tmp) / "scorer")],
                           check=True, env={**os.environ, "PYTHONPATH": str(ROOT), "MPLBACKEND": "Agg"})
            pca_train = np.load(Path(tmp) / "scorer.npy")
            pca_norm = json.loads((Path(tmp) / "scorer.json").read_text())
        same_as_scorer = bool(pca_train.shape[0] == arr.shape[0] and np.allclose(pca_train.reshape(arr.shape), arr, atol=1e-5))
        n = int(row["dataset_size"])
        assert len(ds) == n
        reference[str(n)] = {"baseline_run_name": base_name, "config": row["config"],
                             "config_sha256": hashlib.sha256(cfg_path.read_bytes()).hexdigest(),
                             "dataset_sha256": h.hexdigest(), "map_sha256": maps,
                             "source_counts": row["source_counts"], "zthin": row["zthin"],
                             "normalization": {"method": norm.method,
                                               "kwargs": {k: float(v) for k, v in norm.kwargs.items() if np.isscalar(v)}},
                             "pca_scorer_normalization": pca_norm, "matches_pca_scorer_training_maps": same_as_scorer}
        report["data"][str(n)] = {k: v for k, v in reference[str(n)].items() if k != "map_sha256"}
        print(f"N={n}: dataset {h.hexdigest()[:12]} norm {reference[str(n)]['normalization']} scorer match {same_as_scorer}")
    if reference_path.exists():  # rerun: the recomputed reference must be identical
        previous = json.loads(reference_path.read_text())
        assert {n: r["dataset_sha256"] for n, r in previous.items()} == {n: r["dataset_sha256"] for n, r in reference.items()}, \
            "dataset reference changed since the previous verification"
    else:
        reference_path.write_text(json.dumps(reference, indent=2) + "\n")

    # 3-5. models at the real width-128 configuration
    cfg_2048 = yaml.safe_load((ROOT / fig2["nf_fig2_u128_d2p11_noaug_200k"]["config"]).read_text())
    cfg_2048.setdefault("global", {})["device"] = "cpu"  # CPU-only verification node; training uses the config unchanged
    with tempfile.TemporaryDirectory() as tmp:
        for arm in ARM_DIRS:
            config = json.loads(json.dumps(cfg_2048))
            if arm == "d4_equivariant_unet":
                config["model"]["class"] = "D4ScalarUNet2DModel"
            torch.manual_seed(123)
            model = utils.parse_config_model(config)["model"].eval()
            entry = {"class": type(model).__name__,
                     "block_out_channels": list(model.config.block_out_channels),
                     "norm_num_groups": model.config.norm_num_groups,
                     "stored_trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
                     "independent_parameters": free_parameters(model, torch)}
            assert entry["block_out_channels"] == [32, 64, 128] and entry["norm_num_groups"] == 32
            x, t = torch.randn(1, 1, 128, 128), torch.tensor([137])
            with torch.no_grad():
                y = model(x, t, return_dict=False)[0]
                if arm == "d4_equivariant_unet":
                    errs = [float((model(d4(x, g), t, return_dict=False)[0] - d4(y, g)).abs().max()) for g in range(8)]
                    entry["equivariance_max_abs_error_by_element"] = errs
                    assert max(errs) < 1e-3 * float(y.abs().max()), errs
            # round trip through the sampler loader with a run-style training_config.yaml
            ckpt = Path(tmp) / arm / "checkpoint-epoch-0000"
            model.save_pretrained(ckpt)
            cfg_file = Path(tmp) / arm / "training_config.yaml"
            cfg_file.write_text(yaml.safe_dump(config))
            reloaded, scheduler = _load_for_sampling(ckpt, cfg_file)
            reloaded.eval()
            with torch.no_grad():
                y2 = reloaded(x, t, return_dict=False)[0]
            entry["roundtrip_class"] = type(reloaded).__name__
            entry["roundtrip_max_abs_output_diff"] = float((y - y2).abs().max())
            assert entry["roundtrip_class"] == entry["class"] and entry["roundtrip_max_abs_output_diff"] == 0.0
            report["models"][arm] = entry
            print(arm, entry)

        # 6. actual sampling entry point on the saved width-128 timing checkpoints
        sampler = [sys.executable, str(ROOT / "scripts/sample_cosmodiff.py")]
        for arm, sub in ARM_DIRS.items():
            ckpt = TIMING / sub / "checkpoint-epoch-0003"
            config = json.loads(json.dumps(cfg_2048))
            if arm == "d4_equivariant_unet":
                config["model"]["class"] = "D4ScalarUNet2DModel"
            cfg_file = Path(tmp) / f"timing_{arm}.yaml"
            cfg_file.write_text(yaml.safe_dump(config))
            common = ["--checkpoint", str(ckpt), "--config", str(cfg_file), "--image-size", "128",
                      "--scheduler", "DPMSolverMultistepScheduler", "--seed", "123", "--device", "cpu"]
            pre = subprocess.run(sampler + common + ["--output", str(Path(tmp) / "pre.npz"), "--num-samples", "1",
                                                     "--batch-size", "1", "--num-steps", "50", "--preflight-only"],
                                 capture_output=True, text=True, cwd=ROOT)
            draw_out = Path(tmp) / f"draw_{arm}.npz"
            draw = subprocess.run(sampler + common + ["--output", str(draw_out), "--num-samples", "1",
                                                      "--batch-size", "1", "--num-steps", "2"],
                                  capture_output=True, text=True, cwd=ROOT)
            entry = {"checkpoint": str(ckpt), "preflight_returncode": pre.returncode,
                     "preflight_tail": pre.stdout.strip().splitlines()[-1:] + pre.stderr.strip().splitlines()[-3:],
                     "draw_returncode": draw.returncode, "draw_tail": draw.stderr.strip().splitlines()[-3:]}
            if draw.returncode == 0:
                with np.load(draw_out) as z:
                    s = z["samples"]
                entry["draw_shape"], entry["draw_finite"] = list(s.shape), bool(np.isfinite(s).all())
            if arm == "d4_equivariant_unet":
                plain = Path(tmp) / "plain_unet.yaml"
                plain.write_text(yaml.safe_dump(cfg_2048))
                neg = subprocess.run(sampler + ["--checkpoint", str(ckpt), "--config", str(plain),
                                                "--output", str(Path(tmp) / "neg.npz"), "--num-samples", "1",
                                                "--preflight-only", "--device", "cpu"],
                                     capture_output=True, text=True, cwd=ROOT)
                entry["plain_unet_config_refused"] = neg.returncode != 0 and "D4 checkpoint/config mismatch" in neg.stderr
            report["sampler_entry_point"][arm] = entry
            print(arm, entry)

    ok = (all(e["preflight_returncode"] == 0 and e["draw_returncode"] == 0 and e.get("draw_finite")
              for e in report["sampler_entry_point"].values())
          and report["sampler_entry_point"]["d4_equivariant_unet"]["plain_unet_config_refused"])
    # Informational: whether the PCA95 scorer's own loader reproduces the training maps.
    report["pca_scorer_matches_training_maps"] = {n: d["matches_pca_scorer_training_maps"] for n, d in report["data"].items()}
    report["status"] = "VERIFIED" if ok else "FAILED"
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    print(report["status"], report_path)
    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
