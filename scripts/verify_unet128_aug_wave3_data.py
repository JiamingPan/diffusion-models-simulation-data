#!/usr/bin/env python
"""CPU check of the wave-3 training sets (ledger unet128_aug_wave3). No model, no GPU, no jobs.

For each of the 18 configs in local/unet128_aug_wave3/configs/ and the three base configs
(d2p06 / d2p07 / d2p08 noaug_200k, cross-checked against the wave-1 data_reference.json hashes):
  * training path: cosmodiff.utils.parse_config_data(config) exactly as
    scripts/run_unet128_aug_wave3.py does before training -> N, map shape, fitted normalization,
    normalized-dataset sha256 (same hashing as the runner), and the DataLoader batches the
    training loop sees (torch DataLoader with the config's batch_size / shuffle, no drop_last,
    as cosmodiff/optim.py builds it; accelerate.prepare is not exercised here) -> batch size,
    batches per epoch, updates = batches/epoch * num_epochs;
  * evaluation path: scripts/evaluate_run.py UncondData(config, 8, 512, heldout_zthin=8), the
    settings of evaluate_run_unet128_aug_wave3.sbatch -> train / held-out shapes and its
    normalization (it fits center/xmax on the full selected cubes, not the thinned slices);
  * N <= 64: the raw (pre-log) training slices are slices b*16 + {0, 16/k, ...} (k = N/4 per box)
    of the N = 64 set, and the held-out raw maps equal the N = 64 held-out maps exactly;
  * N = 128 / 256: dataset hash equals the d2p07 / d2p08 base config (= wave-1 hash) and the
    held-out raw maps equal the base config's.
The normalized-dataset hash depends on the torch thread count (float32 mean); run with
OMP_NUM_THREADS=8 as the training task does. --write writes local/unet128_aug_wave3/data_reference.json
(status VERIFIED only if every check passes; refuses to overwrite).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from prepare_unet128_aug_wave3 import BASES, PLAN, WAVE, ZTHIN, base_for, budget, build_configs, config_path, sha256_file  # noqa: E402

WAVE1_REF = "local/unet128_aug_sweep/data_reference.json"
HELDOUT_ZTHIN = 8


def check_config(path: Path, utils, torch, np, UncondData) -> tuple[dict, dict]:
    import yaml
    from torch.utils.data import DataLoader

    config = yaml.safe_load(path.read_text())
    loaded = utils.parse_config_data(config)
    ds = loaded["data"]
    n = len(ds)
    h, maps = hashlib.sha256(), []
    for i in range(n):
        img = ds[i]["images"].cpu().numpy()
        assert img.shape == (1, 128, 128) and np.isfinite(img).all()
        h.update(np.ascontiguousarray(img).tobytes())
        maps.append(hashlib.sha256(np.ascontiguousarray(img).tobytes()).hexdigest())
    tr = config["train"]
    dl = DataLoader(ds, batch_size=tr["batch_size"], shuffle=tr["shuffle"], num_workers=0)
    batch_shapes = [tuple(b["images"].shape) for b in dl]
    norm = loaded["norm"]
    nk = {k: (float(v) if np.isscalar(v) else v) for k, v in norm.kwargs.items()}
    ev = UncondData(path, 8, 512, heldout_zthin=HELDOUT_ZTHIN)
    train_model = np.stack([ds[i]["images"].cpu().numpy()[0] for i in range(n)])
    rec = {"config_sha256": sha256_file(path), "N": n, "zthin": config["data"]["zthin"],
           "map_shape": [1, 128, 128], "dataset_sha256": h.hexdigest(), "map_sha256": maps,
           "normalization": {"method": norm.method, "kwargs": nk},
           "batch_size_config": tr["batch_size"], "effective_batch": max(s[0] for s in batch_shapes),
           "dataloader_batches_per_epoch": len(batch_shapes), "dataloader_batch_shapes": batch_shapes,
           "num_epochs": tr["num_epochs"], "checkpoint_every_n_epochs": tr["checkpoint_every_n_epochs"],
           "updates": len(batch_shapes) * tr["num_epochs"],
           "evaluate_run_uncond": {"heldout_raw_per_source": 8, "max_heldout": 512, "heldout_zthin": HELDOUT_ZTHIN,
                                   "train_shape": list(ev.train.shape), "heldout_shape": list(ev.val.shape),
                                   "heldout_raw_sha256": hashlib.sha256(np.ascontiguousarray(ev.val_raw).tobytes()).hexdigest(),
                                   "center": ev.center, "xmax": ev.xmax,
                                   "max_abs_diff_train_vs_training_space": float(np.abs(ev.train - train_model).max())}}
    return rec, {"train_raw": ev.train_raw, "val_raw": ev.val_raw}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runtime-root", type=Path, default=Path("/home/jiamingp/Diffusion_model/cosmo_diffusion_main"))
    ap.add_argument("--write", action="store_true")
    args = ap.parse_args()
    sys.path.insert(0, str(args.runtime_root.resolve()))
    from simdiff_eval.torch_compat import install_torch_backend_compat
    install_torch_backend_compat(entry_point='verify_unet128_aug_wave3_data')
    import numpy as np
    import torch
    from cosmodiff import utils
    from evaluate_run import UncondData

    problems = []
    out = {"wave": WAVE, "torch_threads": torch.get_num_threads(), "torch": torch.__version__,
           "heldout_zthin": HELDOUT_ZTHIN, "bases": {}, "configs": {}}
    wave1 = json.loads((ROOT / WAVE1_REF).read_text())
    base_recs, base_raws = {}, {}
    for bn, (bpath, bsha, brun) in BASES.items():
        rec, raw = check_config(ROOT / bpath, utils, torch, np, UncondData)
        rec["wave1_dataset_sha256"] = wave1[str(bn)]["dataset_sha256"]
        rec["baseline_run_name"] = brun
        if rec["dataset_sha256"] != rec["wave1_dataset_sha256"] or rec["config_sha256"] != bsha or rec["N"] != bn:
            problems.append(f"base N={bn} {bpath} does not reproduce wave-1 data_reference.json")
        out["bases"][bpath] = rec
        base_recs[bn], base_raws[bn] = rec, raw
        print(f"base {bpath}: N={rec['N']} sha={rec['dataset_sha256'][:12]} wave1={rec['wave1_dataset_sha256'][:12]} "
              f"heldout={rec['evaluate_run_uncond']['heldout_shape']}", flush=True)
    planned = {config_path(name): cfg for name, cfg in build_configs(ROOT)}
    for (arm_tag, n), path in zip(PLAN, planned):
        rec, raw = check_config(ROOT / path, utils, torch, np, UncondData)
        bn = n if n in (128, 256) else 64
        brec, braw = base_recs[bn], base_raws[bn]
        b = budget(n, 32)
        rec["base_config"] = base_for(n)[0]
        checks = {"N": rec["N"] == n, "zthin": rec["zthin"] == ZTHIN[n], "updates": rec["updates"] == 200_000,
                  "num_epochs": rec["num_epochs"] == b["num_epochs"],
                  "batch": all(s[0] == min(32, n) for s in rec["dataloader_batch_shapes"]),
                  "batches_per_epoch": rec["dataloader_batches_per_epoch"] == b["updates_per_epoch"],
                  "uncond_train_N": rec["evaluate_run_uncond"]["train_shape"][0] == n,
                  "heldout_512": rec["evaluate_run_uncond"]["heldout_shape"][0] == 512}
        if n <= 64:
            per_box = n // 4
            idx = np.concatenate([bb * 16 + np.arange(0, 16, 16 // per_box) for bb in range(4)])
            rec["raw_subset_of_N64_indices"] = idx.tolist()
            rec["raw_subset_of_N64"] = bool(np.array_equal(raw["train_raw"], braw["train_raw"][idx]))
            checks["raw_subset"] = rec["raw_subset_of_N64"]
        if n in (64, 128, 256):
            checks["same_data_as_base"] = rec["dataset_sha256"] == brec["dataset_sha256"]
        rec["evaluate_run_uncond"]["heldout_equals_base_heldout"] = bool(np.array_equal(raw["val_raw"], braw["val_raw"]))
        checks["heldout_equals_base"] = rec["evaluate_run_uncond"]["heldout_equals_base_heldout"]
        rec["checks"] = checks
        problems += [f"{path}: {k}" for k, v in checks.items() if not v]
        out["configs"][path] = rec
        e = rec["evaluate_run_uncond"]
        print(f"{path}: N={rec['N']} zthin={rec['zthin']} batch={rec['effective_batch']} "
              f"batches/epoch={rec['dataloader_batches_per_epoch']} num_epochs={rec['num_epochs']} "
              f"updates={rec['updates']} center={rec['normalization']['kwargs']['center']:.6f} "
              f"xmax={rec['normalization']['kwargs']['xmax']:.6f} sha={rec['dataset_sha256'][:12]} "
              f"| UncondData train={e['train_shape']} heldout={e['heldout_shape']} "
              f"heldout==base={e['heldout_equals_base_heldout']} max|diff|={e['max_abs_diff_train_vs_training_space']:.4g} "
              f"checks_ok={all(checks.values())}", flush=True)
    out["status"] = "VERIFIED" if not problems else "FAILED"
    out["problems"] = problems
    print("STATUS", out["status"], problems)
    if args.write:
        dest = ROOT / "local" / WAVE / "data_reference.json"
        with dest.open("x") as f:
            f.write(json.dumps(out, indent=2) + "\n")
        print("Wrote", dest)
    if problems:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
