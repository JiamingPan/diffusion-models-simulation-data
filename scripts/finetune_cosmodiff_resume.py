#!/usr/bin/env python
"""Continue a cosmodiff checkpoint into a NEW output dir with the augmentations of a NEW config.

Ledger nf_cond_omsigc_n256_aug_finetune. Run inside a private cosmodiff runtime (COSMODIFF_DIR on PYTHONPATH).

Why a wrapper: cosmodiff_train.py only resumes from the latest checkpoint inside io.output_dir (so it cannot
continue a checkpoint into a new dir), and optim.train(resume_from_checkpoint=...) does
    model, ..., _aug = utils.load_checkpoint(ckpt); dataset.augmentations = _aug
i.e. it REPLACES the config augmentations with the checkpoint's augmentations.pkl, which is None for the
baseline (it was trained without augmentation), so a plain resume would silently train without augmentation.
Here utils.load_checkpoint is wrapped (in this process only) so that, when the checkpoint has no
augmentations.pkl, it returns the pipeline built from the config; a checkpoint WITH augmentations.pkl is refused.
Optimizer moments, grad scaler, LR-scheduler state and RNG state are restored by accelerator.load_state as in a
normal resume; train.num_epochs counts epochs AFTER the checkpoint (epochs start at checkpoint_epoch + 1).

--check-out JSON runs a short CPU version (train.num_epochs = --check-epochs, force_cpu, no AMP) into
--check-output-dir and records: the pipeline actually used during training (every fetched image compared with
the stored map and decomposed as roll(g(x), s) for g in D4), two post-training fetches of the same index,
the restored scheduler / optimizer step counters, and the counterfactual (what the stock load would return).
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path


def d4_shift_match(img, x0):
    """Return (d4 name, shift) with img == roll(g(x0), shift) exactly, or None."""
    import numpy as np
    import torch

    fa = np.fft.fft2(img[0].double().numpy())
    for t in (False, True):
        y = x0.transpose(-2, -1) if t else x0
        for k in range(4):
            gx = torch.rot90(y, k, dims=(-2, -1))
            c = np.fft.ifft2(fa * np.conj(np.fft.fft2(gx[0].double().numpy()))).real
            s = np.unravel_index(int(np.argmax(c)), c.shape)
            if torch.equal(torch.roll(gx, (int(s[0]), int(s[1])), dims=(-2, -1)), img):
                return f"{'T' if t else 'I'}r{k}", [int(s[0]), int(s[1])]
    return None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--resume-from", required=True, help="checkpoint dir to continue (read only)")
    ap.add_argument("--check-out", help="CPU check: write the JSON report here")
    ap.add_argument("--check-output-dir", help="CPU check: new output dir (instead of io.output_dir)")
    ap.add_argument("--check-epochs", type=int, default=1)
    args = ap.parse_args()

    from simdiff_eval.torch_compat import install_torch_backend_compat

    install_torch_backend_compat(entry_point="scripts.finetune_cosmodiff_resume")
    import torch
    import yaml

    import cosmodiff
    from cosmodiff import optim, utils
    from cosmodiff.utils import ArrayDataset, parse_config_data

    check = args.check_out is not None
    cfg = yaml.safe_load(Path(args.config).read_text())
    if "augmentations" not in cfg:
        raise SystemExit("config has no augmentations block; nothing to fine-tune with")
    src = Path(args.resume_from)
    if not (src / "checkpoint_config.yaml").is_file():
        raise SystemExit(f"{src} is not a cosmodiff checkpoint")
    if (src / "augmentations.pkl").exists():
        raise SystemExit(f"{src} has augmentations.pkl; refusing to choose between it and the config")
    if check:
        if not args.check_output_dir:
            raise SystemExit("--check-output-dir is required with --check-out")
        cfg["io"]["output_dir"] = args.check_output_dir
        cfg["train"].update(num_epochs=args.check_epochs, force_cpu=True, mixed_precision="no")
    out_dir = Path(cfg["io"]["output_dir"])
    if out_dir.exists():
        raise SystemExit(f"Refusing to write into existing {out_dir}")

    data = parse_config_data(cfg)["data"]
    aug = data.augmentations
    if aug is None:
        raise SystemExit("parse_config_data built no augmentation pipeline")
    fetched = []

    class SpyDataset(ArrayDataset):
        def __getitem__(self, idx):
            item = super().__getitem__(idx)
            if len(fetched) < 64:
                fetched.append((int(idx), item["images"].clone()))
            return item

    dataset = (SpyDataset if check else ArrayDataset)(data.arrays, labels=data.labels, augmentations=aug)

    stock_load = utils.load_checkpoint
    stock_aug = stock_load(str(src))[4]

    def load_checkpoint_keep_config_aug(path):
        model, noise_scheduler, optimizer, lr_scheduler, ck_aug = stock_load(path)
        if ck_aug is not None:
            raise SystemExit(f"{path} carries augmentations.pkl; refusing")
        print(f"[finetune] {path}: no augmentations.pkl; keeping the config pipeline {aug!r}", flush=True)
        return model, noise_scheduler, optimizer, lr_scheduler, aug

    utils.load_checkpoint = load_checkpoint_keep_config_aug  # optim.train calls utils.load_checkpoint

    out_dir.mkdir(parents=True)
    shutil.copy2(args.config, out_dir / Path(args.config).name)
    (out_dir / "finetune_provenance.json").write_text(json.dumps({
        "resume_from": str(src), "config": str(Path(args.config).resolve()), "cosmodiff_file": cosmodiff.__file__,
        "augmentations": cfg["augmentations"], "train": cfg["train"], "check": check, "argv": sys.argv}, indent=2))
    print(f"[finetune] cosmodiff from {cosmodiff.__file__}; resume {src} -> {out_dir}", flush=True)
    result = optim.train(dataset, resume_from_checkpoint=str(src), output_dir=str(out_dir), **cfg["train"])
    utils.write_metrics(result["metrics"], str(out_dir / "metrics_finetune.json"))
    if not check:
        return 0

    rep = {"cosmodiff_file": cosmodiff.__file__, "resume_from": str(src), "output_dir": str(out_dir),
           "stock_load_checkpoint_augmentations": repr(stock_aug),
           "config_pipeline": repr(aug),
           "dataset_pipeline_after_train_is_config_pipeline": dataset.augmentations is aug}
    recs = []
    for idx, img in fetched:
        m = d4_shift_match(img, data.arrays[idx])
        recs.append({"index": idx, "equals_stored": bool(torch.equal(img, data.arrays[idx])),
                     "d4": m[0] if m else None, "shift": m[1] if m else None})
    rep["train_fetches_recorded"] = len(recs)
    rep["train_fetches_equal_stored"] = sum(r["equals_stored"] for r in recs)
    rep["train_fetches_in_d4_shift_orbit"] = sum(r["d4"] is not None for r in recs)
    rep["train_fetches_first8"] = recs[:8]
    a, b = dataset[0]["images"], dataset[0]["images"]
    rep["post_fetch_index0_twice_differ"] = not torch.equal(a, b)
    rep["post_fetch_index0"] = [d4_shift_match(a, data.arrays[0]), d4_shift_match(b, data.arrays[0])]
    rep["post_fetch_index0_maxabs_diff"] = float((a - b).abs().max())
    rep["labels_unchanged"] = bool(torch.equal(dataset[0]["labels"], data.labels[0]))

    def counters(ck: Path) -> dict:
        sch = torch.load(ck / "scheduler.bin", map_location="cpu", weights_only=False)
        opt = torch.load(ck / "optimizer.bin", map_location="cpu", weights_only=False)
        st = next(iter(opt["state"].values()))
        g = opt["param_groups"][0]
        return {"scheduler_last_epoch": int(sch["last_epoch"]), "scheduler_T_cur": float(sch.get("T_cur", -1)),
                "adam_step": float(st["step"]), "lr": float(g["lr"]), "weight_decay": float(g["weight_decay"]),
                "betas": list(g["betas"]), "has_augmentations_pkl": (ck / "augmentations.pkl").exists()}

    new_ck = Path(utils.find_latest_checkpoint(str(out_dir)))
    rep["source_checkpoint"], rep["new_checkpoint"] = counters(src), counters(new_ck)
    rep["new_checkpoint_name"] = new_ck.name
    rep["updates_in_check"] = len(result["metrics"]["loss"])
    rep["epoch_lr"] = result["metrics"]["epoch_lr"]
    Path(args.check_out).write_text(json.dumps(rep, indent=2))
    print(json.dumps(rep, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
