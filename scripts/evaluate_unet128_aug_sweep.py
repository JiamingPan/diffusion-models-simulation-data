#!/usr/bin/env python
"""Orbit-aware novelty, P(k) and one-point checks for the width-128 augmentation sweep.

Complements the existing PCA95 score (compute_nf_generalize_pca_full_nn.py on
local/unet128_aug_sweep/manifest.json). Per N, for the noaug baseline and the
three arms:

* nearest-training similarity allowing all 8 D4 elements and all periodic
  shifts (simdiff_eval.orbit_nn), in model space, centred cosine;
* copy thresholds from train-to-train orbit similarities with each map's own
  entire orbit excluded (q95, q99); generated and held-out real maps scored
  against the same thresholds;
* P(k) ratio (ensemble mean power) and one-point PDF L1 against held-out real
  maps, with train-vs-held-out as the reference level.

The warp arm is a distortion control: novelty it gains is not evidence of
generalization unless P(k)/one-point stay at the reference level.
Reads saved samples only; no sampling. Run on a GPU node.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))


def load_source_slices(path, start: int, count: int, zthin: int):
    """Same slicing as compute_nf_generalize_pca_full_nn.load_source_slices (z-thinned 2-D slices)."""
    import numpy as np
    arr = np.load(path, mmap_mode="r")
    stop = min(start + count, len(arr))
    if count <= 0 or stop <= start:
        return np.empty((0, arr.shape[-2], arr.shape[-1]), dtype=np.float32)
    slices = np.asarray(arr[start:stop, ::zthin], dtype=np.float32)
    return slices.reshape(-1, slices.shape[-2], slices.shape[-1])


def evenly_limit(arr, n):
    import numpy as np
    if n is None or n <= 0 or len(arr) <= n:
        return arr
    return arr[np.linspace(0, len(arr) - 1, int(n), dtype=np.int64)]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=ROOT / "results/unet128_aug_sweep/orbit_eval")
    parser.add_argument("--dataset-size", type=int, action="append", help="Restrict to N (repeatable)")
    parser.add_argument("--val-raw-per-source", type=int, default=8)
    parser.add_argument("--max-val-slices", type=int, default=512)
    parser.add_argument("--max-generated", type=int, default=512)
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args()

    from simdiff_eval.torch_compat import install_torch_backend_compat
    install_torch_backend_compat(entry_point="evaluate_unet128_aug_sweep")
    import numpy as np
    import pandas as pd
    import torch
    import yaml
    sys.path.insert(0, str(args.runtime_root))
    from cosmodiff import utils
    from simdiff_eval.orbit_nn import orbit_max_cosine
    from simdiff_eval.conditional_unet_diagnostics import compare_power
    from simdiff_eval.dit_diagnostics import one_point_l1_common_bins

    runs = json.loads((ROOT / "local/unet128_aug_sweep/runs.json").read_text())
    fig2 = {r["run_name"]: r for r in json.loads((ROOT / "local/nf_generalize_fig2/manifest.json").read_text())}
    sizes = sorted({r["dataset_size"] for r in runs} & set(args.dataset_size or [r["dataset_size"] for r in runs]))
    args.out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for n in sizes:
        base = fig2[next(r["baseline_run_name"] for r in runs if r["dataset_size"] == n)]
        config = yaml.safe_load((ROOT / base["config"]).read_text())
        loaded = utils.parse_config_data(config)
        train = loaded["data"].arrays.cpu().numpy()[:, 0].astype(np.float32)
        tform, norm = loaded["tform"], loaded["norm"]

        def to_model_space(raw):
            x = torch.as_tensor(raw[:, None], dtype=torch.float32)
            x = tform(x) if tform is not None else x
            x = norm(x) if norm is not None else x
            return x.numpy()[:, 0]

        # Self-check: the same pipeline on the raw training slices reproduces the training maps.
        zthin = int(base["zthin"])
        raw_train = np.concatenate([load_source_slices(s["path"], 0, int(s["n_samples"]), zthin) for s in base["source_counts"]])
        assert np.allclose(to_model_space(raw_train), train, atol=1e-5), "val normalization would not match training"
        raw_val = np.concatenate([load_source_slices(s["path"], int(s["n_samples"]), args.val_raw_per_source, zthin)
                                  for s in base["source_counts"]])
        val = to_model_space(evenly_limit(raw_val, args.max_val_slices))

        ref = orbit_max_cosine(train, train, exclude_same_index=True)
        q95, q99 = (float(np.quantile(ref["max_cosine"], q)) for q in (0.95, 0.99))
        val_nn = orbit_max_cosine(val, train)["max_cosine"]
        pk_ref, k = compare_power(train, val)
        l1_ref = one_point_l1_common_bins(val, train)
        entries = [("noaug", base["run_name"], ROOT / f"results/nf_generalize_fig2/samples/{base['run_name']}_seed{args.seed}_dpm50.npz")]
        entries += [(r["arm_tag"], r["run_name"], ROOT / r["sample_path"].format(seed=args.seed))
                    for r in runs if r["dataset_size"] == n]
        for arm_tag, name, path in entries:
            row = {"dataset_size": n, "arm": arm_tag, "run_name": name, "sample_path": str(path),
                   "threshold_q95_orbit_excluded": q95, "threshold_q99_orbit_excluded": q99,
                   "val_copy_fraction_q95": float(np.mean(val_nn > q95)),
                   "pk_train_vs_val_max_abs_dev": float(np.max(np.abs(pk_ref - 1))), "one_point_l1_train_vs_val": l1_ref,
                   "distortion_control": arm_tag == "warp"}
            if not path.is_file():
                rows.append({**row, "status": "missing samples"}); continue
            with np.load(path) as z:
                gen = np.asarray(z["samples"], dtype=np.float32).reshape(-1, 128, 128)[:args.max_generated]
            nn = orbit_max_cosine(gen, train)
            pk, _ = compare_power(gen, val)
            transformed = (nn["group_element"] != 0) | (nn["shift_y"] != 0) | (nn["shift_x"] != 0)
            row.update(status="ok", n_generated=len(gen),
                       gen_orbit_nn_median=float(np.median(nn["max_cosine"])),
                       gen_copy_fraction_q95=float(np.mean(nn["max_cosine"] > q95)),
                       gen_copy_fraction_q99=float(np.mean(nn["max_cosine"] > q99)),
                       gen_copies_q95_matched_via_transform=float(np.mean(transformed[nn["max_cosine"] > q95])) if np.any(nn["max_cosine"] > q95) else 0.0,
                       pk_gen_vs_val_max_abs_dev=float(np.max(np.abs(pk - 1))),
                       one_point_l1_gen_vs_val=one_point_l1_common_bins(val, gen))
            np.savez(args.out_dir / f"{name}_orbit_nn.npz", k=k, pk_ratio=pk, pk_ratio_train_ref=pk_ref,
                     train_ref_orbit_nn=ref["max_cosine"], val_orbit_nn=val_nn, **{f"gen_{key}": v for key, v in nn.items()})
            rows.append(row)
            print(json.dumps(row))
    table = pd.DataFrame(rows)
    out = args.out_dir / "orbit_pk_onepoint_summary.csv"
    table.to_csv(out, index=False)
    print("Wrote", out)


if __name__ == "__main__":
    main()
