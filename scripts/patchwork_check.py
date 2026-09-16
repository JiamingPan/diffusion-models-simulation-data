"""Is each junk sample a patchwork of different training maps?

For every generated sample, every 8x8 patch is matched (centered cosine) to the
same-location patch of every map in the exact configured training subset.  A
clean copy draws all its patches from one training map; the hypothesised L16
failure draws them from many.

Per sample it reports
  majority_fraction  fraction of patches whose nearest map is the modal map
  n_distinct         number of distinct nearest maps across the 256 patches
  max_cos            whole-image max cosine to the training set (copy detector)

and writes a figure showing samples next to their patch-assignment maps.

    python scripts/patchwork_check.py --config <run.yaml> --eval-root <simdiff_eval checkout> \
        --samples results/late_start/dit_l16_fresh300k_t499_n128.npz results/late_start/dit_l8_200k_t499_n128.npz \
        --out-dir results/patchwork
"""
from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np

PATCH = 8
COPY_THRESHOLD = 0.98


def scalar(data, key: str):
    if key not in data:
        raise ValueError(f"sample file lacks required provenance field {key!r}")
    value = np.asarray(data[key])
    if value.size != 1:
        raise ValueError(f"sample provenance {key!r} is not scalar: {value.shape}")
    return value.reshape(-1)[0].item()


def validate_provenance(data, reference: np.ndarray) -> None:
    expected_mean = hashlib.sha256(
        reference.mean(axis=0, keepdims=True).tobytes()
    ).hexdigest()
    if str(scalar(data, "training_mean_sha256")) != expected_mean:
        raise ValueError("sample training subset does not match the patch reference")
    expected = {
        "late_start_actual": 499,
        "late_start_steps_run": 50,
        "late_start_init": "mean",
        "seed": 123,
        "num_steps": 50,
        "ema_sigma_rel": -1.0,
        "scheduler_class": "DPMSolverMultistepScheduler",
    }
    for key, wanted in expected.items():
        found = scalar(data, key)
        if found != wanted:
            raise ValueError(f"sample provenance {key}={found!r}, expected {wanted!r}")


def load_reference(config_path: Path, eval_root: Path | None) -> np.ndarray:
    for root in [eval_root, Path.cwd(), Path.cwd() / "scripts"]:
        if root is not None and str(root) not in sys.path:
            sys.path.insert(0, str(root))
    from simdiff_eval.io import iter_real_reference_batches_from_config
    real = np.concatenate(list(iter_real_reference_batches_from_config(config_path))).astype(np.float32)
    assert real.ndim == 4 and np.isfinite(real).all(), real.shape
    return real


def to_patches(images: np.ndarray) -> np.ndarray:
    """(N,1,H,W) -> (N, P, 64) centered, unit-norm patches in raster order."""
    if images.ndim != 4 or images.shape[1] != 1:
        raise ValueError(f"expected NCHW single-channel fields, found {images.shape}")
    if images.shape[2] % PATCH or images.shape[3] % PATCH:
        raise ValueError(f"field dimensions must be divisible by patch size {PATCH}")
    if not np.isfinite(images).all():
        raise ValueError("fields contain non-finite values")
    n, _, h, w = images.shape
    x = images[:, 0].reshape(n, h // PATCH, PATCH, w // PATCH, PATCH).transpose(0, 1, 3, 2, 4)
    x = x.reshape(n, -1, PATCH * PATCH).astype(np.float32)
    x = x - x.mean(axis=2, keepdims=True)
    norms = np.linalg.norm(x, axis=2, keepdims=True)
    return x / np.maximum(norms, 1e-8)


def whole_image_max_cos(samples: np.ndarray, reference: np.ndarray) -> np.ndarray:
    q = samples.reshape(len(samples), -1).astype(np.float32)
    r = reference.reshape(len(reference), -1).astype(np.float32)
    q /= np.maximum(np.linalg.norm(q, axis=1, keepdims=True), 1e-8)
    r /= np.maximum(np.linalg.norm(r, axis=1, keepdims=True), 1e-8)
    return (q @ r.T).max(axis=1)


def patch_assignments(samples: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """(N, P) index of the nearest same-location training patch."""
    sp, rp = to_patches(samples), to_patches(reference)
    out = np.empty((sp.shape[0], sp.shape[1]), dtype=np.int32)
    for p in range(sp.shape[1]):
        out[:, p] = (sp[:, p, :] @ rp[:, p, :].T).argmax(axis=1)
    return out


def summarize(assign: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    modal = np.array([np.bincount(row).argmax() for row in assign])
    majority = (assign == modal[:, None]).mean(axis=1)
    distinct = np.array([len(np.unique(row)) for row in assign])
    return majority, distinct, modal


def corrupted_single_maps(
    reference: np.ndarray, *, cosine: float, rng: np.random.Generator
) -> np.ndarray:
    """Return globally corrupted maps with an exact cosine to their source map."""
    source = reference.astype(np.float64).reshape(len(reference), -1)
    noise = rng.normal(size=source.shape)
    noise -= (
        (noise * source).sum(axis=1, keepdims=True)
        / np.maximum((source * source).sum(axis=1, keepdims=True), 1e-12)
    ) * source
    source_norm = np.linalg.norm(source, axis=1, keepdims=True)
    noise *= source_norm / np.maximum(np.linalg.norm(noise, axis=1, keepdims=True), 1e-12)
    mixed = cosine * source + np.sqrt(1.0 - cosine**2) * noise
    return mixed.reshape(reference.shape).astype(np.float32)


def synthetic_patchworks(
    reference: np.ndarray, *, count: int, rng: np.random.Generator
) -> np.ndarray:
    """Construct fields by selecting each same-location patch from a random map."""
    _, channels, height, width = reference.shape
    out = np.empty((count, channels, height, width), dtype=np.float32)
    for sample_index in range(count):
        for row in range(0, height, PATCH):
            for col in range(0, width, PATCH):
                source_index = int(rng.integers(len(reference)))
                out[sample_index, :, row : row + PATCH, col : col + PATCH] = (
                    reference[source_index, :, row : row + PATCH, col : col + PATCH]
                )
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--samples", type=Path, nargs="+", required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--eval-root", type=Path, default=None)
    parser.add_argument("--n-show", type=int, default=4, help="junk and copy examples per file in the figure")
    args = parser.parse_args()

    import csv
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    reference = load_reference(args.config, args.eval_root)
    if args.out_dir.exists():
        raise FileExistsError(f"refusing to overwrite output directory {args.out_dir}")
    args.out_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(
            prefix=f".{args.out_dir.name}.tmp.", dir=args.out_dir.parent
        )
    )
    side = reference.shape[-1] // PATCH
    try:
        # These controls distinguish an actual mosaic from one globally corrupted map.
        rng = np.random.default_rng(0)
        calibration = [
            ("real_training_maps", reference[:64]),
            (
                "corrupted_single_map_cos050",
                corrupted_single_maps(reference[:64], cosine=0.5, rng=rng),
            ),
            (
                "synthetic_patchwork",
                synthetic_patchworks(reference, count=64, rng=rng),
            ),
            (
                "gaussian_noise",
                rng.normal(size=(64, *reference.shape[1:])).astype(np.float32),
            ),
        ]
        rows = []
        for name, arr in calibration:
            maj, dist, _ = summarize(patch_assignments(arr, reference))
            max_cos = whole_image_max_cos(arr, reference)
            rows.append(
                {
                    "file": name,
                    "population": "calibration",
                    "n": len(arr),
                    "majority_median": float(np.median(maj)),
                    "distinct_median": float(np.median(dist)),
                    "max_cos_median": float(np.median(max_cos)),
                }
            )
            print(
                f"[calibration] {name}: majority median {np.median(maj):.3f}, "
                f"distinct median {np.median(dist):.0f}, "
                f"max cosine median {np.median(max_cos):.3f}"
            )

        for path in args.samples:
            with np.load(path, allow_pickle=False) as data:
                validate_provenance(data, reference)
                samples = data["samples"].astype(np.float32)
            if samples.ndim != 4 or not np.isfinite(samples).all():
                raise ValueError(f"{path}: samples must be finite NCHW fields")
            if samples.shape[1:] != reference.shape[1:]:
                raise ValueError(
                    f"{path}: sample shape {samples.shape} vs reference {reference.shape}"
                )
            assign = patch_assignments(samples, reference)
            majority, distinct, modal = summarize(assign)
            max_cos = whole_image_max_cos(samples, reference)
            copies = max_cos > COPY_THRESHOLD
            per_sample = temporary / f"{path.stem}_patchwork.csv"
            with per_sample.open("w", newline="") as handle:
                writer = csv.writer(handle)
                writer.writerow(["index", "max_cos", "is_copy", "majority_fraction", "n_distinct", "modal_map"])
                for i in range(len(samples)):
                    writer.writerow([i, f"{max_cos[i]:.4f}", int(copies[i]), f"{majority[i]:.4f}", int(distinct[i]), int(modal[i])])
            for pop_name, mask in [("copies", copies), ("noncopies", ~copies)]:
                if mask.any():
                    rows.append({"file": path.name, "population": pop_name, "n": int(mask.sum()),
                                 "majority_median": float(np.median(majority[mask])),
                                 "distinct_median": float(np.median(distinct[mask])),
                                 "max_cos_median": float(np.median(max_cos[mask]))})
                    print(f"{path.name} {pop_name}: n={mask.sum()} majority median {np.median(majority[mask]):.3f}, "
                          f"distinct median {np.median(distinct[mask]):.0f}, "
                          f"max cosine median {np.median(max_cos[mask]):.3f}")

            # Figure: worst junk and best copies, sample beside its patch-assignment map.
            order_junk = np.argsort(max_cos)[: args.n_show]
            order_copy = np.argsort(-max_cos)[: args.n_show]
            picks = list(order_junk) + list(order_copy)
            fig, axes = plt.subplots(2, len(picks), figsize=(2.3 * len(picks), 5), constrained_layout=True)
            for col, i in enumerate(picks):
                axes[0, col].imshow(samples[i, 0], cmap="viridis", vmin=-1, vmax=1)
                axes[0, col].set_title(f"#{i} cos {max_cos[i]:.2f}", fontsize=9)
                ids = assign[i].reshape(side, side)
                # Relabel within one sample so equal colors mean the same source map.
                _, inv = np.unique(ids, return_inverse=True)
                axes[1, col].imshow(inv.reshape(side, side), cmap="tab20", interpolation="nearest")
                axes[1, col].set_title(f"maj {majority[i]:.2f} / {distinct[i]} maps", fontsize=9)
            for ax in axes.flat:
                ax.set_xticks([]); ax.set_yticks([])
            axes[0, 0].set_ylabel("sample"); axes[1, 0].set_ylabel("nearest map per patch")
            fig.suptitle(f"{path.name}: lowest-cosine (left) vs highest-cosine (right) samples", fontsize=11)
            fig.savefig(temporary / f"{path.stem}_patchwork.png", dpi=130)
            plt.close(fig)

        with (temporary / "patchwork_summary.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
            writer.writeheader(); writer.writerows(rows)
        os.replace(temporary, args.out_dir)
        print(f"wrote {args.out_dir / 'patchwork_summary.csv'}")
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)


if __name__ == "__main__":
    main()
