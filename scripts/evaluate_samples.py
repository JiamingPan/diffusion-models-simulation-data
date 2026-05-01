#!/usr/bin/env python
"""Evaluate generated CAMELS field samples against real samples."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


def _ensure_project_imports(project_root: Path) -> None:
    for candidate in [project_root, project_root / "cosmo_diffusion"]:
        if candidate.exists() and str(candidate) not in sys.path:
            sys.path.insert(0, str(candidate))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    real_group = parser.add_mutually_exclusive_group(required=True)
    real_group.add_argument("--real", help="Real samples .npy, already in comparable normalized space.")
    real_group.add_argument("--real-config", help="Training YAML config used to load/normalize real data.")
    parser.add_argument("--generated", required=True, help="Generated samples .npy.")
    parser.add_argument("--output-json", required=True, help="Where to write metrics JSON.")
    parser.add_argument("--fig-dir", default=None, help="Optional directory for histogram and P(k) figures.")
    parser.add_argument("--nbins-pk", type=int, default=25)
    parser.add_argument("--max-real-nn", type=int, default=2048)
    parser.add_argument("--max-generated-nn", type=int, default=256)
    args = parser.parse_args()

    project_root = Path.cwd()
    _ensure_project_imports(project_root)

    from simdiff_eval.io import as_nchw, load_npy, load_real_from_config
    from simdiff_eval.metrics import (
        field_histogram,
        nearest_neighbor_distances,
        power_spectrum_summary,
    )

    if args.real_config:
        real = load_real_from_config(args.real_config)
    else:
        real = as_nchw(load_npy(args.real))
    generated = as_nchw(load_npy(args.generated))

    metrics = {
        "real_shape": list(real.shape),
        "generated_shape": list(generated.shape),
        "real_histogram_stats": field_histogram(real),
        "generated_histogram_stats": field_histogram(generated),
        "power_spectrum": power_spectrum_summary(real, generated, nbins=args.nbins_pk),
        "nearest_neighbor": nearest_neighbor_distances(
            real,
            generated,
            max_real=args.max_real_nn,
            max_generated=args.max_generated_nn,
        ),
    }

    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(metrics, indent=2))
    print(f"Wrote metrics to {output_json}")

    if args.fig_dir is not None:
        from simdiff_eval.plotting import save_histogram_plot, save_power_ratio_plot

        fig_dir = Path(args.fig_dir)
        save_histogram_plot(real, generated, fig_dir / "field_histogram.png")
        save_power_ratio_plot(real, generated, fig_dir / "power_ratio.png", nbins=args.nbins_pk)
        print(f"Wrote figures to {fig_dir}")


if __name__ == "__main__":
    main()
