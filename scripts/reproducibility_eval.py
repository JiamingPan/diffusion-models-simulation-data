#!/usr/bin/env python
"""Compare multiple generated sample sets for reproducibility."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--generated",
        action="append",
        required=True,
        help="Name:path pair for a generated .npy set. Repeat this argument.",
    )
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--nbins-pk", type=int, default=25)
    args = parser.parse_args()

    project_root = Path.cwd()
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    from simdiff_eval.io import as_nchw, load_npy
    from simdiff_eval.metrics import reproducibility_summary

    sample_sets = {}
    for item in args.generated:
        if ":" not in item:
            raise ValueError("--generated must be NAME:PATH")
        name, path = item.split(":", 1)
        sample_sets[name] = as_nchw(load_npy(path))

    metrics = reproducibility_summary(sample_sets, nbins=args.nbins_pk)
    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(metrics, indent=2))
    print(f"Wrote reproducibility metrics to {output_json}")


if __name__ == "__main__":
    main()
