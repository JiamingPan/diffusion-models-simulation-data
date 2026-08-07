#!/usr/bin/env python
"""Build the focused, unexecuted DiT 300k scaling results notebook."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_PATH = ROOT / "notebooks" / "nf_generalize_fig2_dit_300k_scaling.ipynb"


def stable_cell_id(kind: str, section: str, source: str) -> str:
    digest = hashlib.sha1(f"{kind}\0{section}\0{source}".encode()).hexdigest()[:12]
    return f"{kind[:1]}-{digest}"


def markdown_cell(source: str, *, section: str) -> dict[str, Any]:
    return {
        "cell_type": "markdown",
        "id": stable_cell_id("markdown", section, source),
        "metadata": {"analysis_section": section},
        "source": source.splitlines(keepends=True),
    }


def code_cell(source: str, *, section: str) -> dict[str, Any]:
    return {
        "cell_type": "code",
        "execution_count": None,
        "id": stable_cell_id("code", section, source),
        "metadata": {"analysis_section": section},
        "outputs": [],
        "source": source.splitlines(keepends=True),
    }


INTRO = r"""# DiT Memorization-to-Generalization Scaling at 300k

This notebook is the reader-facing DiT scaling and validity analysis. It uses
DiT-L8 200k and DiT-L12 / base 200k as the existing fixed-budget depth
references, DiT-L16 fresh 300k as the clean replacement sweep, and each UNet
curve as a historical UNet reference.

The comparison therefore uses **unequal optimizer-update budgets**. It is an
empirical diagnostic of the available models and does not establish a universal
capacity scaling law.
"""


TLDR = r"""## TL;DR and interpretation rules

1. The novelty curves ask whether generated fields remain close to individual
   training slices. High novelty is necessary for generalization but does not
   establish physical validity.
2. The one-point and power-spectrum sections ask whether the generated
   distribution retains the statistics of the exact training subset used by
   each model.
3. Multiple generated samples and per-sample error tails are shown because a
   mean curve can hide unstable or visibly invalid generations.
4. Every sampler comparison must resolve to the same fresh 300k checkpoint.
"""


SETUP = r"""from __future__ import annotations

import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from IPython.display import Markdown, display


def resolve_project_root(start: Path | None = None) -> Path:
    current = (start or Path.cwd()).resolve()
    for candidate in (current, *current.parents):
        if (candidate / 'scripts').is_dir() and (candidate / 'notebooks').is_dir():
            return candidate
    raise FileNotFoundError('Could not locate the diffusion_models_repo project root')


PROJECT_DIR = resolve_project_root()
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from scripts.dit_300k_scaling_analysis import (
    DATASET_POWERS,
    DATASET_SIZES,
    DATASET_TAGS,
    DIT_LABELS,
    FRESH_SAMPLE_COUNT,
    FRESH_SAMPLE_LABEL,
    FRESH_SAMPLER_STEPS,
    FRESH_SCHEDULER,
    FRESH_SWEEP_NAME,
    FRESH_TRAINING_SEED,
    interpolate_n50,
    require_exact_dataset_sweep,
    validate_sample_archive_metadata,
)

DIT_RESULT_DIR = PROJECT_DIR / 'results' / 'nf_generalize_fig2_dit'
DIT_TABLE_DIR = DIT_RESULT_DIR / 'tables'
UNET_TABLE_DIR = PROJECT_DIR / 'results' / 'nf_generalize_fig2' / 'tables'
FRESH_RESULT_DIR = PROJECT_DIR / 'results' / FRESH_SWEEP_NAME
FRESH_SAMPLE_DIR = FRESH_RESULT_DIR / 'samples'
FRESH_MANIFEST_PATH = PROJECT_DIR / 'local' / FRESH_SWEEP_NAME / 'manifest.json'
QUICKCHECK_DIR = DIT_RESULT_DIR / 'quickcheck' / 'focused_300k_scaling'
CACHE_DIR = FRESH_RESULT_DIR / 'cache'
QUICKCHECK_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'font.size': 12,
    'axes.titlesize': 16,
    'axes.labelsize': 14,
    'legend.fontsize': 11,
    'figure.dpi': 120,
    'savefig.dpi': 180,
    'axes.spines.top': False,
    'axes.spines.right': False,
})
"""


SECTION_MARKDOWN = (
    (
        "input-audit",
        """## Input audit

The audit below establishes table provenance, exact checkpoint identity,
sampler metadata, sample count, and the exact real training subset before any
scientific figure is drawn.
""",
    ),
    (
        "transition",
        """## Generalization transition

PCA and SSCD q95 novelty are shown over the full data-size range and again in
a separate transition view. Historical UNet curves remain visually quiet so
the available DiT depth comparison stays legible.
""",
    ),
    (
        "transition-summary",
        """## Transition summary

`N50` is the interpolated training size where q95 novelty crosses 0.5. The
summary reports censoring or ambiguity instead of forcing a transition value.
""",
    ),
    (
        "optimization",
        """## Fresh L16 optimization across all ten training sizes

All loss curves use optimizer-update coordinates and the same loss definition.
Denoising-loss convergence is not treated as evidence of novelty or correct
physical statistics.
""",
    ),
    (
        "generated-fields",
        """## Generated-field stability across all ten training sizes

Four deterministic sample indices are displayed at every data size with common
normalization. These panels expose unstable outputs without selecting a single
visually convenient example.
""",
    ),
    (
        "nearest-training",
        """## Generated samples versus nearest training slices

Each generated field is compared against the complete configured training
subset for its model. Aggregate nearest-similarity distributions accompany the
example matches.
""",
    ),
    (
        "one-point",
        """## One-point distributions across all ten training sizes

Real and generated histograms use shared bins. Each black reference is computed
from the exact model training subset, not the complete CAMELS collection.
""",
    ),
    (
        "power-spectrum",
        """## Power spectra across all ten training sizes

Mean generated-to-real power ratios use common axes, followed by a
scale-resolved log-ratio heatmap.
""",
    ),
    (
        "outliers",
        """## Per-sample outlier distributions

Median, interquartile range, 95th percentile, and maximum errors distinguish a
systematic shift from a small tail of unstable generations.
""",
    ),
    (
        "sampler",
        """## Sampler audit on the same fresh 300k checkpoints

DPM50, DPM100, DPM200, and DDPM500 are compared only when archive metadata
proves that checkpoint, seed, configuration, sample count, and real reference
are identical.
""",
    ),
    (
        "takeaways",
        """## Takeaways and limitations

Observed transition ordering and physical-statistics failures are reported separately.
This mixed-budget comparison does not establish a universal capacity scaling law and is
not used to fit a universal scaling exponent; sampler sensitivity is audited independently.
""",
    ),
)


def build_notebook() -> dict[str, Any]:
    cells = [
        markdown_cell(INTRO, section="intro"),
        markdown_cell(TLDR, section="tldr"),
        code_cell(SETUP, section="setup"),
    ]
    cells.extend(markdown_cell(text, section=section) for section, text in SECTION_MARKDOWN)
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "version": "3.10"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def main() -> None:
    notebook = build_notebook()
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(notebook, indent=1, ensure_ascii=False) + "\n")
    print(f"Wrote {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
