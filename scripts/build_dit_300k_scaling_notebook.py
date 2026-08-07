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
    MODEL_PARAMETER_COUNTS,
    build_historical_unet_metric_table,
    build_mixed_dit_metric_table,
    interpolate_n50,
    normalize_generalization_table,
    require_exact_dataset_sweep,
    summarize_n50,
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


INPUT_AUDIT_CODE = r"""HISTORICAL_DIT_TABLES = {
    'PCA': DIT_TABLE_DIR / 'nf_generalize_fig2_dit_pca_full_nn_metrics.csv',
    'SSCD': DIT_TABLE_DIR / 'nf_generalize_fig2_dit_sscd_full_nn_metrics.csv',
}
FRESH_L16_TABLES = {
    'PCA': DIT_TABLE_DIR / 'nf_generalize_fig2_dit_l16_fresh300k_v2_pca_full_nn_metrics.csv',
    'SSCD': DIT_TABLE_DIR / 'nf_generalize_fig2_dit_l16_fresh300k_v2_sscd_full_nn_metrics.csv',
}
UNET_TABLES = {
    'PCA': UNET_TABLE_DIR / 'nf_generalize_fig2_pca_full_nn_metrics.csv',
    'SSCD': UNET_TABLE_DIR / 'nf_generalize_fig2_sscd_full_nn_metrics.csv',
}


def read_required_csv(path: Path, *, context: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f'{context}: missing {path}')
    table = pd.read_csv(path)
    if table.empty:
        raise ValueError(f'{context}: empty table {path}')
    return table


audit_rows = []
for family, paths in (
    ('historical DiT 200k', HISTORICAL_DIT_TABLES),
    ('fresh independent 300k v2', FRESH_L16_TABLES),
    ('historical UNet reference', UNET_TABLES),
):
    for feature, path in paths.items():
        audit_rows.append({
            'family': family,
            'feature': feature,
            'exists': path.exists(),
            'bytes': path.stat().st_size if path.exists() else np.nan,
            'path': str(path.relative_to(PROJECT_DIR)),
        })

if not FRESH_MANIFEST_PATH.exists():
    raise FileNotFoundError(f'Missing frozen fresh-sweep manifest: {FRESH_MANIFEST_PATH}')
fresh_manifest_rows = json.loads(FRESH_MANIFEST_PATH.read_text())
fresh_manifest = pd.DataFrame(fresh_manifest_rows).sort_values('dataset_size').reset_index(drop=True)
manifest_tags = tuple(fresh_manifest['dataset_tag'].astype(str))
if manifest_tags != DATASET_TAGS:
    raise ValueError(f'Fresh manifest tags {manifest_tags} do not equal {DATASET_TAGS}')
if not (pd.to_numeric(fresh_manifest['target_total_updates']) == 300_000).all():
    raise ValueError('Fresh manifest does not target exactly 300000 updates for every run')
if not (fresh_manifest['sample_label'].astype(str) == FRESH_SAMPLE_LABEL).all():
    raise ValueError('Fresh manifest sample labels do not match the fixed fresh label')

file_audit = pd.DataFrame(audit_rows)
display(file_audit)
display(fresh_manifest[['dataset_tag', 'dataset_size', 'run_name', 'target_total_updates', 'sample_label']])
if not file_audit['exists'].all():
    missing = file_audit.loc[~file_audit['exists'], 'path'].tolist()
    raise FileNotFoundError(f'Required metric tables are missing: {missing}')
"""


TRANSITION_CODE = r"""mixed_metric_by_feature = {}
unet_metric_by_feature = {}
for feature in ('PCA', 'SSCD'):
    historical = read_required_csv(
        HISTORICAL_DIT_TABLES[feature], context=f'historical {feature} DiT'
    )
    fresh = read_required_csv(
        FRESH_L16_TABLES[feature], context=f'fresh independent 300k v2 {feature}'
    )
    mixed_metric_by_feature[feature] = build_mixed_dit_metric_table(
        historical, fresh, feature=feature
    )
    unet_metric_by_feature[feature] = build_historical_unet_metric_table(
        read_required_csv(UNET_TABLES[feature], context=f'historical UNet reference {feature}'),
        feature=feature,
    )

transition_rows = pd.concat(mixed_metric_by_feature.values(), ignore_index=True)
display(
    transition_rows[
        ['feature', 'model_label', 'updates_k', 'dataset_tag', 'dataset_size',
         'gen_gl_q95', 'source']
    ].sort_values(['feature', 'updates_k', 'model_label', 'dataset_size'])
)

DIT_STYLES = {
    'dit_l8': dict(color='#009E73', marker='P', lw=2.8, ms=7.5),
    'dit_base': dict(color='#0072B2', marker='D', lw=2.8, ms=7.5),
    'dit_l16': dict(color='#B33C86', marker='X', lw=3.2, ms=8.5),
}
UNET_STYLES = {
    'u64': dict(color='#B8B8B8', marker='^', ls=(0, (5, 3))),
    'u128': dict(color='#858585', marker='o', ls=(0, (2, 2))),
    'u256': dict(color='#505050', marker='s', ls=(0, (8, 3))),
}
UNET_LABELS = {'u64': 'UNet-64 historical', 'u128': 'UNet-128 historical', 'u256': 'UNet-256 historical'}


def plot_mixed_budget_transition(*, zoom: bool) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(15.8, 5.9), sharey=True)
    for axis, feature in zip(axes, ('PCA', 'SSCD')):
        unet = unet_metric_by_feature[feature]
        mixed = mixed_metric_by_feature[feature]
        for arch, style in UNET_STYLES.items():
            subset = unet[unet['arch'].astype(str) == arch].sort_values('dataset_size')
            if subset.empty:
                continue
            axis.plot(
                np.log2(subset['dataset_size']), subset['gen_gl_q95'],
                color=style['color'], marker=style['marker'], linestyle=style['ls'],
                lw=1.6, ms=5.5, markerfacecolor='white', label=UNET_LABELS[arch], zorder=1,
            )
        for arch, style in DIT_STYLES.items():
            subset = mixed[mixed['arch'] == arch].sort_values('dataset_size')
            axis.plot(
                np.log2(subset['dataset_size']), subset['gen_gl_q95'],
                label=DIT_LABELS[arch], zorder=3, **style,
            )
        axis.axhline(0.5, color='0.3', lw=1.2, ls=':', zorder=0)
        powers = range(6, 12) if zoom else DATASET_POWERS
        axis.set_xticks(list(powers), [rf'$2^{{{power}}}$' for power in powers])
        axis.set_xlim(5.7, 11.3 if zoom else 15.3)
        axis.set_ylim(-0.035, 1.035)
        axis.set_xlabel(r'Training images $N_{2D}$')
        axis.set_title(f'{feature} q95 novelty')
        axis.grid(axis='y', alpha=0.16)
    axes[0].set_ylabel('Novelty score')
    handles, labels = axes[0].get_legend_handles_labels()
    fig.suptitle(
        'Memorization-to-generalization transition' + (' (transition zoom)' if zoom else ''),
        fontsize=21, fontweight='semibold', y=0.98,
    )
    fig.text(
        0.5, 0.91,
        'L8/L12 use 200k updates; fresh independent L16 uses 300k. UNets are historical references.',
        ha='center', fontsize=12.5, color='0.3',
    )
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0.865), ncol=3, frameon=False)
    fig.subplots_adjust(left=0.07, right=0.99, bottom=0.14, top=0.72, wspace=0.10)
    filename = (
        'dit_300k_mixed_budget_transition_zoom.png'
        if zoom else 'dit_300k_mixed_budget_transition_full.png'
    )
    output = QUICKCHECK_DIR / filename
    fig.savefig(output, bbox_inches='tight')
    plt.show()
    print('wrote', output)
    return output


transition_full_path = plot_mixed_budget_transition(zoom=False)
transition_zoom_path = plot_mixed_budget_transition(zoom=True)
"""


TRANSITION_SUMMARY_CODE = r"""transition_n50 = summarize_n50(transition_rows)
transition_n50_path = QUICKCHECK_DIR / 'dit_300k_transition_n50.csv'
transition_n50.to_csv(transition_n50_path, index=False)
display(transition_n50)
print('wrote', transition_n50_path)

status_notes = {
    'crossing': 'unique upward q95=0.5 crossing',
    'left_censored': 'already above 0.5 at the smallest measured data size',
    'right_censored': 'still below 0.5 at the largest measured data size',
    'ambiguous': 'multiple or downward crossings; no single transition is reported',
}
for _, row in transition_n50.iterrows():
    print(
        f"{row['feature']:4s} | {row['model_label']:24s} | "
        f"status={row['status']}: {status_notes[row['status']]}"
    )

unet_transition_rows = pd.concat(unet_metric_by_feature.values(), ignore_index=True)
capacity_transition_rows = pd.concat(
    [transition_rows, unet_transition_rows], ignore_index=True
)
capacity_n50 = summarize_n50(capacity_transition_rows)
capacity_n50['model_parameters'] = capacity_n50['arch'].map(MODEL_PARAMETER_COUNTS)
capacity_n50['family'] = np.where(
    capacity_n50['arch'].str.startswith('u'), 'UNet', 'DiT'
)
capacity_table_path = QUICKCHECK_DIR / 'dit_300k_capacity_n50_diagnostic.csv'
capacity_n50.to_csv(capacity_table_path, index=False)
display(capacity_n50)

fig, axes = plt.subplots(1, 2, figsize=(13.8, 5.5), sharey=True)
family_styles = {
    'UNet': dict(color='0.45', marker='o'),
    'DiT': dict(color='#0072B2', marker='D'),
}
for axis, feature in zip(axes, ('PCA', 'SSCD')):
    subset = capacity_n50[capacity_n50['feature'] == feature]
    for family, style in family_styles.items():
        points = subset[subset['family'] == family]
        axis.scatter(
            points['model_parameters'], points['n50'], s=88,
            edgecolor='white', linewidth=0.8, label=family, **style,
        )
        for _, point in points.iterrows():
            short_label = point['model_label'].replace(' historical 200k', '').replace(' fresh 300k', '')
            suffix = {'left_censored': ' <=', 'right_censored': ' >=', 'ambiguous': ' ?'}.get(
                point['status'], ''
            )
            axis.annotate(
                short_label + suffix,
                (point['model_parameters'], point['n50']),
                xytext=(6, 6), textcoords='offset points', fontsize=9.5,
            )
    axis.set_xscale('log')
    axis.set_yscale('log', base=2)
    axis.set_xlabel('Trainable parameters')
    axis.set_title(f'{feature} q95 N50')
    axis.grid(axis='y', alpha=0.16)
axes[0].set_ylabel(r'$N_{50}$ training images')
handles, labels = axes[0].get_legend_handles_labels()
fig.suptitle('Exploratory capacity versus transition diagnostic', fontsize=20, fontweight='semibold')
fig.text(
    0.5, 0.89,
    'Points are not connected or fit: L16 uses 300k updates while the other models use 200k.',
    ha='center', fontsize=12, color='0.3',
)
fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0.84), ncol=2, frameon=False)
fig.subplots_adjust(left=0.09, right=0.98, bottom=0.14, top=0.72, wspace=0.12)
capacity_plot_path = QUICKCHECK_DIR / 'dit_300k_capacity_n50_diagnostic.png'
fig.savefig(capacity_plot_path, bbox_inches='tight')
plt.show()
print('wrote', capacity_table_path)
print('wrote', capacity_plot_path)
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


SECTION_CODE = {
    "input-audit": (INPUT_AUDIT_CODE,),
    "transition": (TRANSITION_CODE,),
    "transition-summary": (TRANSITION_SUMMARY_CODE,),
}


def build_notebook() -> dict[str, Any]:
    cells = [
        markdown_cell(INTRO, section="intro"),
        markdown_cell(TLDR, section="tldr"),
        code_cell(SETUP, section="setup"),
    ]
    for section, source in SECTION_MARKDOWN:
        cells.append(markdown_cell(source, section=section))
        for index, code in enumerate(SECTION_CODE.get(section, ())):
            cells.append(code_cell(code, section=f"{section}-{index + 1}"))
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
