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
import re
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from IPython.display import Markdown, display
from matplotlib.lines import Line2D

def resolve_project_root(start: Path | None = None) -> Path:
    current = (start or Path.cwd()).resolve()
    for candidate in (current, *current.parents):
        if (candidate / 'scripts').is_dir() and (candidate / 'notebooks').is_dir():
            return candidate
    raise FileNotFoundError('Could not locate the diffusion_models_repo project root')


PROJECT_DIR = resolve_project_root()
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from simdiff_eval.io import (
    as_nchw,
    configured_training_reference_info,
    iter_real_reference_batches_from_config,
)
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
    evenly_spaced_indices,
    interpolate_n50,
    normalize_generalization_table,
    prepare_loss_history,
    require_exact_dataset_sweep,
    streaming_nearest_neighbors,
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


OPTIMIZATION_CODE = r"""LOSS_EXPECTED_TAGS = (
    'd2p06', 'd2p07', 'd2p08', 'd2p09', 'd2p10',
    'd2p11', 'd2p12', 'd2p13', 'd2p14', 'd2p15',
)
if LOSS_EXPECTED_TAGS != DATASET_TAGS:
    raise AssertionError('Loss-plot dataset order no longer matches the full sweep')


def checkpoint_epoch(path: Path) -> int:
    match = re.search(r'checkpoint-epoch-(\d+)', str(path))
    return int(match.group(1)) if match else -1


def loss_metric_candidates(row: pd.Series) -> list[Path]:
    roots: list[Path] = []
    for key in ('checkpoint_dir', 'expected_checkpoint'):
        value = str(row.get(key, '') or '')
        if value:
            roots.append(Path(value))

    candidates: set[Path] = set()
    for root in roots:
        if not root.exists():
            continue
        if root.is_file() and root.suffix == '.json':
            candidates.add(root)
            continue
        candidates.update(root.glob('metrics_epoch_*.json'))
        if (root / 'metrics.json').exists():
            candidates.add(root / 'metrics.json')
        candidates.update(root.glob('checkpoint-epoch-*/metrics*.json'))
        if root.name.startswith('checkpoint-epoch-'):
            candidates.update(root.glob('metrics*.json'))
            candidates.update(root.parent.glob('metrics_epoch_*.json'))
            if (root.parent / 'metrics.json').exists():
                candidates.add(root.parent / 'metrics.json')
    return sorted(candidates)


def read_latest_loss_metrics(row: pd.Series) -> tuple[dict[str, Any], Path]:
    candidates = loss_metric_candidates(row)
    if not candidates:
        raise FileNotFoundError(
            f"{row['dataset_tag']}: no training metrics found below "
            f"{row.get('checkpoint_dir', row.get('expected_checkpoint'))}"
        )

    def score(path: Path) -> tuple[int, float]:
        return checkpoint_epoch(path), path.stat().st_mtime

    latest = max(candidates, key=score)
    with latest.open() as stream:
        metrics = json.load(stream)
    return metrics, latest


fresh_loss_by_tag: dict[str, dict[str, Any]] = {}
fresh_loss_audit_rows: list[dict[str, Any]] = []
for _, row in fresh_manifest.sort_values('dataset_size').iterrows():
    dataset_tag = str(row['dataset_tag'])
    target_updates = int(row['target_total_updates'])
    if target_updates != 300_000:
        raise ValueError(f'{dataset_tag}: expected 300000 updates, found {target_updates}')
    metrics, metrics_path = read_latest_loss_metrics(row)
    history = prepare_loss_history(
        metrics,
        steps_per_epoch=int(row['steps_per_epoch']),
        target_updates=target_updates,
        restart_updates=4_000,
        minimum_fraction=0.98,
    )
    fresh_loss_by_tag[dataset_tag] = history
    fresh_loss_audit_rows.append({
        'dataset_tag': dataset_tag,
        'dataset_size': int(row['dataset_size']),
        'metrics_path': str(metrics_path),
        'target_total_updates': target_updates,
        'optimizer_updates_recorded': history['recorded_updates'],
        'epochs_completed': history['epochs_completed'],
        'steps_per_epoch': history['steps_per_epoch'],
        'tail_loss_median': history['tail_median_loss'],
        'tail_loss_q25': history['tail_q25_loss'],
        'tail_loss_q75': history['tail_q75_loss'],
        'best_loss': history['best_loss'],
    })

if tuple(fresh_loss_by_tag) != LOSS_EXPECTED_TAGS:
    raise ValueError(f'Fresh loss histories are incomplete: {tuple(fresh_loss_by_tag)}')

fresh_loss_audit = pd.DataFrame(fresh_loss_audit_rows)
fresh_loss_audit_path = QUICKCHECK_DIR / 'fresh_loss_audit.csv'
fresh_loss_audit.to_csv(fresh_loss_audit_path, index=False)
display(fresh_loss_audit)
print('wrote', fresh_loss_audit_path)

loss_colors = plt.cm.viridis(np.linspace(0.08, 0.92, len(DATASET_TAGS)))
fig, axes = plt.subplots(2, 5, figsize=(18.2, 8.0), sharex=True, sharey=True)
for axis, dataset_tag, power, color in zip(
    axes.flat, DATASET_TAGS, DATASET_POWERS, loss_colors
):
    history = fresh_loss_by_tag[dataset_tag]
    axis.plot(
        history['updates'] / 1_000,
        history['cycle_averaged_loss'],
        color=color,
        lw=2.2,
    )
    axis.set_title(rf'$N_{{2D}}=2^{{{power}}}$', fontweight='semibold')
    axis.set_yscale('log')
    axis.set_xlim(0, 305)
    axis.grid(alpha=0.16)
for axis in axes[-1, :]:
    axis.set_xlabel('Optimizer updates (thousands)')
for axis in axes[:, 0]:
    axis.set_ylabel('Cycle-averaged denoising loss')
fig.suptitle(
    'Fresh DiT-L16 optimization across all ten training sizes',
    fontsize=21,
    fontweight='semibold',
    y=0.985,
)
fig.text(
    0.5,
    0.945,
    'Every panel is an independent 300k-update run; curves are averaged over one 4k-update LR cycle.',
    ha='center',
    fontsize=12.5,
    color='0.3',
)
fig.subplots_adjust(left=0.07, right=0.99, bottom=0.09, top=0.88, hspace=0.34, wspace=0.16)
fresh_loss_plot_path = QUICKCHECK_DIR / 'dit_l16_fresh300k_loss_all_sizes.png'
fig.savefig(fresh_loss_plot_path, bbox_inches='tight')
plt.show()
print('wrote', fresh_loss_plot_path)

fig, axis = plt.subplots(figsize=(10.8, 5.6))
x = np.asarray(DATASET_POWERS)
median = fresh_loss_audit['tail_loss_median'].to_numpy(float)
q25 = fresh_loss_audit['tail_loss_q25'].to_numpy(float)
q75 = fresh_loss_audit['tail_loss_q75'].to_numpy(float)
axis.plot(x, median, color='#0072B2', marker='o', lw=2.6, ms=7, label='final 5% median')
axis.fill_between(x, q25, q75, color='#56B4E9', alpha=0.28, label='final 5% IQR')
axis.set_yscale('log')
axis.set_xticks(x, [rf'$2^{{{power}}}$' for power in x])
axis.set_xlabel(r'Training images $N_{2D}$')
axis.set_ylabel('Denoising loss')
axis.set_title('Fresh DiT-L16 tail loss after 300k optimizer updates', fontweight='semibold')
axis.grid(axis='y', alpha=0.18)
axis.legend(frameon=False)
fig.tight_layout()
fresh_tail_plot_path = QUICKCHECK_DIR / 'dit_l16_fresh300k_tail_loss_summary.png'
fig.savefig(fresh_tail_plot_path, bbox_inches='tight')
plt.show()
print('wrote', fresh_tail_plot_path)
"""


GENERATED_FIELDS_CODE = r"""DISPLAY_SAMPLE_COUNT = 4


def resolve_repo_path(value: str | Path) -> Path:
    path = Path(str(value)).expanduser()
    return path.resolve() if path.is_absolute() else (PROJECT_DIR / path).resolve()


def fresh_sample_path(row: dict[str, Any]) -> Path:
    return resolve_repo_path(
        str(row['sample_path']).format(
            seed=FRESH_TRAINING_SEED,
            sample_label=FRESH_SAMPLE_LABEL,
        )
    )


fresh_sample_bundles: dict[str, dict[str, Any]] = {}
fresh_sample_audit_rows = []
for row in sorted(fresh_manifest_rows, key=lambda item: int(item['dataset_size'])):
    tag = str(row['dataset_tag'])
    if str(row['sample_label']) != 'dpm50_fresh300k_v2':
        raise ValueError(f'{tag}: expected the controlled dpm50_fresh300k_v2 archive')
    sample_path = fresh_sample_path(row)
    config_path = resolve_repo_path(row['config'])
    expected_checkpoint = resolve_repo_path(row['expected_checkpoint'])
    if not sample_path.exists():
        raise FileNotFoundError(f'Missing fresh 300k sample archive for {tag}: {sample_path}')
    if not config_path.exists():
        raise FileNotFoundError(f'Missing frozen fresh configuration for {tag}: {config_path}')
    if not expected_checkpoint.is_dir():
        raise FileNotFoundError(f'Missing exact fresh 300k checkpoint for {tag}: {expected_checkpoint}')

    with np.load(sample_path, allow_pickle=False) as payload:
        archive = {name: np.asarray(payload[name]) for name in payload.files}
    metadata = validate_sample_archive_metadata(
        archive,
        expected_checkpoint=expected_checkpoint,
        expected_config_path=config_path,
        expected_scheduler=FRESH_SCHEDULER,
        expected_num_steps=FRESH_SAMPLER_STEPS,
        expected_seed=FRESH_TRAINING_SEED,
        expected_samples=FRESH_SAMPLE_COUNT,
    )
    generated = as_nchw(np.asarray(archive['samples'], dtype=np.float32))
    reference_info = configured_training_reference_info(config_path)
    configured_slices = int(reference_info['configured_slices'])
    if configured_slices != int(row['dataset_size']):
        raise ValueError(
            f'{tag}: configured_slices={configured_slices} does not equal '
            f'manifest dataset_size={row["dataset_size"]}'
        )

    fresh_sample_bundles[tag] = {
        'manifest': row,
        'generated': generated,
        'sample_path': sample_path,
        'config_path': config_path,
        'expected_checkpoint': expected_checkpoint,
        'reference_info': reference_info,
    }
    fresh_sample_audit_rows.append({
        'dataset_tag': tag,
        'dataset_size': int(row['dataset_size']),
        'target_updates': int(row['target_total_updates']),
        'sample_label': FRESH_SAMPLE_LABEL,
        'n_generated': metadata['n_generated'],
        'configured_slices': configured_slices,
        'real_reference': 'exact model training subset',
        'scheduler': metadata['scheduler'],
        'num_steps': metadata['num_steps'],
        'seed': metadata['seed'],
        'sample_path': str(sample_path),
        'config_path': str(config_path),
        'checkpoint': metadata['resolved_checkpoint'],
    })

if tuple(fresh_sample_bundles) != DATASET_TAGS:
    raise ValueError('Fresh sample bundle does not cover d2p06 through d2p15 in order')
fresh_sample_audit = pd.DataFrame(fresh_sample_audit_rows)
fresh_sample_audit_path = QUICKCHECK_DIR / 'fresh_sample_audit.csv'
fresh_sample_audit.to_csv(fresh_sample_audit_path, index=False)
display(fresh_sample_audit)
print('wrote', fresh_sample_audit_path)

display_indices = evenly_spaced_indices(
    total=FRESH_SAMPLE_COUNT,
    count=DISPLAY_SAMPLE_COUNT,
)
display_values = np.concatenate([
    fresh_sample_bundles[tag]['generated'][display_indices, 0].reshape(-1)
    for tag in DATASET_TAGS
])
display_vmin, display_vmax = np.quantile(display_values, [0.005, 0.995])

fig, axes = plt.subplots(
    2 * DISPLAY_SAMPLE_COUNT,
    5,
    figsize=(16.5, 22.0),
    constrained_layout=True,
)
for block, tags in enumerate((DATASET_TAGS[:5], DATASET_TAGS[5:])):
    for column, tag in enumerate(tags):
        power = int(tag[-2:])
        generated = fresh_sample_bundles[tag]['generated']
        for row_index, sample_index in enumerate(display_indices):
            axis = axes[block * DISPLAY_SAMPLE_COUNT + row_index, column]
            axis.imshow(
                generated[sample_index, 0],
                cmap='viridis',
                vmin=display_vmin,
                vmax=display_vmax,
            )
            axis.set_xticks([])
            axis.set_yticks([])
            if row_index == 0:
                axis.set_title(rf'$2^{{{power}}}$', fontweight='semibold')
        axes[block * DISPLAY_SAMPLE_COUNT, 0].set_ylabel(
            f'samples {display_indices[0]}–{display_indices[-1]}',
            fontweight='semibold',
        )
fig.suptitle(
    'Fresh DiT-L16 generated fields at 300k updates: complete data-size sweep',
    fontsize=21,
    fontweight='semibold',
)
generated_plot_path = QUICKCHECK_DIR / 'dit_l16_fresh300k_generated_all_sizes.png'
fig.savefig(generated_plot_path, bbox_inches='tight')
plt.show()
print('wrote', generated_plot_path)
"""


NEAREST_TRAINING_CODE = r"""NEAREST_QUERY_COUNT = 16
REAL_REFERENCE_RAW_BATCH_SIZE = 4

nearest_results: dict[str, dict[str, Any]] = {}
nearest_query_rows = []
for tag in DATASET_TAGS:
    bundle = fresh_sample_bundles[tag]
    query_indices = evenly_spaced_indices(
        total=len(bundle['generated']),
        count=NEAREST_QUERY_COUNT,
    )
    queries = bundle['generated'][query_indices]
    result = streaming_nearest_neighbors(
        queries,
        (
            as_nchw(np.asarray(batch, dtype=np.float32))
            for batch in iter_real_reference_batches_from_config(
                bundle['config_path'],
                raw_batch_size=REAL_REFERENCE_RAW_BATCH_SIZE,
            )
        ),
    )
    expected_training = int(bundle['reference_info']['configured_slices'])
    if result['n_training'] != expected_training:
        raise ValueError(
            f'{tag}: nearest search scanned {result["n_training"]} slices; '
            f'expected {expected_training}'
        )
    result['query_indices'] = query_indices
    result['queries'] = queries
    nearest_results[tag] = result
    for local_index, query_index in enumerate(query_indices):
        nearest_query_rows.append({
            'dataset_tag': tag,
            'dataset_size': int(bundle['manifest']['dataset_size']),
            'query_index': int(query_index),
            'nearest_training_index': int(result['nearest_index'][local_index]),
            'nearest_mse': float(result['mse'][local_index]),
            'nearest_cosine_similarity': float(result['cosine_similarity'][local_index]),
            'training_slices_scanned': int(result['n_training']),
        })

nearest_queries = pd.DataFrame(nearest_query_rows)
nearest_query_path = QUICKCHECK_DIR / 'dit_l16_fresh300k_nearest_queries.csv'
nearest_queries.to_csv(nearest_query_path, index=False)
display(nearest_queries.groupby(['dataset_tag', 'dataset_size']).agg(
    nearest_mse_median=('nearest_mse', 'median'),
    nearest_mse_max=('nearest_mse', 'max'),
    nearest_cosine_median=('nearest_cosine_similarity', 'median'),
    nearest_cosine_max=('nearest_cosine_similarity', 'max'),
).reset_index())
print('wrote', nearest_query_path)

fig, axes = plt.subplots(6, 5, figsize=(16.5, 19.0), constrained_layout=True)
for block, tags in enumerate((DATASET_TAGS[:5], DATASET_TAGS[5:])):
    for column, tag in enumerate(tags):
        power = int(tag[-2:])
        result = nearest_results[tag]
        generated = result['queries'][0, 0]
        nearest = result['nearest_images'][0, 0]
        difference = np.abs(generated - nearest)
        row_start = block * 3
        axes[row_start, column].imshow(
            generated, cmap='viridis', vmin=display_vmin, vmax=display_vmax
        )
        axes[row_start + 1, column].imshow(
            nearest, cmap='viridis', vmin=display_vmin, vmax=display_vmax
        )
        axes[row_start + 2, column].imshow(
            difference,
            cmap='magma',
            vmin=0,
            vmax=max(float(np.quantile(difference, 0.995)), 1e-8),
        )
        axes[row_start + 2, column].text(
            0.02,
            0.03,
            f'MSE={result["mse"][0]:.3g}\ncos={result["cosine_similarity"][0]:.3f}',
            transform=axes[row_start + 2, column].transAxes,
            fontsize=8.5,
            color='white',
            va='bottom',
            bbox={'facecolor': 'black', 'alpha': 0.6, 'pad': 2},
        )
        axes[row_start, column].set_title(rf'$2^{{{power}}}$', fontweight='semibold')
        for row_offset in range(3):
            axes[row_start + row_offset, column].set_xticks([])
            axes[row_start + row_offset, column].set_yticks([])
    axes[block * 3, 0].set_ylabel('generated', fontweight='semibold')
    axes[block * 3 + 1, 0].set_ylabel('nearest training', fontweight='semibold')
    axes[block * 3 + 2, 0].set_ylabel('absolute difference', fontweight='semibold')
fig.suptitle(
    'Fresh DiT-L16 generated fields and exact nearest training slices',
    fontsize=21,
    fontweight='semibold',
)
nearest_example_path = QUICKCHECK_DIR / 'dit_l16_fresh300k_nearest_examples.png'
fig.savefig(nearest_example_path, bbox_inches='tight')
plt.show()
print('wrote', nearest_example_path)

fig, axes = plt.subplots(1, 2, figsize=(14.0, 5.4))
offsets = np.linspace(-0.16, 0.16, NEAREST_QUERY_COUNT)
for power, tag in zip(DATASET_POWERS, DATASET_TAGS):
    rows = nearest_queries[nearest_queries['dataset_tag'] == tag]
    x = np.full(len(rows), power, dtype=float) + offsets
    axes[0].scatter(x, rows['nearest_cosine_similarity'], s=26, alpha=0.58, color='#0072B2')
    axes[1].scatter(x, rows['nearest_mse'], s=26, alpha=0.58, color='#D55E00')
    axes[0].plot(power, rows['nearest_cosine_similarity'].median(), 'D', ms=7, color='black')
    axes[1].plot(power, rows['nearest_mse'].median(), 'D', ms=7, color='black')
for axis in axes:
    axis.set_xticks(DATASET_POWERS, [rf'$2^{{{power}}}$' for power in DATASET_POWERS])
    axis.set_xlabel(r'Training images $N_{2D}$')
    axis.grid(axis='y', alpha=0.18)
axes[0].set_ylabel('Cosine similarity to nearest training slice')
axes[0].set_title('Nearest-training cosine similarity', fontweight='semibold')
axes[1].set_yscale('log')
axes[1].set_ylabel('MSE to nearest training slice')
axes[1].set_title('Nearest-training pixel MSE', fontweight='semibold')
fig.suptitle(
    'Complete-subset nearest-neighbor audit (16 deterministic generations per size)',
    fontsize=18,
    fontweight='semibold',
)
fig.tight_layout(rect=(0, 0, 1, 0.91))
nearest_distribution_path = QUICKCHECK_DIR / 'dit_l16_fresh300k_nearest_distribution.png'
fig.savefig(nearest_distribution_path, bbox_inches='tight')
plt.show()
print('wrote', nearest_distribution_path)
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
    "optimization": (OPTIMIZATION_CODE,),
    "generated-fields": (GENERATED_FIELDS_CODE,),
    "nearest-training": (NEAREST_TRAINING_CODE,),
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
