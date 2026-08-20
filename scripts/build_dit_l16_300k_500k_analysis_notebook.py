#!/usr/bin/env python
"""Build the standalone audited DiT-L16 300k-to-500k analysis notebook."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from textwrap import dedent
from typing import Any

try:
    from scripts.update_dit_l16_300k_500k_outlier_notebook import transform_notebook
except ModuleNotFoundError:  # Direct execution places scripts/ on sys.path.
    from update_dit_l16_300k_500k_outlier_notebook import transform_notebook


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_PATH = ROOT / "notebooks" / "nf_generalize_fig2_dit_l16_300k_500k_analysis.ipynb"


def _source(text: str) -> list[str]:
    return (dedent(text).strip("\n") + "\n").splitlines(keepends=True)


def _cell_id(kind: str, section: str, text: str) -> str:
    digest = hashlib.sha1(f"{kind}\0{section}\0{text}".encode()).hexdigest()[:12]
    return f"{kind[0]}-{digest}"


def markdown(text: str, *, section: str) -> dict[str, Any]:
    return {
        "cell_type": "markdown",
        "id": _cell_id("markdown", section, text),
        "metadata": {"analysis_section": section},
        "source": _source(text),
    }


def code(text: str, *, section: str) -> dict[str, Any]:
    return {
        "cell_type": "code",
        "execution_count": None,
        "id": _cell_id("code", section, text),
        "metadata": {"analysis_section": section},
        "outputs": [],
        "source": _source(text),
    }


INTRO = r"""
# DiT-L16 from 300k to 500k Optimizer Updates

This notebook is the standalone analysis of the audited
`nf_generalize_fig2_dit_l16_continue500k_v2` sweep. It follows ten independently
trained DiT-L16 models, one for each training-set size from $2^6$ through
$2^{15}$, at 300k, 340k, 380k, 420k, 460k, and 500k optimizer updates.

The analysis keeps three questions separate:

1. **Optimization:** does the denoising objective continue to improve?
2. **Novelty:** do generated maps remain distinct from individual training maps?
3. **Scientific validity:** do generated maps retain the one-point distribution,
   power spectrum, scale-dependent variance, and expected patch-grid behavior?

The notebook refuses to plot results unless the final artifact audit passes. It
does not use the superseded L16 run family. Historical UNet and DiT-L8/L12
results appear only in the explicitly labeled architecture-context section.
"""


SETUP = r"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.display import Markdown, display


def resolve_project_root(start: Path | None = None) -> Path:
    current = (start or Path.cwd()).resolve()
    for candidate in (current, *current.parents):
        if (candidate / 'scripts').is_dir() and (candidate / 'notebooks').is_dir():
            return candidate
    raise FileNotFoundError('Could not locate diffusion_models_repo')


PROJECT_DIR = resolve_project_root()
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from scripts.dit_300k_scaling_analysis import (
    build_historical_unet_metric_table,
    evenly_spaced_indices,
    normalize_generalization_table,
    prepare_stitched_loss_history,
    require_exact_dataset_sweep,
    robust_log_ratio_outliers,
    stage_loss_metrics_from_logs,
    streaming_nearest_neighbors,
    summarize_filtered_power_ratios,
    summarize_n50,
    validate_sampler_endpoint,
)
from simdiff_eval.io import (
    as_nchw,
    configured_training_reference_info,
    iter_real_reference_batches_from_config,
    load_real_reference_from_config,
)
from simdiff_eval.metrics import batch_power_spectra


CONT_SWEEP = 'nf_generalize_fig2_dit_l16_continue500k_v2'
CONT_UPDATES_K = (300, 340, 380, 420, 460, 500)
CONT_TAGS = (
    'd2p06', 'd2p07', 'd2p08', 'd2p09', 'd2p10',
    'd2p11', 'd2p12', 'd2p13', 'd2p14', 'd2p15',
)
CONT_POWERS = tuple(range(6, 16))
CONT_SIZES = tuple(2 ** power for power in range(6, 16))
CONT_FEATURES = ('PCA', 'SSCD')
OUTLIER_K_BIN = 60
OUTLIER_THRESHOLD = 4.5
FEATURE_TITLES = {
    'PCA': 'PCA q95 novelty',
    'SSCD': 'SSCD q95 novelty',
}

LOCAL_DIR = PROJECT_DIR / 'local' / CONT_SWEEP
LOG_DIR = PROJECT_DIR / 'logs' / CONT_SWEEP
TABLE_DIR = PROJECT_DIR / 'results' / 'nf_generalize_fig2_dit' / 'tables'
PHYSICS_DIR = PROJECT_DIR / 'results' / 'nf_generalize_fig2_dit' / 'physics'
SAMPLE_DIR = PROJECT_DIR / 'results' / CONT_SWEEP / 'samples'
OUTPUT_DIR = PROJECT_DIR / 'results' / CONT_SWEEP / 'notebook_analysis'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

COLORS = dict(zip(CONT_UPDATES_K, plt.cm.viridis(np.linspace(0.08, 0.92, 6))))

plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'font.size': 12,
    'axes.titlesize': 16,
    'axes.labelsize': 14,
    'legend.fontsize': 11,
    'figure.dpi': 120,
    'savefig.dpi': 220,
    'axes.spines.top': False,
    'axes.spines.right': False,
})


def project_path(value: str | Path) -> Path:
    path = Path(str(value))
    return path if path.is_absolute() else PROJECT_DIR / path


def dataset_label(size: int) -> str:
    return rf'$2^{{{int(np.log2(size))}}}$'


def save_figure(fig: plt.Figure, name: str) -> Path:
    path = OUTPUT_DIR / name
    fig.savefig(path, bbox_inches='tight')
    print('wrote', path)
    return path


def expected_sample_label(updates_k: int) -> str:
    return 'dpm50_source_300k' if int(updates_k) == 300 else f'dpm50_cont_{int(updates_k)}k'


def source_config_for_row(row: pd.Series) -> Path:
    '''Resolve the frozen source config that defines the model's training subset.'''
    for key in ('source_config', 'config'):
        value = row.get(key)
        if value is None or pd.isna(value):
            continue
        text = str(value).strip()
        if text:
            return project_path(text)
    raise ValueError('Manifest row has neither source_config nor config')


def load_samples(path: str | Path) -> np.ndarray:
    resolved = project_path(path)
    if not resolved.is_file():
        raise FileNotFoundError(f'Missing sample archive: {resolved}')
    with np.load(resolved, allow_pickle=False) as archive:
        if 'samples' not in archive.files:
            raise ValueError(f'Archive has no samples tensor: {resolved}')
        samples = as_nchw(np.asarray(archive['samples'], dtype=np.float32))
    if samples.shape != (512, 1, 128, 128) or not np.isfinite(samples).all():
        raise ValueError(f'Invalid sample tensor {samples.shape}: {resolved}')
    return samples


def archive_scalar(path: str | Path, key: str) -> Any:
    with np.load(project_path(path), allow_pickle=False) as archive:
        if key not in archive.files:
            raise KeyError(f'{project_path(path)} has no metadata field {key}')
        value = np.asarray(archive[key])
    if value.size != 1:
        raise ValueError(f'{key} is not scalar in {project_path(path)}')
    return value.reshape(()).item()


def exact_reference(config_path: str | Path, max_slices: int | None = None) -> np.ndarray:
    '''Load the exact configured training subset used by one model.'''
    return load_real_reference_from_config(project_path(config_path), max_slices=max_slices)
"""


AUDIT_MARKDOWN = r"""
## 1. Scope, provenance, and mandatory audit

The final sweep audit is a hard prerequisite, not a display-only status check.
It verifies final 500k weights, samples, corrected checkpoint-specific novelty
tables, physical statistics, selected-$k$, patch-boundary, and sampler-control
artifacts. Intermediate weight retention is reported separately: missing
historical weights prevent exact resumption, but do not invalidate already
audited samples and statistics. The code below also checks the manifest pairs
and confirms that each PCA/SSCD table points to its own checkpoint sample.
"""


AUDIT_CODE = r"""
audit_path = LOCAL_DIR / 'final_audit.json'
manifest_path = LOCAL_DIR / 'analysis_manifest.json'
if not audit_path.is_file() or not manifest_path.is_file():
    raise FileNotFoundError('Missing final_audit.json or analysis_manifest.json')

continuation_audit = json.loads(audit_path.read_text())
if continuation_audit.get('analysis_status') != 'PASS':
    raise RuntimeError('Mandatory continuation analysis audit did not pass:\n' + json.dumps(continuation_audit, indent=2))

audit_counts = continuation_audit.get('counts', {})
expected_counts = {
    'expected_checkpoints': 50,
    'expected_final_checkpoints': 10,
    'valid_final_checkpoints': 10,
    'expected_dpm_samples': 60,
    'expected_ddpm_controls': 4,
    'valid_sample_files': 64,
    'expected_metric_tables': 12,
    'valid_metric_tables': 12,
    'physics_summary_rows': 60,
    'selected_bin_rows': 180,
    'patch_boundary_rows': 90,
    'physics_curve_arrays': 361,
}
for key, expected in expected_counts.items():
    observed = int(audit_counts.get(key, -1))
    if observed != expected:
        raise RuntimeError(f'Audit count mismatch for {key}: {observed} != {expected}')

continuation_manifest = pd.DataFrame(json.loads(manifest_path.read_text()))
continuation_manifest['analysis_updates'] = pd.to_numeric(
    continuation_manifest['analysis_updates'], errors='raise'
).astype(int)
continuation_manifest['updates_k'] = continuation_manifest['analysis_updates'] // 1000
expected_pairs = {(tag, updates) for tag in CONT_TAGS for updates in CONT_UPDATES_K}
actual_pairs = set(zip(continuation_manifest['dataset_tag'], continuation_manifest['updates_k']))
if len(continuation_manifest) != 60 or actual_pairs != expected_pairs:
    raise RuntimeError('The analysis manifest does not contain exactly the 60 expected pairs')


def continuation_row(tag: str, updates_k: int) -> pd.Series:
    selected = continuation_manifest[
        (continuation_manifest['dataset_tag'] == tag)
        & (continuation_manifest['updates_k'] == int(updates_k))
    ]
    if len(selected) != 1:
        raise RuntimeError(f'Expected one manifest row for {tag}/{updates_k}k; found {len(selected)}')
    return selected.iloc[0]


novelty_parts = []
metric_audit_rows = []
for feature in CONT_FEATURES:
    for updates_k in CONT_UPDATES_K:
        table_path = TABLE_DIR / f'{CONT_SWEEP}_{updates_k}k_{feature.lower()}_full_nn_metrics.csv'
        if not table_path.is_file():
            raise FileNotFoundError(table_path)
        table = normalize_generalization_table(pd.read_csv(table_path), context=f'{feature} {updates_k}k')
        table = require_exact_dataset_sweep(
            table,
            arch='dit_l16',
            value_columns=('gen_gl_q95',),
            context=f'{feature} {updates_k}k corrected novelty',
        )
        label = expected_sample_label(updates_k)
        if 'sample_path' not in table.columns:
            raise ValueError(f'{table_path} has no sample_path provenance column')
        expected_paths = {
            tag: project_path(continuation_row(tag, updates_k)['sample_path']).resolve()
            for tag in CONT_TAGS
        }
        for _, metric_row in table.iterrows():
            tag = str(metric_row['dataset_tag'])
            observed_path = project_path(metric_row['sample_path']).resolve()
            if label not in str(observed_path) or observed_path != expected_paths[tag]:
                raise RuntimeError(
                    f'{feature} {updates_k}k {tag} points to {observed_path}, '
                    f'expected {expected_paths[tag]} with label {label}'
                )
        table['feature'] = feature
        table['updates_k'] = updates_k
        table['model_label'] = f'DiT-L16 {updates_k}k'
        table['arch'] = f'dit_l16_{updates_k}k'
        novelty_parts.append(table)
        metric_audit_rows.append({
            'feature': feature,
            'updates_k': updates_k,
            'rows': len(table),
            'expected_sample_label': label,
            'path': str(table_path),
        })

continuation_novelty = pd.concat(novelty_parts, ignore_index=True)
physics_path = TABLE_DIR / f'{CONT_SWEEP}_physics_summary.csv'
selected_bins_path = TABLE_DIR / f'{CONT_SWEEP}_pk_selected_bins.csv'
patch_path = TABLE_DIR / f'{CONT_SWEEP}_patch_boundaries.csv'
curves_path = PHYSICS_DIR / f'{CONT_SWEEP}_curves.npz'
continuation_physics = pd.read_csv(physics_path)
continuation_selected_bins = pd.read_csv(selected_bins_path)
continuation_patch = pd.read_csv(patch_path)
continuation_curves = np.load(curves_path)

if len(continuation_novelty) != 120:
    raise RuntimeError(f'Expected 120 novelty rows; found {len(continuation_novelty)}')
if len(continuation_physics) != audit_counts['physics_summary_rows']:
    raise RuntimeError('physics_summary_rows does not match the loaded table')
if len(continuation_selected_bins) != audit_counts['selected_bin_rows']:
    raise RuntimeError('selected_bin_rows does not match the loaded table')
if len(continuation_patch) != audit_counts['patch_boundary_rows']:
    raise RuntimeError('patch_boundary_rows does not match the loaded table')
if len(continuation_curves.files) != audit_counts['physics_curve_arrays']:
    raise RuntimeError('physics_curve_arrays does not match the loaded archive')

corrected_physics_columns = (
    'k_max', 'hist_bins', 'hist_min', 'hist_max',
    'real_pixel_coverage', 'generated_pixel_coverage',
    'hist_l1', 'hist_l1_lo', 'hist_l1_hi',
    'pk_log10_mae', 'pk_log10_mae_lo', 'pk_log10_mae_hi',
    'real_vs_real_hist_l1', 'real_vs_real_pk_log10_mae',
    'bootstrap_resamples', 'bootstrap_seed',
)
missing_corrected_columns = sorted(set(corrected_physics_columns) - set(continuation_physics.columns))
if missing_corrected_columns:
    raise RuntimeError(
        'Physics summary predates the corrected metric definitions; missing columns: '
        + ', '.join(missing_corrected_columns)
    )

corrected_numeric = continuation_physics.loc[:, corrected_physics_columns].apply(
    pd.to_numeric, errors='raise'
)
if not np.isfinite(corrected_numeric.to_numpy(dtype=float)).all():
    raise RuntimeError('Corrected physics columns contain non-finite values')
if not (
    (corrected_numeric['hist_l1_lo'] <= corrected_numeric['hist_l1'])
    & (corrected_numeric['hist_l1'] <= corrected_numeric['hist_l1_hi'])
    & (corrected_numeric['pk_log10_mae_lo'] <= corrected_numeric['pk_log10_mae'])
    & (corrected_numeric['pk_log10_mae'] <= corrected_numeric['pk_log10_mae_hi'])
).all():
    raise RuntimeError('A corrected physics point estimate lies outside its bootstrap interval')
for coverage_column in ('real_pixel_coverage', 'generated_pixel_coverage'):
    if not corrected_numeric[coverage_column].between(0.0, 1.0).all():
        raise RuntimeError(f'{coverage_column} must lie in [0, 1]')
if not (
    (corrected_numeric['k_max'] > 0).all()
    and (corrected_numeric['hist_bins'] > 0).all()
    and (corrected_numeric['bootstrap_resamples'] > 0).all()
    and (corrected_numeric['real_vs_real_hist_l1'] >= 0).all()
    and (corrected_numeric['real_vs_real_pk_log10_mae'] >= 0).all()
):
    raise RuntimeError('Corrected metric metadata or real-vs-real floors are invalid')

corrected_metric_audit = pd.DataFrame([{
    'rows': len(continuation_physics),
    'k_max': ', '.join(map(str, sorted(corrected_numeric['k_max'].astype(int).unique()))),
    'hist_bins': ', '.join(map(str, sorted(corrected_numeric['hist_bins'].astype(int).unique()))),
    'hist_range': (
        f"[{corrected_numeric['hist_min'].min():.3g}, "
        f"{corrected_numeric['hist_max'].max():.3g}]"
    ),
    'min_real_pixel_coverage': float(corrected_numeric['real_pixel_coverage'].min()),
    'min_generated_pixel_coverage': float(corrected_numeric['generated_pixel_coverage'].min()),
    'bootstrap_resamples': ', '.join(map(
        str, sorted(corrected_numeric['bootstrap_resamples'].astype(int).unique())
    )),
    'bootstrap_intervals_ordered': True,
    'real_vs_real_floors_present': True,
}])

reference_audit = []
for tag in CONT_TAGS:
    row = continuation_row(tag, 300)
    info = configured_training_reference_info(source_config_for_row(row))
    reference_audit.append({'dataset_tag': tag, 'dataset_size': int(row['dataset_size']), **info})

display(Markdown(f"### Final analysis audit: **{continuation_audit['status']}**"))
if continuation_audit.get('checkpoint_retention_status') != 'PASS':
    display(Markdown(
        '**Checkpoint-retention warning:** intermediate 340k--460k weight '
        'directories are incomplete. The audited samples, novelty tables, and '
        'physics products remain usable, but those exact intermediate models '
        'cannot be resumed or sampled again without retraining.'
    ))
display(Markdown('### Corrected physical-statistics definition audit'))
display(corrected_metric_audit)
display(pd.DataFrame(metric_audit_rows))
display(pd.DataFrame(reference_audit))
"""


LOSS_MARKDOWN = r"""
## 2. Optimization history

Each panel shows the cycle-averaged denoising loss for one training-set size.
The history is reconstructed from the exact epoch interval in the completed
Slurm logs for each 40k continuation stage; overwritten run-level metric files
are not used. Falling loss shows that the optimizer is still changing the
denoising objective; it does not by itself establish novelty or physical
agreement.
"""


LOSS_CODE = r"""
def checkpoint_epoch(path: Path) -> int:
    match = re.fullmatch(r'checkpoint-epoch-(\d+)', path.name)
    if match is None:
        raise ValueError(f'Cannot read checkpoint epoch from {path}')
    return int(match.group(1))


def read_stage_loss_metrics(
    row: pd.Series,
) -> tuple[dict[str, Any], tuple[Path, ...], int, int]:
    previous_epoch = checkpoint_epoch(project_path(row['previous_expected_checkpoint']))
    final_epoch = int(row['expected_final_epoch'])
    task_index = CONT_TAGS.index(str(row['dataset_tag']))
    candidates = sorted(LOG_DIR.glob(f'train_stage*_{task_index}.out'))
    if not candidates:
        raise FileNotFoundError(
            f"No continuation training logs found for {row['dataset_tag']} "
            f"under {LOG_DIR}"
        )
    metrics, used_paths = stage_loss_metrics_from_logs(
        candidates,
        first_epoch=previous_epoch + 1,
        final_epoch=final_epoch,
    )
    return metrics, used_paths, previous_epoch, final_epoch


loss_histories = {}
loss_audit_rows = []
for tag in CONT_TAGS:
    final_row = continuation_row(tag, 500)
    steps_per_epoch = int(final_row['steps_per_epoch'])
    segments = []
    metrics_paths = []
    for updates_k in CONT_UPDATES_K[1:]:
        row = continuation_row(tag, updates_k)
        metrics, used_paths, previous_epoch, final_epoch = read_stage_loss_metrics(row)
        stage_start_updates = (previous_epoch + 1) * steps_per_epoch
        stage_end_updates = (final_epoch + 1) * steps_per_epoch
        segments.append((metrics, stage_start_updates, stage_end_updates))
        metrics_paths.extend(str(path) for path in used_paths)
    history = prepare_stitched_loss_history(
        segments,
        steps_per_epoch=steps_per_epoch,
        restart_updates=4_000,
        minimum_fraction=0.98,
    )
    loss_histories[tag] = history
    loss_audit_rows.append({
        'dataset_tag': tag,
        'dataset_size': int(final_row['dataset_size']),
        'start_update': history['start_updates'],
        'final_update': history['recorded_updates'],
        'stage_recorded_updates': history['stage_recorded_updates'].tolist(),
        'tail_median_loss': history['tail_median_loss'],
        'training_log_paths': list(dict.fromkeys(metrics_paths)),
    })

fig, axes = plt.subplots(2, 5, figsize=(19, 8.4), sharex=True, constrained_layout=True)
for axis, tag in zip(axes.flat, CONT_TAGS):
    history = loss_histories[tag]
    x = np.asarray(history['updates']) / 1000
    y = np.asarray(history['cycle_averaged_loss'])
    keep = (x >= 295) & (x <= 505)
    axis.plot(x[keep], y[keep], color='#8E2A68', lw=2.2)
    for updates_k in CONT_UPDATES_K:
        axis.axvline(updates_k, color='0.82', lw=0.8, zorder=0)
    axis.set_yscale('log')
    axis.set_title(dataset_label(int(continuation_row(tag, 500)['dataset_size'])))
    axis.set_xlabel('Optimizer updates (thousands)')
    axis.grid(alpha=0.18)
for axis in axes[:, 0]:
    axis.set_ylabel('Cycle-averaged denoising loss')
fig.suptitle('DiT-L16 optimization from 300k to 500k', fontsize=21, fontweight='semibold')
save_figure(fig, 'loss_trajectory.png')
plt.show()
display(pd.DataFrame(loss_audit_rows))
"""


MAPS_MARKDOWN = r"""
## 3. Generated maps across the full sweep

Each row is a checkpoint and each column is a training-set size. All panels use
sample index 0 from the audited 512-sample archive and the same display range.
Two figures are used so all ten data sizes remain large enough to inspect.
These are visual diagnostics, not substitutes for the quantitative tests below.
"""


MAPS_CODE = r"""
def plot_generated_map_block(checkpoints: tuple[int, ...], filename: str) -> None:
    fig, axes = plt.subplots(len(checkpoints), 10, figsize=(24, 2.45 * len(checkpoints)), constrained_layout=True)
    axes = np.atleast_2d(axes)
    for row_index, updates_k in enumerate(checkpoints):
        for column_index, tag in enumerate(CONT_TAGS):
            row = continuation_row(tag, updates_k)
            image = load_samples(row['sample_path'])[0, 0]
            axes[row_index, column_index].imshow(image, cmap='viridis', vmin=-1, vmax=1)
            axes[row_index, column_index].set_xticks([])
            axes[row_index, column_index].set_yticks([])
            if row_index == 0:
                axes[row_index, column_index].set_title(dataset_label(int(row['dataset_size'])))
            if column_index == 0:
                axes[row_index, column_index].set_ylabel(f'{updates_k}k', fontweight='bold')
    fig.suptitle('Fresh DiT-L16 generated maps: sample 0, seed 123', fontsize=22, fontweight='semibold')
    save_figure(fig, filename)
    plt.show()


plot_generated_map_block((300, 340, 380), 'generated_maps_300k_380k.png')
plot_generated_map_block((420, 460, 500), 'generated_maps_420k_500k.png')
"""


NOVELTY_MARKDOWN = r"""
## 4. PCA and SSCD generalization trajectories

The q95 score is the fraction of generated samples below the train-versus-train
copy-similarity threshold. A value near one means the samples are unlike
individual training maps in that representation. It does not imply that they
are in distribution or physically accurate.
"""


NOVELTY_CODE = r"""
fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharey=True, constrained_layout=True)
for axis, feature in zip(axes, CONT_FEATURES):
    feature_rows = continuation_novelty[continuation_novelty['feature'] == feature]
    for updates_k in CONT_UPDATES_K:
        current = feature_rows[feature_rows['updates_k'] == updates_k].sort_values('dataset_size')
        axis.plot(
            np.log2(current['dataset_size']), current['gen_gl_q95'],
            color=COLORS[updates_k], marker='o', lw=2.1, ms=5.5, label=f'{updates_k}k',
        )
    axis.axhline(0.5, color='0.4', ls=':', lw=1.2)
    axis.set_title(FEATURE_TITLES[feature], fontweight='semibold')
    axis.set_xlabel(r'Training images $N_{2D}$')
    axis.set_xticks(CONT_POWERS, [rf'$2^{{{power}}}$' for power in CONT_POWERS])
    axis.set_ylim(-0.03, 1.04)
    axis.grid(alpha=0.18)
axes[0].set_ylabel('q95 novelty score')
axes[1].legend(title='Optimizer updates', ncol=2, frameon=False, loc='lower right')
fig.suptitle('DiT-L16 novelty across training time', fontsize=21, fontweight='semibold')
save_figure(fig, 'generalization_trajectories.png')
plt.show()
"""


HEATMAP_MARKDOWN = r"""
## 5. Generalization phase diagrams

These phase diagrams show the same novelty scores without connecting data sizes
or checkpoints into a fitted law. Horizontal structure indicates dependence on
data size; vertical structure indicates dependence on optimizer updates.
"""


HEATMAP_CODE = r"""
fig, axes = plt.subplots(1, 2, figsize=(16, 5.8), constrained_layout=True)
for axis, feature in zip(axes, CONT_FEATURES):
    current = continuation_novelty[continuation_novelty['feature'] == feature]
    matrix = current.pivot(index='updates_k', columns='dataset_size', values='gen_gl_q95').reindex(
        index=CONT_UPDATES_K, columns=CONT_SIZES
    )
    image = axis.imshow(matrix.to_numpy(dtype=float), origin='lower', aspect='auto', vmin=0, vmax=1, cmap='viridis')
    axis.set_title(FEATURE_TITLES[feature], fontweight='semibold')
    axis.set_xlabel(r'Training images $N_{2D}$')
    axis.set_ylabel('Optimizer updates')
    axis.set_xticks(range(10), [rf'$2^{{{power}}}$' for power in CONT_POWERS])
    axis.set_yticks(range(6), [f'{updates}k' for updates in CONT_UPDATES_K])
    plt.colorbar(image, ax=axis, shrink=0.86, label='q95 novelty')
fig.suptitle('DiT-L16 generalization phase diagrams', fontsize=21, fontweight='semibold')
save_figure(fig, 'generalization_heatmaps.png')
plt.show()
"""


N50_MARKDOWN = r"""
## 6. Transition-location summary

The q95 crossing of 0.5 is summarized only when the curve has one valid crossing.
Left-censored, right-censored, and nonmonotonic ambiguous curves remain labeled
as such. This prevents an anomalous low-data point from being converted into a
spurious transition or scaling law.
"""


N50_CODE = r"""
n50_summary = summarize_n50(continuation_novelty)
display(n50_summary)

fig, axes = plt.subplots(1, 2, figsize=(14, 5.4), sharey=True, constrained_layout=True)
for axis, feature in zip(axes, CONT_FEATURES):
    current = n50_summary[n50_summary['feature'] == feature].sort_values('updates_k')
    crossing = current[current['status'] == 'crossing']
    axis.plot(crossing['updates_k'], crossing['log2_n50'], color='#8E2A68', marker='o', lw=2)
    for _, row in current.iterrows():
        if row['status'] != 'crossing':
            axis.annotate(row['status'], (row['updates_k'], 6.15), rotation=45, fontsize=9, color='0.35')
    axis.set_title(feature, fontweight='semibold')
    axis.set_xlabel('Optimizer updates (thousands)')
    axis.set_ylim(5.8, 15.3)
    axis.set_yticks(CONT_POWERS, [rf'$2^{{{power}}}$' for power in CONT_POWERS])
    axis.grid(alpha=0.18)
axes[0].set_ylabel(r'Estimated $N_{50}$')
fig.suptitle('q95 transition summary with censoring retained', fontsize=20, fontweight='semibold')
save_figure(fig, 'transition_location_summary.png')
plt.show()
"""


CONTEXT_MARKDOWN = r"""
## 7. Architecture context

**Historical context only.** UNet, DiT-L8, and DiT-L12/base references use their
existing 200k analyses. The audited DiT-L16 curve is shown at 300k and 500k.
Because the optimizer budgets differ, this panel is descriptive context and is
not a controlled capacity-scaling fit. The superseded L16 family is excluded.
"""


CONTEXT_CODE = r"""
unet_paths = {
    feature: TABLE_DIR.parent.parent / 'nf_generalize_fig2' / 'tables' / f'nf_generalize_fig2_{feature.lower()}_full_nn_metrics.csv'
    for feature in CONT_FEATURES
}
historical_dit_paths = {
    feature: TABLE_DIR / f'nf_generalize_fig2_dit_{feature.lower()}_full_nn_metrics.csv'
    for feature in CONT_FEATURES
}

fig, axes = plt.subplots(1, 2, figsize=(17, 6.2), sharey=True, constrained_layout=True)
for axis, feature in zip(axes, CONT_FEATURES):
    unet = build_historical_unet_metric_table(pd.read_csv(unet_paths[feature]), feature=feature)
    historical_dit = normalize_generalization_table(
        pd.read_csv(historical_dit_paths[feature]), context=f'historical {feature} DiT context'
    )
    for arch, label, color, marker in (
        ('u64', 'UNet-64', '#C2C2C2', '^'),
        ('u128', 'UNet-128', '#8D8D8D', 'o'),
        ('u256', 'UNet-256', '#575757', 's'),
    ):
        current = unet[unet['arch'] == arch].sort_values('dataset_size')
        axis.plot(np.log2(current['dataset_size']), current['gen_gl_q95'], color=color, marker=marker, ls='--', lw=1.7, label=label)
    for arch, label, color, marker in (
        ('dit_l8', 'DiT-L8, 200k', '#009E73', 'P'),
        ('dit_base', 'DiT-L12/base, 200k', '#0072B2', 'D'),
    ):
        current = require_exact_dataset_sweep(
            historical_dit, arch=arch, value_columns=('gen_gl_q95',), context=f'{feature} {label}'
        )
        axis.plot(np.log2(current['dataset_size']), current['gen_gl_q95'], color=color, marker=marker, lw=2.3, label=label)
    for updates_k, label, linestyle in ((300, 'DiT-L16, 300k', '--'), (500, 'DiT-L16, 500k', '-')):
        current = continuation_novelty[
            (continuation_novelty['feature'] == feature) & (continuation_novelty['updates_k'] == updates_k)
        ].sort_values('dataset_size')
        axis.plot(np.log2(current['dataset_size']), current['gen_gl_q95'], color='#B33C86', marker='X', ls=linestyle, lw=2.8, label=label)
    axis.axhline(0.5, color='0.4', ls=':', lw=1.1)
    axis.set_title(feature, fontweight='semibold')
    axis.set_xlabel(r'Training images $N_{2D}$')
    axis.set_xticks(CONT_POWERS, [rf'$2^{{{power}}}$' for power in CONT_POWERS])
    axis.set_ylim(-0.03, 1.04)
    axis.grid(alpha=0.16)
axes[0].set_ylabel('q95 novelty score')
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0.98), ncol=4, frameon=False)
fig.suptitle('Historical architecture context only: unequal optimizer budgets', y=1.08, fontsize=21, fontweight='semibold')
save_figure(fig, 'architecture_context.png')
plt.show()
"""


ONE_POINT_MARKDOWN = r"""
## 8. One-point distributions

The heatmap covers every data size and checkpoint. The detailed figures compare
the generated pixel distribution with the **exact configured training subset**
for that model at 300k and 500k. Shared histogram edges are stored in the audited
curve archive, so black and generated curves use identical bins.
"""


ONE_POINT_CODE = r"""
def metric_heatmap(frame: pd.DataFrame, column: str, title: str, filename: str, label: str) -> None:
    matrix = frame.pivot(index='updates_k', columns='dataset_size', values=column).reindex(
        index=CONT_UPDATES_K, columns=CONT_SIZES
    )
    fig, axis = plt.subplots(figsize=(12.5, 5.2), constrained_layout=True)
    image = axis.imshow(matrix.to_numpy(dtype=float), origin='lower', aspect='auto', cmap='magma')
    axis.set_title(title, fontweight='semibold')
    axis.set_xlabel(r'Training images $N_{2D}$')
    axis.set_ylabel('Optimizer updates')
    axis.set_xticks(range(10), [rf'$2^{{{power}}}$' for power in CONT_POWERS])
    axis.set_yticks(range(6), [f'{updates}k' for updates in CONT_UPDATES_K])
    plt.colorbar(image, ax=axis, shrink=0.87, label=label)
    save_figure(fig, filename)
    plt.show()


metric_heatmap(continuation_physics, 'hist_l1', 'One-point distribution error', 'one_point_error_heatmap.png', r'$L_1$ error')

hist_edges = np.asarray(continuation_curves['histogram_edges'])
hist_centers = 0.5 * (hist_edges[:-1] + hist_edges[1:])


def plot_one_point_checkpoint(updates_k: int) -> None:
    fig, axes = plt.subplots(2, 5, figsize=(19, 8), constrained_layout=True)
    for axis, tag in zip(axes.flat, CONT_TAGS):
        key = f'{tag}_{updates_k}k'
        real = continuation_curves[f'{key}_real_hist_probability']
        generated = continuation_curves[f'{key}_generated_hist_probability']
        axis.plot(hist_centers, real, color='black', lw=2.1, label='exact configured training subset')
        axis.plot(hist_centers, generated, color='#B33C86', lw=2.1, label='generated')
        axis.set_yscale('log')
        axis.set_title(dataset_label(int(continuation_row(tag, updates_k)['dataset_size'])))
        axis.set_xlabel('Normalized field value')
        axis.grid(alpha=0.16)
    for axis in axes[:, 0]:
        axis.set_ylabel('Pixel probability')
    axes[0, 0].legend(frameon=False, fontsize=9)
    fig.suptitle(f'DiT-L16 one-point distributions at {updates_k}k updates', fontsize=21, fontweight='semibold')
    save_figure(fig, f'one_point_distributions_{updates_k}k.png')
    plt.show()


plot_one_point_checkpoint(300)
plot_one_point_checkpoint(500)
"""


PK_MARKDOWN = r"""
## 9. Power-spectrum fidelity

The summary metric is the mean absolute log-ratio across radial bins. Detailed
ratio panels show where the disagreement occurs in scale. One is exact
agreement. Every panel within a checkpoint uses a shared vertical scale so that
large errors cannot be hidden by panel-specific zooming.
"""


PK_CODE = r"""
metric_heatmap(
    continuation_physics,
    'pk_log10_mae',
    'Power-spectrum error',
    'power_spectrum_error_heatmap.png',
    r'mean $|\log_{10}(P_g/P_r)|$',
)


def plot_power_checkpoint(updates_k: int) -> None:
    ratios = [np.asarray(continuation_curves[f'{tag}_{updates_k}k_pk_ratio']) for tag in CONT_TAGS]
    finite_values = np.concatenate([ratio[np.isfinite(ratio)] for ratio in ratios])
    ymax = max(2.0, min(8.0, float(np.quantile(finite_values, 0.995) * 1.08)))
    fig, axes = plt.subplots(2, 5, figsize=(19, 8), sharey=True, constrained_layout=True)
    for axis, tag, ratio in zip(axes.flat, CONT_TAGS, ratios):
        key = f'{tag}_{updates_k}k'
        kbins = continuation_curves[f'{key}_kbins']
        axis.plot(kbins, ratio, color='#B33C86', marker='o', ms=3.2, lw=1.9)
        axis.axhline(1, color='black', ls='--', lw=1.1)
        axis.set_title(dataset_label(int(continuation_row(tag, updates_k)['dataset_size'])))
        axis.set_xlabel(r'$k$ bin')
        axis.set_ylim(0, ymax)
        axis.grid(alpha=0.16)
    for axis in axes[:, 0]:
        axis.set_ylabel(r'$P_{generated}(k)/P_{real}(k)$')
    fig.suptitle(f'DiT-L16 power-spectrum ratios at {updates_k}k updates', fontsize=21, fontweight='semibold')
    save_figure(fig, f'power_spectrum_ratios_{updates_k}k.png')
    plt.show()


plot_power_checkpoint(300)
plot_power_checkpoint(500)
"""


UNCERTAINTY_MARKDOWN = r"""
## 10. Power-spectrum uncertainty

This section implements the requested scale-resolved check at **k-bin 20, 40, and 60**.
The top row reports the mean generated-to-real ratio with deterministic
bootstrap intervals. The bottom row reports variance across the 512 inference
samples. A correct mean can coexist with excessive sample-to-sample variance.
"""


UNCERTAINTY_CODE = r"""
fig, axes = plt.subplots(2, 3, figsize=(18, 9.5), constrained_layout=True)
for column, k_bin in enumerate((20, 40, 60)):
    selected = continuation_selected_bins[continuation_selected_bins['k_bin'] == k_bin]
    for updates_k in CONT_UPDATES_K:
        current = selected[selected['updates_k'] == updates_k].sort_values('dataset_size')
        x = np.log2(current['dataset_size'])
        mean = current['ratio_mean'].to_numpy(dtype=float)
        low = current['ratio_mean_ci_low'].to_numpy(dtype=float)
        high = current['ratio_mean_ci_high'].to_numpy(dtype=float)
        axes[0, column].plot(x, mean, color=COLORS[updates_k], marker='o', ms=4, lw=1.8, label=f'{updates_k}k')
        axes[0, column].fill_between(x, low, high, color=COLORS[updates_k], alpha=0.12)
        axes[1, column].plot(x, current['ratio_variance'], color=COLORS[updates_k], marker='o', ms=4, lw=1.8)
    axes[0, column].axhline(1, color='black', ls='--', lw=1.1)
    axes[0, column].set_title(f'k-bin {k_bin}', fontweight='semibold')
    axes[1, column].set_xlabel(r'Training images $N_{2D}$')
    for axis in axes[:, column]:
        axis.set_xticks(CONT_POWERS, [rf'$2^{{{power}}}$' for power in CONT_POWERS])
        axis.grid(alpha=0.16)
axes[0, 0].set_ylabel(r'Mean $P_g/P_r$')
axes[1, 0].set_ylabel(r'Variance of $P_g/P_r$')
axes[0, 2].legend(title='Updates', ncol=2, frameon=False)
fig.suptitle('Scale-resolved power-spectrum mean and inference variance', fontsize=21, fontweight='semibold')
save_figure(fig, 'power_spectrum_selected_bins.png')
plt.show()
"""


OUTLIER_MARKDOWN = r"""
## 11. Conservative k=60 outlier sensitivity analysis

The unusually large low-data variance could reflect either a few catastrophic
samples or a broadly unstable generated distribution. We distinguish these
possibilities using every one of the 512 generated samples in every
dataset-size/checkpoint group.

For each group, the selection variable is
$\log_{10}[P_g(k=60)/P_r(k=60)]$. A sample is flagged only when its absolute
deviation from the group median exceeds **4.5 robust standard deviations**, where
one robust standard deviation is $1.4826\,\mathrm{MAD}$. The rule is two-sided
and is evaluated independently within each group. If the MAD is zero, no sample
is removed. This is deliberately stricter than the common 3-MAD convention.

The original all-sample result remains the primary result. The outlier-excluded
curves are a sensitivity analysis only. Every excluded sample is identified,
plotted with its complete spectrum, and retained in the audit tables. Filtered
plots always state `n_kept / 512`.
"""


OUTLIER_CODE = r"""
outlier_sample_parts = []
outlier_group_rows = []
filtered_selected_rows = []
outlier_analysis = {}

for updates_k in CONT_UPDATES_K:
    for tag in CONT_TAGS:
        row = continuation_row(tag, updates_k)
        sample_path = project_path(row['sample_path'])
        samples = load_samples(sample_path)
        key = f'{tag}_{updates_k}k'
        real_pk = np.asarray(continuation_curves[f'{key}_real_pk_mean'], dtype=float)
        expected_kbins = np.asarray(continuation_curves[f'{key}_kbins'], dtype=float)
        generated_pk, kbins = batch_power_spectra(samples, nbins=len(real_pk))
        if not np.allclose(kbins, expected_kbins, equal_nan=True):
            raise RuntimeError(f'k-bin mismatch for {tag}/{updates_k}k')
        pk_ratio = generated_pk / np.clip(real_pk[None, :], 1e-30, None)
        selection = robust_log_ratio_outliers(
            pk_ratio[:, OUTLIER_K_BIN], threshold=OUTLIER_THRESHOLD
        )
        mask = np.asarray(selection['outlier_mask'], dtype=bool)
        score = np.asarray(selection['robust_score'], dtype=float)
        log_ratio = np.asarray(selection['log_ratio'], dtype=float)
        median_log = float(selection['median_log_ratio'])
        scaled_mad = float(selection['scaled_mad'])
        lower_log = median_log - OUTLIER_THRESHOLD * scaled_mad
        upper_log = median_log + OUTLIER_THRESHOLD * scaled_mad

        outlier_sample_parts.append(pd.DataFrame({
            'dataset_tag': tag,
            'dataset_size': int(row['dataset_size']),
            'updates_k': updates_k,
            'sample_index': np.arange(len(samples), dtype=int),
            'k_bin': OUTLIER_K_BIN,
            'k60_ratio': pk_ratio[:, OUTLIER_K_BIN],
            'log10_k60_ratio': log_ratio,
            'robust_score': score,
            'flagged_outlier': mask,
            'sample_path': str(sample_path),
        }))
        outlier_group_rows.append({
            'dataset_tag': tag,
            'dataset_size': int(row['dataset_size']),
            'updates_k': updates_k,
            'k_bin': OUTLIER_K_BIN,
            'threshold_robust_sd': OUTLIER_THRESHOLD,
            'median_log10_ratio': median_log,
            'mad_log10_ratio': float(selection['mad_log_ratio']),
            'scaled_mad_log10_ratio': scaled_mad,
            'lower_ratio_threshold': float(10 ** lower_log),
            'upper_ratio_threshold': float(10 ** upper_log),
            'n_total': int(selection['n_total']),
            'n_flagged': int(selection['n_flagged']),
            'n_kept': int(selection['n_total'] - selection['n_flagged']),
        })
        for summary in summarize_filtered_power_ratios(
            pk_ratio, outlier_mask=mask, bin_indices=(20, 40, 60)
        ):
            filtered_selected_rows.append({
                'dataset_tag': tag,
                'dataset_size': int(row['dataset_size']),
                'updates_k': updates_k,
                **summary,
            })
        outlier_analysis[(tag, updates_k)] = {
            'kbins': kbins,
            'pk_ratio': pk_ratio,
            'outlier_mask': mask,
            'sample_path': sample_path,
        }

outlier_samples = pd.concat(outlier_sample_parts, ignore_index=True)
outlier_groups = pd.DataFrame(outlier_group_rows).sort_values(['updates_k', 'dataset_size'])
filtered_selected_bins = pd.DataFrame(filtered_selected_rows).sort_values(
    ['k_bin', 'updates_k', 'dataset_size']
)

sample_audit_path = OUTPUT_DIR / 'k60_outlier_sample_audit.csv'
group_audit_path = OUTPUT_DIR / 'k60_outlier_group_audit.csv'
filtered_bins_path = OUTPUT_DIR / 'k60_outlier_filtered_selected_bins.csv'
outlier_samples.to_csv(sample_audit_path, index=False)
outlier_groups.to_csv(group_audit_path, index=False)
filtered_selected_bins.to_csv(filtered_bins_path, index=False)
print('wrote', sample_audit_path)
print('wrote', group_audit_path)
print('wrote', filtered_bins_path)
display(outlier_groups)

# Distribution of the actual selection variable. Flagged points remain visible.
fig, axes = plt.subplots(2, 3, figsize=(18, 10), sharex=True, constrained_layout=True)
for axis, updates_k in zip(axes.flat, CONT_UPDATES_K):
    current = outlier_samples[outlier_samples['updates_k'] == updates_k]
    for power, tag in zip(CONT_POWERS, CONT_TAGS):
        group = current[current['dataset_tag'] == tag]
        retained = group[~group['flagged_outlier']]
        flagged = group[group['flagged_outlier']]
        jitter = np.random.default_rng(10_000 + updates_k + power).uniform(-0.12, 0.12, len(retained))
        axis.scatter(
            power + jitter, retained['k60_ratio'], s=8, color='0.45', alpha=0.22,
            rasterized=True,
        )
        if len(flagged):
            axis.scatter(
                np.full(len(flagged), power), flagged['k60_ratio'], s=34,
                color='#D55E00', edgecolor='black', linewidth=0.4, zorder=5,
            )
    axis.axhline(1, color='black', ls='--', lw=1)
    axis.set_yscale('log')
    axis.set_title(f'{updates_k}k updates', fontweight='semibold')
    axis.set_xticks(CONT_POWERS, [rf'$2^{{{power}}}$' for power in CONT_POWERS])
    axis.set_xlabel(r'Training images $N_{2D}$')
    axis.grid(alpha=0.14)
for axis in axes[:, 0]:
    axis.set_ylabel(r'Per-sample $P_g(k=60)/P_r(k=60)$')
fig.suptitle('k=60 sample distribution; orange points are flagged by the 4.5-MAD rule', fontsize=20, fontweight='semibold')
save_figure(fig, 'k60_outlier_distributions.png')
plt.show()

# Plot every flagged sample. Pages are bounded so no image is silently omitted.
flagged_rows = outlier_samples[outlier_samples['flagged_outlier']].sort_values(
    ['updates_k', 'dataset_size', 'robust_score'], ascending=[True, True, False]
).reset_index(drop=True)
if flagged_rows.empty:
    display(Markdown('No samples passed the conservative 4.5-MAD outlier rule.'))
else:
    rows_per_page = 6
    for page_start in range(0, len(flagged_rows), rows_per_page):
        page = flagged_rows.iloc[page_start:page_start + rows_per_page]
        page_sample_cache = {}
        fig, axes = plt.subplots(len(page), 2, figsize=(12, 4.1 * len(page)), squeeze=False, constrained_layout=True)
        for row_index, (_, flagged) in enumerate(page.iterrows()):
            tag = str(flagged['dataset_tag'])
            updates_k = int(flagged['updates_k'])
            sample_index = int(flagged['sample_index'])
            analysis = outlier_analysis[(tag, updates_k)]
            sample_path = str(analysis['sample_path'])
            if sample_path not in page_sample_cache:
                page_sample_cache[sample_path] = load_samples(sample_path)
            image = page_sample_cache[sample_path][sample_index, 0]
            ratio = analysis['pk_ratio'][sample_index]
            axes[row_index, 0].imshow(image, cmap='viridis', vmin=-1, vmax=1)
            axes[row_index, 0].set_title(
                f'{tag}, {updates_k}k, sample {sample_index}; '
                f'k60={float(flagged["k60_ratio"]):.3g}, score={float(flagged["robust_score"]):.1f}',
                fontsize=11,
            )
            axes[row_index, 0].set_xticks([])
            axes[row_index, 0].set_yticks([])
            axes[row_index, 1].plot(analysis['kbins'], ratio, color='#D55E00', lw=2)
            axes[row_index, 1].axhline(1, color='black', ls='--', lw=1)
            axes[row_index, 1].axvline(analysis['kbins'][OUTLIER_K_BIN], color='#0072B2', ls=':', lw=1.2)
            axes[row_index, 1].set_xlabel(r'$k$ bin')
            axes[row_index, 1].set_ylabel(r'$P_g/P_r$')
            axes[row_index, 1].grid(alpha=0.16)
        page_number = page_start // rows_per_page + 1
        fig.suptitle(f'Flagged k=60 samples: page {page_number}', fontsize=19, fontweight='semibold')
        save_figure(fig, f'k60_flagged_sample_gallery_page_{page_number:03d}.png')
        plt.show()

# Original, robust-center, and outlier-excluded summaries at the selected bins.
fig, axes = plt.subplots(2, 3, figsize=(18, 10), constrained_layout=True)
for column, k_bin in enumerate((20, 40, 60)):
    selected = filtered_selected_bins[filtered_selected_bins['k_bin'] == k_bin]
    for updates_k in CONT_UPDATES_K:
        current = selected[selected['updates_k'] == updates_k].sort_values('dataset_size')
        x = np.log2(current['dataset_size'])
        color = COLORS[updates_k]
        axes[0, column].plot(x, current['original_mean'], color=color, alpha=0.25, lw=1.3)
        axes[0, column].plot(x, current['original_median'], color=color, ls=':', lw=1.4)
        axes[0, column].plot(x, current['filtered_mean'], color=color, marker='o', ms=3.8, lw=2, label=f'{updates_k}k')
        axes[1, column].plot(x, current['original_variance'], color=color, alpha=0.25, lw=1.3)
        axes[1, column].plot(x, current['filtered_variance'], color=color, marker='o', ms=3.8, lw=2)
    axes[0, column].axhline(1, color='black', ls='--', lw=1)
    axes[0, column].set_title(f'k-bin {k_bin}', fontweight='semibold')
    axes[1, column].set_xlabel(r'Training images $N_{2D}$')
    for axis in axes[:, column]:
        axis.set_xticks(CONT_POWERS, [rf'$2^{{{power}}}$' for power in CONT_POWERS])
        axis.grid(alpha=0.14)
axes[0, 0].set_ylabel(r'Mean or median $P_g/P_r$')
axes[1, 0].set_ylabel(r'Variance of $P_g/P_r$')
axes[0, 2].legend(title='Filtered mean', ncol=2, frameon=False, fontsize=9)
fig.text(0.5, 0.01, 'Faint: original mean/variance; dotted: sample median; solid: 4.5-MAD filtered result', ha='center', fontsize=11)
fig.suptitle('Selected-bin outlier sensitivity analysis', fontsize=20, fontweight='semibold')
save_figure(fig, 'power_spectrum_selected_bins_outlier_sensitivity.png')
plt.show()

# Full 500k spectra show whether filtering changes only the high-k tail or the entire curve.
fig, axes = plt.subplots(2, 5, figsize=(19, 8), sharey=True, constrained_layout=True)
for axis, tag in zip(axes.flat, CONT_TAGS):
    analysis = outlier_analysis[(tag, 500)]
    ratios = analysis['pk_ratio']
    kept = ~analysis['outlier_mask']
    original_mean = np.nanmean(ratios, axis=0)
    sample_median = np.nanmedian(ratios, axis=0)
    filtered_mean = np.nanmean(ratios[kept], axis=0)
    axis.plot(analysis['kbins'], original_mean, color='0.55', lw=1.5, label='all-sample mean')
    axis.plot(analysis['kbins'], sample_median, color='#0072B2', ls=':', lw=1.7, label='sample median')
    axis.plot(analysis['kbins'], filtered_mean, color='#B33C86', lw=2, label='filtered mean')
    axis.axhline(1, color='black', ls='--', lw=1)
    axis.set_title(f'{dataset_label(int(continuation_row(tag, 500)["dataset_size"]))}\n{int(kept.sum())}/512 kept')
    axis.set_xlabel(r'$k$ bin')
    axis.grid(alpha=0.14)
for axis in axes[:, 0]:
    axis.set_ylabel(r'$P_g(k)/P_r(k)$')
axes[0, 0].legend(frameon=False, fontsize=9)
fig.suptitle('500k full-spectrum outlier sensitivity analysis', fontsize=20, fontweight='semibold')
save_figure(fig, 'power_spectrum_ratios_500k_outlier_sensitivity.png')
plt.show()
"""


SAMPLER_MARKDOWN = r"""
## 12. Sampler control

The sampler control compares DPM-Solver 50 with DDPM 500 at the same resolved
checkpoint, data configuration, generation seed, and number of samples. It uses
$2^8$ and $2^{11}$ at 300k and 500k. Agreement would argue against premature
termination by the 50-step sampler as the primary explanation. Endpoint
completion is scheduler-specific: DPM-Solver exposes a terminal $\sigma=0$,
whereas DDPM exposes no sigma array and is verified by all 500 executed steps
ending at diffusion timestep $t=0$.
"""


SAMPLER_CODE = r"""
sampler_rows = []
sampler_cases = [(tag, updates) for tag in ('d2p08', 'd2p11') for updates in (300, 500)]
fig, axes = plt.subplots(2, 2, figsize=(14, 10), constrained_layout=True)
map_fig, map_axes = plt.subplots(4, 3, figsize=(10.5, 13), constrained_layout=True)

for case_index, (axis, (tag, updates_k)) in enumerate(zip(axes.flat, sampler_cases)):
    row = continuation_row(tag, updates_k)
    dpm_path = project_path(row['sample_path'])
    ddpm_label = 'ddpm500_source_300k' if updates_k == 300 else 'ddpm500_cont_500k'
    ddpm_path = SAMPLE_DIR / f"{row['run_name']}_seed123_{ddpm_label}.npz"
    expected_checkpoint = project_path(row['analysis_checkpoint']).resolve()
    resolved = {
        'DPM-Solver 50': project_path(str(archive_scalar(dpm_path, 'resolved_checkpoint'))).resolve(),
        'DDPM 500': project_path(str(archive_scalar(ddpm_path, 'resolved_checkpoint'))).resolve(),
    }
    if set(resolved.values()) != {expected_checkpoint}:
        raise RuntimeError(f'Sampler checkpoint mismatch for {tag}/{updates_k}k: {resolved}')

    key = f'{tag}_{updates_k}k'
    kbins = continuation_curves[f'{key}_kbins']
    real_pk = continuation_curves[f'{key}_real_pk_mean']
    sample_arrays = {}
    for path, label, color, expected_steps, expected_scheduler in (
        (dpm_path, 'DPM-Solver 50', '#0072B2', 50, 'DPMSolverMultistepScheduler'),
        (ddpm_path, 'DDPM 500', '#D55E00', 500, 'DDPMScheduler'),
    ):
        samples = load_samples(path)
        sample_arrays[label] = samples
        spectra, current_kbins = batch_power_spectra(samples, nbins=len(real_pk))
        if not np.allclose(current_kbins, kbins, equal_nan=True):
            raise RuntimeError(f'k-bin mismatch for {path}')
        ratio = np.nanmean(spectra, axis=0) / np.clip(real_pk, 1e-30, None)
        axis.plot(kbins, ratio, color=color, marker='o', ms=3, lw=2, label=label)
        observed_steps = int(archive_scalar(path, 'executed_inference_steps'))
        scheduler_class = str(archive_scalar(path, 'scheduler_class'))
        if scheduler_class != expected_scheduler:
            raise RuntimeError(
                f'{path} used {scheduler_class}; expected {expected_scheduler}'
            )
        final_timestep = float(archive_scalar(path, 'final_timestep'))
        terminal_sigma = float(archive_scalar(path, 'terminal_sigma'))
        terminal_sigma_is_zero = bool(archive_scalar(path, 'terminal_sigma_is_zero'))
        terminal_sigma_verifiable = bool(archive_scalar(path, 'terminal_sigma_verifiable'))
        try:
            endpoint_evidence = validate_sampler_endpoint(
                scheduler_class=scheduler_class,
                executed_steps=observed_steps,
                expected_steps=expected_steps,
                final_timestep=final_timestep,
                terminal_sigma=terminal_sigma,
                terminal_sigma_is_zero=terminal_sigma_is_zero,
                terminal_sigma_verifiable=terminal_sigma_verifiable,
            )
        except ValueError as error:
            raise RuntimeError(f'Sampler endpoint audit failed for {path}: {error}') from error
        sampler_rows.append({
            'dataset_tag': tag,
            'updates_k': updates_k,
            'sampler': label,
            'scheduler_class': scheduler_class,
            'executed_inference_steps': observed_steps,
            'final_timestep': final_timestep,
            'terminal_sigma': terminal_sigma,
            'terminal_sigma_is_zero': terminal_sigma_is_zero,
            'terminal_sigma_verifiable': terminal_sigma_verifiable,
            'endpoint_evidence': endpoint_evidence,
            'resolved_checkpoint': str(resolved[label]),
            'sample_path': str(path),
        })
    axis.axhline(1, color='black', ls='--', lw=1.1)
    axis.set_title(f'{dataset_label(int(row["dataset_size"]))}, {updates_k}k')
    axis.set_xlabel(r'$k$ bin')
    axis.set_ylabel(r'$P_g/P_r$')
    axis.grid(alpha=0.16)

    dpm_image = sample_arrays['DPM-Solver 50'][0, 0]
    ddpm_image = sample_arrays['DDPM 500'][0, 0]
    difference = np.abs(dpm_image - ddpm_image)
    map_axes[case_index, 0].imshow(dpm_image, cmap='viridis', vmin=-1, vmax=1)
    map_axes[case_index, 1].imshow(ddpm_image, cmap='viridis', vmin=-1, vmax=1)
    map_axes[case_index, 2].imshow(difference, cmap='magma', vmin=0, vmax=max(float(np.quantile(difference, 0.995)), 1e-8))
    map_axes[case_index, 0].set_ylabel(f'{tag}, {updates_k}k', fontweight='bold')

axes[0, 0].legend(frameon=False)
fig.suptitle('Same-checkpoint sampler control', fontsize=21, fontweight='semibold')
save_figure(fig, 'sampler_control_power_spectrum.png')
plt.show()
for column, title in enumerate(('DPM-Solver 50', 'DDPM 500', 'absolute difference')):
    map_axes[0, column].set_title(title, fontweight='semibold')
for axis in map_axes.flat:
    axis.set_xticks([])
    axis.set_yticks([])
map_fig.suptitle('Same-seed sampler map control: sample 0', fontsize=20, fontweight='semibold')
save_figure(map_fig, 'sampler_control_maps.png')
plt.show()
display(pd.DataFrame(sampler_rows))
"""


PATCH_MARKDOWN = r"""
## 13. Patch-boundary diagnostics

The Patch-boundary statistic compares adjacent-pixel discontinuity on the
8-pixel DiT patch grid with an equal-size neighboring control. The trajectory
tests whether visible checkerboard structure changes during continuation. Real
subsets and historical L8/L12 values are controls, not evidence for an L16
scaling law.
"""


PATCH_CODE = r"""
fig, axes = plt.subplots(1, 2, figsize=(17, 6), constrained_layout=True)
for architecture, updates_k, label, color, linestyle in (
    ('real_reference', None, 'exact training subsets', 'black', ':'),
    ('dit_l8', 200, 'DiT-L8 context', '#009E73', '--'),
    ('dit_base', 200, 'DiT-L12 context', '#0072B2', '--'),
    ('dit_l16', 300, 'DiT-L16 300k', '#CC79A7', '-'),
    ('dit_l16', 500, 'DiT-L16 500k', '#8E2A68', '-'),
):
    current = continuation_patch[continuation_patch['architecture'] == architecture].copy()
    if updates_k is not None:
        current = current[pd.to_numeric(current['updates_k'], errors='coerce') == updates_k]
    current = current.sort_values('dataset_size')
    axes[0].plot(
        np.log2(current['dataset_size']), current['boundary_to_control_ratio'],
        color=color, marker='o', ls=linestyle, lw=2, label=label,
    )
axes[0].axhline(1, color='0.5', ls=':', lw=1)
axes[0].set_title('Reference and architecture context', fontweight='semibold')
axes[0].set_xlabel(r'Training images $N_{2D}$')
axes[0].set_ylabel('Boundary / local-control discontinuity')
axes[0].set_xticks(CONT_POWERS, [rf'$2^{{{power}}}$' for power in CONT_POWERS])
axes[0].legend(frameon=False, fontsize=9)
axes[0].grid(alpha=0.16)

l16_patch = continuation_patch[continuation_patch['architecture'] == 'dit_l16']
matrix = l16_patch.pivot(index='updates_k', columns='dataset_size', values='boundary_to_control_ratio').reindex(
    index=CONT_UPDATES_K, columns=CONT_SIZES
)
image = axes[1].imshow(matrix.to_numpy(dtype=float), origin='lower', aspect='auto', cmap='coolwarm')
axes[1].set_title('DiT-L16 continuation', fontweight='semibold')
axes[1].set_xlabel(r'Training images $N_{2D}$')
axes[1].set_ylabel('Optimizer updates')
axes[1].set_xticks(range(10), [rf'$2^{{{power}}}$' for power in CONT_POWERS])
axes[1].set_yticks(range(6), [f'{updates}k' for updates in CONT_UPDATES_K])
plt.colorbar(image, ax=axes[1], shrink=0.86, label='Boundary / control ratio')
fig.suptitle('Patch-boundary audit for patch size 8', fontsize=21, fontweight='semibold')
save_figure(fig, 'patch_boundary_diagnostics.png')
plt.show()
"""


NEAREST_MARKDOWN = r"""
## 14. Nearest-training audit

Four deterministic generated indices are checked for $2^8$ and $2^{11}$ at
300k and 500k. Nearest neighbors are searched across the complete exact
configured training subset in bounded batches. A dark difference map indicates
copying. A large difference establishes novelty only, not physical validity.
"""


NEAREST_CODE = r"""
nearest_rows = []
for tag in ('d2p08', 'd2p11'):
    queries_by_update = {}
    indices_by_update = {}
    for updates_k in (300, 500):
        samples = load_samples(continuation_row(tag, updates_k)['sample_path'])
        selected_indices = evenly_spaced_indices(total=len(samples), count=4)
        indices_by_update[updates_k] = selected_indices
        queries_by_update[updates_k] = samples[selected_indices]
    combined_queries = np.concatenate([queries_by_update[300], queries_by_update[500]], axis=0)
    config_path = source_config_for_row(continuation_row(tag, 300))
    matches = streaming_nearest_neighbors(
        combined_queries,
        iter_real_reference_batches_from_config(config_path, raw_batch_size=4),
    )

    fig, axes = plt.subplots(4, 6, figsize=(18, 12.5), constrained_layout=True)
    for update_offset, updates_k in enumerate((300, 500)):
        for local_index in range(4):
            query_index = update_offset * 4 + local_index
            generated = combined_queries[query_index, 0]
            nearest = matches['nearest_images'][query_index, 0]
            difference = np.abs(generated - nearest)
            column = update_offset * 3
            axes[local_index, column].imshow(generated, cmap='viridis', vmin=-1, vmax=1)
            axes[local_index, column + 1].imshow(nearest, cmap='viridis', vmin=-1, vmax=1)
            axes[local_index, column + 2].imshow(
                difference, cmap='magma', vmin=0, vmax=max(float(np.quantile(difference, 0.995)), 1e-8)
            )
            axes[local_index, column].set_ylabel(f'generated {indices_by_update[updates_k][local_index]}', fontweight='bold')
            axes[local_index, column + 2].text(
                0.03, 0.03,
                f"MSE={matches['mse'][query_index]:.3g}\ncos={matches['cosine_similarity'][query_index]:.3f}",
                transform=axes[local_index, column + 2].transAxes,
                color='white', fontsize=8,
                bbox={'facecolor': 'black', 'alpha': 0.65, 'pad': 2},
            )
            nearest_rows.append({
                'dataset_tag': tag,
                'updates_k': updates_k,
                'generated_index': int(indices_by_update[updates_k][local_index]),
                'nearest_training_index': int(matches['nearest_index'][query_index]),
                'nearest_mse': float(matches['mse'][query_index]),
                'nearest_cosine': float(matches['cosine_similarity'][query_index]),
            })
    for column, title in enumerate((
        '300k generated', '300k nearest training', '300k absolute difference',
        '500k generated', '500k nearest training', '500k absolute difference',
    )):
        axes[0, column].set_title(title, fontweight='semibold')
    for axis in axes.flat:
        axis.set_xticks([])
        axis.set_yticks([])
    fig.suptitle(f'DiT-L16 {tag}: complete-subset nearest-training audit', fontsize=21, fontweight='semibold')
    save_figure(fig, f'nearest_training_{tag}.png')
    plt.show()

nearest_summary = pd.DataFrame(nearest_rows)
display(nearest_summary)
"""


JOINT_MARKDOWN = r"""
## 15. Joint novelty and physical validity

Novelty and physical validity answer different questions. The scatter plots
join q95 novelty to one-point and power-spectrum error at the same checkpoint
and data size. “Novel but physically inaccurate” is a descriptive flag for
novel points in the upper quartile of the displayed physical error, not a
scientific acceptance threshold.
"""


JOINT_CODE = r"""
joint = continuation_novelty.merge(
    continuation_physics[
        ['dataset_tag', 'dataset_size', 'updates_k', 'hist_l1', 'pk_log10_mae']
    ],
    on=['dataset_tag', 'dataset_size', 'updates_k'],
    how='validate',
    validate='many_to_one',
)

fig, axes = plt.subplots(2, 2, figsize=(15, 11), constrained_layout=True)
for row_index, metric in enumerate(('hist_l1', 'pk_log10_mae')):
    for column_index, feature in enumerate(CONT_FEATURES):
        axis = axes[row_index, column_index]
        current = joint[joint['feature'] == feature].copy()
        novel = current['gen_gl_q95'] >= 0.5
        high_error_cut = current.loc[novel, metric].quantile(0.75) if novel.any() else np.nan
        current['verification_class'] = np.where(
            novel & (current[metric] >= high_error_cut),
            'Novel but physically inaccurate',
            'Other audited point',
        )
        for updates_k in CONT_UPDATES_K:
            subset = current[current['updates_k'] == updates_k]
            axis.scatter(subset['gen_gl_q95'], subset[metric], color=COLORS[updates_k], s=48, alpha=0.82, label=f'{updates_k}k')
        flagged = current[current['verification_class'] == 'Novel but physically inaccurate']
        axis.scatter(flagged['gen_gl_q95'], flagged[metric], facecolors='none', edgecolors='black', s=110, lw=1.3)
        axis.axvline(0.5, color='0.5', ls=':', lw=1)
        axis.set_title(f'{feature}: novelty vs {metric}', fontweight='semibold')
        axis.set_xlabel('q95 novelty score')
        axis.set_ylabel(metric)
        axis.grid(alpha=0.16)
axes[0, 1].legend(title='Updates', ncol=2, frameon=False)
fig.suptitle('Novelty does not certify physical validity', fontsize=21, fontweight='semibold')
save_figure(fig, 'joint_novelty_physical_validity.png')
plt.show()

joint_path = OUTPUT_DIR / 'joint_verification.csv'
joint.to_csv(joint_path, index=False)
print('wrote', joint_path)
"""


SUMMARY_MARKDOWN = r"""
## 16. Evidence summary

The final digest uses the corrected, all-sample physical metrics as the primary
result. It classifies a 300k-to-500k change as an improvement or degradation
only when the corresponding bootstrap intervals are separated. Overlapping
intervals remain unresolved. A ``reaches real floor'' flag means that the 500k
bootstrap interval reaches the empirical real-versus-real discrepancy; it is a
descriptive benchmark, not a formal equivalence test.

Median spectra and the fixed $4.5$-MAD outlier exclusion remain sensitivity
analyses. They can reveal whether a mean is driven by a small generated tail,
but they do not replace the primary all-sample population result. Novelty is
reported separately because removing physically unusual samples can bias a
memorization estimate upward or downward.

Interpret the output using the following rules:

- falling loss plus improving physical error supports the “needs more training”
  explanation for that data size;
- increasing novelty without improving physical error is not successful
  scientific generalization;
- agreement between DPM-Solver 50 and DDPM 500 weakens the sampler-truncation
  explanation;
- a transition-location claim is supported only for curves with one valid
  crossing and must be reported with its checkpoint budget.
"""


SUMMARY_CODE = r"""
def classify_interval_change(old_lo: float, old_hi: float, new_lo: float, new_hi: float) -> str:
    # Lower physical discrepancy is better; only separated intervals get a direction.
    if new_hi < old_lo:
        return 'CI-separated improvement'
    if new_lo > old_hi:
        return 'CI-separated degradation'
    return 'CI-overlapping / unresolved'


novelty_delta_columns = {
    'PCA': 'PCA_novelty_delta_500k_minus_300k',
    'SSCD': 'SSCD_novelty_delta_500k_minus_300k',
}
results_rows = []
for tag, size in zip(CONT_TAGS, CONT_SIZES):
    physics_rows = continuation_physics[continuation_physics['dataset_tag'] == tag].set_index('updates_k')
    if not {300, 500}.issubset(physics_rows.index):
        raise RuntimeError(f'Missing 300k or 500k corrected physics row for {tag}')
    at_300 = physics_rows.loc[300]
    at_500 = physics_rows.loc[500]

    row = {
        'dataset_tag': tag,
        'dataset_size': size,
        'hist_l1_300k': float(at_300['hist_l1']),
        'hist_l1_500k': float(at_500['hist_l1']),
        'hist_l1_delta_500k_minus_300k': float(at_500['hist_l1'] - at_300['hist_l1']),
        'hist_change_300k_to_500k': classify_interval_change(
            float(at_300['hist_l1_lo']), float(at_300['hist_l1_hi']),
            float(at_500['hist_l1_lo']), float(at_500['hist_l1_hi']),
        ),
        'pk_log10_mae_300k': float(at_300['pk_log10_mae']),
        'pk_log10_mae_500k': float(at_500['pk_log10_mae']),
        'pk_error_delta_500k_minus_300k': float(at_500['pk_log10_mae'] - at_300['pk_log10_mae']),
        'pk_change_300k_to_500k': classify_interval_change(
            float(at_300['pk_log10_mae_lo']), float(at_300['pk_log10_mae_hi']),
            float(at_500['pk_log10_mae_lo']), float(at_500['pk_log10_mae_hi']),
        ),
        'hist_reaches_real_floor_500k': bool(
            float(at_500['hist_l1_lo']) <= float(at_500['real_vs_real_hist_l1'])
        ),
        'pk_reaches_real_floor_500k': bool(
            float(at_500['pk_log10_mae_lo']) <= float(at_500['real_vs_real_pk_log10_mae'])
        ),
    }
    for feature in CONT_FEATURES:
        selected = continuation_novelty[
            (continuation_novelty['dataset_tag'] == tag) & (continuation_novelty['feature'] == feature)
        ].set_index('updates_k')['gen_gl_q95']
        row[novelty_delta_columns[feature]] = float(selected.loc[500] - selected.loc[300])

    filtered = outlier_excluded_physics[
        (outlier_excluded_physics['dataset_tag'] == tag)
        & (outlier_excluded_physics['updates_k'] == 500)
    ]
    if len(filtered) != 1:
        raise RuntimeError(f'Expected one outlier-excluded 500k row for {tag}; found {len(filtered)}')
    filtered = filtered.iloc[0]
    row.update({
        'filtered_hist_l1_500k': float(filtered['hist_l1']),
        'filtered_pk_log10_mae_500k': float(filtered['pk_log10_mae']),
        'outliers_removed_500k': int(filtered['n_removed']),
        'retention_fraction_500k': float(filtered['retention_fraction']),
    })
    results_rows.append(row)

results_digest = pd.DataFrame(results_rows)
results_digest_path = OUTPUT_DIR / 'results_digest.csv'
results_digest.to_csv(results_digest_path, index=False)

change_counts = pd.concat([
    results_digest['hist_change_300k_to_500k'].value_counts().rename('one-point PDF'),
    results_digest['pk_change_300k_to_500k'].value_counts().rename('power spectrum'),
], axis=1).fillna(0).astype(int)

display(Markdown('### 300k-to-500k corrected-results digest'))
display(results_digest)
display(Markdown('### Bootstrap-interval change classifications'))
display(change_counts)

evidence_summary = results_digest.copy()
evidence_path = OUTPUT_DIR / 'evidence_summary.csv'
evidence_summary.to_csv(evidence_path, index=False)
display(n50_summary)
display(pd.DataFrame(sampler_rows))
print('wrote', results_digest_path)
print('wrote', evidence_path)
"""


def build_notebook() -> dict[str, Any]:
    cells = [
        markdown(INTRO, section="intro"),
        code(SETUP, section="setup"),
        markdown(AUDIT_MARKDOWN, section="01-audit"),
        code(AUDIT_CODE, section="01-audit"),
        markdown(LOSS_MARKDOWN, section="02-loss"),
        code(LOSS_CODE, section="02-loss"),
        markdown(MAPS_MARKDOWN, section="03-maps"),
        code(MAPS_CODE, section="03-maps"),
        markdown(NOVELTY_MARKDOWN, section="04-novelty"),
        code(NOVELTY_CODE, section="04-novelty"),
        markdown(HEATMAP_MARKDOWN, section="05-phase"),
        code(HEATMAP_CODE, section="05-phase"),
        markdown(N50_MARKDOWN, section="06-n50"),
        code(N50_CODE, section="06-n50"),
        markdown(CONTEXT_MARKDOWN, section="07-context"),
        code(CONTEXT_CODE, section="07-context"),
        markdown(ONE_POINT_MARKDOWN, section="08-one-point"),
        code(ONE_POINT_CODE, section="08-one-point"),
        markdown(PK_MARKDOWN, section="09-power"),
        code(PK_CODE, section="09-power"),
        markdown(UNCERTAINTY_MARKDOWN, section="10-uncertainty"),
        code(UNCERTAINTY_CODE, section="10-uncertainty"),
        markdown(OUTLIER_MARKDOWN, section="11-outliers"),
        code(OUTLIER_CODE, section="11-outliers"),
        markdown(SAMPLER_MARKDOWN, section="12-sampler"),
        code(SAMPLER_CODE, section="12-sampler"),
        markdown(PATCH_MARKDOWN, section="13-patch"),
        code(PATCH_CODE, section="13-patch"),
        markdown(NEAREST_MARKDOWN, section="14-nearest"),
        code(NEAREST_CODE, section="14-nearest"),
        markdown(JOINT_MARKDOWN, section="15-joint"),
        code(JOINT_CODE, section="15-joint"),
        markdown(SUMMARY_MARKDOWN, section="16-summary"),
        code(SUMMARY_CODE, section="16-summary"),
    ]
    notebook = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "version": "3"},
            "analysis_contract": {
                "sweep": "nf_generalize_fig2_dit_l16_continue500k_v2",
                "updates_k": [300, 340, 380, 420, 460, 500],
                "dataset_powers": list(range(6, 16)),
                "requires_final_audit": True,
            },
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    return transform_notebook(notebook)


def main() -> None:
    notebook = build_notebook()
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(notebook, indent=1, ensure_ascii=True) + "\n")
    print(f"wrote {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
