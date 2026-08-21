#!/usr/bin/env python
"""Insert the audited DiT-L16 300k-to-500k analysis into the results notebook."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


TAG = "dit-l16-continue500k-v2"
RERUN_HEADING = "## Great Lakes Rerun Command"


def _source_lines(text: str) -> list[str]:
    text = text.strip("\n") + "\n"
    return text.splitlines(keepends=True)


def _markdown(cell_id: str, text: str) -> dict[str, Any]:
    return {
        "cell_type": "markdown",
        "id": cell_id,
        "metadata": {"tags": [TAG]},
        "source": _source_lines(text),
    }


def _code(cell_id: str, text: str) -> dict[str, Any]:
    return {
        "cell_type": "code",
        "execution_count": None,
        "id": cell_id,
        "metadata": {"tags": [TAG]},
        "outputs": [],
        "source": _source_lines(text),
    }


def build_cells() -> list[dict[str, Any]]:
    """Return the deterministic, self-auditing continuation analysis block."""
    return [
        _markdown(
            "l16c500-overview",
            r"""
## Audited DiT-L16 Continuation: 300k to 500k Updates

This section tests whether the fresh DiT-L16 sweep simply needed more optimization. Every one of the ten models is continued from its clean 300k checkpoint to 340k, 380k, 420k, 460k, and 500k optimizer updates with full model, EMA, optimizer, scheduler, scaler, and random-number state restored.

The analysis keeps three questions separate:

1. Does the denoising loss continue to decrease?
2. Do PCA and SSCD novelty curves move with additional training?
3. Do the exact-subset one-point distribution, power spectrum, and patch-boundary diagnostics improve?

The final audit is mandatory. **Do not infer a scaling law** from this section unless every expected checkpoint, sample, metric table, sampler record, and physical-statistics output passes that audit.
""",
        ),
        _code(
            "l16c500-audit",
            r"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import Markdown, display

CONT_SWEEP = 'nf_generalize_fig2_dit_l16_continue500k_v2'
CONT_LOCAL_DIR = PROJECT_DIR / 'local' / CONT_SWEEP
CONT_SAMPLE_DIR = PROJECT_DIR / 'results' / CONT_SWEEP / 'samples'
CONT_TABLE_DIR = PROJECT_DIR / 'results' / 'nf_generalize_fig2_dit' / 'tables'
CONT_PHYSICS_DIR = PROJECT_DIR / 'results' / 'nf_generalize_fig2_dit' / 'physics'
CONT_QUICKCHECK_DIR = PROJECT_DIR / 'results' / CONT_SWEEP / 'quickcheck'
CONT_QUICKCHECK_DIR.mkdir(parents=True, exist_ok=True)

CONT_UPDATES_K = [300, 340, 380, 420, 460, 500]
CONT_TAGS = [f'd2p{i:02d}' for i in range(6, 16)]
CONT_FEATURES = ('PCA', 'SSCD')

audit_path = CONT_LOCAL_DIR / 'final_audit.json'
if not audit_path.is_file():
    raise FileNotFoundError(
        f'Missing mandatory final audit: {audit_path}. Run the sweep audit before this notebook.'
    )
continuation_audit = json.loads(audit_path.read_text())
if continuation_audit.get('status') != 'PASS':
    raise RuntimeError(
        'DiT-L16 continuation audit did not pass:\n' +
        json.dumps(continuation_audit, indent=2, sort_keys=True)
    )
display(Markdown('### Final artifact audit: **PASS**'))
display(pd.json_normalize(continuation_audit, sep='.'))

analysis_manifest_path = CONT_LOCAL_DIR / 'analysis_manifest.json'
continuation_manifest = pd.DataFrame(json.loads(analysis_manifest_path.read_text()))
continuation_manifest['analysis_updates'] = continuation_manifest['analysis_updates'].astype(int)
continuation_manifest['updates_k'] = continuation_manifest['analysis_updates'] // 1000
expected_pairs = {(tag, updates) for tag in CONT_TAGS for updates in CONT_UPDATES_K}
actual_pairs = set(zip(continuation_manifest['dataset_tag'], continuation_manifest['updates_k']))
if len(continuation_manifest) != 60 or actual_pairs != expected_pairs:
    raise RuntimeError('analysis_manifest.json does not contain all 60 dataset/checkpoint rows')

def continuation_row(tag: str, updates_k: int) -> pd.Series:
    rows = continuation_manifest[
        (continuation_manifest['dataset_tag'] == tag) &
        (continuation_manifest['updates_k'] == int(updates_k))
    ]
    if len(rows) != 1:
        raise RuntimeError(f'Expected one manifest row for {tag} at {updates_k}k; found {len(rows)}')
    return rows.iloc[0]

continuation_novelty_frames = []
for feature in CONT_FEATURES:
    for updates_k in CONT_UPDATES_K:
        path = CONT_TABLE_DIR / f'{CONT_SWEEP}_{updates_k}k_{feature.lower()}_full_nn_metrics.csv'
        frame = pd.read_csv(path)
        frame = add_generalization_columns(frame)
        if len(frame) != 10 or set(frame['dataset_tag']) != set(CONT_TAGS):
            raise RuntimeError(f'Incomplete novelty table: {path}')
        frame['feature'] = feature
        frame['updates_k'] = updates_k
        continuation_novelty_frames.append(frame)
continuation_novelty = pd.concat(continuation_novelty_frames, ignore_index=True)

physics_summary_path = CONT_TABLE_DIR / f'{CONT_SWEEP}_physics_summary.csv'
selected_bins_path = CONT_TABLE_DIR / f'{CONT_SWEEP}_pk_selected_bins.csv'
patch_table_path = CONT_TABLE_DIR / f'{CONT_SWEEP}_patch_boundaries.csv'
physics_curves_path = CONT_PHYSICS_DIR / f'{CONT_SWEEP}_curves.npz'
continuation_physics = pd.read_csv(physics_summary_path)
continuation_selected_bins = pd.read_csv(selected_bins_path)
continuation_patch = pd.read_csv(patch_table_path)
continuation_curves = np.load(physics_curves_path)


def physics_k_max(dataset_tag: str, updates_k: int) -> float:
    rows = continuation_physics[
        (continuation_physics['dataset_tag'] == dataset_tag)
        & (continuation_physics['updates_k'] == int(updates_k))
    ]
    if len(rows) != 1:
        raise RuntimeError(
            f'Expected exactly one physics row for {dataset_tag}/{updates_k}k; found {len(rows)}'
        )
    try:
        value = float(rows.iloc[0]['k_max'])
    except (KeyError, TypeError, ValueError) as error:
        raise RuntimeError(
            f'Invalid k_max for {dataset_tag}/{updates_k}k; expected a finite positive value'
        ) from error
    if not np.isfinite(value) or value <= 0:
        raise RuntimeError(
            f'Invalid k_max for {dataset_tag}/{updates_k}k; expected a finite positive value'
        )
    return value

if len(continuation_physics) != 60:
    raise RuntimeError(f'Expected 60 physics rows; found {len(continuation_physics)}')
if len(continuation_selected_bins) != 180:
    raise RuntimeError(f'Expected 180 selected-k rows; found {len(continuation_selected_bins)}')

display(Markdown(
    f'Loaded **{len(continuation_manifest)}** manifest rows, '
    f'**{len(continuation_novelty)}** novelty rows, '
    f'**{len(continuation_physics)}** physical-summary rows, and '
    f'**{len(continuation_selected_bins)}** selected-k rows.'
))
""",
        ),
        _markdown(
            "l16c500-loss-note",
            r"""
### Training loss from 300k to 500k

Each panel is one training-set size. The curves show the cycle-averaged denoising objective over the continuation interval, with vertical guides at the six sampled checkpoints. A declining loss establishes that optimization continued; it does not by itself establish novelty or physical validity.
""",
        ),
        _code(
            "l16c500-loss",
            r"""
fig, axes = plt.subplots(2, 5, figsize=(18, 8.6), sharex=True, constrained_layout=True)
loss_audit_rows = []
for axis, tag in zip(axes.flat, CONT_TAGS):
    row = continuation_row(tag, 500)
    metrics, metrics_path = read_latest_metrics(row)
    epoch_loss = flatten_numeric(metrics.get('epoch_loss'))
    steps_per_epoch = int(row.get('steps_per_epoch', 1) or 1)
    update_axis, smoothed_loss = cycle_average_epoch_loss(
        epoch_loss, steps_per_epoch, restart_updates=4000
    )
    keep = (update_axis >= 295_000) & (update_axis <= 505_000)
    if not np.any(keep):
        raise RuntimeError(f'No 300k-500k loss history found for {tag}: {metrics_path}')
    axis.plot(update_axis[keep] / 1000, smoothed_loss[keep], color='#b83280', lw=2.2)
    for updates_k in CONT_UPDATES_K:
        axis.axvline(updates_k, color='0.78', lw=0.8, zorder=0)
    axis.set_yscale('log')
    axis.set_title(dataset_size_label(int(row['dataset_size'])), fontsize=15)
    axis.set_xlabel('Optimizer updates (thousands)')
    axis.grid(alpha=0.2)
    loss_audit_rows.append({
        'dataset_tag': tag,
        'dataset_size': int(row['dataset_size']),
        'history_last_update': float(np.nanmax(update_axis)),
        'loss_near_300k': float(smoothed_loss[keep][0]),
        'loss_near_500k': float(smoothed_loss[keep][-1]),
        'relative_change': float(smoothed_loss[keep][-1] / smoothed_loss[keep][0] - 1),
        'metrics_path': str(metrics_path),
    })
for axis in axes[:, 0]:
    axis.set_ylabel('Cycle-averaged denoising loss')
fig.suptitle('Fresh DiT-L16 optimization after the 300k checkpoint', fontsize=21, fontweight='semibold')
loss_path = CONT_QUICKCHECK_DIR / 'dit_l16_continue500k_loss.png'
fig.savefig(loss_path, dpi=250, bbox_inches='tight')
plt.show()
display(pd.DataFrame(loss_audit_rows))
print('wrote', loss_path)
""",
        ),
        _markdown(
            "l16c500-novelty-note",
            r"""
### PCA and SSCD novelty

Each colored line is one optimizer checkpoint. The complete $2^6$ through $2^{15}$ sweep is retained, so any movement of the transition can be compared directly. The dotted line is a visual 0.5 reference, not a fitted phase boundary. High novelty can still describe an out-of-distribution or physically incorrect sample.
""",
        ),
        _code(
            "l16c500-novelty",
            r"""
checkpoint_colors = plt.cm.viridis(np.linspace(0.08, 0.92, len(CONT_UPDATES_K)))
fig, axes = plt.subplots(1, 2, figsize=(16.5, 6.4), sharey=True, constrained_layout=True)
for axis, feature in zip(axes, CONT_FEATURES):
    feature_frame = continuation_novelty[continuation_novelty['feature'] == feature]
    for color, updates_k in zip(checkpoint_colors, CONT_UPDATES_K):
        current = feature_frame[feature_frame['updates_k'] == updates_k].sort_values('dataset_size')
        axis.plot(
            np.log2(current['dataset_size']), current['gen_gl_q95'],
            marker='o', ms=5.5, lw=2.2, color=color, label=f'{updates_k}k'
        )
    axis.axhline(0.5, color='0.35', lw=1.2, ls=':')
    axis.set_title(f'{feature} q95 novelty', fontsize=18, fontweight='semibold')
    axis.set_xlabel(r'Training images $N_{2D}$')
    axis.set_xticks(range(6, 16), [rf'$2^{{{i}}}$' for i in range(6, 16)])
    axis.set_ylim(-0.03, 1.04)
    axis.grid(alpha=0.2)
axes[0].set_ylabel('q95 novelty score')
axes[1].legend(title='Optimizer updates', ncol=2, frameon=False, loc='lower right')
fig.suptitle('DiT-L16 novelty across the full continuation trajectory', fontsize=22, fontweight='semibold')
novelty_path = CONT_QUICKCHECK_DIR / 'dit_l16_continue500k_novelty.png'
fig.savefig(novelty_path, dpi=250, bbox_inches='tight')
plt.show()
print('wrote', novelty_path)
""",
        ),
        _markdown(
            "l16c500-physical-note",
            r"""
### One-point and power-spectrum trajectories

The black one-point and power-spectrum references are recomputed from the **exact training subset configured for each model**. The heatmaps below summarize all ten data sizes and all six checkpoints. Lower error is better. This makes it possible to see whether longer training repairs the intermediate-data failure or only changes novelty.
""",
        ),
        _code(
            "l16c500-physical-summary",
            r"""
def intervals_overlap(low_a, high_a, low_b, high_b):
    values = np.asarray([low_a, high_a, low_b, high_b], dtype=float)
    if not np.isfinite(values).all():
        raise RuntimeError('Bootstrap interval contains a non-finite endpoint.')
    return max(low_a, low_b) <= min(high_a, high_b)


def _metric_heatmap(axis, frame, column, title, colorbar_label):
    low_column = f'{column}_lo'
    high_column = f'{column}_hi'
    missing = [name for name in (column, low_column, high_column) if name not in frame.columns]
    if missing:
        raise KeyError(f'Missing mandatory heatmap columns: {missing}')
    matrix = (
        frame.pivot(index='updates_k', columns='dataset_size', values=column)
        .reindex(index=CONT_UPDATES_K, columns=[2**i for i in range(6, 16)])
    )
    low = frame.pivot(index='updates_k', columns='dataset_size', values=low_column).reindex(
        index=CONT_UPDATES_K, columns=[2**i for i in range(6, 16)]
    )
    high = frame.pivot(index='updates_k', columns='dataset_size', values=high_column).reindex(
        index=CONT_UPDATES_K, columns=[2**i for i in range(6, 16)]
    )
    image = axis.imshow(matrix.to_numpy(dtype=float), aspect='auto', origin='lower', cmap='magma')
    axis.set_title(title, fontsize=17, fontweight='semibold')
    axis.set_xlabel(r'Training images $N_{2D}$')
    axis.set_ylabel('Optimizer updates')
    axis.set_xticks(range(10), [rf'$2^{{{i}}}$' for i in range(6, 16)])
    axis.set_yticks(range(6), [f'{value}k' for value in CONT_UPDATES_K])
    for column_index, dataset_size in enumerate([2**i for i in range(6, 16)]):
        overlap = intervals_overlap(
            low.loc[300, dataset_size], high.loc[300, dataset_size],
            low.loc[500, dataset_size], high.loc[500, dataset_size],
        )
        axis.text(
            column_index, CONT_UPDATES_K.index(500), 'o' if overlap else 'x',
            ha='center', va='center', color='white', fontsize=14, fontweight='bold',
        )
    axis.text(
        0.0, -0.20, '500k row: o = overlapping, x = separated 300k/500k 95% CIs',
        transform=axis.transAxes, fontsize=10.5, color='0.25',
    )
    plt.colorbar(image, ax=axis, shrink=0.85, label=colorbar_label)

required_ci_columns = (
    'hist_l1_lo', 'hist_l1_hi', 'pk_log10_mae_lo', 'pk_log10_mae_hi',
)
missing_ci_columns = [
    column for column in required_ci_columns if column not in continuation_physics.columns
]
if missing_ci_columns:
    raise KeyError(f'Missing mandatory bootstrap CI columns: {missing_ci_columns}')

fig, axes = plt.subplots(1, 2, figsize=(17, 6.4), constrained_layout=True)
_metric_heatmap(axes[0], continuation_physics, 'hist_l1', 'One-point PDF error', r'$L_1$ error')
_metric_heatmap(axes[1], continuation_physics, 'pk_log10_mae', r'Power-spectrum error', r'mean $|\log_{10}(P_g/P_r)|$')
fig.suptitle('Physical agreement over data size and training time', fontsize=22, fontweight='semibold')
physical_heatmap_path = CONT_QUICKCHECK_DIR / 'dit_l16_continue500k_physical_heatmaps.png'
fig.savefig(physical_heatmap_path, dpi=250, bbox_inches='tight')
plt.show()
print('wrote', physical_heatmap_path)

hist_edges = continuation_curves['histogram_edges']
hist_centers = 0.5 * (hist_edges[:-1] + hist_edges[1:])

def plot_physical_checkpoint(updates_k: int):
    checkpoint_ratios = [
        np.asarray(continuation_curves[f'{tag}_{updates_k}k_pk_ratio'], dtype=float)
        for tag in CONT_TAGS
    ]
    ratio_values = np.concatenate([ratio.reshape(-1) for ratio in checkpoint_ratios])
    if not np.isfinite(ratio_values).all() or np.any(ratio_values <= 0):
        raise RuntimeError(f'Invalid non-positive P(k) ratio at {updates_k}k updates.')
    ratio_limits = (float(ratio_values.min() / 1.15), float(ratio_values.max() * 1.15))
    fig, axes = plt.subplots(4, 5, figsize=(19, 13.5), constrained_layout=True)
    for column, tag in enumerate(CONT_TAGS[:5]):
        for block, current_tag in enumerate((tag, CONT_TAGS[column + 5])):
            row = 2 * block
            key = f'{current_tag}_{updates_k}k'
            axes[row, column].plot(hist_centers, continuation_curves[f'{key}_real_hist_probability'], color='black', lw=2, label='exact training subset')
            axes[row, column].plot(hist_centers, continuation_curves[f'{key}_generated_hist_probability'], color='#b83280', lw=2, label='generated')
            axes[row, column].set_yscale('log')
            axes[row, column].set_title(dataset_size_label(2 ** (column + 6 + 5 * block)))
            axes[row, column].set_xlabel('Normalized field value')
            kbins = continuation_curves[f'{key}_kbins']
            axes[row + 1, column].plot(kbins, continuation_curves[f'{key}_pk_ratio'], color='#b83280', marker='o', ms=2.8, lw=1.8)
            axes[row + 1, column].axhline(1.0, color='black', lw=1.1, ls='--')
            axes[row + 1, column].set_yscale('log')
            axes[row + 1, column].set_ylim(*ratio_limits)
            axes[row + 1, column].set_xlabel(r'$k$ bin')
    axes[0, 0].set_ylabel('Pixel probability')
    axes[1, 0].set_ylabel(r'$P_{generated}/P_{real}$')
    axes[2, 0].set_ylabel('Pixel probability')
    axes[3, 0].set_ylabel(r'$P_{generated}/P_{real}$')
    axes[0, 0].legend(frameon=False, fontsize=10)
    fig.suptitle(f'Fresh DiT-L16 exact-subset physical checks at {updates_k}k updates', fontsize=22, fontweight='semibold')
    path = CONT_QUICKCHECK_DIR / f'dit_l16_continue500k_physics_{updates_k}k.png'
    fig.savefig(path, dpi=230, bbox_inches='tight')
    plt.show()
    print('wrote', path)

plot_physical_checkpoint(300)
plot_physical_checkpoint(500)
""",
        ),
        _markdown(
            "l16c500-kbins-note",
            r"""
### Scale-resolved uncertainty at k-bin 20, 40, and 60

The left row shows the mean generated-to-real power ratio with bootstrap uncertainty. The right row shows the variance across the 512 generated inference samples. A mean near one can conceal excessive sample-to-sample scatter, so both quantities are required.
""",
        ),
        _code(
            "l16c500-kbins",
            r"""
fig, axes = plt.subplots(2, 3, figsize=(18, 10), constrained_layout=True)
for column, k_bin in enumerate((20, 40, 60)):
    selected = continuation_selected_bins[continuation_selected_bins['k_bin'] == k_bin]
    for color, updates_k in zip(checkpoint_colors, CONT_UPDATES_K):
        current = selected[selected['updates_k'] == updates_k].sort_values('dataset_size')
        x = np.log2(current['dataset_size'])
        mean = current['ratio_mean'].to_numpy(dtype=float)
        low = current['ratio_mean_ci_low'].to_numpy(dtype=float)
        high = current['ratio_mean_ci_high'].to_numpy(dtype=float)
        if (current['real_reference_mean'] <= 0).any():
            raise RuntimeError(f'Non-positive real power at k-bin {k_bin}, {updates_k}k')
        ratio_variance = current['generated_variance'] / current['real_reference_mean'].pow(2)
        axes[0, column].plot(x, mean, marker='o', ms=4.5, color=color, lw=1.8, label=f'{updates_k}k')
        axes[0, column].fill_between(x, low, high, color=color, alpha=0.12)
        axes[1, column].plot(x, ratio_variance, marker='o', ms=4.5, color=color, lw=1.8)
    axes[0, column].axhline(1.0, color='black', lw=1.1, ls='--')
    axes[0, column].set_title(f'k-bin {k_bin}', fontsize=16, fontweight='semibold')
    axes[1, column].set_xlabel(r'Training images $N_{2D}$')
    for axis in axes[:, column]:
        axis.set_xticks(range(6, 16), [rf'$2^{{{i}}}$' for i in range(6, 16)])
        axis.grid(alpha=0.2)
axes[0, 0].set_ylabel(r'Mean $P_{generated}/P_{real}$')
axes[1, 0].set_ylabel('Variance across generated samples')
axes[0, 2].legend(title='Updates', ncol=2, frameon=False)
fig.suptitle('Scale-resolved power-spectrum accuracy and inference variance', fontsize=22, fontweight='semibold')
selected_path = CONT_QUICKCHECK_DIR / 'dit_l16_continue500k_selected_kbins.png'
fig.savefig(selected_path, dpi=250, bbox_inches='tight')
plt.show()
print('wrote', selected_path)
""",
        ),
        _markdown(
            "l16c500-sampler-note",
            r"""
### DPM-Solver 50 versus DDPM 500

This controlled comparison uses the same resolved checkpoint and seed for $2^8$ and $2^{11}$ at 300k and 500k. It tests whether the visible artifacts or power-spectrum errors are caused by the 50-step sampler. Scheduler metadata and the executed terminal step are displayed with the curves.
""",
        ),
        _code(
            "l16c500-sampler",
            r"""
def _npz_scalar(payload, key, default=None):
    if key not in payload.files:
        return default
    value = np.asarray(payload[key])
    return value.item() if value.size == 1 else value.tolist()

sampler_rows = []
fig, axes = plt.subplots(2, 2, figsize=(14, 10), constrained_layout=True)
for axis, (tag, updates_k) in zip(axes.flat, [(tag, updates) for tag in ('d2p08', 'd2p11') for updates in (300, 500)]):
    row = continuation_row(tag, updates_k)
    dpm_path = Path(row['sample_path'])
    if not dpm_path.is_absolute():
        dpm_path = PROJECT_DIR / dpm_path
    run_name = str(row['run_name'])
    ddpm_label = f'ddpm500_{"source_300k" if updates_k == 300 else "cont_500k"}'
    ddpm_path = CONT_SAMPLE_DIR / f'{run_name}_seed123_{ddpm_label}.npz'
    key = f'{tag}_{updates_k}k'
    real_pk = continuation_curves[f'{key}_real_pk_mean']
    kbins = continuation_curves[f'{key}_kbins']
    k_max = physics_k_max(tag, updates_k)
    for path, label, color in ((dpm_path, 'DPM-Solver 50', '#0072B2'), (ddpm_path, 'DDPM 500', '#D55E00')):
        samples = load_npz_array(path)
        spectra, current_kbins = batch_power_spectra(
            samples,
            nbins=len(real_pk),
            k_max=k_max,
        )
        if not np.allclose(current_kbins, kbins, equal_nan=True):
            raise RuntimeError(f'k-bin mismatch for {path}')
        ratio = np.nanmean(spectra, axis=0) / np.clip(real_pk, 1e-30, None)
        axis.plot(kbins, ratio, marker='o', ms=3, lw=2, color=color, label=label)
        with np.load(path, allow_pickle=False) as payload:
            sampler_rows.append({
                'dataset_tag': tag,
                'updates_k': updates_k,
                'sampler': label,
                'scheduler': _npz_scalar(payload, 'scheduler', 'missing'),
                'executed_inference_steps': _npz_scalar(payload, 'executed_inference_steps', 'missing'),
                'terminal_sigma': _npz_scalar(payload, 'terminal_sigma', 'missing'),
                'terminal_sigma_verifiable': _npz_scalar(payload, 'terminal_sigma_verifiable', 'missing'),
                'resolved_checkpoint': _npz_scalar(payload, 'resolved_checkpoint', 'missing'),
                'sample_path': str(path),
            })
    axis.axhline(1, color='black', lw=1.1, ls='--')
    axis.set_title(f'{dataset_size_label(int(row["dataset_size"]))}, {updates_k}k updates')
    axis.set_xlabel(r'$k$ bin')
    axis.set_ylabel(r'$P_{generated}/P_{real}$')
    axis.grid(alpha=0.2)
axes[0, 0].legend(frameon=False)
fig.suptitle('Controlled sampler check', fontsize=22, fontweight='semibold')
sampler_path = CONT_QUICKCHECK_DIR / 'dit_l16_continue500k_sampler_control.png'
fig.savefig(sampler_path, dpi=250, bbox_inches='tight')
plt.show()
sampler_audit = pd.DataFrame(sampler_rows)
display(sampler_audit)
print('wrote', sampler_path)
""",
        ),
        _markdown(
            "l16c500-patch-note",
            r"""
### Patch-boundary diagnostic

DiT-L8, DiT-L12, and DiT-L16 all use patch size 8. This diagnostic measures discontinuity at patch boundaries relative to interior neighboring pixels. The real-reference line and 200k L8/L12 curves provide scale controls; the L16 heatmap shows whether the checkerboard pattern changes from 300k to 500k.
""",
        ),
        _code(
            "l16c500-patch",
            r"""
fig, axes = plt.subplots(1, 2, figsize=(17, 6.2), constrained_layout=True)
line_specs = [
    ('real_reference', None, 'exact real subset', 'black', '-'),
    ('dit_l8', 200, 'DiT-L8, 200k', '#009E73', '--'),
    ('dit_base', 200, 'DiT-L12, 200k', '#0072B2', '--'),
    ('dit_l16', 300, 'DiT-L16, 300k', '#CC79A7', '-'),
    ('dit_l16', 500, 'DiT-L16, 500k', '#8E2A68', '-'),
]
for architecture, updates_k, label, color, linestyle in line_specs:
    current = continuation_patch[continuation_patch['architecture'] == architecture].copy()
    if updates_k is not None:
        current = current[current['updates_k'] == updates_k]
    current = current.sort_values('dataset_size')
    axes[0].plot(np.log2(current['dataset_size']), current['patch_boundary_ratio'], marker='o', lw=2, color=color, ls=linestyle, label=label)
axes[0].axhline(1, color='0.5', lw=1, ls=':')
axes[0].set_title('Architecture and real-reference comparison', fontsize=16, fontweight='semibold')
axes[0].set_xlabel(r'Training images $N_{2D}$')
axes[0].set_ylabel('Patch-boundary / interior discontinuity')
axes[0].set_xticks(range(6, 16), [rf'$2^{{{i}}}$' for i in range(6, 16)])
axes[0].legend(frameon=False, fontsize=10)
axes[0].grid(alpha=0.2)

l16_patch = continuation_patch[continuation_patch['architecture'] == 'dit_l16']
matrix = l16_patch.pivot(index='updates_k', columns='dataset_size', values='patch_boundary_ratio').reindex(index=CONT_UPDATES_K, columns=[2**i for i in range(6, 16)])
image = axes[1].imshow(matrix.to_numpy(dtype=float), aspect='auto', origin='lower', cmap='coolwarm')
axes[1].set_title('DiT-L16 continuation', fontsize=16, fontweight='semibold')
axes[1].set_xlabel(r'Training images $N_{2D}$')
axes[1].set_ylabel('Optimizer updates')
axes[1].set_xticks(range(10), [rf'$2^{{{i}}}$' for i in range(6, 16)])
axes[1].set_yticks(range(6), [f'{value}k' for value in CONT_UPDATES_K])
plt.colorbar(image, ax=axes[1], shrink=0.86, label='Boundary ratio')
fig.suptitle('Patch-boundary audit (patch size 8)', fontsize=22, fontweight='semibold')
patch_path = CONT_QUICKCHECK_DIR / 'dit_l16_continue500k_patch_boundary.png'
fig.savefig(patch_path, dpi=250, bbox_inches='tight')
plt.show()
print('wrote', patch_path)
""",
        ),
        _markdown(
            "l16c500-nearest-note",
            r"""
### Generated samples versus nearest training slices

For the transition candidate $2^8$ and the higher-data control $2^{11}$, four generated samples are compared with their nearest training slice at 300k and 500k. A nearly blank absolute-difference map indicates copying. A large difference establishes novelty only; it must be read together with the physical-statistics and sampler panels above.
""",
        ),
        _code(
            "l16c500-nearest",
            r"""
nearest_summary_rows = []
for tag in ('d2p08', 'd2p11'):
    for updates_k in (300, 500):
        row = continuation_row(tag, updates_k)
        sample_path = Path(row['sample_path'])
        if not sample_path.is_absolute():
            sample_path = PROJECT_DIR / sample_path
        config_value = row.get('source_config') if updates_k == 300 else row.get('config')
        config_path = Path(str(config_value))
        if not config_path.is_absolute():
            config_path = PROJECT_DIR / config_path
        generated = load_npz_array(sample_path)
        training = load_real_reference_from_config(config_path, max_slices=None)
        matches = nearest_training_matches(
            generated, training, max_generated=4, max_training=None, training_chunk=256
        )
        generated_index = np.asarray(matches['generated_index'], dtype=int)
        nearest_index = np.asarray(matches['nearest_training_index'], dtype=int)
        selected_generated = generated[generated_index, 0]
        selected_training = training[nearest_index, 0]
        difference = np.abs(selected_generated - selected_training)

        fig, axes = plt.subplots(4, 3, figsize=(10.5, 13.5), constrained_layout=True)
        for index in range(4):
            combined = np.concatenate([selected_generated[index].ravel(), selected_training[index].ravel()])
            vmin, vmax = np.quantile(combined, [0.005, 0.995])
            dmax = max(float(np.quantile(difference[index], 0.995)), 1e-8)
            axes[index, 0].imshow(selected_generated[index], cmap='viridis', vmin=vmin, vmax=vmax)
            axes[index, 1].imshow(selected_training[index], cmap='viridis', vmin=vmin, vmax=vmax)
            axes[index, 2].imshow(difference[index], cmap='magma', vmin=0, vmax=dmax)
            axes[index, 0].set_ylabel(f'generated {generated_index[index]}', fontweight='bold')
            axes[index, 2].text(
                0.03, 0.03,
                f'MSE={matches["nearest_mse"][index]:.3g}; cos={matches["nearest_cosine"][index]:.3f}',
                transform=axes[index, 2].transAxes, color='white', fontsize=9,
                bbox={'facecolor': 'black', 'alpha': 0.62, 'pad': 2},
            )
            for axis in axes[index]:
                axis.set_xticks([])
                axis.set_yticks([])
            nearest_summary_rows.append({
                'dataset_tag': tag,
                'updates_k': updates_k,
                'generated_index': int(generated_index[index]),
                'nearest_training_index': int(nearest_index[index]),
                'nearest_mse': float(matches['nearest_mse'][index]),
                'nearest_cosine': float(matches['nearest_cosine'][index]),
            })
        for axis, title in zip(axes[0], ('generated', 'nearest training', 'absolute difference')):
            axis.set_title(title, fontsize=14, fontweight='semibold')
        fig.suptitle(f'DiT-L16 {dataset_size_label(int(row["dataset_size"]))}: nearest-training audit at {updates_k}k', fontsize=19, fontweight='semibold')
        path = CONT_QUICKCHECK_DIR / f'dit_l16_{tag}_{updates_k}k_nearest_training.png'
        fig.savefig(path, dpi=230, bbox_inches='tight')
        plt.show()
        print('wrote', path)

nearest_summary = pd.DataFrame(nearest_summary_rows)
display(nearest_summary)
""",
        ),
        _markdown(
            "l16c500-conclusion",
            r"""
### Interpretation checklist

- If loss falls while one-point and $P(k)$ errors remain flat, insufficient optimizer updates alone are not the explanation.
- If DPM-Solver 50 and DDPM 500 agree at the same checkpoint, sampler truncation is not the main cause of the artifacts.
- If patch-boundary ratios are elevated specifically for DiT-L16, investigate patch-token optimization or reconstruction rather than the CAMELS preprocessing.
- If the nearest-training difference is large but physical error is also large, the sample is novel but invalid; it must not be counted as successful generalization.
- Only a repeated, physically valid shift of the PCA and SSCD curves supports a depth-dependent transition. **Do not infer a scaling law** merely because the novelty score crosses 0.5.
""",
        ),
    ]


def update_notebook(input_path: Path, output_path: Path) -> None:
    """Replace this updater's tagged block while preserving every other cell."""
    notebook = json.loads(Path(input_path).read_text())
    cells = notebook.get("cells")
    if not isinstance(cells, list):
        raise ValueError(f"Notebook has no cells list: {input_path}")

    untagged = [
        cell
        for cell in cells
        if TAG not in cell.get("metadata", {}).get("tags", [])
    ]
    anchors = [
        index
        for index, cell in enumerate(untagged)
        if RERUN_HEADING in "".join(cell.get("source", []))
    ]
    if len(anchors) != 1:
        raise RuntimeError(
            f"Expected exactly one '{RERUN_HEADING}' anchor; found {len(anchors)}"
        )
    insert_at = anchors[0]
    notebook["cells"] = untagged[:insert_at] + build_cells() + untagged[insert_at:]

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(notebook, indent=1, ensure_ascii=False) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    default = Path("notebooks/nf_generalize_fig2_dit_results.ipynb")
    parser.add_argument("--input", type=Path, default=default)
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = args.output or args.input
    update_notebook(args.input, output)
    print(f"updated {output}")


if __name__ == "__main__":
    main()
