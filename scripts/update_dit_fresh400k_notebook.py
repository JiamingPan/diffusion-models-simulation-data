#!/usr/bin/env python
"""Insert the audited fresh DiT-L16 400k analysis into the results notebook."""

from __future__ import annotations

import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = REPO_ROOT / "notebooks" / "nf_generalize_fig2_dit_results.ipynb"
SECTION_HEADING = "## Fresh DiT-L16 sweep through 400k updates"
LEGACY_SECTION_HEADING = "## Fresh DiT-L16 sweep through 300k updates"


def markdown_cell(source: str) -> dict:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": source.splitlines(keepends=True),
    }


def code_cell(source: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source.splitlines(keepends=True),
    }


INTRO = r"""## Fresh DiT-L16 sweep through 400k updates

This is the clean replacement for the legacy continuation diagnostic. Ten DiT-L16
models are initialized from scratch with seed 123, one for every training-set size
from $2^6$ through $2^{15}$, and trained continuously to 400k optimizer updates.

- **200k is the equal-budget comparison** with DiT-L8 and DiT-L12/base.
- **300k is the intermediate L16 curve** and includes all ten data sizes.
- **400k is the final L16 curve** and includes all ten data sizes.
- **No legacy continuation fallback** is allowed in this section.
- A q95 novelty score measures distance from training examples. **q95 novelty does not guarantee physical fidelity**; read it with the image, one-point, and power-spectrum checks.
"""


AUDIT_CODE = r"""FRESH_SWEEP_NAME = 'nf_generalize_fig2_dit_l16_fresh400k'
FRESH_RESULT_DIR = PROJECT_DIR / 'results' / FRESH_SWEEP_NAME
FRESH_SAMPLE_DIR = FRESH_RESULT_DIR / 'samples'
FRESH_MANIFEST_PATH = PROJECT_DIR / 'local' / FRESH_SWEEP_NAME / 'manifest.json'
FRESH_ANALYSIS_MANIFEST_PATH = PROJECT_DIR / 'local' / FRESH_SWEEP_NAME / 'analysis_manifest.json'
FRESH_UPDATES_K = [200, 300, 400]
FRESH_EXPECTED_POWERS = list(range(6, 16))
FRESH_EXPECTED_TAGS = [f'd2p{power:02d}' for power in FRESH_EXPECTED_POWERS]
FRESH_EXPECTED_SIZES = [2 ** power for power in FRESH_EXPECTED_POWERS]
fresh_equal_budget_updates_k = 200
fresh_intermediate_updates_k = 300
fresh_final_updates_k = 400


def fresh_metric_path(feature: str, updates_k: int) -> Path:
    return TABLE_DIR / (
        f'{FRESH_SWEEP_NAME}_{updates_k}k_{feature.lower()}_full_nn_metrics.csv'
    )


def audit_fresh_metric_table(feature: str, updates_k: int) -> tuple[pd.DataFrame, dict[str, Any]]:
    path = fresh_metric_path(feature, updates_k)
    audit_row: dict[str, Any] = {
        'feature': feature,
        'updates_k': updates_k,
        'table_path': rel(path),
        'exists': path.exists(),
        'complete': False,
        'row_count': 0,
        'missing_tags': list(FRESH_EXPECTED_TAGS),
        'extra_tags': [],
    }
    if not path.exists():
        return pd.DataFrame(), audit_row

    table = add_generalization_columns(pd.read_csv(path))
    table = ensure_arch_columns(table)
    table = table[table['arch'].astype(str) == 'dit_l16'].copy()
    tags = sorted(table['dataset_tag'].dropna().astype(str).unique().tolist())
    missing = sorted(set(FRESH_EXPECTED_TAGS) - set(tags))
    extra = sorted(set(tags) - set(FRESH_EXPECTED_TAGS))
    duplicate_tags = sorted(
        table.loc[table['dataset_tag'].duplicated(keep=False), 'dataset_tag']
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )
    complete = (
        len(table) == 10
        and not missing
        and not extra
        and not duplicate_tags
        and 'gen_gl_q95' in table.columns
        and table['gen_gl_q95'].notna().all()
    )
    audit_row.update({
        'complete': bool(complete),
        'row_count': len(table),
        'missing_tags': missing,
        'extra_tags': extra,
        'duplicate_tags': duplicate_tags,
    })
    table['feature'] = feature
    table['updates_k'] = updates_k
    table['fresh_sweep'] = True
    return table.sort_values('dataset_size'), audit_row


fresh_metrics_by_update: dict[int, dict[str, pd.DataFrame]] = {}
fresh_audit_rows: list[dict[str, Any]] = []
for updates_k in FRESH_UPDATES_K:
    fresh_metrics_by_update[updates_k] = {}
    for feature in ('PCA', 'SSCD'):
        table, audit_row = audit_fresh_metric_table(feature, updates_k)
        fresh_metrics_by_update[updates_k][feature] = table
        fresh_audit_rows.append(audit_row)

fresh_table_audit_df = pd.DataFrame(fresh_audit_rows)
display(Markdown('### Fresh exact-checkpoint table audit'))
display(fresh_table_audit_df)

fresh_400k_rows = fresh_table_audit_df[
    fresh_table_audit_df['updates_k'] == fresh_final_updates_k
]
fresh_400k_complete = (
    len(fresh_400k_rows) == 2
    and bool(fresh_400k_rows['complete'].all())
)
fresh_300k_rows = fresh_table_audit_df[
    fresh_table_audit_df['updates_k'] == fresh_intermediate_updates_k
]
fresh_300k_complete = (
    len(fresh_300k_rows) == 2
    and bool(fresh_300k_rows['complete'].all())
)
fresh_200k_rows = fresh_table_audit_df[
    fresh_table_audit_df['updates_k'] == fresh_equal_budget_updates_k
]
fresh_200k_complete = (
    len(fresh_200k_rows) == 2
    and bool(fresh_200k_rows['complete'].all())
)

if fresh_400k_complete:
    display(Markdown(
        '**Fresh 400k audit passed.** All ten data sizes '
        r'$2^6,\ldots,2^{15}$ are present in both PCA and SSCD.'
    ))
else:
    display(Markdown(
        '**Fresh 400k audit incomplete: not drawing the fresh final curve.** '
        'All ten data sizes must be present in both PCA and SSCD. '
        'No legacy continuation fallback will be used.'
    ))
"""


PLOT_CODE = r"""FRESH_DEPTH_COLORS = {
    'dit_l8': '#009E73',
    'dit_base': '#0072B2',
    'dit_l16': '#B33C86',
}
FRESH_DEPTH_MARKERS = {'dit_l8': 'P', 'dit_base': 'D', 'dit_l16': 'X'}


def require_complete_arch_curve(
    table: pd.DataFrame,
    arch: str,
    *,
    context: str,
) -> pd.DataFrame:
    sub = table[table['arch'].astype(str) == arch].copy().sort_values('dataset_size')
    tags = set(sub['dataset_tag'].dropna().astype(str))
    missing = sorted(set(FRESH_EXPECTED_TAGS) - tags)
    if len(sub) != 10 or missing or sub['dataset_tag'].duplicated().any():
        raise ValueError(
            f'{context}: {arch} does not contain exactly the ten expected data sizes; '
            f'missing={missing}, rows={len(sub)}'
        )
    if 'gen_gl_q95' not in sub.columns or sub['gen_gl_q95'].isna().any():
        raise ValueError(f'{context}: {arch} has incomplete q95 novelty values')
    return sub


def fresh_depth_sources(feature: str, l16_updates_k: int) -> dict[str, pd.DataFrame]:
    base_table = pca_metrics if feature == 'PCA' else sscd_metrics
    fresh_table = fresh_metrics_by_update[l16_updates_k][feature]
    return {
        'dit_l8': require_complete_arch_curve(
            base_table, 'dit_l8', context=f'{feature} DiT-L8 200k'
        ),
        'dit_base': require_complete_arch_curve(
            base_table, 'dit_base', context=f'{feature} DiT-L12 200k'
        ),
        'dit_l16': require_complete_arch_curve(
            fresh_table, 'dit_l16', context=f'{feature} fresh DiT-L16 {l16_updates_k}k'
        ),
    }


def plot_fresh_depth_comparison(
    *,
    l16_updates_k: int,
    output_name: str,
    heading: str,
    subtitle: str,
    zoom_max_power: int | None = None,
) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(15.8, 6.0), sharey=True, constrained_layout=True)
    for ax, feature in zip(axes, ('PCA', 'SSCD')):
        sources = fresh_depth_sources(feature, l16_updates_k)
        for arch in ('dit_l8', 'dit_base', 'dit_l16'):
            sub = sources[arch]
            budget = l16_updates_k if arch == 'dit_l16' else 200
            label = f"{DIT_ARCH_LABELS[arch]} {budget}k"
            ax.plot(
                sub['dataset_size'],
                sub['gen_gl_q95'],
                color=FRESH_DEPTH_COLORS[arch],
                marker=FRESH_DEPTH_MARKERS[arch],
                ms=8.5,
                lw=3.0,
                label=label,
            )
        ax.axhline(0.5, color='0.35', ls=':', lw=1.5)
        ax.set_xscale('log', base=2)
        shown_powers = [
            power for power in FRESH_EXPECTED_POWERS
            if zoom_max_power is None or power <= zoom_max_power
        ]
        shown_sizes = [2 ** power for power in shown_powers]
        ax.set_xticks(shown_sizes)
        ax.set_xticklabels([rf'$2^{{{power}}}$' for power in shown_powers])
        if zoom_max_power is not None:
            ax.set_xlim(2 ** 5.75, 2 ** (zoom_max_power + 0.25))
        ax.set_ylim(-0.04, 1.04)
        ax.set_xlabel(r'Training images $N_{2D}$')
        ax.set_title(f'{feature} q95 novelty', fontsize=19, pad=12)
        ax.grid(axis='y', alpha=0.16)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
    axes[0].set_ylabel('q95 novelty score')
    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0.91),
               ncol=3, frameon=False)
    fig.suptitle(heading, fontsize=23, fontweight='semibold', y=1.01)
    fig.text(0.5, 0.945, subtitle, ha='center', color='0.35', fontsize=12.5)
    out = QUICKCHECK_DIR / output_name
    fig.savefig(out, bbox_inches='tight', dpi=300)
    plt.show()
    print('wrote', out)
    return out


fresh_equal_budget_outputs = []
if fresh_200k_complete:
    fresh_equal_budget_outputs.append(plot_fresh_depth_comparison(
        l16_updates_k=fresh_equal_budget_updates_k,
        output_name='nf_generalize_fig2_dit_l16_fresh400k_equal_budget_200k_full.png',
        heading='Fresh DiT depth sweep at a fixed 200k-update budget',
        subtitle='L8, L12, and fresh L16 use the same optimizer-update budget.',
    ))
    fresh_equal_budget_outputs.append(plot_fresh_depth_comparison(
        l16_updates_k=fresh_equal_budget_updates_k,
        output_name='nf_generalize_fig2_dit_l16_fresh400k_equal_budget_200k_zoom.png',
        heading='Fresh DiT depth sweep at 200k: transition region',
        subtitle=r'Zoomed to $2^6$--$2^{11}$; all ten sizes are still required by the audit.',
        zoom_max_power=11,
    ))
else:
    display(Markdown(
        '**Equal-budget plot skipped.** The fresh L16 200k PCA/SSCD tables '
        'must each contain all ten data sizes.'
    ))

fresh_intermediate_outputs = []
if fresh_300k_complete:
    fresh_intermediate_outputs.append(plot_fresh_depth_comparison(
        l16_updates_k=fresh_intermediate_updates_k,
        output_name='nf_generalize_fig2_dit_l16_fresh400k_intermediate_300k_full.png',
        heading='Intermediate fresh DiT-L16 result at 300k updates',
        subtitle='L8 and L12 use 200k updates; L16 uses 300k. This is not an equal-compute comparison.',
    ))
    fresh_intermediate_outputs.append(plot_fresh_depth_comparison(
        l16_updates_k=fresh_intermediate_updates_k,
        output_name='nf_generalize_fig2_dit_l16_fresh400k_intermediate_300k_zoom.png',
        heading='Intermediate fresh DiT-L16 result: transition region',
        subtitle=r'Zoomed to $2^6$--$2^{11}$; L16 uses 300k updates.',
        zoom_max_power=11,
    ))

fresh_final_outputs = []
if fresh_400k_complete:
    fresh_final_outputs.append(plot_fresh_depth_comparison(
        l16_updates_k=fresh_final_updates_k,
        output_name='nf_generalize_fig2_dit_l16_fresh400k_final_outcome_full.png',
        heading='Final fresh DiT-L16 result through 400k updates',
        subtitle='L8 and L12 use 200k updates; L16 uses 400k. This is not an equal-compute comparison.',
    ))
    fresh_final_outputs.append(plot_fresh_depth_comparison(
        l16_updates_k=fresh_final_updates_k,
        output_name='nf_generalize_fig2_dit_l16_fresh400k_final_outcome_zoom.png',
        heading='Final fresh DiT-L16 result: transition region',
        subtitle=r'Zoomed to $2^6$--$2^{11}$; L16 uses 400k updates.',
        zoom_max_power=11,
    ))
"""


TRAJECTORY_CODE = r"""def plot_fresh_l16_checkpoint_trajectories() -> Path | None:
    complete_updates = []
    for updates_k in FRESH_UPDATES_K:
        rows = fresh_table_audit_df[fresh_table_audit_df['updates_k'] == updates_k]
        if len(rows) == 2 and bool(rows['complete'].all()):
            complete_updates.append(updates_k)
    if not complete_updates:
        display(Markdown(
            '**Fresh trajectory unavailable.** No optimizer milestone has complete '
            'PCA and SSCD tables across all ten data sizes.'
        ))
        return None

    colors = plt.cm.viridis(np.linspace(0.12, 0.90, len(complete_updates)))
    fig, axes = plt.subplots(1, 2, figsize=(15.8, 6.0), sharey=True, constrained_layout=True)
    for ax, feature in zip(axes, ('PCA', 'SSCD')):
        for color, updates_k in zip(colors, complete_updates):
            sub = require_complete_arch_curve(
                fresh_metrics_by_update[updates_k][feature],
                'dit_l16',
                context=f'{feature} fresh DiT-L16 {updates_k}k trajectory',
            )
            ax.plot(
                sub['dataset_size'],
                sub['gen_gl_q95'],
                color=color,
                marker='o',
                ms=7.0,
                lw=2.6,
                label=f'{updates_k}k',
            )
        ax.axhline(0.5, color='0.35', ls=':', lw=1.5)
        ax.set_xscale('log', base=2)
        ax.set_xticks(FRESH_EXPECTED_SIZES)
        ax.set_xticklabels([rf'$2^{{{power}}}$' for power in FRESH_EXPECTED_POWERS])
        ax.set_ylim(-0.04, 1.04)
        ax.set_xlabel(r'Training images $N_{2D}$')
        ax.set_title(f'{feature} q95 novelty', fontsize=19, pad=12)
        ax.grid(axis='y', alpha=0.16)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
    axes[0].set_ylabel('q95 novelty score')
    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0.91),
               ncol=len(labels), frameon=False)
    fig.suptitle('Fresh DiT-L16 transition across optimizer milestones',
                 fontsize=23, fontweight='semibold', y=1.01)
    fig.text(
        0.5,
        0.945,
        'Every displayed line contains all ten data sizes; no legacy continuation files are used.',
        ha='center',
        color='0.35',
        fontsize=12.5,
    )
    out = QUICKCHECK_DIR / 'nf_generalize_fig2_dit_l16_fresh400k_checkpoint_trajectories_full.png'
    fig.savefig(out, bbox_inches='tight', dpi=300)
    plt.show()
    print('wrote', out)
    return out


fresh_checkpoint_trajectory = plot_fresh_l16_checkpoint_trajectories()
"""


INTERPRETATION = r"""### How to read the fresh sweep

The fixed-200k figure is the fair depth comparison. The final figure answers a
different question: what does the L16 transition look like after allowing it to
train through 400k updates? The DiT-L16 400k curve is the final longer-training
result; the 300k curve is an intermediate checkpoint. A rightward shift relative
to L12 would be evidence
consistent with a depth-dependent data requirement, but the code does not assume
or enforce that result. If the fresh L16 curve remains nonmonotonic, the next step
is to inspect generated maps, one-point distributions, power spectra, and
checkpoint loss histories before estimating any scaling relation.
"""

TAKEAWAYS_CODE = r"""def best_transition_lines(feature: str) -> list[str]:
    if transition_df.empty:
        return [f'- {feature}: fixed-200k transition table missing.']
    result = []
    for arch in DIT_ARCH_ORDER:
        sub = transition_df[
            (transition_df['feature'] == feature)
            & (transition_df['arch'] == arch)
            & (transition_df['score_col'] == 'gen_gl_q95')
        ]
        label = arch_label(arch)
        if sub.empty:
            result.append(f'- {feature} {label}: fixed-200k q95 value missing.')
            continue
        row = sub.iloc[0]
        if pd.isna(row['n_cross']):
            result.append(f'- {feature} {label}: fixed-200k N50 unavailable ({row["status"]}).')
        else:
            result.append(
                f'- {feature} {label}: fixed-200k q95 N50 = '
                f'2^{row["log2_n_cross"]:.2f} = {row["n_cross"]:.0f} '
                f'2D images ({row["status"]}).'
            )
    return result


sample_ok = None if manifest_df.empty else int(manifest_df['sample_exists'].sum())
sample_total = None if manifest_df.empty else len(manifest_df)

lines = ['### Notebook summary']
lines.append(
    '- Fixed-budget comparison: DiT-L8, DiT-L12/base, and fresh DiT-L16 '
    'are compared at 200k optimizer updates across all ten data sizes.'
)
if fresh_400k_complete:
    lines.append(
        '- **Fresh 400k status: complete.** The final longer-training L16 diagnostic '
        'uses all ten data sizes from $2^6$ through $2^{15}$ in '
        'both PCA and SSCD.'
    )
else:
    lines.append(
        '- **Fresh 400k status: incomplete.** The final longer-training L16 diagnostic '
        'is withheld until all ten data sizes are present in both '
        'PCA and SSCD; no legacy result is substituted.'
    )
lines.append(
    '- **Legacy continuation:** retained only as a failure-analysis record. '
    'It is not used for the fresh 400k curve or for a depth-scaling claim.'
)
if sample_ok is not None:
    lines.append(f'- Original fixed-200k sample audit: {sample_ok}/{sample_total} files found.')
lines.extend(best_transition_lines('PCA'))
lines.extend(best_transition_lines('SSCD'))
lines.extend([
    '- Read PCA and SSCD together. Agreement strengthens a novelty claim, '
    'but neither metric establishes physical validity.',
    '- A rightward L16 transition would be consistent with a larger data '
    'requirement. The fresh experiment measures this outcome; it does not '
    'assume or enforce it.',
    '- Inspect generated maps, one-point distributions, and power spectra '
    'before interpreting any high novelty score as successful generalization.',
])

display(Markdown('\n'.join(lines)))
"""


def update_notebook(path: Path = NOTEBOOK_PATH) -> None:
    notebook = json.loads(path.read_text())
    cells = notebook.get("cells", [])
    filtered = []
    skipping = False
    for cell in cells:
        source = "".join(cell.get("source", []))
        if source.startswith((SECTION_HEADING, LEGACY_SECTION_HEADING)):
            skipping = True
            continue
        if skipping and source.startswith("## Takeaways"):
            skipping = False
        if not skipping:
            filtered.append(cell)

    insert_at = next(
        index
        for index, cell in enumerate(filtered)
        if "".join(cell.get("source", [])).startswith("## Takeaways")
    )
    new_cells = [
        markdown_cell(INTRO),
        code_cell(AUDIT_CODE),
        code_cell(PLOT_CODE),
        code_cell(TRAJECTORY_CODE),
        markdown_cell(INTERPRETATION),
    ]
    updated_cells = filtered[:insert_at] + new_cells + filtered[insert_at:]
    takeaways_index = next(
        index
        for index, cell in enumerate(updated_cells)
        if "".join(cell.get("source", [])).startswith("## Takeaways")
    )
    if (
        takeaways_index + 1 >= len(updated_cells)
        or updated_cells[takeaways_index + 1].get("cell_type") != "code"
    ):
        raise ValueError("Expected a code cell immediately after the Takeaways heading")
    updated_cells[takeaways_index + 1] = code_cell(TAKEAWAYS_CODE)
    notebook["cells"] = updated_cells
    path.write_text(json.dumps(notebook, indent=1) + "\n")


if __name__ == "__main__":
    update_notebook()
    print(f"Updated {NOTEBOOK_PATH}")
