#!/usr/bin/env python
"""Replace the stale DiT-L16 section with the clean 300k v2 result audit."""

from __future__ import annotations

import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = REPO_ROOT / "notebooks" / "nf_generalize_fig2_dit_results.ipynb"
SECTION_HEADINGS = (
    "## Fresh DiT-L16 sweep through 400k updates",
    "## Fresh DiT-L16 sweep through 300k updates",
    "## Fresh DiT-L16 300k replacement sweep",
)


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


INTRO = r"""## Fresh DiT-L16 300k replacement sweep

This section uses the clean replacement experiment, not the failed staged
continuation. Ten DiT-L16 models start from new seed-123 initializations and
train directly to 300k optimizer updates, one for every training-set size from
$2^6$ through $2^{15}$.

DiT-L8 and DiT-L12 use their original 200k runs. DiT-L16 uses the clean 300k
replacement sweep. This is therefore a depth comparison at unequal training
budgets, labeled explicitly in the figure. No failed continuation or old L16
table is substituted. The q95 novelty score tests proximity to training
examples; **q95 novelty does not guarantee physical fidelity**.
"""


AUDIT_CODE = r"""FRESH_SWEEP_NAME = 'nf_generalize_fig2_dit_l16_fresh300k_v2'
FRESH_EXPECTED_POWERS = list(range(6, 16))
FRESH_EXPECTED_TAGS = [f'd2p{power:02d}' for power in FRESH_EXPECTED_POWERS]
FRESH_EXPECTED_SIZES = [2 ** power for power in FRESH_EXPECTED_POWERS]


def fresh_300k_v2_metric_path(feature: str) -> Path:
    return TABLE_DIR / (
        f'{FRESH_SWEEP_NAME}_{feature.lower()}_full_nn_metrics.csv'
    )


def audit_fresh_300k_v2_table(feature: str) -> tuple[pd.DataFrame, dict[str, Any]]:
    path = fresh_300k_v2_metric_path(feature)
    audit_row: dict[str, Any] = {
        'feature': feature,
        'path': rel(path),
        'exists': path.exists(),
        'rows': 0,
        'missing_tags': list(FRESH_EXPECTED_TAGS),
        'duplicate_tags': [],
        'complete': False,
    }
    if not path.exists():
        return pd.DataFrame(), audit_row

    table = ensure_arch_columns(add_generalization_columns(pd.read_csv(path)))
    table = table[table['arch'].astype(str) == 'dit_l16'].copy()
    table = table.sort_values('dataset_size')
    tags = table['dataset_tag'].dropna().astype(str)
    missing = sorted(set(FRESH_EXPECTED_TAGS) - set(tags))
    extra = sorted(set(tags) - set(FRESH_EXPECTED_TAGS))
    duplicates = sorted(tags[tags.duplicated(keep=False)].unique().tolist())
    complete = (
        len(table) == 10
        and not missing
        and not extra
        and not duplicates
        and 'gen_gl_q95' in table.columns
        and table['gen_gl_q95'].notna().all()
    )
    audit_row.update({
        'rows': len(table),
        'missing_tags': missing,
        'extra_tags': extra,
        'duplicate_tags': duplicates,
        'complete': bool(complete),
    })
    return table, audit_row


fresh_300k_v2_metrics: dict[str, pd.DataFrame] = {}
fresh_300k_v2_audit_rows: list[dict[str, Any]] = []
for feature in ('PCA', 'SSCD'):
    table, audit_row = audit_fresh_300k_v2_table(feature)
    fresh_300k_v2_metrics[feature] = table
    fresh_300k_v2_audit_rows.append(audit_row)

fresh_300k_v2_audit_df = pd.DataFrame(fresh_300k_v2_audit_rows)
display(Markdown('### Clean replacement table audit'))
display(fresh_300k_v2_audit_df)

fresh_300k_v2_complete = (
    len(fresh_300k_v2_audit_df) == 2
    and bool(fresh_300k_v2_audit_df['complete'].all())
)
if fresh_300k_v2_complete:
    display(Markdown(
        '**Fresh 300k v2 audit passed.** All ten data sizes are present exactly '
        'once in both PCA and SSCD.'
    ))
else:
    display(Markdown(
        '**Fresh 300k v2 audit incomplete: not drawing the replacement L16 '
        'curve.** Both tables must contain all ten data sizes exactly once.'
    ))
"""


PLOT_CODE = r"""FRESH_DEPTH_COLORS = {
    'dit_l8': '#009E73',
    'dit_base': '#0072B2',
    'dit_l16': '#B33C86',
}
FRESH_DEPTH_MARKERS = {'dit_l8': 'P', 'dit_base': 'D', 'dit_l16': 'X'}
FRESH_DEPTH_LABELS = {
    'dit_l8': 'DiT-L8 200k',
    'dit_base': 'DiT-L12 / base 200k',
    'dit_l16': 'DiT-L16 300k',
}


def require_fresh_complete_curve(
    table: pd.DataFrame,
    arch: str,
    *,
    context: str,
) -> pd.DataFrame:
    sub = table[table['arch'].astype(str) == arch].copy()
    sub = sub.sort_values('dataset_size')
    tags = sub['dataset_tag'].dropna().astype(str)
    missing = sorted(set(FRESH_EXPECTED_TAGS) - set(tags))
    extra = sorted(set(tags) - set(FRESH_EXPECTED_TAGS))
    if (
        len(sub) != 10
        or missing
        or extra
        or tags.duplicated().any()
        or 'gen_gl_q95' not in sub.columns
        or sub['gen_gl_q95'].isna().any()
    ):
        raise ValueError(
            f'{context} is incomplete: rows={len(sub)}, '
            f'missing={missing}, extra={extra}'
        )
    return sub


def fresh_depth_sources(feature: str) -> dict[str, pd.DataFrame]:
    baseline = pca_metrics if feature == 'PCA' else sscd_metrics
    return {
        'dit_l8': require_fresh_complete_curve(
            baseline, 'dit_l8', context=f'{feature} DiT-L8 200k'
        ),
        'dit_base': require_fresh_complete_curve(
            baseline, 'dit_base', context=f'{feature} DiT-L12 200k'
        ),
        'dit_l16': require_fresh_complete_curve(
            fresh_300k_v2_metrics[feature],
            'dit_l16',
            context=f'{feature} fresh DiT-L16 300k v2',
        ),
    }


def plot_fresh_300k_v2_depth_comparison(
    *,
    output_name: str,
    zoom_max_power: int | None = None,
) -> Path:
    fig, axes = plt.subplots(
        1, 2, figsize=(16.2, 6.2), sharey=True, constrained_layout=True
    )
    for ax, feature in zip(axes, ('PCA', 'SSCD')):
        for arch, sub in fresh_depth_sources(feature).items():
            ax.plot(
                sub['dataset_size'],
                sub['gen_gl_q95'],
                color=FRESH_DEPTH_COLORS[arch],
                marker=FRESH_DEPTH_MARKERS[arch],
                ms=9,
                lw=3,
                label=FRESH_DEPTH_LABELS[arch],
            )
        ax.axhline(0.5, color='0.35', ls=':', lw=1.6)
        shown_powers = [
            power for power in FRESH_EXPECTED_POWERS
            if zoom_max_power is None or power <= zoom_max_power
        ]
        ax.set_xscale('log', base=2)
        ax.set_xticks([2 ** power for power in shown_powers])
        ax.set_xticklabels([rf'$2^{{{power}}}$' for power in shown_powers])
        if zoom_max_power is not None:
            ax.set_xlim(2 ** 5.75, 2 ** (zoom_max_power + 0.25))
        ax.set_ylim(-0.04, 1.04)
        ax.set_xlabel(r'Training images $N_{2D}$')
        ax.set_title(f'{feature} q95 novelty', fontsize=19, pad=12)
        ax.grid(axis='y', alpha=0.18)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
    axes[0].set_ylabel('q95 novelty score')
    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(
        handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0.91),
        ncol=3, frameon=False
    )
    title = 'Clean DiT depth comparison'
    if zoom_max_power is not None:
        title += ': transition region'
    fig.suptitle(title, fontsize=23, fontweight='semibold', y=1.01)
    fig.text(
        0.5, 0.945,
        'L8 and L12 use 200k updates; fresh L16 uses 300k updates.',
        ha='center', color='0.35', fontsize=12.5
    )
    out = QUICKCHECK_DIR / output_name
    fig.savefig(out, bbox_inches='tight', dpi=300)
    plt.show()
    print('wrote', out)
    return out


fresh_300k_v2_outputs = []
if fresh_300k_v2_complete:
    fresh_300k_v2_outputs.append(plot_fresh_300k_v2_depth_comparison(
        output_name='nf_generalize_fig2_dit_l16_fresh300k_v2_depth_comparison_full.png'
    ))
    fresh_300k_v2_outputs.append(plot_fresh_300k_v2_depth_comparison(
        output_name='nf_generalize_fig2_dit_l16_fresh300k_v2_depth_comparison_zoom.png',
        zoom_max_power=11,
    ))
"""


INTERPRETATION = r"""### Interpretation

The full-range figure answers whether the clean L16 transition is monotonic
across all ten data sizes. The zoomed figure shows the transition region without
hiding the high-data runs. A rightward L16 transition would be consistent with a
larger data requirement, but the experiment measures that outcome; it does not
assume or enforce it. Read PCA and SSCD together and verify generated maps,
one-point distributions, and power spectra before treating high novelty as a
successful scientific model.
"""


TAKEAWAYS_CODE = r"""lines = ['### Notebook summary']
if fresh_300k_v2_complete:
    lines.append(
        '- **Fresh 300k v2 status: complete.** PCA and SSCD both contain '
        'all ten data sizes from $2^6$ through $2^{15}$.'
    )
else:
    lines.append(
        '- **Fresh 300k v2 status: incomplete.** The replacement L16 depth '
        'curve is withheld until PCA and SSCD both contain all ten data sizes.'
    )
lines.extend([
    '- The plotted L16 line comes only from the clean replacement sweep.',
    '- DiT-L8 and DiT-L12 use 200k updates; DiT-L16 uses 300k, so this is '
    'not an equal-compute comparison.',
    '- PCA and SSCD measure novelty relative to training examples. They do '
    'not establish physical validity by themselves.',
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
        if source.startswith(SECTION_HEADINGS):
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
        markdown_cell(INTERPRETATION),
    ]
    updated = filtered[:insert_at] + new_cells + filtered[insert_at:]
    takeaways_index = next(
        index
        for index, cell in enumerate(updated)
        if "".join(cell.get("source", [])).startswith("## Takeaways")
    )
    if (
        takeaways_index + 1 >= len(updated)
        or updated[takeaways_index + 1].get("cell_type") != "code"
    ):
        raise ValueError("Expected a code cell after the Takeaways heading")
    updated[takeaways_index + 1] = code_cell(TAKEAWAYS_CODE)
    notebook["cells"] = updated
    path.write_text(json.dumps(notebook, indent=1) + "\n")


if __name__ == "__main__":
    update_notebook()
    print(f"Updated {NOTEBOOK_PATH}")
