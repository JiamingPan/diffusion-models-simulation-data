#!/usr/bin/env python3
"""Make the DiT loss plot use audited fresh 300k L16 histories."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS = [
    ROOT / "notebooks" / "nf_generalize_fig2_dit_results.ipynb",
    ROOT / "notebooks" / "nf_generalize_fig2_dit_results_explained.ipynb",
]
OUTPUT_NAME = "nf_generalize_fig2_dit_training_curves.png"


FRESH_BLOCK = '''    # Load the independent L16 replacement sweep. A history is used only
    # when its recorded optimizer-update count reaches the requested budget.
    FRESH_LOSS_SWEEP_NAME = 'nf_generalize_fig2_dit_l16_fresh300k_v2'
    fresh_loss_manifest_path = (
        PROJECT_DIR / 'local' / FRESH_LOSS_SWEEP_NAME / 'manifest.json'
    )
    fresh_loss_obj = read_json(fresh_loss_manifest_path)
    if isinstance(fresh_loss_obj, dict):
        fresh_loss_rows = fresh_loss_obj.get('runs', [])
    elif isinstance(fresh_loss_obj, list):
        fresh_loss_rows = fresh_loss_obj
    else:
        fresh_loss_rows = []

    def explicit_update_count(metrics: dict[str, Any]) -> int | None:
        for key in ('optimizer_updates', 'optimizer_step', 'global_step', 'num_updates', 'updates'):
            if key in metrics:
                values = flatten_numeric(metrics.get(key))
                if len(values):
                    return int(round(values[-1]))
        return None

    fresh_loss_by_tag: dict[str, dict[str, Any]] = {}
    fresh_loss_audit_rows: list[dict[str, Any]] = []
    fresh_loss_df = pd.DataFrame(fresh_loss_rows)
    if not fresh_loss_df.empty:
        if 'run_name' not in fresh_loss_df.columns and 'name' in fresh_loss_df.columns:
            fresh_loss_df['run_name'] = fresh_loss_df['name']
        if 'dataset_tag' not in fresh_loss_df.columns:
            fresh_loss_df['dataset_tag'] = fresh_loss_df['run_name'].map(dataset_tag_from_name)
        if 'dataset_size' not in fresh_loss_df.columns:
            fresh_loss_df['dataset_size'] = fresh_loss_df['dataset_tag'].map(dataset_size_from_tag)

        for _, fresh_row in fresh_loss_df.sort_values('dataset_size').iterrows():
            dataset_tag = str(fresh_row.get('dataset_tag', '') or '')
            steps_per_epoch = max(1, int(fresh_row.get('steps_per_epoch', 1) or 1))
            target_updates = int(fresh_row.get('target_total_updates', 300000) or 300000)
            metrics, metrics_path = read_latest_metrics(fresh_row)
            epoch_loss = flatten_numeric(metrics.get('epoch_loss'))
            computed_updates = len(epoch_loss) * steps_per_epoch
            recorded_updates = explicit_update_count(metrics) or computed_updates
            use_fresh_300k = recorded_updates >= 0.98 * target_updates
            if use_fresh_300k:
                fresh_loss_by_tag[dataset_tag] = {
                    'metrics': metrics,
                    'metrics_path': metrics_path,
                    'epoch_loss': epoch_loss,
                    'batch_loss': flatten_numeric(metrics.get('loss', metrics.get('batch_loss'))),
                    'epoch_lr': flatten_numeric(metrics.get('epoch_lr', metrics.get('lr'))),
                    'steps_per_epoch': steps_per_epoch,
                    'optimizer_updates_recorded': recorded_updates,
                    'source_label': 'fresh 300k v2',
                }
            fresh_loss_audit_rows.append({
                'run_name': fresh_row.get('run_name'),
                'dataset_tag': dataset_tag,
                'target_updates': target_updates,
                'optimizer_updates_recorded': recorded_updates,
                'using_fresh_300k_history': use_fresh_300k,
                'metrics_path': rel(metrics_path) if metrics_path else None,
            })

    fresh_loss_audit_df = pd.DataFrame(fresh_loss_audit_rows)
    display(Markdown('### Fresh 300k v2 loss-source audit'))
    if fresh_loss_audit_df.empty:
        display(Markdown(
            f'**Missing fresh loss manifest:** `{rel(fresh_loss_manifest_path)}`. '
            'The plot will retain verified 200k histories and will not claim 300k.'
        ))
    else:
        display(fresh_loss_audit_df)

    loss_plot_summary_df = loss_df.copy()
    loss_plot_summary_df['loss_source'] = '200k original'
    for dataset_tag, fresh_source in fresh_loss_by_tag.items():
        mask = (
            (loss_plot_summary_df['arch'].astype(str) == 'dit_l16')
            & (loss_plot_summary_df['dataset_tag'].astype(str) == dataset_tag)
        )
        epoch_loss = np.asarray(fresh_source['epoch_loss'], dtype=float)
        tail_window = max(1, min(len(epoch_loss), max(5, len(epoch_loss) // 20)))
        loss_plot_summary_df.loc[mask, 'epochs_completed'] = len(epoch_loss)
        loss_plot_summary_df.loc[mask, 'optimizer_updates_recorded'] = fresh_source[
            'optimizer_updates_recorded'
        ]
        loss_plot_summary_df.loc[mask, 'tail_median_epoch_loss'] = float(
            np.nanmedian(epoch_loss[-tail_window:])
        )
        loss_plot_summary_df.loc[mask, 'best_epoch_loss'] = float(np.nanmin(epoch_loss))
        loss_plot_summary_df.loc[mask, 'loss_source'] = 'fresh 300k v2'
'''


def find_loss_cell(notebook: dict) -> dict:
    for cell in notebook["cells"]:
        source = "".join(cell.get("source", []))
        if OUTPUT_NAME in source and "fig.savefig" in source:
            return cell
    raise RuntimeError("Could not find the DiT training-loss plotting cell")


def replace_once(source: str, old: str, new: str, label: str) -> str:
    count = source.count(old)
    if count != 1:
        raise RuntimeError(f"Expected one {label} block, found {count}")
    return source.replace(old, new, 1)


def update_loss_source(source: str) -> str:
    legacy_marker = "    # L16 continuation metrics through 300k live in isolated checkpoint directories."
    fresh_marker = "    FRESH_LOSS_SWEEP_NAME = 'nf_generalize_fig2_dit_l16_fresh300k_v2'"
    if legacy_marker in source:
        start = source.index(legacy_marker)
        end = source.index("    loss_summary_cols", start)
        source = source[:start] + FRESH_BLOCK + source[end:]
    elif fresh_marker not in source:
        anchor = "    loss_df = pd.DataFrame(loss_rows).sort_values(['arch', 'dataset_size'])\n"
        source = replace_once(source, anchor, anchor + "\n" + FRESH_BLOCK, "loss dataframe")

    source = source.replace(
        "fig, axes = plt.subplots(1, len(selected_tags), figsize=(18.0, 6.8), sharex=True, sharey=True)",
        "fig, axes = plt.subplots(1, len(selected_tags), figsize=(18.0, 6.8), sharex=False, sharey=True)",
    )

    old_choice = '''                loss_source = loss_by_run.get(run_name, {})
                if arch == 'dit_l16' and run_name in continuation_loss_by_run:
                    loss_source = continuation_loss_by_run[run_name]
'''
    new_choice = '''                loss_source = loss_by_run.get(run_name, {})
                if arch == 'dit_l16' and dataset_tag in fresh_loss_by_tag:
                    loss_source = fresh_loss_by_tag[dataset_tag]
'''
    if old_choice in source:
        source = replace_once(source, old_choice, new_choice, "L16 history selection")
    elif new_choice not in source:
        raise RuntimeError("Could not find the L16 history-selection block")

    old_steps = "                steps_per_epoch = max(1, int(row.get('steps_per_epoch', 1) or 1))\n"
    new_steps = """                steps_per_epoch = max(1, int(
                    loss_source.get('steps_per_epoch', row.get('steps_per_epoch', 1)) or 1
                ))
"""
    if old_steps in source:
        source = replace_once(source, old_steps, new_steps, "steps per epoch")
    elif new_steps not in source:
        raise RuntimeError("Could not find the steps-per-epoch block")

    old_title = '''            l16_source = next((source for arch, source, _ in panel_sources if arch == 'dit_l16'), '200k original')
            representative_epochs = next((epochs for arch, _, epochs in panel_sources if arch == 'dit_l16'), 0)
            budget_note = 'L16: 300k' if l16_source == '300k continuation' else 'all: 200k'
            ax.set_title(
                f'$N_{{2D}}=2^{{{dataset_exponent}}}$\\n{budget_note}; L16 {representative_epochs:,} epochs',
                fontsize=17, pad=12,
            )
'''
    new_title = '''            l16_source = fresh_loss_by_tag.get(dataset_tag)
            if l16_source is not None:
                representative_epochs = len(l16_source['epoch_loss'])
                l16_budget_k = int(round(l16_source['optimizer_updates_recorded'] / 1000.0))
                budget_note = f'L16: {l16_budget_k}k'
            else:
                representative_epochs = next(
                    (epochs for arch, _, epochs in panel_sources if arch == 'dit_l16'), 0
                )
                budget_note = 'L16: verified 200k only'
            ax.set_title(
                f'$N_{{2D}}=2^{{{dataset_exponent}}}$\\n{budget_note}; {representative_epochs:,} epochs',
                fontsize=17, pad=12,
            )
'''
    if old_title in source:
        source = replace_once(source, old_title, new_title, "panel budget title")
    elif new_title not in source:
        raise RuntimeError("Could not find the panel budget-title block")

    source = source.replace(
        "label=(arch_label(a) + (' (300k where available)' if a == 'dit_l16' else ' (200k)'))",
        "label=(arch_label(a) + (' (fresh 300k)' if a == 'dit_l16' else ' (200k)'))",
        1,
    )
    source = source.replace(
        "fig.suptitle('DiT optimization with L16 continued to 300k'",
        "fig.suptitle('DiT optimization: L8/L12 at 200k, fresh L16 at 300k'",
        1,
    )
    source = source.replace(
        "r'L16 uses 300k updates for $2^6$--$2^{10}$; L8, L12, and the $2^{15}$ panel use 200k.'",
        "f'Fresh L16 histories accepted for {len(fresh_loss_by_tag)}/10 data sizes; x-axes show recorded optimizer updates.'",
        1,
    )
    source = source.replace(
        "sub = loss_df[loss_df['arch'].astype(str) == arch].sort_values('dataset_size')",
        "sub = loss_plot_summary_df[loss_plot_summary_df['arch'].astype(str) == arch].sort_values('dataset_size')",
        1,
    )
    source = source.replace(
        "ax.set_title('Final training loss is driven mainly by dataset size'",
        "ax.set_title('Tail loss at verified budgets: L8/L12 200k, fresh L16 300k'",
        1,
    )
    source = source.replace(
        "selected_summary = loss_df[\n            loss_df['arch'].isin(DIT_ARCH_ORDER) & loss_df['dataset_tag'].isin(selected_tags)\n        ][['arch_label', 'dataset_tag', 'dataset_size', 'epochs_completed', 'tail_median_epoch_loss']]",
        "selected_summary = loss_plot_summary_df[\n            loss_plot_summary_df['arch'].isin(DIT_ARCH_ORDER)\n            & loss_plot_summary_df['dataset_tag'].isin(selected_tags)\n        ][['arch_label', 'dataset_tag', 'dataset_size', 'epochs_completed',\n           'optimizer_updates_recorded', 'tail_median_epoch_loss', 'loss_source']]",
        1,
    )
    source = source.replace(
        """            '**Interpretation:** the first two panels intentionally show the extended 300k L16 history; '
            'the rightmost panel remains a fixed-200k comparison because no $2^{15}$ continuation exists. '
            'The inset is a linear-scale zoom of that rightmost panel after 20k updates. '
""",
        """            '**Interpretation:** each L16 panel uses the independent fresh 300k history only after '
            'the audit verifies its recorded update count. L8 and L12 remain at 200k. '
            'The inset is a linear-scale zoom of the rightmost panel after 20k updates. '
""",
        1,
    )

    required = [
        fresh_marker,
        "recorded_updates >= 0.98 * target_updates",
        "loss_source = fresh_loss_by_tag[dataset_tag]",
        "L16: {l16_budget_k}k",
        "Fresh 300k v2 loss-source audit",
    ]
    missing = [token for token in required if token not in source]
    if missing:
        raise RuntimeError(f"Updated cell is missing required tokens: {missing}")
    return source


def main() -> None:
    template_notebook = json.loads(NOTEBOOKS[1].read_text())
    template_cell = find_loss_cell(template_notebook)
    updated_source = update_loss_source("".join(template_cell.get("source", [])))

    for path in NOTEBOOKS:
        notebook = json.loads(path.read_text())
        cell = find_loss_cell(notebook)
        cell["source"] = updated_source.splitlines(keepends=True)
        cell["outputs"] = []
        cell["execution_count"] = None
        path.write_text(json.dumps(notebook, indent=1, ensure_ascii=False) + "\n")
        print(f"updated {path}")


if __name__ == "__main__":
    main()
