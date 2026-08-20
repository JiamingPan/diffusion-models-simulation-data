#!/usr/bin/env python3
"""Repair and extend the standalone DiT-L16 300k--500k notebook."""

from __future__ import annotations

import argparse
import copy
import json
import textwrap
from pathlib import Path


TAG = "dit-l16-outlier-excluded-v1"


def _source(text: str) -> list[str]:
    return textwrap.dedent(text).strip("\n").splitlines(keepends=True)


def _cell(cell_type: str, cell_id: str, source: str) -> dict:
    cell = {
        "cell_type": cell_type,
        "id": cell_id,
        "metadata": {"tags": [TAG]},
        "source": _source(source),
    }
    if cell_type == "code":
        cell.update({"execution_count": None, "outputs": []})
    return cell


def build_cells() -> list[dict]:
    return [
        _cell(
            "markdown",
            "outlier-excluded-method",
            r"""
            ### 11.1 Outlier-excluded sensitivity: novelty, one-point PDF, and power spectrum

            This section applies the **same predeclared two-sided 4.5-MAD rule at k-bin 60**
            to every dataset size and checkpoint, then excludes every flagged generated sample.
            It is a post-hoc sensitivity analysis; the unfiltered ensemble result above remains
            the primary result.

            The physical reference is still the **exact configured training-subset mean**.
            A generated-sample median is shown only as a robustness diagnostic and is not used
            as the truth. The one-point PDF and power spectrum can be recomputed exactly after
            exclusion. The PCA/SSCD tables contain only aggregate q95 novelty, not per-sample
            copy labels, so the filtered novelty score is reported as a mathematically exact
            **feasible interval** rather than an invented point estimate. Exact filtered novelty
            would require saving or recomputing the per-sample nearest-neighbor scores.
            """,
        ),
        _cell(
            "code",
            "outlier-excluded-analysis",
            r"""
            from simdiff_eval.outlier_sensitivity import (
                filtered_histogram_probability,
                filtered_power_summary,
                novelty_bounds_after_filtering,
            )

            filtered_physics_rows = []
            filtered_physics_curves = {}

            for updates_k in CONT_UPDATES_K:
                for tag in CONT_TAGS:
                    row = continuation_row(tag, updates_k)
                    key = f'{tag}_{updates_k}k'
                    analysis = outlier_analysis[(tag, updates_k)]
                    samples = load_samples(analysis['sample_path'])
                    keep = ~np.asarray(analysis['outlier_mask'], dtype=bool)
                    if len(keep) != len(samples):
                        raise RuntimeError(f'Outlier mask length mismatch for {tag}/{updates_k}k')

                    generated_hist = filtered_histogram_probability(
                        samples,
                        bins=hist_edges,
                        keep_mask=keep,
                    )
                    real_hist = np.asarray(
                        continuation_curves[f'{key}_real_hist_probability'], dtype=float
                    )
                    power = filtered_power_summary(
                        analysis['pk_ratio'],
                        keep_mask=keep,
                    )
                    filtered_physics_rows.append({
                        'dataset_tag': tag,
                        'dataset_size': int(row['dataset_size']),
                        'updates_k': updates_k,
                        'n_total': int(len(keep)),
                        'n_removed': int((~keep).sum()),
                        'n_kept': int(keep.sum()),
                        'retention_fraction': float(keep.mean()),
                        'hist_l1': float(np.abs(real_hist - generated_hist).sum()),
                        'pk_log10_mae': power['log10_mae'],
                    })
                    filtered_physics_curves[(tag, updates_k)] = {
                        'generated_hist_probability': generated_hist,
                        'pk_mean_ratio': power['mean'],
                        'pk_median_ratio': power['median'],
                        'pk_variance': power['variance'],
                        'kbins': np.asarray(analysis['kbins'], dtype=float),
                    }

            outlier_excluded_physics = pd.DataFrame(filtered_physics_rows).sort_values(
                ['updates_k', 'dataset_size']
            )
            if len(outlier_excluded_physics) != 60:
                raise RuntimeError(
                    f'Expected 60 outlier-excluded physics rows; found {len(outlier_excluded_physics)}'
                )

            filter_counts = outlier_excluded_physics[
                ['dataset_tag', 'dataset_size', 'updates_k', 'n_total', 'n_removed', 'n_kept']
            ]
            novelty_with_counts = continuation_novelty.merge(
                filter_counts,
                on=['dataset_tag', 'dataset_size', 'updates_k'],
                how='inner',
                validate='many_to_one',
            )
            novelty_bound_rows = []
            for _, novelty_row in novelty_with_counts.iterrows():
                bounds = novelty_bounds_after_filtering(
                    n_total=int(novelty_row['n_total']),
                    n_removed=int(novelty_row['n_removed']),
                    novelty_score=float(novelty_row['gen_gl_q95']),
                )
                novelty_bound_rows.append({
                    'dataset_tag': str(novelty_row['dataset_tag']),
                    'dataset_size': int(novelty_row['dataset_size']),
                    'updates_k': int(novelty_row['updates_k']),
                    'feature': str(novelty_row['feature']),
                    'original_gen_gl_q95': float(novelty_row['gen_gl_q95']),
                    **bounds,
                })
            outlier_excluded_novelty_bounds = pd.DataFrame(novelty_bound_rows).sort_values(
                ['feature', 'updates_k', 'dataset_size']
            )
            if len(outlier_excluded_novelty_bounds) != 120:
                raise RuntimeError(
                    'Expected 120 filtered novelty-bound rows; found '
                    f'{len(outlier_excluded_novelty_bounds)}'
                )

            filtered_physics_path = OUTPUT_DIR / 'outlier_excluded_physics_summary.csv'
            filtered_novelty_path = OUTPUT_DIR / 'outlier_excluded_novelty_bounds.csv'
            outlier_excluded_physics.to_csv(filtered_physics_path, index=False)
            outlier_excluded_novelty_bounds.to_csv(filtered_novelty_path, index=False)
            print('wrote', filtered_physics_path)
            print('wrote', filtered_novelty_path)
            display(outlier_excluded_physics[
                ['dataset_tag', 'updates_k', 'n_removed', 'n_kept', 'retention_fraction']
            ])

            # The new novelty curve is an identified interval because the saved PCA/SSCD
            # tables do not identify which generated samples were classified as copies.
            fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharey=True, constrained_layout=True)
            for axis, feature in zip(axes, CONT_FEATURES):
                current = outlier_excluded_novelty_bounds[
                    (outlier_excluded_novelty_bounds['feature'] == feature)
                    & (outlier_excluded_novelty_bounds['updates_k'] == 500)
                ].sort_values('dataset_size')
                x = np.log2(current['dataset_size'].to_numpy(dtype=float))
                original = current['original_gen_gl_q95'].to_numpy(dtype=float)
                lower = current['novelty_lower'].to_numpy(dtype=float)
                upper = current['novelty_upper'].to_numpy(dtype=float)
                axis.fill_between(
                    x, lower, upper, color='#B33C86', alpha=0.25,
                    label='feasible interval after exclusion',
                )
                axis.plot(
                    x, original, color='black', ls='--', marker='o', lw=1.7,
                    label='original aggregate score',
                )
                axis.axhline(0.5, color='0.5', ls=':', lw=1)
                axis.set_title(f'{feature}: 500k novelty after outlier exclusion', fontweight='semibold')
                axis.set_xlabel(r'Training images $N_{2D}$')
                axis.set_xticks(CONT_POWERS, [rf'$2^{{{power}}}$' for power in CONT_POWERS])
                axis.set_ylim(-0.03, 1.03)
                axis.grid(alpha=0.15)
            axes[0].set_ylabel('q95 novelty score')
            axes[0].legend(frameon=False, fontsize=10)
            fig.suptitle(
                'Outlier-excluded novelty is bounded, not point-identified by aggregate tables',
                fontsize=19, fontweight='semibold',
            )
            save_figure(fig, 'outlier_excluded_novelty_bounds_500k.png')
            plt.show()

            # Exact one-point recomputation on retained samples.
            fig, axes = plt.subplots(2, 5, figsize=(19, 8), constrained_layout=True)
            for axis, tag in zip(axes.flat, CONT_TAGS):
                key = f'{tag}_500k'
                real = np.asarray(continuation_curves[f'{key}_real_hist_probability'], dtype=float)
                generated = filtered_physics_curves[(tag, 500)]['generated_hist_probability']
                retained = outlier_excluded_physics[
                    (outlier_excluded_physics['dataset_tag'] == tag)
                    & (outlier_excluded_physics['updates_k'] == 500)
                ].iloc[0]
                axis.plot(hist_centers, real, color='black', lw=2.1, label='exact training-subset mean')
                axis.plot(hist_centers, generated, color='#B33C86', lw=2.1, label='retained generated samples')
                axis.set_yscale('log')
                axis.set_title(
                    f'{dataset_label(int(retained["dataset_size"]))}\n'
                    f'{int(retained["n_kept"])}/{int(retained["n_total"])} kept'
                )
                axis.set_xlabel('Normalized field value')
                axis.grid(alpha=0.15)
            for axis in axes[:, 0]:
                axis.set_ylabel('Pixel probability')
            axes[0, 0].legend(frameon=False, fontsize=9)
            fig.suptitle('500k one-point distributions after fixed outlier exclusion', fontsize=20, fontweight='semibold')
            save_figure(fig, 'outlier_excluded_one_point_500k.png')
            plt.show()

            # Exact retained-sample mean P(k); median is diagnostic only.
            all_power_values = np.concatenate([
                filtered_physics_curves[(tag, 500)]['pk_mean_ratio']
                for tag in CONT_TAGS
            ])
            finite_power_values = all_power_values[np.isfinite(all_power_values)]
            ymax = max(2.0, float(finite_power_values.max() * 1.08))
            fig, axes = plt.subplots(2, 5, figsize=(19, 8), sharey=True, constrained_layout=True)
            for axis, tag in zip(axes.flat, CONT_TAGS):
                curves = filtered_physics_curves[(tag, 500)]
                retained = outlier_excluded_physics[
                    (outlier_excluded_physics['dataset_tag'] == tag)
                    & (outlier_excluded_physics['updates_k'] == 500)
                ].iloc[0]
                axis.plot(
                    curves['kbins'], curves['pk_mean_ratio'], color='#B33C86', lw=2.1,
                    label='retained-sample mean',
                )
                axis.plot(
                    curves['kbins'], curves['pk_median_ratio'], color='#0072B2', ls=':', lw=1.7,
                    label='generated-sample median (diagnostic)',
                )
                axis.axhline(1, color='black', ls='--', lw=1.1, label='exact real-subset mean')
                axis.set_title(
                    f'{dataset_label(int(retained["dataset_size"]))}\n'
                    f'{int(retained["n_kept"])}/{int(retained["n_total"])} kept'
                )
                axis.set_xlabel(r'$k$ bin')
                axis.set_ylim(0, ymax)
                axis.grid(alpha=0.15)
            for axis in axes[:, 0]:
                axis.set_ylabel(r'$P_g(k)/P_r(k)$')
            axes[0, 0].legend(frameon=False, fontsize=8.5)
            fig.suptitle('500k power spectra after fixed outlier exclusion', fontsize=20, fontweight='semibold')
            save_figure(fig, 'outlier_excluded_power_spectrum_500k.png')
            plt.show()

            metric_heatmap(
                outlier_excluded_physics,
                'hist_l1',
                'Outlier-excluded one-point distribution error',
                'outlier_excluded_one_point_error_heatmap.png',
                r'$L_1$ error',
            )
            metric_heatmap(
                outlier_excluded_physics,
                'pk_log10_mae',
                'Outlier-excluded power-spectrum error',
                'outlier_excluded_power_spectrum_error_heatmap.png',
                r'mean $|\log_{10}(P_g/P_r)|$',
            )
            """,
        ),
        _cell(
            "markdown",
            "outlier-excluded-interpretation",
            r"""
            **How to read this sensitivity analysis.** If the retained-sample mean remains far
            from one after the flagged tail is removed, the mismatch is not explained by a few
            catastrophic generations; it is a broader distributional failure. A median closer
            to one than the retained mean diagnoses skewness, but it does not replace the
            ensemble mean required for power-spectrum fidelity. The novelty band is deliberately
            an interval because the current metric tables do not preserve sample-level copy labels.
            """,
        ),
    ]


def _repair_merge(cell: dict) -> bool:
    source = "".join(cell.get("source", []))
    if "how='validate'" not in source:
        return False
    cell["source"] = source.replace("how='validate'", "how='inner'").splitlines(keepends=True)
    if cell.get("cell_type") == "code":
        cell["execution_count"] = None
        cell["outputs"] = []
    return True


def transform_notebook(notebook: dict) -> dict:
    """Return the canonical notebook with the outlier sensitivity section applied."""

    notebook = copy.deepcopy(notebook)
    cells = [
        copy.deepcopy(cell)
        for cell in notebook["cells"]
        if TAG not in cell.get("metadata", {}).get("tags", [])
    ]
    repaired = sum(_repair_merge(cell) for cell in cells)
    corrected_merges = sum(
        "how='inner'" in "".join(cell.get("source", []))
        and "validate='many_to_one'" in "".join(cell.get("source", []))
        for cell in cells
    )
    if corrected_merges != 1:
        raise RuntimeError(
            "Expected exactly one erroneous or already-corrected validated merge; "
            f"found {corrected_merges} after repairing {repaired}"
        )

    anchors = [
        index for index, cell in enumerate(cells)
        if "outlier_analysis = {}" in "".join(cell.get("source", []))
    ]
    if len(anchors) != 1:
        raise RuntimeError(f"Expected one outlier-analysis anchor; found {len(anchors)}")
    insertion = anchors[0] + 1
    cells[insertion:insertion] = build_cells()
    notebook["cells"] = cells
    return notebook


def update_notebook(input_path: Path, output_path: Path) -> None:
    notebook = transform_notebook(json.loads(Path(input_path).read_text()))
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(notebook, indent=1, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    update_notebook(args.input, args.output)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
