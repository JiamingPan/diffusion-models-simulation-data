#!/usr/bin/env python
"""Add paper-ready PDF exports to the conditional VGG results notebook."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


TAG = "conditional-paper-figures"


def _cell(cell_type: str, source: str) -> dict:
    cell = {
        "cell_type": cell_type,
        "metadata": {"tags": [TAG]},
        "source": [line + "\n" for line in source.strip().splitlines()],
    }
    if cell_type == "code":
        cell.update({"execution_count": None, "outputs": []})
    return cell


def _cells() -> list[dict]:
    return [
        _cell(
            "markdown",
            r"""
## Paper-ready verification figures

This cell exports the conditional-recovery result and its comparison figures at the exact paper width.
The conditional figure reports full empirical calibration curves for three representative training-set sizes,
including examples from the memorization and generalization regimes. Coverage is computed directly from the individual generated-map
probe recoveries over the 32 held-out cosmologies; the plotted markers identify the nominal 68\% and 95\%
intervals. It also
saves the frozen VGG16+MLP heldout-real slope/$R^2$ summary used to establish which conditional directions
the probe can verify. The U-Net-128 audit uses $N_{2D}=2^6,2^8,2^{10},2^{12},2^{15}$
to show copying, the intermediate degradation, and recovery at high data. The normalized SSCD Fréchet
annotation compares 512 generated maps with 512 held-out real maps and divides by the distance between two
independent 512-map held-out-real splits. The CSV records both raw distances and the ratio. This held-out-real
reference tests whether generated maps remain in distribution. By contrast, the third-row Nyquist-limited
power-spectrum ratio uses each model's exact configured training subset as its real reference, because that
row tests whether the model reproduces the statistics of the distribution on which it was trained. Its Fourier
coordinate is converted to physical units with the CAMELS map width $L=25\,h^{-1}\mathrm{Mpc}$.
""",
        ),
        _cell(
            "code",
            r"""
from pathlib import Path
import sys
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import Image, Markdown, display

PAPER_PROJECT_DIR = Path.cwd().resolve()
while PAPER_PROJECT_DIR != PAPER_PROJECT_DIR.parent and not (PAPER_PROJECT_DIR / '.git').exists():
    PAPER_PROJECT_DIR = PAPER_PROJECT_DIR.parent
if not (PAPER_PROJECT_DIR / '.git').exists():
    raise RuntimeError(f'Could not locate repository root from {Path.cwd().resolve()}')

paper_scripts = PAPER_PROJECT_DIR / 'scripts'
if str(paper_scripts) not in sys.path:
    sys.path.insert(0, str(paper_scripts))

from plot_nf_conditional_bias_paper_figures import (
    build_conditional_coverage_figure,
    build_generalization_figure,
    build_nearest_training_panels,
    build_probe_summary_figure,
    export_nearest_training_outputs,
    save_figure,
)

paper_figure_dir = PAPER_PROJECT_DIR / 'paper' / 'ai4science_verification' / 'figures'
paper_figure_dir.mkdir(parents=True, exist_ok=True)
full_sweep_dir = (
    PAPER_PROJECT_DIR / 'results' / 'nf_conditional_bias_fresh_full_sweep_200k'
    / 'calibration_vgg'
)
paper_inputs = {
    'samples': full_sweep_dir / 'bias_probe_per_sample_predictions.csv',
    'generalization': (
        PAPER_PROJECT_DIR / 'results' / 'nf_generalize_fig2' / 'tables'
        / 'nf_generalize_fig2_pca_full_nn_metrics.csv'
    ),
    'nearest_manifest': (
        PAPER_PROJECT_DIR / 'local' / 'nf_generalize_fig2' / 'manifest.json'
    ),
    'sscd_cache': (
        PAPER_PROJECT_DIR / 'results' / 'nf_generalize_fig2' / 'cache'
        / 'sscd_full_nn'
    ),
    'probe': (
        PAPER_PROJECT_DIR / 'results' / 'nf_conditional_bias_probe' / 'encoder'
        / 'vgg_real_probe_slope_r2_summary.csv'
    ),
}
paper_file_inputs = {
    key: path for key, path in paper_inputs.items() if key != 'sscd_cache'
}
missing_paper_inputs = [str(path) for path in paper_file_inputs.values() if not path.is_file()]
if not paper_inputs['sscd_cache'].is_dir():
    missing_paper_inputs.append(str(paper_inputs['sscd_cache']))
if missing_paper_inputs:
    raise FileNotFoundError('Missing exact paper inputs:\n' + '\n'.join(missing_paper_inputs))

paper_outputs = {
    'conditional': paper_figure_dir / 'conditional_recovery_transition.pdf',
    'conditional_coverage_table': (
        paper_figure_dir / 'conditional_recovery_coverage_curves.csv'
    ),
    'generalization': paper_figure_dir / 'generalization_transition.pdf',
    'nearest': paper_figure_dir / 'nearest_training_u128.pdf',
    'nearest_preview': paper_figure_dir / 'nearest_training_u128_preview.png',
    'nearest_table': paper_figure_dir / 'nearest_training_u128.csv',
    'probe': paper_figure_dir / 'vgg_probe_heldout_real.pdf',
}

paper_samples = pd.read_csv(paper_inputs['samples'])
paper_generalization = pd.read_csv(paper_inputs['generalization'])
conditional_figure, conditional_coverage_report = build_conditional_coverage_figure(
    paper_samples,
    bootstrap=2000,
    seed=123,
)
paper_dimensions = {
    'conditional': save_figure(conditional_figure, paper_outputs['conditional'])
}
conditional_coverage_report.to_csv(
    paper_outputs['conditional_coverage_table'], index=False
)
display(Markdown('### Conditional recovery coverage'))
display(conditional_figure)
plt.close(conditional_figure)

generalization_figure = build_generalization_figure(paper_generalization)
paper_dimensions['generalization'] = save_figure(
    generalization_figure,
    paper_outputs['generalization'],
)
display(Markdown('### Memorization-to-novelty transition'))
display(generalization_figure)
plt.close(generalization_figure)

nearest_panels = build_nearest_training_panels(
    PAPER_PROJECT_DIR,
    paper_inputs['nearest_manifest'],
    paper_inputs['sscd_cache'],
    seed=123,
    sample_label='dpm50',
)
paper_dimensions['nearest'], nearest_training_report = export_nearest_training_outputs(
    nearest_panels,
    paper_outputs['nearest'],
    paper_outputs['nearest_table'],
    preview_path=paper_outputs['nearest_preview'],
)
display(Markdown('### Generated samples, nearest training slices, and in-distribution check'))
display(Image(filename=str(paper_outputs['nearest_preview']), width=1100))
display(nearest_training_report)

probe_figure = build_probe_summary_figure(pd.read_csv(paper_inputs['probe']))
paper_dimensions['probe'] = save_figure(probe_figure, paper_outputs['probe'])
display(Markdown('### Frozen VGG16+MLP heldout-real validation'))
display(probe_figure)
plt.close(probe_figure)

display(Markdown('### Saved paper figures'))
for paper_name in ('conditional', 'generalization', 'nearest', 'probe'):
    paper_path = paper_outputs[paper_name]
    width, height = paper_dimensions[paper_name]
    print(f'{paper_name}: {paper_path} ({width:.3f} x {height:.3f} in)')
print(f"nearest_table: {paper_outputs['nearest_table']} ({len(nearest_training_report)} rows)")
print(f"nearest_preview: {paper_outputs['nearest_preview']} (300 dpi)")
print(
    f"conditional_coverage_table: {paper_outputs['conditional_coverage_table']} "
    f"({len(conditional_coverage_report)} rows)"
)

display(Markdown('### Exact saved $\\Omega_m$ coverage'))
display(
    conditional_coverage_report[
        conditional_coverage_report['plotted']
        & conditional_coverage_report['nominal_coverage'].isin([0.68, 0.95])
    ][
        [
            'dataset_size', 'nominal_coverage', 'empirical_coverage',
            'coverage_ci16', 'coverage_ci84', 'n_heldout',
            'draws_per_cosmology',
        ]
    ].sort_values(['dataset_size', 'nominal_coverage']).reset_index(drop=True)
)
""",
        ),
        _cell(
            "markdown",
            r"""
The PDFs above contain no figure-level titles; put the scientific description, ideal-calibration diagonal,
and training protocol in the LaTeX captions. The conditional coverage curve uses the measured recovered-$\Omega_m$
draws at $N_{2D}=2^7,2^{10},2^{14}$; curves below the diagonal are overconfident.
The U-Net-128 table records the exact-subset configuration, copying similarity, power-spectrum error,
normalized SSCD Fr\'echet distance, and equal evaluation sample counts for every displayed column.
""",
        ),
    ]


def update(path: Path) -> None:
    notebook = json.loads(path.read_text())
    notebook["cells"] = [
        cell
        for cell in notebook.get("cells", [])
        if TAG not in cell.get("metadata", {}).get("tags", [])
    ]
    notebook["cells"].extend(_cells())
    path.write_text(json.dumps(notebook, indent=1, ensure_ascii=True) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "notebook",
        nargs="?",
        type=Path,
        default=Path("notebooks/nf_conditional_bias_vgg_results.ipynb"),
    )
    args = parser.parse_args()
    update(args.notebook)
    print(f"Updated {args.notebook}")


if __name__ == "__main__":
    main()
