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

This cell exports the conditional-recovery result and its two comparison figures at the exact paper width.
It reads the completed ten-size calibration tables without refitting or changing any data selection. The
three representative panels use $N_{2D}=2^6,2^{11},2^{15}$ with shared limits; the companion panel reports
all ten saved slopes and their 16th--84th percentile bootstrap intervals. It also saves the frozen VGG16+MLP
heldout-real slope/$R^2$ summary used to establish which conditional directions the probe can verify.
""",
        ),
        _cell(
            "code",
            r"""
from pathlib import Path
import sys
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import Markdown, display

PAPER_PROJECT_DIR = Path.cwd().resolve()
while PAPER_PROJECT_DIR != PAPER_PROJECT_DIR.parent and not (PAPER_PROJECT_DIR / '.git').exists():
    PAPER_PROJECT_DIR = PAPER_PROJECT_DIR.parent
if not (PAPER_PROJECT_DIR / '.git').exists():
    raise RuntimeError(f'Could not locate repository root from {Path.cwd().resolve()}')

paper_scripts = PAPER_PROJECT_DIR / 'scripts'
if str(paper_scripts) not in sys.path:
    sys.path.insert(0, str(paper_scripts))

from plot_nf_conditional_bias_paper_figures import (
    build_conditional_recovery_figure,
    build_generalization_figure,
    build_probe_summary_figure,
    export_nearest_training_pdf,
    save_figure,
)

paper_figure_dir = PAPER_PROJECT_DIR / 'paper' / 'ai4science_verification' / 'figures'
paper_figure_dir.mkdir(parents=True, exist_ok=True)
full_sweep_dir = (
    PAPER_PROJECT_DIR / 'results' / 'nf_conditional_bias_fresh_full_sweep_200k'
    / 'calibration_vgg'
)
paper_inputs = {
    'points': full_sweep_dir / 'bias_probe_per_cosmology_points.csv',
    'slopes': full_sweep_dir / 'bias_probe_regime_slopes.csv',
    'generalization': (
        PAPER_PROJECT_DIR / 'results' / 'nf_generalize_fig2' / 'tables'
        / 'nf_generalize_fig2_pca_full_nn_metrics.csv'
    ),
    'nearest': (
        PAPER_PROJECT_DIR / 'results' / 'nf_generalize_fig2' / 'quickcheck'
        / 'nf_generalize_fig2_u128_generated_vs_pixel_nn.png'
    ),
    'probe': (
        PAPER_PROJECT_DIR / 'results' / 'nf_conditional_bias_probe' / 'encoder'
        / 'vgg_real_probe_slope_r2_summary.csv'
    ),
}
missing_paper_inputs = [str(path) for path in paper_inputs.values() if not path.is_file()]
if missing_paper_inputs:
    raise FileNotFoundError('Missing exact paper inputs:\n' + '\n'.join(missing_paper_inputs))

paper_outputs = {
    'conditional': paper_figure_dir / 'conditional_recovery_transition.pdf',
    'generalization': paper_figure_dir / 'generalization_transition.pdf',
    'nearest': paper_figure_dir / 'nearest_training_unet128.pdf',
    'probe': paper_figure_dir / 'vgg_probe_heldout_real.pdf',
}

paper_points = pd.read_csv(paper_inputs['points'])
paper_slopes = pd.read_csv(paper_inputs['slopes'])
conditional_figure, conditional_slope_report = build_conditional_recovery_figure(
    paper_points,
    paper_slopes,
)
paper_dimensions = {
    'conditional': save_figure(conditional_figure, paper_outputs['conditional'])
}
plt.close(conditional_figure)

generalization_figure = build_generalization_figure(pd.read_csv(paper_inputs['generalization']))
paper_dimensions['generalization'] = save_figure(
    generalization_figure,
    paper_outputs['generalization'],
)
plt.close(generalization_figure)

paper_dimensions['nearest'] = export_nearest_training_pdf(
    paper_inputs['nearest'],
    paper_outputs['nearest'],
)

probe_figure = build_probe_summary_figure(pd.read_csv(paper_inputs['probe']))
paper_dimensions['probe'] = save_figure(probe_figure, paper_outputs['probe'])
plt.close(probe_figure)

display(Markdown('### Saved paper figures'))
for paper_name, paper_path in paper_outputs.items():
    width, height = paper_dimensions[paper_name]
    print(f'{paper_name}: {paper_path} ({width:.3f} x {height:.3f} in)')

display(Markdown('### Exact saved $\\Omega_m$ slopes'))
display(
    conditional_slope_report[
        ['dataset_size', 'slope', 'slope_ci16', 'slope_ci84']
    ].sort_values('dataset_size').reset_index(drop=True)
)
""",
        ),
        _cell(
            "markdown",
            r"""
The PDFs above contain no figure-level titles; put the scientific description, reference-line definition,
and training protocol in the LaTeX captions. The conditional and generalization figures use the same
$2^6$--$2^{15}$ horizontal axis so their transitions can be compared directly when stacked in the paper.
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
