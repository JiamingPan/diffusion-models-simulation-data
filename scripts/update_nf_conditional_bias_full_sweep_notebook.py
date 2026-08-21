#!/usr/bin/env python
"""Add the full conditional training-size sweep to the existing VGG notebook."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


TAG = "conditional-full-sweep"


def markdown_cell(source: str) -> dict:
    return {
        "cell_type": "markdown",
        "metadata": {"tags": [TAG]},
        "source": [line + "\n" for line in source.strip().splitlines()],
    }


def code_cell(source: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {"tags": [TAG]},
        "outputs": [],
        "source": [line + "\n" for line in source.strip().splitlines()],
    }


def update(path: Path) -> None:
    notebook = json.loads(path.read_text())
    notebook["cells"] = [
        cell
        for cell in notebook.get("cells", [])
        if TAG not in cell.get("metadata", {}).get("tags", [])
    ]
    notebook["cells"].extend(
        [
            markdown_cell(
                r"""
## Full training-size sweep

This section replaces the two-endpoint comparison with conditional UNet-128 models trained on every
$N_{2D}=2^6,\ldots,2^{15}$. In particular, all ten generators are trained from clean initializations;
no generator checkpoint from the earlier two-size comparison is reused. All models use the same 200k
optimizer-update target, fixed RNG seed, full six-dimensional CAMELS conditioning vector, heldout
simulations 900--931, and the same frozen VGG16+MLP probe.

The first figure overlays all ten fitted $\Omega_m$ responses and places their slopes directly against
training-set size. The second keeps the underlying per-size points and uncertainty intervals visible in
separate panels. The third summarizes the response slope for all six parameters. A slope of one is ideal;
a small slope means generated maps respond weakly to changes in the requested parameter.
"""
            ),
            code_cell(
                r"""
from pathlib import Path
import sys
import pandas as pd
from IPython.display import Image, Markdown, display

# Resolve the repository independently so this cell is safe to run by itself.
PROJECT_DIR = Path.cwd().resolve()
while PROJECT_DIR != PROJECT_DIR.parent and not (PROJECT_DIR / '.git').exists():
    PROJECT_DIR = PROJECT_DIR.parent
if not (PROJECT_DIR / '.git').exists():
    raise RuntimeError(f'Could not locate the repository root from {Path.cwd().resolve()}')

full_sweep_root = PROJECT_DIR / 'results' / 'nf_conditional_bias_fresh_full_sweep_200k' / 'calibration_vgg'
full_points_path = full_sweep_root / 'bias_probe_per_cosmology_points.csv'
full_slopes_path = full_sweep_root / 'bias_probe_regime_slopes.csv'
omega_figure = full_sweep_root / 'bias_probe_omega_m_all_dataset_sizes.png'
omega_transition_figure = full_sweep_root / 'bias_probe_omega_m_transition_vs_dataset_size.png'
slope_figure = full_sweep_root / 'bias_probe_all_parameter_slopes_vs_dataset_size.png'
expected_sizes = [2**power for power in range(6, 16)]

if full_points_path.exists() and full_slopes_path.exists():
    full_points = pd.read_csv(full_points_path)
    full_slopes = pd.read_csv(full_slopes_path)
    present_sizes = sorted(full_points['dataset_size'].astype(int).unique().tolist())
    missing_sizes = sorted(set(expected_sizes) - set(present_sizes))
    if missing_sizes:
        raise RuntimeError(f'missing dataset sizes: {missing_sizes}')
    display(pd.DataFrame({
        'dataset_size': expected_sizes,
        'log2_size': list(range(6, 16)),
        'point_rows': [int((full_points.dataset_size == size).sum()) for size in expected_sizes],
        'slope_rows': [int((full_slopes.dataset_size == size).sum()) for size in expected_sizes],
    }))
    scripts_dir = PROJECT_DIR / 'scripts'
    if str(scripts_dir) not in sys.path:
        sys.path.insert(0, str(scripts_dir))
    from plot_nf_conditional_bias_full_sweep import (
        plot_omega_m_grid,
        plot_omega_m_transition,
        plot_parameter_slope_summary,
    )
    plot_omega_m_grid(full_points, full_slopes, omega_figure)
    plot_omega_m_transition(full_points, full_slopes, omega_transition_figure)
    plot_parameter_slope_summary(full_slopes, slope_figure)
    display(Image(filename=str(omega_transition_figure)))
    display(Image(filename=str(omega_figure)))
    display(Image(filename=str(slope_figure)))
else:
    display(Markdown(
        '**Full-sweep results are not present yet.** Run the Great Lakes full-sweep pipeline, then rerun '
        'this cell. No sizes are interpolated or silently omitted.'
    ))
"""
            ),
            markdown_cell(
                r"""
### Reading the sweep

Read the panels horizontally, not as ten independent anecdotes. The key question is whether the fitted
response moves smoothly toward the ideal line as training data grows, and whether different cosmological
and astrophysical parameters reach that regime at the same dataset size. The error bars describe variation
across generated samples at each heldout cosmology; the slope interval describes uncertainty across the
heldout cosmologies themselves.
"""
            ),
        ]
    )
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
