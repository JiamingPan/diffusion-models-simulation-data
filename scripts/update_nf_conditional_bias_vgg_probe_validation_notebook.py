#!/usr/bin/env python
"""Replace the VGG notebook's Omega-only real check with all-parameter validation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


TAG = "vgg-real-probe-validation"


def _cell(cell_type: str, source: str) -> dict:
    cell = {
        "cell_type": cell_type,
        "metadata": {"tags": [TAG]},
        "source": [line + "\n" for line in source.strip().splitlines()],
    }
    if cell_type == "code":
        cell.update({"execution_count": None, "outputs": []})
    return cell


def _replacement_cells() -> list[dict]:
    return [
        _cell(
            "markdown",
            r"""
## Real Held-Out Probe Validation

Before interpreting generated-map calibration, measure what the frozen VGG16+MLP probe can recover from
real held-out CAMELS simulations 900--931. For every parameter, the point is the median probe prediction
over real slices from one held-out cosmology; bars show the 16th--84th percentile slice spread.

The fitted **slope** measures response to that parameter, while $R^2$ measures absolute predictive agreement.
Both metrics are computed from the same held-out per-cosmology medians. Weak generated-map response is only
strong evidence of a generator failure when the real-map probe itself has a useful slope and $R^2$.
""",
        ),
        _cell(
            "code",
            r"""
REAL_METADATA_PATH = ENCODER_DIR / 'vgg_real_test_metadata.json'
FULL_SWEEP_METADATA_PATH = (
    ROOT / 'results' / 'nf_conditional_bias_fresh_full_sweep_200k'
    / 'calibration_vgg' / 'bias_probe_eval_metadata.json'
)

real_metrics = read_csv_or_none(REAL_METRICS_PATH)
real_pred = read_csv_or_none(REAL_PRED_PATH)
real_probe_metadata = read_json_or_none(REAL_METADATA_PATH)
full_sweep_metadata = read_json_or_none(FULL_SWEEP_METADATA_PATH)

real_encoder = str(real_probe_metadata.get('encoder_path', 'unknown encoder'))
sweep_encoder = str(full_sweep_metadata.get('encoder_path', 'unknown encoder'))
probe_label = f'probe provenance: {real_encoder}'

if real_encoder != 'unknown encoder' and sweep_encoder != 'unknown encoder':
    if Path(real_encoder).name != Path(sweep_encoder).name:
        display(Markdown(
            '**Probe provenance mismatch:** the saved held-out-real predictions came from '
            f'`{real_encoder}`, while the full generated-map sweep used `{sweep_encoder}`. '
            'The plots below remain valid for the named probe, but they must not be used as the '
            'generator sweep baseline until matching held-out-real predictions are available.'
        ))
    else:
        display(Markdown(f'**Probe provenance matched:** `{Path(real_encoder).name}`'))

if real_metrics is not None:
    real_summary = real_metrics[
        (real_metrics['split'] == 'test') & (real_metrics['grain'] == 'per_cosmology')
    ].copy()
    real_summary['parameter'] = pd.Categorical(real_summary['parameter'], PARAM_ORDER, ordered=True)
    display(real_summary.sort_values('parameter')[['parameter', 'n', 'mae', 'rmse', 'bias', 'r2']])
else:
    display(Markdown('Live `vgg_real_test_metrics.csv` is not present in this checkout.'))
""",
        ),
        _cell(
            "code",
            r"""
if real_pred is None:
    display(Markdown(
        'Cannot compute all-parameter held-out slopes and $R^2$ because '
        '`vgg_real_test_per_cosmology_predictions.csv` is missing.'
    ))
else:
    import sys
    scripts_dir = ROOT / 'scripts'
    if str(scripts_dir) not in sys.path:
        sys.path.insert(0, str(scripts_dir))
    from plot_nf_conditional_bias_vgg_probe_validation import write_probe_validation_outputs

    real_probe_panel = ENCODER_DIR / 'vgg_real_probe_all_parameters_1to1.png'
    real_probe_summary_figure = ENCODER_DIR / 'vgg_real_probe_slope_r2_summary.png'
    real_probe_summary_table = ENCODER_DIR / 'vgg_real_probe_slope_r2_summary.csv'
    real_probe_summary = write_probe_validation_outputs(
        real_pred,
        panel_path=real_probe_panel,
        summary_path=real_probe_summary_figure,
        table_path=real_probe_summary_table,
        probe_label=probe_label,
    )
    display(real_probe_summary)
    display(Image(filename=str(real_probe_panel)))
    display(Image(filename=str(real_probe_summary_figure)))
    print('wrote', real_probe_panel)
    print('wrote', real_probe_summary_figure)
    print('wrote', real_probe_summary_table)
""",
        ),
        _cell(
            "markdown",
            r"""
### How to use this check

Do not compare generated slopes to one for a parameter the probe cannot read reliably from real maps.
For example, a strong real-map $Omega_m$ slope and $R^2$ make a weak generated $Omega_m$ response
diagnostic of the generator. Conversely, weak real-map performance for an astrophysical parameter means
the VGG probe is not a reliable verifier for that conditional direction.
""",
        ),
    ]


def update(path: Path) -> None:
    notebook = json.loads(path.read_text())
    cells = notebook.get("cells", [])
    start = next(
        index
        for index, cell in enumerate(cells)
        if "## Real Held-Out" in "".join(cell.get("source", []))
    )
    end = next(
        index
        for index in range(start + 1, len(cells))
        if "## Generated-Field Calibration" in "".join(cells[index].get("source", []))
    )
    notebook["cells"] = cells[:start] + _replacement_cells() + cells[end:]
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
