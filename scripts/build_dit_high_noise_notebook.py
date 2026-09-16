#!/usr/bin/env python
"""Build the reader-facing notebook for the DiT high-noise diagnostic."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

def _cell(cell_type: str, source: str) -> dict:
    cell = {"cell_type": cell_type, "metadata": {}, "source": source.splitlines(keepends=True)}
    if cell_type == "code":
        cell.update({"execution_count": None, "outputs": []})
    return cell


def build_notebook() -> dict:
    nb = {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {},
        "cells": [],
    }
    nb["metadata"]["kernelspec"] = {
        "display_name": "Python (cosmodiff_nf_class)",
        "language": "python",
        "name": "python3",
    }
    cells = []
    cells.append(_cell("markdown", """# Does DiT-L16 fail at the noisiest timesteps?

## tl;dr

Run all cells after the L8 and both L16 diagnostics finish. L12 is optional because its checkpoint weights are unavailable. Missing L12 results are reported and omitted from plots. These diagnostics investigate high-noise behavior but cannot alone establish that loss weighting caused the failure.

This is **not training**. It reads frozen checkpoints and answers three questions:

1. Can each model reconstruct a known real map at different noise levels?
2. Does skipping the earliest, noisiest DPM steps remove the blocky failure?
3. At the terminal noise level, does the model return the training-set mean as the MSE-optimal solution, or an unstable blocky field?

The three required models must pass the checks below. L8 has a different training length, so this comparison does not isolate depth alone."""))
    cells.append(_cell("code", """from pathlib import Path
import json
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_DIR = Path(os.environ.get('PROJECT_DIR', '/home/jiamingp/diffusion_models_repo'))
RESULT_DIR = PROJECT_DIR / 'results/dit_l16_high_noise_diagnostic'
EXPECTED = ['dit_l8_200k', 'dit_l12_200k', 'dit_l16_fresh300k', 'dit_l16_seed456_500k']
OPTIONAL = {'dit_l12_200k'}
EXPECTED_LAYERS = dict(zip(EXPECTED, [8, 12, 16, 16]))
COLORS = {
    'dit_l8_200k': '#009E73',
    'dit_l12_200k': '#0072B2',
    'dit_l16_fresh300k': '#CC79A7',
    'dit_l16_seed456_500k': '#8E2A68',
}
LABELS = {
    'dit_l8_200k': 'L8, 200k',
    'dit_l12_200k': 'L12, 200k',
    'dit_l16_fresh300k': 'L16, fresh 300k',
    'dit_l16_seed456_500k': 'L16, seed456 500k',
}
plt.rcParams.update({'figure.dpi': 120, 'font.size': 11, 'axes.grid': True, 'grid.alpha': .22})"""))
    cells.append(_cell("markdown", """## Context & methods

All models use the same 128×128 normalized fields, patch size 8, v-prediction, zero-terminal-SNR cosine schedule, and Min-SNR γ=5. The L8/L12 checkpoints are the original 200k depth controls; L16 is shown at fresh 300k and after the independent seed456 continuation to 500k.

### Key assumptions

- A large-timestep error localized to small SNR supports the proposed mechanism; it does not by itself prove the loss weighting is the only cause.
- Late-start denoising is a reconstruction test, not unconditional generation. Similarity is measured against the known source map.
- Raw weights are used because the observed failed samples were generated from raw checkpoints. EMA is a separate follow-up if this mechanism is present."""))
    cells.append(_cell("code", """summaries, arrays = {}, {}
for name in EXPECTED:
    json_path = RESULT_DIR / f'{name}.json'
    npz_path = RESULT_DIR / f'{name}.npz'
    if name in OPTIONAL and not json_path.exists() and not npz_path.exists():
        print(f'UNAVAILABLE: {LABELS[name]} — checkpoint weights missing; excluded from plots.')
        continue
    if not json_path.is_file() or not npz_path.is_file():
        raise FileNotFoundError(f'Missing completed diagnostic for {name}: {json_path}, {npz_path}')
    summaries[name] = json.loads(json_path.read_text())
    if summaries[name].get('status') != 'complete':
        raise RuntimeError(f'{name} does not have terminal complete status')
    arrays[name] = np.load(npz_path, allow_pickle=False)

EXPECTED = [name for name in EXPECTED if name in summaries]
assert all(summaries[name]['num_layers'] == EXPECTED_LAYERS[name] for name in EXPECTED)
assert {summaries[name]['patch_size'] for name in EXPECTED} == {8}
assert {summaries[name]['prediction_type'] for name in EXPECTED} == {'v_prediction'}
assert {summaries[name]['weights'] for name in EXPECTED} == {'raw'}
assert len({summaries[name]['code_revision'] for name in EXPECTED}) == 1

pd.DataFrame(summaries).T[['num_layers', 'checkpoint', 'num_reference',
                           'complete_training_reference_size', 'terminal_snr',
                           'terminal_v_weight', 'code_revision']]"""))
    cells.append(_cell("markdown", """## Results

### 1. What training weight did each noise level receive?

Large timestep means more noise. A weight near zero means that timestep contributed almost no gradient during training."""))
    cells.append(_cell("code", """schedule = arrays[EXPECTED[0]]
timesteps = np.arange(len(schedule['schedule_snr']))
fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), constrained_layout=True)
axes[0].semilogy(timesteps, np.maximum(schedule['schedule_snr'], 1e-12), color='#355C7D')
axes[0].set(xlabel='Training timestep (larger = noisier)', ylabel='SNR', title='Noise schedule')
axes[1].semilogy(timesteps, np.maximum(schedule['schedule_v_weight'], 1e-12), color='#D97706')
axes[1].set(xlabel='Training timestep (larger = noisier)', ylabel='v-loss weight', title='Actual Min-SNR weight')
for axis in axes:
    axis.axvline(400, color='black', linestyle=':', linewidth=1)
plt.show()

weights = schedule['schedule_v_weight']
pd.DataFrame({
    'criterion': ['weight < 0.1', 'weight < 0.01', 'weight = 0'],
    'number_of_500_timesteps': [(weights < .1).sum(), (weights < .01).sum(), (weights == 0).sum()],
})"""))
    cells.append(_cell("markdown", """### 2. Direct reconstruction error by timestep

For each real map, the script draws known Gaussian noise, forms the exact forward-diffusion state, asks the frozen model for v, and analytically converts that prediction back to x₀. Lower normalized MSE and higher cosine similarity are better. A patch-boundary ratio of 1 means no special discontinuity at the 8-pixel grid."""))
    cells.append(_cell("code", """fig, axes = plt.subplots(1, 3, figsize=(17, 4.8), constrained_layout=True)
for name in EXPECTED:
    a = arrays[name]
    t = a['timesteps']
    snr = a['schedule_snr'][t]
    order = np.argsort(snr)
    axes[0].plot(np.maximum(snr[order], 1e-12), np.median(a['x0_nmse'], axis=1)[order], marker='o', color=COLORS[name], label=LABELS[name])
    axes[1].plot(np.maximum(snr[order], 1e-12), np.median(a['x0_cosine'], axis=1)[order], marker='o', color=COLORS[name])
    axes[2].plot(np.maximum(snr[order], 1e-12), np.median(a['x0_patch_ratio'], axis=1)[order], marker='o', color=COLORS[name])
axes[0].set(xscale='log', yscale='log', xlabel='SNR', ylabel='Median normalized MSE', title='Can the model recover x₀?')
axes[1].set(xscale='log', xlabel='SNR', ylabel='Median cosine to true x₀', title='Recovered structure')
axes[2].set(xscale='log', xlabel='SNR', ylabel='Median boundary / local-control jump', title='Patch-grid artifact')
axes[2].axhline(1, color='black', linestyle=':')
axes[0].legend(frameon=False)
plt.show()"""))
    cells.append(_cell("markdown", """### 3. Skip the earliest high-noise steps

These runs start from a known real map noised to several points in the *same* DPM-50 schedule. If terminal-weight starvation is the main cause, L16 should improve sharply when starting near t≈400 instead of t≈499. This is not an unconditional novelty test."""))
    cells.append(_cell("code", """fig, axes = plt.subplots(1, 3, figsize=(17, 4.8), constrained_layout=True)
for name in EXPECTED:
    a = arrays[name]
    t = a['late_actual_timesteps']
    axes[0].plot(t, np.median(a['late_nmse'], axis=1), marker='o', color=COLORS[name], label=LABELS[name])
    axes[1].plot(t, np.median(a['late_cosine'], axis=1), marker='o', color=COLORS[name])
    axes[2].plot(t, np.median(a['late_patch_ratio'], axis=1), marker='o', color=COLORS[name])
axes[0].set(yscale='log', xlabel='Actual starting timestep', ylabel='Median normalized MSE', title='Late-start reconstruction error')
axes[1].set(xlabel='Actual starting timestep', ylabel='Median cosine to source map', title='Late-start recovered structure')
axes[2].set(xlabel='Actual starting timestep', ylabel='Median boundary / control jump', title='Late-start patch artifact')
axes[2].axhline(1, color='black', linestyle=':')
axes[0].legend(frameon=False)
plt.show()"""))
    cells.append(_cell("code", """def show_late_gallery(name):
    a = arrays[name]
    rows = len(a['late_actual_timesteps']) + 1
    cols = a['reference'].shape[0]
    fig, axes = plt.subplots(rows, cols, figsize=(2.25 * cols, 2.2 * rows), constrained_layout=True)
    vmin, vmax = -1, 1
    for col in range(cols):
        axes[0, col].imshow(a['reference'][col, 0], vmin=vmin, vmax=vmax, cmap='viridis')
        axes[0, col].set_title(f'Real source {col}')
    for row, timestep in enumerate(a['late_actual_timesteps'], start=1):
        for col in range(cols):
            axes[row, col].imshow(a['late_gallery'][row - 1, col, 0], vmin=vmin, vmax=vmax, cmap='viridis')
            axes[row, col].set_title(f'start t={timestep}')
    for axis in axes.flat:
        axis.axis('off')
    fig.suptitle(f'{LABELS[name]}: same source maps, different starting noise levels', fontsize=15)
    plt.show()

show_late_gallery('dit_l16_fresh300k')
show_late_gallery('dit_l16_seed456_500k')"""))
    cells.append(_cell("markdown", """### 4. Pure noise at the terminal timestep

At zero SNR, xₜ contains no information about its paired training map. Under MSE, the optimal predicted x₀ is therefore the training-set mean. Large output-to-output variation or patch-grid structure is direct evidence of uncontrolled terminal extrapolation."""))
    cells.append(_cell("code", """terminal_rows = []
fig, axes = plt.subplots(len(EXPECTED), 7, figsize=(15, 8.5), constrained_layout=True)
for row, name in enumerate(EXPECTED):
    a = arrays[name]
    axes[row, 0].imshow(a['training_mean'][0, 0], vmin=-1, vmax=1, cmap='viridis')
    axes[row, 0].set_title('Training mean')
    for col in range(1, 7):
        axes[row, col].imshow(a['terminal_gallery'][col - 1, 0], vmin=-1, vmax=1, cmap='viridis')
        axes[row, col].set_title(f'noise {col - 1}')
    axes[row, 0].set_ylabel(LABELS[name])
    terminal_rows.append({
        'model': LABELS[name],
        'mean MSE to training mean': float(a['terminal_mean_mse']),
        'variation across noise inputs (RMS)': float(a['terminal_between_noise_rms']),
        'training-map variation (RMS)': float(a['training_between_map_rms']),
        'median patch ratio': float(np.median(a['terminal_patch_ratio'])),
        'median output RMS': float(np.median(a['terminal_output_rms'])),
    })
for axis in axes.flat:
    axis.set_xticks([]); axis.set_yticks([])
plt.show()
pd.DataFrame(terminal_rows).set_index('model')"""))
    cells.append(_cell("markdown", """## Takeaways

Read the evidence jointly:

- **Supports the high-noise-weighting mechanism:** L16 becomes much worse than L8/L12 only as SNR approaches zero; late-start reconstruction improves sharply by t≈400; and pure-noise terminal outputs from L16 vary strongly or show patch-grid structure.
- **Weakens the mechanism:** L16 remains comparably bad at moderate/low noise, or skipping the terminal region does not remove its artifacts. The problem then lies deeper in the denoising trajectory or representation.
- **Sampler contribution:** If direct one-step x₀ estimates are sound but late-start DPM reconstructions fail, the solver trajectory—not missing gradient alone—is implicated.

This notebook intentionally does not turn those conditions into an arbitrary single pass/fail number. Record the observed values and images before deciding whether to change the loss."""))
    nb["cells"] = cells
    for index, cell in enumerate(cells):
        cell["id"] = f"dit-high-noise-{index:02d}"
    return nb


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    notebook = build_notebook()
    args.output.write_text(json.dumps(notebook, indent=1) + "\n")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
