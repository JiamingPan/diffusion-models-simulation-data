#!/usr/bin/env python
"""Build the read-only reader for Nick's nearest-training-parameter control."""
from pathlib import Path
import nbformat as nbf

ROOT=Path(__file__).resolve().parents[1]
nb=nbf.v4.new_notebook()
md=nbf.v4.new_markdown_cell
code=nbf.v4.new_code_cell
nb.cells=[
md("""# Conditional recovery: nearest-training-parameter sanity check

## tl;dr

This notebook reads saved control results; it never trains, samples, fits the probe, or submits jobs.
The actual bias diagnosis is **pending until the control tables are available**. A small cosine
distance to an image is not used to select this baseline: neighbors are chosen in the six-dimensional
**physical parameter space**, standardized with frozen training-only scales.

The new constant-label DiT zero-init runs cannot replace these conditional recovery runs.
They are a separate image-quality/memorization experiment."""),
md("""## Context & Methods

At each requested held-out cosmology, compare:

1. **Generated-field recovery**: the existing frozen probe applied to saved generated fields.
2. **Nearest training cosmology, true parameters**: the training-parameter coverage baseline, with no probe.
3. **Nearest training fields, probe recovery**: the same frozen probe on all exact selected fields of that nearest cosmology.
4. **Held-out real fields, probe recovery**: probe behavior on real fields at the requested cosmology.

### Key Assumptions

Distances use all six raw parameters divided by the existing training-only standard deviations.
Parameter-space proximity is a baseline choice, not a claim that the generator uses nearest-neighbor retrieval.
Exact ties use the smallest simulation ID and are recorded. Multiple slices from one cube are correlated;
64 slices are not 64 independent cosmologies. Error bars are the central 68% field/probe spread,
**not Bayesian posterior uncertainty**. Interval inclusion is a recovery diagnostic, not SBC."""),
md("### 1. Select the saved results"),
code("""import json
import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display, Markdown

candidates = [Path.cwd(), Path.cwd().parent, Path('/home/jiamingp/diffusion_models_repo'),
              Path('/Users/apple/AI/Diffusion_model')]
PROJECT_DIR = next(p.resolve() for p in candidates if (p/'simdiff_eval'/'parameter_neighbor_control.py').is_file())
sys.path.insert(0,str(PROJECT_DIR))
from simdiff_eval.parameter_neighbor_plots import plot_recovery

# Change this to the new output folder produced by evaluate_parameter_neighbor_control.py.
CONTROL_DIR = Path(os.environ.get('NEIGHBOR_CONTROL_DIR',str(PROJECT_DIR/'results'/'parameter_neighbor_control_v1')))
plt.rcParams.update({'font.size':11,'axes.titlesize':13,'figure.facecolor':'white'})
display(Markdown(f'Result folder: `{CONTROL_DIR}`'))"""),
md("""## Data

### 2. Audit availability and provenance

Missing results stay pending. Malformed or mismatched saved results raise an error;
they are not silently replaced by a different experiment."""),
code("""required = ['control_points.csv','parameter_matches.csv','control_summary.csv','metadata.json']
available = all((CONTROL_DIR/name).is_file() for name in required)
display(pd.DataFrame({'artifact':required,'available':[(CONTROL_DIR/name).is_file() for name in required]}))
points = pd.DataFrame()
if available:
    metadata = json.loads((CONTROL_DIR/'metadata.json').read_text())
    if metadata.get('intervals') != 'field/probe recovery spread; not Bayesian posterior coverage':
        raise ValueError('Unexpected control methodology; review metadata before interpreting results.')
    points = pd.read_csv(CONTROL_DIR/'control_points.csv')
    matches = pd.read_csv(CONTROL_DIR/'parameter_matches.csv')
    summary = pd.read_csv(CONTROL_DIR/'control_summary.csv')
    if points.duplicated(['run_name','heldout_sim','parameter','kind']).any():
        raise ValueError('Duplicate control points.')
    display(Markdown('Distance metric: '+metadata['distance']))
    display(matches[['run_name','heldout_sim','nearest_sim','parameter_distance',
                     'selected_training_fields','tied_cosmologies']].head(12))
else:
    display(Markdown('**PENDING:** run the CPU control in the existing review environment and point this notebook at its output. No bias conclusion is available locally.'))"""),
md("""## Results

### 3. Requested versus recovered parameters

The dashed diagonal is equality. Black crosses show whether even an oracle returning the nearest
training cosmology would be offset. Gold triangles add the frozen probe. Blue squares check that
probe on held-out real fields. Red circles are the existing generator."""),
code("""if available:
    for run in points.run_name.unique():
        fig=plot_recovery(points,run)
        display(fig)
        plt.close(fig)"""),
md("""### 4. Recovery residuals

The horizontal zero line means no offset. Compare the red generator residuals with the gold
neighbor-control residuals and blue held-out-real residuals at the **same requested parameters**.
Do not infer a cause from a fitted slope alone."""),
code("""if available:
    for run in points.run_name.unique():
        fig=plot_recovery(points,run,residual=True)
        display(fig)
        plt.close(fig)"""),
md("""### 5. Exact summaries and residual accounting

The identity below separates comparisons, not independent causal effects:

`generated − requested = (neighbor true − requested) + (neighbor probe − neighbor true) + (generated − neighbor probe)`

These differences use per-cosmology medians. Recovery-interval inclusion is **not posterior coverage**."""),
code("""if available:
    display(summary[summary.parameter.isin(['Omega_m','sigma_8'])].round(4))
    decomposition_path=CONTROL_DIR/'residual_decomposition.csv'
    if decomposition_path.exists():
        decomposition=pd.read_csv(decomposition_path)
        display(decomposition[decomposition.parameter=='Omega_m'].head(12).round(4))
    else:
        display(Markdown('Generated predictions were not included; the three-way residual comparison remains pending.'))"""),
md("""## Takeaways

Fill in the diagnosis only after reading real outputs:

- **True nearest-training parameters already biased:** finite six-dimensional training coverage can produce an offset even without a probe.
- **Nearest-field probe adds bias:** the recovery estimator contributes additional distortion.
- **Held-out real probe is biased too:** do not attribute all generator-versus-diagonal differences to diffusion.
- **Generator differs from both real controls:** investigate learned conditioning, sample morphology, and parameter normalization next.

For genuine posterior calibration, use samples from an actual `p(theta | observed field)` inference pipeline
and validate it separately. This notebook does not turn forward-generated fields into posterior samples.

The DiT zero-init refresh needs its own manifests and exact-subset audits. Keep old native-init depth
curves and new zero-init curves separate, and report the update budget and seed for each."""),
]
nb.metadata={'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},'language_info':{'name':'python'}}
path=ROOT/'notebooks'/'conditional_parameter_neighbor_control.ipynb'
nbf.validate(nb)
nbf.write(nb,path)
print(path)
