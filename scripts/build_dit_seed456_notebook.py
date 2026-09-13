"""Build the Great Lakes-only interactive review notebook (no training)."""
import json
from pathlib import Path
import textwrap

cells = []


def cell(kind, source):
    item = dict(cell_type=kind, id=f"review-{len(cells):02d}", metadata={},
                source=textwrap.dedent(source).strip().splitlines(keepends=True))
    if kind == "code":
        item.update(execution_count=None, outputs=[])
    cells.append(item)


cell("markdown", """
# DiT-L16: inspect the seed-456 continuation
## tl;dr
Run **Kernel → Restart & Run All**, then save this notebook to keep the plots.
This notebook computes its figures on Great Lakes from the four existing sample
files. No training, sampling, downloads, or model fitting occur here.
Results have not been inspected by the author: draw conclusions after running.

## Context & Methods
Compare 300k versus approximately 500k for **256 and 1,024 training maps**.
Both use raw weights, 512 samples, sampling seed 123, and DPM-Solver (50 steps).
Seed 456 refers to the training continuation, not the sampling seed.

### Key assumptions and limits
All plots use the model's log/tanh-normalized field, **not physical HI units**.
Real reference maps are the exact configured **training subset**, not held-out
data. Agreement is a useful check but does not prove out-of-distribution or
physical validity. Pixel cosine similarity below is **not the paper's PCA/SSCD G**.
Use your existing allocated Jupyter session, not a login-node workload.
""")
cell("code", """
import os, sys, json, hashlib
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import yaml
from IPython.display import display, Markdown

PROJECT = Path('/home/jiamingp/diffusion_models_repo')
# This notebook can live in PROJECT/notebooks or in the evaluation checkout.
CODE = Path(os.environ.get('DIT_REVIEW_CODE_ROOT',
    '/scratch/huterer_root/huterer0/jiamingp/dit_eval_3dff836'))
sys.path.insert(0, str(CODE))
sys.path.insert(0, str(CODE / 'scripts'))
from simdiff_eval.io import iter_real_reference_batches_from_config
from simdiff_eval.metrics import batch_power_spectra
from validate_nf_generalize_fig2_dit_sample import validate_sample_file
SWEEP = 'nf_generalize_fig2_dit_l16_seed_restart500k_v1'
ROOT = PROJECT / 'results' / SWEEP / 'review_samples'
TAGS = {'d2p08':256, 'd2p10':1024}
COLORS = {'300k':'#0072B2', '500k':'#D55E00'}
plt.rcParams.update({'figure.dpi':110, 'font.size':11})
print('Reading:', ROOT)
print('Python:', sys.executable)
""")
cell("markdown", """
## Data
### Verify all four files before plotting
Check sample dimensions, finite values, checkpoint, sampling settings, and config
hash against each sampling sidecar. Fail rather than silently show an older run.
""")
cell("code", """
samples, configs = {}, {}
for tag in TAGS:
    for stage in ('300k','500k'):
        path = ROOT / f'{tag}_{stage}_raw_seed123_dpm50.npz'
        sidecar = json.loads(path.with_suffix('.validated.json').read_text())
        plan = sidecar['plan']
        assert plan['dataset'] == tag and plan['checkpoint_kind'] == stage
        assert plan['weights'] == 'raw' and plan['sampling_seed'] == 123
        config = Path(plan['config'])
        assert hashlib.sha256(config.read_bytes()).hexdigest() == plan['config_sha256']
        validate_sample_file(path, requested_checkpoint=Path(plan['checkpoint']),
            scheduler='DPMSolverMultistepScheduler', requested_steps=50)
        with np.load(path, allow_pickle=False) as data:
            assert int(data['seed']) == 123
            samples[tag,stage] = data['samples'].copy()
        configs[tag,stage] = config
        print(tag, stage, samples[tag,stage].shape, plan['checkpoint'])
    a = yaml.safe_load(configs[tag,'300k'].read_text())['data']
    b = yaml.safe_load(configs[tag,'500k'].read_text())['data']
    assert a == b, 'Data configuration changed: inspect before comparing'
    assert not np.array_equal(samples[tag,'300k'], samples[tag,'500k'])
""")
cell("markdown", """
## Results
### 1. What do the generated maps look like?
Six fixed indices, not selected for visual quality. The same color range is used
throughout. Rows are before/after continuation; columns are sample indices.
""")
cell("code", """
indices = np.linspace(0,511,6,dtype=int)
for tag,n in TAGS.items():
    fig, axes = plt.subplots(2,6,figsize=(14,5),constrained_layout=True)
    for row,stage in enumerate(('300k','500k')):
        for col,i in enumerate(indices):
            im = axes[row,col].imshow(samples[tag,stage][i,0],cmap='viridis',vmin=-1,vmax=1)
            axes[row,col].set_title(f'{stage}, sample {i}')
            axes[row,col].set_xticks([]); axes[row,col].set_yticks([])
    fig.suptitle(f'{n} training maps — generated normalized fields')
    fig.colorbar(im,ax=axes.ravel().tolist(),label='Normalized field',shrink=.7)
    plt.show()
""")
cell("markdown", """
### 2. Load the matching training maps
Normalization uses the complete configured subset. Do not estimate normalization
from a small preview. Counts must equal 256 and 1,024 respectively.
""")
cell("code", """
real = {}
for tag,n in TAGS.items():
    real[tag] = np.concatenate(list(iter_real_reference_batches_from_config(configs[tag,'300k'])))
    assert real[tag].shape == (n,1,128,128), real[tag].shape
    assert np.isfinite(real[tag]).all()
    print(tag, 'loaded exact training reference:', real[tag].shape)
""")
cell("markdown", """
### 3. Pixel-value distribution and power ratio
Left: how often different normalized pixel values occur. Right: mean generated
power divided by mean training-reference power. One means agreement at that
scale; larger k means smaller structures. Curves use all maps. No uncertainty
intervals are estimated here, so small differences are not significance claims.
""")
cell("code", """
for tag,n in TAGS.items():
    fig,axes = plt.subplots(1,2,figsize=(12,4),constrained_layout=True)
    arrays = [real[tag]] + [samples[tag,s] for s in ('300k','500k')]
    edges = np.linspace(min(float(x.min()) for x in arrays),max(float(x.max()) for x in arrays),101)
    axes[0].hist(real[tag].ravel(),bins=edges,density=True,histtype='step',color='black',label='Training reference')
    rp,k = batch_power_spectra(real[tag],nbins=25)
    denominator = rp.mean(axis=0)
    assert np.all(denominator > 0) and np.isfinite(denominator).all()
    for stage,style in [('300k','--'),('500k','-')]:
        arr = samples[tag,stage]
        axes[0].hist(arr.ravel(),bins=edges,density=True,histtype='step',color=COLORS[stage],linestyle=style,label=stage)
        gp,gk = batch_power_spectra(arr,nbins=25)
        assert np.allclose(k,gk)
        axes[1].plot(k,gp.mean(axis=0)/denominator,style,color=COLORS[stage],label=stage)
    axes[0].set(xlabel='Normalized pixel value',ylabel='Probability density')
    axes[1].axhline(1,color='black',linestyle=':')
    axes[1].set(xlabel='k (Fourier grid units)',ylabel='Mean generated / mean training power')
    for ax in axes: ax.legend(); ax.grid(alpha=.2)
    fig.suptitle(f'{n} training maps — normalized-field checks')
    plt.show()
""")
cell("markdown", """
### 4. Is each generated map close to a training map?
Compute the maximum **raw-pixel cosine similarity** over every training map,
without image rotation or translation. This is a diagnostic, not PCA/SSCD G.
Below each generated example is its highest-cosine training match.
""")
cell("code", """
def unit_rows(images):
    flat = images.reshape(len(images),-1).astype(np.float32)
    norms = np.linalg.norm(flat,axis=1,keepdims=True)
    assert np.all(norms > 0), 'Zero-norm image: cosine undefined'
    return flat / norms

for tag,n in TAGS.items():
    reference = unit_rows(real[tag])
    fig,ax = plt.subplots(figsize=(7,3),constrained_layout=True)
    matches = {}
    for stage in ('300k','500k'):
        q = unit_rows(samples[tag,stage])
        scores, ids = [], []
        for start in range(0,len(q),32):
            similarities = q[start:start+32] @ reference.T
            ids.extend(similarities.argmax(axis=1))
            scores.extend(similarities.max(axis=1))
        matches[stage] = (np.asarray(ids),np.asarray(scores))
        ax.hist(scores,bins=np.linspace(-1,1,61),histtype='step',label=stage,color=COLORS[stage])
        print(n, stage, 'median maximum pixel cosine:',float(np.median(scores)))
    ax.set(xlabel='Maximum pixel cosine similarity',ylabel='Generated-map count',title=f'{n} training maps')
    ax.legend(); plt.show()
    for stage in ('300k','500k'):
        ids,scores = matches[stage]
        fig,axes = plt.subplots(2,6,figsize=(14,5),constrained_layout=True)
        for col,i in enumerate(indices):
            for row,image in enumerate((samples[tag,stage][i,0],real[tag][ids[i],0])):
                axes[row,col].imshow(image,cmap='viridis',vmin=-1,vmax=1)
                axes[row,col].set_xticks([]); axes[row,col].set_yticks([])
            axes[0,col].set_title(f'Generated {i}')
            axes[1,col].set_title(f'Train {ids[i]} / cos {scores[i]:.3f}')
        fig.suptitle(f'{n} maps / {stage}: generated (top), nearest training (bottom)')
        plt.show()
""")
cell("markdown", """
## Takeaways
Inspect the plots before concluding that continuation improved quality.
- Do maps look plausible, and do they closely reproduce individual training maps?
- Does the 500k histogram better match the reference? Does its power ratio move toward one?
- Novelty alone is insufficient: noise can be novel and still wrong.

This notebook does not compute PCA/SSCD G, SSCD Frechet distance, held-out
distribution tests, physical-unit spectra, or uncertainty estimates. Do not
label these diagnostics as those measurements. Save the executed notebook to
retain the figures, and use a separate filename for any annotated version.
""")

if __name__ == '__main__':
    notebook = dict(cells=cells, metadata={'kernelspec':{'display_name':'Python 3',
        'language':'python','name':'python3'},'language_info':{'name':'python'}},nbformat=4,nbformat_minor=5)
    target = Path(__file__).resolve().parents[1] / 'notebooks/dit_l16_seed456_review.ipynb'
    target.write_text(json.dumps(notebook,indent=1)+'\n')
    print(target)
