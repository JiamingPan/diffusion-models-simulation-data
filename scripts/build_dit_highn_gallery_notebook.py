"""Build the read-only saved-sample gallery notebook (stdlib only)."""
import json
from pathlib import Path
from textwrap import dedent

cells = []
def md(s):
    cells.append(dict(cell_type='markdown', metadata={}, source=dedent(s).strip()+'\n'))
def code(s):
    cells.append(dict(cell_type='code', metadata={}, execution_count=None, outputs=[], source=dedent(s).strip()+'\n'))

md('''
# L16 zero-init: inspect the saved high-N samples

**No training, sampling, EMA reconstruction, or file overwrites.** Run this on Great Lakes,
where the saved samples and original training data live. Select a Python kernel with
NumPy, Matplotlib and PyYAML, then Run All. Reading the real references may take several minutes.

We inspect N=2,048, 8,192 and 32,768. A *draw* is one saved generated map;
draw 0 means the first map, not a training iteration. Galleries show consecutive draws,
without selecting for appearance. Change `PAGE` to see the next 16.

Low cosine similarity means **low similarity to training maps**, not “junk.”
Look separately for patch seams, missing structure, and mismatched power.
All values are in each run's normalized model space, not physical density units.
Real controls are training maps, **not held-out validation data**.
''')
code('''
from pathlib import Path
import sys, json, hashlib
import numpy as np
import matplotlib.pyplot as plt

ROOT = Path('/scratch/huterer_root/huterer0/jiamingp/dit_l16_adalnzero_highn_screen_v1')
# Existing evaluation checkout contains the same streaming reference loader used by the evaluator.
CODE_ROOT = Path('/scratch/huterer_root/huterer0/jiamingp/dit_l16_eval_72c4fbd')
SIZES = (2048, 8192, 32768)
PAGE = 0                  # 0..31 for 512 saved draws, 16 per page
INSPECT_N = 2048
INSPECT_DRAW = 0
PLAN_SHA = '339b119b6bb5d80bdfd5abac222d1d166c9198fb202bd5bb93e0ccb05733cffd'

if not (CODE_ROOT / 'simdiff_eval/io.py').is_file():
    raise FileNotFoundError(f'Set CODE_ROOT to the evaluation checkout: {CODE_ROOT}')
sys.path.insert(0, str(CODE_ROOT))
from simdiff_eval.io import iter_real_reference_batches_from_config
plt.rcParams.update({'figure.dpi': 110, 'font.size': 10, 'axes.spines.top': False,
                     'axes.spines.right': False})

def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()

assert sha(ROOT/'plan.json') == PLAN_SHA, 'Unexpected plan'
plan = json.loads((ROOT/'plan.json').read_text())
''')
md('''
## Load saved samples and matching real controls

Sample receipts and file hashes are checked before plotting. Power means and boundary
statistics use **all configured real maps**, streamed in batches. The gray power band
uses up to 512 evenly spaced real maps and represents their 10th–90th percentile
spread, not a confidence interval. No GPU is required.
''')
code('''
def planes(a):
    a = np.asarray(a, dtype=np.float32)
    if a.ndim == 4 and a.shape[1] == 1:
        a = a[:, 0]
    assert a.ndim == 3 and a.shape[1:] == (128, 128), a.shape
    assert np.isfinite(a).all(), 'Nonfinite pixels'
    return a

def jump(a):
    # Edge position j compares columns j-1,j and rows j-1,j; positions are 1..127.
    return (np.abs(np.diff(a, axis=2)).mean(axis=1) +
            np.abs(np.diff(a, axis=1)).mean(axis=2)) / 2

def boundary(a):
    j = jump(a)
    edge = np.arange(1, a.shape[-1]) % 8 == 0
    return j[:, edge].mean(axis=1) / np.maximum(j[:, ~edge].mean(axis=1), 1e-30)

freq = np.fft.fftfreq(128)*128
radius = np.hypot(freq[:, None], freq[None, :])
edges = np.linspace(0, radius.max(), 26)
k = (edges[:-1] + edges[1:])/2
# Match the evaluator's 25 radial bins; exclude the DC mode from displayed bins below.
masks = [(radius >= lo) & (radius < hi) for lo, hi in zip(edges[:-1], edges[1:])]
def power(a):
    p = np.abs(np.fft.fft2(a, axes=(-2, -1)))**2 / (128*128)
    return np.stack([p[:, m].mean(axis=1) for m in masks], axis=1)

results = {}
for task, row in enumerate(plan['runs']):
    n = int(row['dataset_size'])
    if n not in SIZES:
        continue
    config = Path(row['config'])
    assert sha(config) == row['config_sha256'], 'Config changed'
    path = Path(row['sample_path'].format(seed=123, sample_label='dpm50_n512'))
    receipt = json.loads(path.with_suffix('.complete.json').read_text())
    assert receipt['status'] == 'complete' and receipt['plan_sha256'] == PLAN_SHA
    assert receipt['run_name'] == row['run_name']
    assert receipt['samples_sha256'] == sha(path), 'Samples changed'
    with np.load(path, allow_pickle=False) as z:
        samples = planes(z['samples']).copy()
    ev = json.loads((ROOT/'evaluation'/f'task_{task}.json').read_text())
    assert ev['run_name'] == row['run_name'] and ev['n'] == len(samples)
    cosine = np.asarray(ev['max_cosine'])
    assert cosine.shape == (len(samples),) and np.isfinite(cosine).all()
    wanted = np.unique(np.linspace(0, n-1, min(512, n), dtype=int))
    controls, boundaries, total, count = [], [], np.zeros(len(k)), 0
    for batch in iter_real_reference_batches_from_config(config, raw_batch_size=2):
        a = planes(batch)
        take = wanted[(wanted >= count) & (wanted < count+len(a))] - count
        if len(take):
            controls.append(a[take].copy())
        # Bound FFT temporary memory independently of raw-volume batch size.
        for start in range(0, len(a), 64):
            small = a[start:start+64]
            total += power(small).sum(axis=0)
            boundaries.append(boundary(small))
        count += len(a)
    assert count == n, f'Reference count {count} != configured N={n}'
    real = np.concatenate(controls)
    gp = np.concatenate([power(samples[i:i+64]) for i in range(0, len(samples), 64)])
    results[n] = dict(samples=samples, cosine=cosine, real=real, real_ids=wanted,
                      gp=gp, rp=power(real), mean_real=total/count,
                      rb=np.concatenate(boundaries), gb=boundary(samples))
    print(f'N={n:,}: {len(samples)} saved maps; {count:,} real references; {path}')
assert set(results) == set(SIZES), 'Missing requested run'

# One numerical color range for every gallery, derived only from real controls.
vmin, vmax = np.quantile(np.concatenate([r['real'].ravel() for r in results.values()]), [.005, .995])
for n, r in results.items():
    clipped = np.mean((r['samples'] < vmin) | (r['samples'] > vmax))
    print(f'N={n:,}: {clipped:.2%} of generated pixels outside displayed color range')
''')
md('''
## Galleries: real controls above, generated maps below

The first row contains four evenly spaced real controls. The next four rows contain
16 consecutive generated draws. Every image uses the same color range; colors outside
that range saturate (the fraction is printed above). Different N may use different
normalization parameters, so this is not a comparison of absolute physical densities.
`cos` is maximum pixel cosine to any training map; `B8` is the raw patch-edge/inside jump ratio.
''')
code('''
def gallery(n, page=0):
    r = results[n]
    ids = np.arange(page*16, min((page+1)*16, len(r['samples'])))
    if not len(ids) or page < 0:
        raise ValueError('PAGE must select existing saved draws')
    fig, axs = plt.subplots(5, 4, figsize=(12, 14), layout='constrained')
    for ax in axs.flat:
        ax.axis('off')
    for ax, i in zip(axs[0], np.linspace(0, len(r['real'])-1, 4, dtype=int)):
        im = ax.imshow(r['real'][i], vmin=vmin, vmax=vmax, cmap='viridis')
        ax.set_title(f'Real control {r["real_ids"][i]}')
    for ax, i in zip(axs[1:].flat, ids):
        ax.imshow(r['samples'][i], vmin=vmin, vmax=vmax, cmap='viridis')
        ax.set_title(f'Draw {i} | cos={r["cosine"][i]:.3f} | B8={r["gb"][i]:.2f}', fontsize=9)
    fig.colorbar(im, ax=axs, shrink=.65, label='Normalized model-space pixel value')
    fig.suptitle(f'L16 · patch 8 · zero-init · N={n:,} · page {page}')
    plt.show()
''')
for n in (2048,8192,32768):
    code(f'gallery({n}, PAGE)')
md('''
## Power and patch boundaries: separate checks

**Power plot:** x is Fourier frequency k (larger k = finer structures); y is generated
mean power divided by mean real power. A value of 0.3 means 30% of the reference power,
not 30% image accuracy. Dashed line = equal mean power. Gray = real-map spread, not a CI.
The first radial bin containing the DC mode is omitted here.

**Boundary plot:** every dot is one generated map. x is training-map similarity.
y is B8 divided by the real-map median B8. Elevated values indicate stronger jumps
on the 8-pixel grid relative to interiors. This tests seams, not overall validity.
''')
code('''
fig, axs = plt.subplots(2, 3, figsize=(15, 8), layout='constrained', sharex='row', sharey='row')
valid = (k > edges[1]) & (k <= 64)
for col, n in enumerate(SIZES):
    r = results[n]
    denom = r['mean_real']
    assert np.all(denom > 0)
    lo, hi = np.quantile(r['rp']/denom, [.1, .9], axis=0)
    ax = axs[0, col]
    ax.fill_between(k[valid], lo[valid], hi[valid], color='.85', label='Real controls p10–p90')
    ax.plot(k[valid], (r['gp'].mean(axis=0)/denom)[valid], color='#bc5731', label='Generated mean')
    ax.axhline(1, color='.25', linestyle='--')
    ax.set(title=f'N={n:,}', xlabel='k (Fourier grid units)', ylim=(0, None))
    ax.legend(fontsize=8)
    ax = axs[1, col]
    base = np.median(r['rb'])
    ax.scatter(r['cosine'], r['gb']/base, s=12, alpha=.6, color='#31688e')
    ax.axhline(1, color='.25', linestyle=':', label='Real median')
    ax.axhline(np.quantile(r['rb'], .99)/base, color='.25', linestyle='--', label='Real p99')
    ax.set(xlabel='Maximum cosine to training maps', xlim=(0, 1.005))
    ax.legend(fontsize=8)
axs[0, 0].set_ylabel('Mean generated power / mean real power')
axs[1, 0].set_ylabel('B8 / real-map median B8')
plt.show()
''')
md('''
## Inspect any individual draw

Edit `INSPECT_N` and `INSPECT_DRAW` below. The zoom overlays the model's 8-pixel patch grid.
The jump profile averages horizontal and vertical absolute adjacent-pixel differences.
**At x=20, compare columns 19 and 20, averaging over rows, and rows 19 and 20,
averaging over columns; then average those two numbers.** Indices here start at zero.
Guides at 8, 16, … mark patch edges. Repeated peaks at those guides suggest seams;
an isolated peak can also be a real filament.
''')
code('''
INSPECT_N = 2048
INSPECT_DRAW = 0
r = results[INSPECT_N]
assert 0 <= INSPECT_DRAW < len(r['samples'])
a = r['samples'][INSPECT_DRAW]
fig, axs = plt.subplots(1, 3, figsize=(15, 4.5), layout='constrained')
im = axs[0].imshow(a, cmap='viridis', vmin=vmin, vmax=vmax)
axs[0].set_title(f'N={INSPECT_N:,} · draw {INSPECT_DRAW}')
axs[1].imshow(a[32:96, 32:96], cmap='viridis', vmin=vmin, vmax=vmax)
for edge in np.arange(8, 64, 8)-.5:
    axs[1].axhline(edge, color='white', alpha=.4, linewidth=.6)
    axs[1].axvline(edge, color='white', alpha=.4, linewidth=.6)
axs[1].set_title('Central zoom · 8-pixel patch grid')
for ax in axs[:2]:
    ax.axis('off')
fig.colorbar(im, ax=list(axs[:2]), shrink=.7, label='Normalized pixel value')
x = np.arange(1, 128)
profiles = jump(r['real'])
lo, hi = np.quantile(profiles, [.1, .9], axis=0)
axs[2].fill_between(x, lo, hi, color='.85', label='Real controls p10–p90')
axs[2].plot(x, profiles.mean(axis=0), color='.3', label='Real controls mean')
axs[2].plot(x, jump(a[None])[0], color='#bc5731', label='Generated draw')
for edge in range(8, 128, 8):
    axs[2].axvline(edge, color='#31688e', alpha=.15, linewidth=.6)
axs[2].set(xlabel='Pixel edge position (row/column averaged)', ylabel='Mean absolute neighboring-pixel jump')
axs[2].legend(fontsize=8)
plt.show()
''')
md('''
## What this can—and cannot—establish

- Coherent-looking, low-similarity maps are possible; similarity alone cannot label them invalid.
- Near-normal patch-boundary statistics do not guarantee correct power or inference calibration.
- A power deficit remains a distributional discrepancy even if a gallery looks plausible.
- These are existing raw DPM-50 samples; this notebook does not test EMA or explain the cause of any discrepancy.
- The controls come from training data. Generalization and posterior-calibration claims need their own checks.
''')

out = Path(__file__).resolve().parents[1]/'notebooks/dit_l16_zero_init_highn_gallery.ipynb'
nb = dict(nbformat=4, nbformat_minor=5, metadata={'kernelspec': {'display_name':'Python 3', 'language':'python', 'name':'python3'}, 'language_info':{'name':'python','version':'3.10'}}, cells=cells)
for i, cell in enumerate(cells):
    cell['id'] = f'highn-{i:02d}'
out.write_text(json.dumps(nb, indent=1)+'\n')
print(out)
