"""Prepare exactly 25 remaining models, without submitting jobs."""
import argparse
import json
from pathlib import Path
from prepare_dit_adaln_zero_refresh import draft_runs

def build(out, checkpoints):
    plan, configs = draft_runs(out, checkpoints, 300000)
    existing_l16 = {256, 512, 2048, 8192, 32768}
    plan['runs'] = sorted(
        [r for r in plan['runs'] if not (r['num_layers'] == 16 and r['dataset_size'] in existing_l16)],
        key=lambda r: (-r['num_layers'], r['dataset_size']))
    assert len(plan['runs']) == 25
    plan['selection'] = {'purpose': 'remaining 300k zero-init depth sweep',
                         'completed_l16_sizes_excluded': sorted(existing_l16),
                         'qualification': 'N256 ablation comparability must be checked before pooling'}
    return plan, {Path(r['config']): configs[Path(r['config'])] for r in plan['runs']}

if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out-dir', type=Path, required=True)
    p.add_argument('--checkpoint-root', type=Path, required=True)
    a = p.parse_args()
    out = a.out_dir.resolve()
    if out.exists():
        raise FileExistsError(f'Preserving existing directory: {out}')
    plan, configs = build(out, a.checkpoint_root)
    (out/'configs').mkdir(parents=True)
    for path, content in configs.items():
        path.write_text(content)
    (out/'plan.json').write_text(json.dumps(plan, indent=2)+'\n')
    for i, r in enumerate(plan['runs']):
        print(f'{i:2d}: L{r["num_layers"]} N={r["dataset_size"]} nominal_updates={r["nominal_updates"]}')
    print('PREPARED 25 RUNS; NO JOBS SUBMITTED')
