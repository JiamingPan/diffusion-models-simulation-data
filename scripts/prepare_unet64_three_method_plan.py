"""Write a non-executable experiment inventory; never submit or train.

Copies only the selected existing 200k baseline configs into a new directory.
No speculative equivariant model config is passed off as runnable.
"""
import argparse
import hashlib
import json
from pathlib import Path
import yaml

ARMS = ('d4_periodic_shift','smooth_periodic_warp','d4_equivariant_unet')

def build(root):
    records=[]
    for power in range(6,16):
        n=2**power
        path=root/'local/nf_generalize_fig2/configs'/f'nf_fig2_u64_d2p{power:02d}_noaug_200k.yaml'
        cfg=yaml.safe_load(path.read_text())
        assert cfg['model']['class']=='UNet2DModel'
        assert cfg['model']['kwargs']['block_out_channels']==[16,32,64]
        assert not cfg.get('augmentations') and cfg['data']['label_path'] is None
        assert cfg['data']['zthin']==8 and sum(cfg['data']['n_samples'])*16==n
        train=cfg['train']
        assert train['batch_size']==32 and train['gradient_accumulation_steps']==1
        updates=train['num_epochs']*(n//32)
        assert 200000<=updates<201000
        for arm in ARMS:
            records.append(dict(arm=arm,N=n,baseline_config=str(path.relative_to(root)),
                baseline_config_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                baseline_checkpoint_dir=cfg['io']['output_dir'],baseline_reuse='pending remote verification',
                updates=updates,initialization='fresh; not a checkpoint continuation',
                status='NOT RUNNABLE: training/sampling integration and remote preflight required',
                output_dir=f'/scratch/huterer_root/huterer0/jiamingp/unet64_three_method_v1/{arm}/n{n}'))
    return records

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--project-dir',type=Path,default=Path(__file__).resolve().parents[1])
    p.add_argument('--output-dir',type=Path,required=True)
    args=p.parse_args(); records=build(args.project_dir)
    args.output_dir.mkdir(parents=True,exist_ok=False)
    (args.output_dir/'plan.json').write_text(json.dumps(dict(status='draft; no jobs submitted',runs=records),indent=2)+'\n')
    configs=args.output_dir/'baseline_configs'; configs.mkdir()
    for r in records[:]:
        src=args.project_dir/r['baseline_config']; dest=configs/src.name
        if not dest.exists(): dest.write_bytes(src.read_bytes())
    print('Draft tasks:',len(records),'Total updates:',sum(r['updates'] for r in records))
    print('GPU-hours unknown until measured; no training performed.')

if __name__=='__main__': main()
