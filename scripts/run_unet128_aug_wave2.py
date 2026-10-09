"""Fresh-training runner for augmentation wave 2 (ledger unet128_aug_wave2). Default mode only validates.

Same procedure as scripts/run_unet64_three_method.py (preflight receipt with code/runtime
hashes; pinned config and normalized-dataset hashes; new scratch output; per-fetch
augmentation via dataset.augmentations seeded args.seed + 1000; sampler-loader reload
check), restricted to the UNet2DModel width-128 config and the three wave-2 arms:

  shift_only : simdiff_eval.aug_wave2_transforms.ShiftOnly
  d4_only    : D4Only
  flip_only  : FlipOnly

Never submits jobs. Training needs --execute inside a GPU allocation.
"""
import argparse
import hashlib
import inspect
import json
import os
from pathlib import Path
import random
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'scripts'))

from run_unet64_three_method import digest  # noqa: E402  (same sha256 helper as the wave-1 runner)

ARM_CHOICES = ('shift_only', 'd4_only', 'flip_only')
TEST_FILES = ('tests/test_aug_wave2_transforms.py',)


def fingerprint(runtime):
    paths = [ROOT / 'simdiff_eval/aug_wave2_transforms.py', ROOT / 'scripts/run_unet128_aug_wave2.py',
             ROOT / 'scripts/run_unet64_three_method.py', ROOT / 'simdiff_eval/torch_compat.py']
    paths += [ROOT / t for t in TEST_FILES]
    paths += [runtime / 'cosmodiff' / f for f in ('utils.py', 'optim.py', 'transform.py', 'augment.py')]
    return {str(p): digest(p) for p in paths}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--runtime-root', type=Path, required=True)
    parser.add_argument('--receipt', type=Path, required=True)
    parser.add_argument('--arm', choices=ARM_CHOICES)
    parser.add_argument('--config', type=Path)
    parser.add_argument('--expected-config-sha256')
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--expected-dataset-sha256',
                        help='Normalized-dataset hash from local/unet128_aug_sweep/data_reference.json for this config')
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    os.environ['HF_HUB_OFFLINE'] = '1'; os.environ['TRANSFORMERS_OFFLINE'] = '1'
    from simdiff_eval.torch_compat import install_torch_backend_compat
    install_torch_backend_compat(entry_point='unet128_aug_wave2')
    import torch
    import diffusers
    import numpy as np
    import yaml
    from simdiff_eval.aug_wave2_transforms import ARMS
    runtime = args.runtime_root.resolve()
    sys.path.insert(0, str(runtime))
    from cosmodiff import utils, optim
    assert runtime in Path(inspect.getfile(utils)).resolve().parents
    hashes = fingerprint(runtime)
    if not args.execute:
        if args.receipt.exists(): raise FileExistsError(args.receipt)
        import pytest
        started = time.monotonic()
        status = pytest.main(['-q', '-p', 'no:cacheprovider'] + [str(ROOT / t) for t in TEST_FILES])
        if status: raise SystemExit(status)
        args.receipt.parent.mkdir(parents=True, exist_ok=True)
        with args.receipt.open('x') as f:
            json.dump(dict(status='implementation_tests_passed', hashes=hashes,
                           torch=torch.__version__, diffusers=diffusers.__version__,
                           seconds=time.monotonic() - started,
                           limitations='Transform unit tests only; no model, GPU or baseline artifact check'), f, indent=2)
        print('PREFLIGHT PASSED:', args.receipt, 'No training or sampling performed.')
        return
    if not all([args.arm, args.config, args.output_dir, args.expected_config_sha256, args.expected_dataset_sha256]):
        raise ValueError('Training requires arm, config, output-dir, expected-config-sha256 and expected-dataset-sha256')
    if not os.environ.get('SLURM_JOB_ID') or not torch.cuda.is_available():
        raise RuntimeError('Existing GPU allocation required')
    if int(os.environ.get('WORLD_SIZE', '1')) != 1: raise RuntimeError('Single GPU only')
    receipt = json.loads(args.receipt.read_text())
    assert receipt['status'] == 'implementation_tests_passed' and receipt['hashes'] == hashes, 'Changed code/runtime: rerun preflight'
    assert receipt['torch'] == torch.__version__ and receipt['diffusers'] == diffusers.__version__
    assert digest(args.config) == args.expected_config_sha256, 'Source config changed'
    output = args.output_dir.resolve()
    if not str(output).startswith(('/scratch/', '/gpfs/accounts/')): raise ValueError('New scratch output required')
    if output.exists(): raise FileExistsError(output)
    config = yaml.safe_load(args.config.read_text())
    assert config['model']['class'] == 'UNet2DModel'
    # Width-128 sweep: must match the existing nf_fig2_u128_*_noaug_200k baseline configs.
    assert config['model']['kwargs']['block_out_channels'] == [32, 64, 128]
    assert config['model']['kwargs']['norm_num_groups'] == 32
    assert config['model']['kwargs']['sample_size'] == 128
    assert not config.get('augmentations') and config['data']['label_path'] is None
    assert config['data']['seed'] is None, 'Review selection semantics before seeding'
    assert config['data']['keep_on_cpu'] and config['train']['dataloader_num_workers'] == 0
    # Build exact original dataset before overriding retrieval augmentation.
    loaded = utils.parse_config_data(config); dataset = loaded['data']
    n = len(dataset)
    assert n in (64, 128, 256), n
    assert dataset.augmentations is None
    data_hash = hashlib.sha256(); map_hashes = []
    for i in range(n):
        image = dataset[i]['images'].cpu().numpy()
        assert image.shape == (1, 128, 128) and np.isfinite(image).all()
        data_hash.update(np.ascontiguousarray(image).tobytes())
        map_hashes.append(hashlib.sha256(np.ascontiguousarray(image).tobytes()).hexdigest())
    assert data_hash.hexdigest() == args.expected_dataset_sha256, 'Selected maps/normalization differ from the verified baseline'
    norm = loaded['norm']
    normalization = None if norm is None else dict(method=norm.method, kwargs={k: (float(v) if np.isscalar(v) else v) for k, v in norm.kwargs.items()})
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed)
    dataset.augmentations = ARMS[args.arm](args.seed + 1000)
    config['io']['output_dir'] = str(output)
    model_out = utils.parse_config_model(config)
    model = model_out['model']
    assert type(model).__name__ == 'UNet2DModel'
    assert tuple(model.config.block_out_channels) == (32, 64, 128) and model.config.norm_num_groups == 32
    output.mkdir(parents=True, exist_ok=False)
    with (output / 'data_identity.json').open('x') as f:
        json.dump(dict(N=n, dataset_sha256=data_hash.hexdigest(), map_sha256=map_hashes,
                       normalization=normalization, source_config=str(args.config)), f, indent=2)
    with (output / 'training_config.yaml').open('x') as f: yaml.safe_dump(config, f)
    with (output / 'experiment.json').open('x') as f:
        json.dump(dict(arm=args.arm, N=n, seed=args.seed, augmentation_seed=args.seed + 1000,
                       source_config=str(args.config), source_sha256=args.expected_config_sha256,
                       dataset_sha256=data_hash.hexdigest(),
                       model_parameters=sum(p.numel() for p in model_out['model'].parameters()),
                       source_hashes=hashes, augmentation=repr(dataset.augmentations)), f, indent=2)
    result = optim.train(dataset, model_out['model'], optimizer=model_out['optimizer'],
                         noise_scheduler=model_out['noise_scheduler'], lr_scheduler=model_out['lr_scheduler'],
                         output_dir=str(output), **config['train'])
    utils.write_metrics(result['metrics'], str(output / 'metrics.json'))
    # Reload the final checkpoint through the sampler's own loader, as sampling will.
    from sample_cosmodiff import _load_for_sampling, _find_latest_checkpoint
    final = _find_latest_checkpoint(output)
    reloaded, _ = _load_for_sampling(final, output / 'training_config.yaml')
    assert type(reloaded).__name__ == 'UNet2DModel', type(reloaded)
    assert tuple(reloaded.config.block_out_channels) == (32, 64, 128) and reloaded.config.norm_num_groups == 32
    trained = model.module if hasattr(model, 'module') else model
    trained_state = {k: v.detach().cpu() for k, v in trained.state_dict().items()}
    reloaded_state = reloaded.state_dict()
    assert set(trained_state) == set(reloaded_state), 'Reloaded parameter names differ'
    max_abs_diff = max(float((trained_state[k].float() - v.cpu().float()).abs().max()) for k, v in reloaded_state.items())
    with (output / 'reload_check.json').open('x') as f:
        json.dump(dict(status='sampler_loader_reload_passed', checkpoint=str(final),
                       model_class=type(reloaded).__name__, block_out_channels=list(reloaded.config.block_out_channels),
                       norm_num_groups=reloaded.config.norm_num_groups, max_abs_weight_diff_vs_in_memory=max_abs_diff,
                       parameters=sum(p.numel() for p in reloaded.parameters())), f, indent=2)


if __name__ == '__main__': main()
