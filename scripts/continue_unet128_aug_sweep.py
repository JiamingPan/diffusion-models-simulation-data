"""Continue four UNet-128 augmentation-sweep runs from 200k to 400k updates in NEW dirs.

Rows: local/unet128_aug_sweep/cont400k_runs.json. Original runs are never written to.

Modes (run in this order):
  seed     CPU, no torch. Create the new output dir, copy the base 200k checkpoint into it,
           verify sha256 of every copied file, write seed_receipt.json. Refuses if the dir exists.
  dry-run  CPU. Rebuild the dataset exactly as scripts/run_unet64_three_method.py does, check its
           hash, load the seeded checkpoint through cosmodiff.optim.train(resume_from_checkpoint=...)
           with num_epochs=0 (zero updates) into a throwaway --work-dir, and report the restored
           step counter, LR schedule, optimizer state and augmentation RNG. Writes only the
           dry-run receipt (local/unet128_aug_sweep/cont400k_dryrun/<run>.runner-<sha12>.json,
           refuses if it exists) and the throwaway work dir. Nothing is written under saved_runs.
  execute  GPU allocation only. Requires the dry-run receipt with unchanged code hashes and a
           new dir holding exactly the seeded checkpoint + seed_receipt.json; then resumes
           training in the new dir for additional_epochs (200k more updates).

How cosmodiff resumes (cosmo_diffusion_main, checked 2026-10-08, optim.py sha d3cb398a):
  optim.py:353-360  resume_from_checkpoint -> utils.load_checkpoint() replaces model, scheduler,
                    optimizer, lr_scheduler, and sets dataset.augmentations to the pickled
                    augmentations (ArrayDataset only); start_epoch = int(path.split('-')[-1]) + 1.
  optim.py:422-423  accelerator.load_state() restores weights (via hook), optimizer moments and
                    param_groups, LR scheduler, grad scaler, python/numpy/torch/cuda RNG.
  optim.py:425-436  post-hoc EMA is NOT restored; it restarts at the resume point.
  optim.py:456      global_step restarts at 0 (EMA burn-in and tensorboard steps count from resume).
  optim.py:466      runs num_epochs ADDITIONAL epochs: range(start_epoch, start_epoch + num_epochs).
  utils.py:458-462  load_checkpoint builds UNet2DModel for any non-cross-attention checkpoint, so a
                    D4ScalarUNet2DModel checkpoint would not be rebuilt as D4. This script replaces
                    utils.load_checkpoint in-process with a loader that builds the class named in
                    config.json and checks every tensor against the safetensors file. No file in
                    cosmodiff is modified.
"""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
MANIFEST = ROOT / 'local/unet128_aug_sweep/cont400k_runs.json'
DATA_REF = ROOT / 'local/unet128_aug_sweep/data_reference.json'
RECEIPT_DIR = ROOT / 'local/unet128_aug_sweep/cont400k_dryrun'
NEW_ROOT = '/scratch/huterer_root/huterer0/jiamingp/saved_runs/unet128_aug_sweep_cont400k/'
D4_FILE = str(ROOT / 'simdiff_eval/d4_unet64.py')
# The only code change since the original training (docs/decisions.md 2026-10-07, commit 8de7c9e):
# a torch-compat import + call added to d4_unet64.py. Removing exactly these lines reproduces the
# hash recorded in each original experiment.json byte for byte; checked at runtime below.
D4_COMPAT_BLOCK = ('\nfrom simdiff_eval.torch_compat import install_torch_backend_compat\n\n'
                   'install_torch_backend_compat(entry_point="d4_unet64")\n\n')


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def tree_hashes(root):
    root = Path(root)
    return {str(p.relative_to(root)): digest(p) for p in sorted(root.rglob('*')) if p.is_file()}


def receipt_for(row):
    """Dry-run receipt keyed by this script's own sha256: any edit to the runner needs a new dry run."""
    return RECEIPT_DIR / f"{row['run_name']}.runner-{digest(Path(__file__).resolve())[:12]}.json"


def load_row(args):
    rows = json.loads(MANIFEST.read_text())
    if args.run_name:
        match = [r for r in rows if r['run_name'] == args.run_name]
    else:
        match = [r for r in rows if r['array_index'] == args.index]
    if len(match) != 1:
        raise ValueError(f'No unique manifest row for {args.run_name or args.index}')
    row = match[0]
    assert row['output_dir'].startswith(NEW_ROOT) and row['output_dir'] != row['base_output_dir']
    assert Path(row['seed_checkpoint']).parent == Path(row['output_dir'])
    assert Path(row['seed_checkpoint']).name == Path(row['base_checkpoint']).name
    assert row['base_updates'] + row['additional_epochs'] * row['updates_per_epoch'] == row['target_updates']
    assert row['resume_start_epoch'] * row['updates_per_epoch'] == row['base_updates']
    assert row['dataset_size'] == row['batch_size'] * row['updates_per_epoch']
    return row


def seed(row):
    src, out = Path(row['base_checkpoint']), Path(row['output_dir'])
    dst = Path(row['seed_checkpoint'])
    base_dir = Path(row['base_output_dir'])
    if out.exists():
        raise FileExistsError(out)
    if not src.is_dir():
        raise FileNotFoundError(src)
    src_hashes = tree_hashes(src)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.mkdir(exist_ok=False)
    shutil.copytree(src, dst, copy_function=shutil.copy2)
    dst_hashes = tree_hashes(dst)
    if dst_hashes != src_hashes or tree_hashes(src) != src_hashes:
        raise RuntimeError(f'Copy mismatch for {dst}; leave it for manual inspection')
    for p in dst.rglob('*'):
        if p.is_file():
            p.chmod(0o444)
    receipt = dict(status='seeded', run_name=row['run_name'], source_checkpoint=str(src), seed_checkpoint=str(dst),
                   files=dst_hashes, weights_sha256=dst_hashes['diffusion_pytorch_model.safetensors'],
                   base_run_files={f: digest(base_dir / f) for f in
                                   ('training_config.yaml', 'experiment.json', 'data_identity.json', 'reload_check.json')},
                   created=time.strftime('%Y-%m-%dT%H:%M:%S%z'))
    with (out / 'seed_receipt.json').open('x') as f:
        json.dump(receipt, f, indent=2)
    print(f'SEEDED {dst}  weights sha256 {receipt["weights_sha256"]}  ({len(dst_hashes)} files, all hashes match source)')


def check_seed(row):
    out, dst = Path(row['output_dir']), Path(row['seed_checkpoint'])
    present = sorted(p.name for p in out.iterdir())
    expected = sorted([dst.name, 'seed_receipt.json'])
    if present != expected:
        raise FileExistsError(f'{out} must contain exactly {expected}; found {present}. '
                              'Refusing: never resume over existing continuation output.')
    receipt = json.loads((out / 'seed_receipt.json').read_text())
    assert receipt['status'] == 'seeded' and receipt['seed_checkpoint'] == str(dst)
    hashes = tree_hashes(dst)
    assert hashes == receipt['files'], 'Seeded checkpoint changed since seeding'
    assert hashes == tree_hashes(row['base_checkpoint']), 'Seeded checkpoint differs from base checkpoint'
    return receipt


def code_hashes(runtime):
    """Original runner's fingerprint (same file list) plus this script and the sampler."""
    spec = importlib.util.spec_from_file_location('run_unet64_three_method', ROOT / 'scripts/run_unet64_three_method.py')
    orig = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(orig)
    hashes = orig.fingerprint(runtime)
    extra = {str(ROOT / 'scripts/continue_unet128_aug_sweep.py'): digest(ROOT / 'scripts/continue_unet128_aug_sweep.py'),
             str(ROOT / 'scripts/sample_cosmodiff.py'): digest(ROOT / 'scripts/sample_cosmodiff.py')}
    return hashes, extra


def check_code_against_original(hashes, base_experiment):
    recorded = base_experiment['source_hashes']
    assert set(recorded) == set(hashes), 'Fingerprint file list differs from the original run'
    changed = sorted(k for k in hashes if hashes[k] != recorded[k])
    if changed not in ([], [D4_FILE]):
        raise AssertionError(f'Code changed since the original training: {changed}')
    if changed:
        text = Path(D4_FILE).read_text()
        assert text.count(D4_COMPAT_BLOCK) == 1
        restored = hashlib.sha256(text.replace(D4_COMPAT_BLOCK, '', 1).encode()).hexdigest()
        assert restored == recorded[D4_FILE], 'd4_unet64.py differs beyond the documented torch-compat lines'
    return dict(changed_since_original=changed,
                note='d4_unet64.py differs from the training-time file only by the torch-compat import+call '
                     '(verified: removing those lines reproduces the recorded sha256)' if changed else 'identical')


def build_dataset(row, utils, np, torch):
    """Same construction and checks as scripts/run_unet64_three_method.py:86-108."""
    import yaml
    cfg_path = ROOT / row['config']
    assert digest(cfg_path) == row['config_sha256'], 'Source config changed'
    config = yaml.safe_load(cfg_path.read_text())
    assert config['model']['class'] == 'UNet2DModel'
    assert config['model']['kwargs']['block_out_channels'] == [32, 64, 128]
    assert config['model']['kwargs']['norm_num_groups'] == 32
    assert config['model']['kwargs']['sample_size'] == 128
    assert not config.get('augmentations') and config['data']['label_path'] is None
    assert config['data']['seed'] is None
    assert config['data']['keep_on_cpu'] and config['train']['dataloader_num_workers'] == 0
    loaded = utils.parse_config_data(config)
    dataset = loaded['data']
    assert isinstance(dataset, utils.ArrayDataset), type(dataset)  # optim.py:357 only sets augs on ArrayDataset
    n = len(dataset)
    assert n == row['dataset_size'] and dataset.augmentations is None
    data_hash = hashlib.sha256()
    map_hashes = []
    for i in range(n):
        image = dataset[i]['images'].cpu().numpy()
        assert image.shape == (1, 128, 128) and np.isfinite(image).all()
        data_hash.update(np.ascontiguousarray(image).tobytes())
        map_hashes.append(hashlib.sha256(np.ascontiguousarray(image).tobytes()).hexdigest())
    ref = json.loads(DATA_REF.read_text())[str(n)]
    base_identity = json.loads((Path(row['base_output_dir']) / 'data_identity.json').read_text())
    assert data_hash.hexdigest() == ref['dataset_sha256'], 'Dataset hash differs from data_reference.json'
    assert map_hashes == ref['map_sha256'] == base_identity['map_sha256']
    assert base_identity['dataset_sha256'] == ref['dataset_sha256']
    norm = loaded['norm']
    normalization = None if norm is None else dict(method=norm.method, kwargs={
        k: (float(v) if np.isscalar(v) else v) for k, v in norm.kwargs.items()})
    return dataset, data_hash.hexdigest(), map_hashes, normalization


def install_loader(row, utils, torch, captured):
    """Replace utils.load_checkpoint (looked up at call time, optim.py:355) for this process only."""
    from safetensors.torch import load_file
    stock = utils.load_checkpoint

    def load(ckpt_path):
        assert Path(ckpt_path) == Path(row['seed_checkpoint']), ckpt_path
        cls_name = json.loads((Path(ckpt_path) / 'config.json').read_text())['_class_name']
        assert cls_name == row['model_class'], (cls_name, row['model_class'])
        if cls_name == 'D4ScalarUNet2DModel':
            import yaml
            from simdiff_eval.d4_unet64 import D4ScalarUNet2DModel
            model = D4ScalarUNet2DModel.from_pretrained(ckpt_path)
            cfg = yaml.safe_load((Path(ckpt_path) / 'checkpoint_config.yaml').read_text())
            noise_scheduler = utils._import_class(cfg['noise_scheduler']['class']).from_pretrained(ckpt_path)
            optimizer = utils._import_class(cfg['optimizer']['class'])(model.parameters())
            lr_scheduler = utils._import_class(cfg['lr_scheduler']['class'])(optimizer, **cfg['lr_scheduler']['kwargs'])
            assert not (Path(ckpt_path) / 'augmentations.pkl').exists()
            augmentations = None
        else:
            model, noise_scheduler, optimizer, lr_scheduler, augmentations = stock(ckpt_path)
        assert type(model).__name__ == row['model_class'], type(model)
        assert tuple(model.config.block_out_channels) == (32, 64, 128) and model.config.norm_num_groups == 32
        saved = load_file(str(Path(ckpt_path) / 'diffusion_pytorch_model.safetensors'))
        state = model.state_dict()
        assert set(saved) == set(state), 'Checkpoint keys differ from the rebuilt model'
        max_diff = max(float((state[k].float() - saved[k].float()).abs().max()) for k in saved)
        assert max_diff == 0.0, max_diff
        if row['arm_tag'] == 'd4shift':
            assert type(augmentations).__name__ == 'SymmetryPreserving'
        else:
            assert augmentations is None
        captured.update(model=model, optimizer=optimizer, lr_scheduler=lr_scheduler,
                        noise_scheduler=noise_scheduler, augmentations=augmentations,
                        load_weight_max_abs_diff=max_diff)
        return model, noise_scheduler, optimizer, lr_scheduler, augmentations

    utils.load_checkpoint = load
    return stock


def optimizer_summary(optimizer, lr_scheduler):
    steps = [float(s['step']) for s in optimizer.state.values() if 'step' in s]
    g = optimizer.param_groups[0]
    return dict(optimizer_class=type(optimizer).__name__, n_param_states=len(optimizer.state),
                adam_step_min=min(steps) if steps else None, adam_step_max=max(steps) if steps else None,
                param_group={k: (v if isinstance(v, (int, float, str, bool, type(None))) else list(v) if isinstance(v, tuple) else str(v))
                             for k, v in g.items() if k != 'params'},
                lr_scheduler_class=type(lr_scheduler).__name__, lr_scheduler_last_epoch=lr_scheduler.last_epoch,
                lr_scheduler_T_cur=getattr(lr_scheduler, 'T_cur', None), lr_scheduler_T_0=getattr(lr_scheduler, 'T_0', None),
                lr_scheduler_base_lrs=list(lr_scheduler.base_lrs), lr_scheduler_eta_min=getattr(lr_scheduler, 'eta_min', None),
                lr_now=lr_scheduler.get_last_lr()[0])


def aug_replay_check(row, aug, n_fetches):
    """Is the pickled RNG the seed+1000 stream advanced by exactly the 200k-update fetch count?"""
    from simdiff_eval.unet64_sweep_transforms import SymmetryPreserving
    fresh = SymmetryPreserving(row['training_seed'] + 1000)
    start_equal = fresh.rng.bit_generator.state == aug.rng.bit_generator.state
    for _ in range(n_fetches):
        fresh.rng.integers(8)
        fresh.rng.integers(128, size=2)
    return dict(pickled_state_equals_fresh_seed=start_equal, n_fetches_simulated=n_fetches,
                pickled_state_equals_seed_advanced_by_n_fetches=fresh.rng.bit_generator.state == aug.rng.bit_generator.state)


def d4_equivariance(model, torch):
    from simdiff_eval.d4_unet64 import d4
    torch.manual_seed(0)
    x = torch.randn(1, 1, 128, 128)
    t = torch.tensor([137])
    model.eval()
    with torch.no_grad():
        base = model(x, t).sample
        err = max(float((model(d4(x, g), t).sample - d4(base, g)).abs().max()) for g in range(1, 8))
    return dict(max_abs_equivariance_error=err, output_abs_max=float(base.abs().max()))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--mode', choices=('seed', 'dry-run', 'execute'), required=True)
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument('--index', type=int)
    g.add_argument('--run-name')
    p.add_argument('--runtime-root', type=Path, default=Path('/home/jiamingp/Diffusion_model/cosmo_diffusion_main'))
    p.add_argument('--work-dir', type=Path, help='dry-run only: throwaway dir for train(num_epochs=0) logs')
    p.add_argument('--skip-aug-replay', action='store_true', help='dry-run only: skip the 6.4M-fetch RNG replay')
    args = p.parse_args()
    row = load_row(args)
    if args.mode == 'seed':
        return seed(row)

    os.environ['HF_HUB_OFFLINE'] = '1'; os.environ['TRANSFORMERS_OFFLINE'] = '1'
    if os.environ.get('OMP_NUM_THREADS') != '8':
        raise RuntimeError('OMP_NUM_THREADS=8 required (dataset hash depends on the thread count)')
    from simdiff_eval.torch_compat import install_torch_backend_compat
    install_torch_backend_compat(entry_point='unet64_three_method')
    import inspect
    import numpy as np
    import torch
    import diffusers
    import yaml
    from simdiff_eval.d4_unet64 import register_model
    register_model()
    runtime = args.runtime_root.resolve()
    sys.path.insert(0, str(runtime))
    from cosmodiff import utils, optim
    assert runtime in Path(inspect.getfile(utils)).resolve().parents
    assert torch.get_num_threads() == 8, torch.get_num_threads()

    base_dir = Path(row['base_output_dir'])
    base_experiment = json.loads((base_dir / 'experiment.json').read_text())
    assert base_experiment['N'] == row['dataset_size'] and base_experiment['seed'] == row['training_seed']
    assert base_experiment['arm'] == row['arm']
    hashes, extra = code_hashes(runtime)
    code_status = check_code_against_original(hashes, base_experiment)
    preflight = json.loads((ROOT / 'local/unet128_aug_sweep/preflight_receipt.json').read_text())
    assert preflight['torch'] == torch.__version__ and preflight['diffusers'] == diffusers.__version__
    seed_receipt = check_seed(row)
    base_train_cfg = yaml.safe_load((base_dir / 'training_config.yaml').read_text())
    assert base_train_cfg['model']['class'] == row['model_class']
    assert base_train_cfg['train']['checkpoint_every_n_epochs'] == row['checkpoint_every_n_epochs']
    assert base_train_cfg['train']['batch_size'] == row['batch_size']
    # Doubling 200k -> 400k: the additional epochs equal the original num_epochs, so train() gets the
    # original train block unchanged (num_epochs counts epochs after the resume point, optim.py:466).
    assert base_train_cfg['train']['num_epochs'] == row['additional_epochs']
    train_kwargs = dict(base_train_cfg['train'])

    dataset, data_sha, map_hashes, normalization = build_dataset(row, utils, np, torch)
    random_seed_note = 'not re-seeded: python/numpy/torch/cuda RNG restored from random_states_0.pkl by accelerator.load_state'
    captured = {}
    stock_loader = install_loader(row, utils, torch, captured)

    if args.mode == 'dry-run':
        receipt_path = receipt_for(row)
        if receipt_path.exists():
            raise FileExistsError(receipt_path)
        if args.work_dir is None or str(args.work_dir.resolve()).startswith(('/scratch/', '/gpfs/')):
            raise ValueError('--work-dir must be a throwaway dir outside scratch saved_runs')
        work = args.work_dir.resolve() / row['run_name']
        work.mkdir(parents=True, exist_ok=False)
        stock_class = None
        if row['model_class'] == 'D4ScalarUNet2DModel':
            try:
                stock_class = type(stock_loader(row['seed_checkpoint'])[0]).__name__
            except Exception as exc:
                stock_class = f'raised {type(exc).__name__}: {str(exc)[:300]}'
        kw = dict(train_kwargs, num_epochs=0, force_cpu=True, mixed_precision='no', verbose=False)
        t0 = time.monotonic()
        optim.train(dataset, None, output_dir=str(work), resume_from_checkpoint=row['seed_checkpoint'], **kw)
        model = captured['model']
        from safetensors.torch import load_file
        saved = load_file(str(Path(row['seed_checkpoint']) / 'diffusion_pytorch_model.safetensors'))
        state = model.state_dict()
        weight_diff_after_load_state = max(float((state[k].float() - saved[k].float()).abs().max()) for k in saved)
        rng_file = torch.load(str(Path(row['seed_checkpoint']) / 'random_states_0.pkl'), map_location='cpu')
        scaler = torch.load(str(Path(row['seed_checkpoint']) / 'scaler.pt'), map_location='cpu')
        report = dict(
            status='dry_run_passed', run_name=row['run_name'], mode='CPU, optim.train(num_epochs=0): zero updates',
            seed_checkpoint=row['seed_checkpoint'], seed_weights_sha256=seed_receipt['weights_sha256'],
            dataset_sha256=data_sha, dataset_matches_data_reference=True, N=len(dataset),
            normalization=normalization, code_status=code_status, code_hashes=hashes, extra_hashes=extra,
            torch=torch.__version__, diffusers=diffusers.__version__,
            model_class=type(model).__name__, stock_load_checkpoint_class_for_this_checkpoint=stock_class,
            load_weight_max_abs_diff=captured['load_weight_max_abs_diff'],
            weight_max_abs_diff_after_accelerator_load_state=weight_diff_after_load_state,
            resumed=optimizer_summary(captured['optimizer'], captured['lr_scheduler']),
            grad_scaler_saved=dict((k, (float(v) if hasattr(v, '__float__') else v)) for k, v in scaler.items()),
            saved_rng_keys=sorted(rng_file) if isinstance(rng_file, dict) else str(type(rng_file)),
            start_epoch=row['resume_start_epoch'], final_epoch=row['final_epoch'],
            additional_epochs=row['additional_epochs'], additional_updates=row['additional_epochs'] * row['updates_per_epoch'],
            checkpoint_every_n_epochs=row['checkpoint_every_n_epochs'],
            planned_checkpoint_updates=[c['updates'] for c in row['planned_checkpoints']],
            train_kwargs_execute=train_kwargs, global_rng=random_seed_note,
            ema='fresh PostHocEMA from the resume point (optim.py:425-436), burn-in 1000 steps of the continuation; not used at sampling',
            dataset_augmentations_after_resume=repr(type(dataset.augmentations).__name__ if dataset.augmentations is not None else None),
            seconds=time.monotonic() - t0, work_dir=str(work))
        if row['arm_tag'] == 'd4shift':
            assert dataset.augmentations is captured['augmentations']
            n_fetches = row['base_updates'] * row['batch_size']
            report['augmentation_rng'] = (dict(skipped=True) if args.skip_aug_replay
                                          else aug_replay_check(row, dataset.augmentations, n_fetches))
        else:
            assert dataset.augmentations is None
            report['d4_equivariance'] = d4_equivariance(model, torch)
        RECEIPT_DIR.mkdir(parents=True, exist_ok=True)
        with receipt_path.open('x') as f:
            json.dump(report, f, indent=2, default=str)
        print(json.dumps({k: report[k] for k in ('status', 'run_name', 'model_class', 'stock_load_checkpoint_class_for_this_checkpoint',
                                               'load_weight_max_abs_diff', 'weight_max_abs_diff_after_accelerator_load_state',
                                               'resumed', 'start_epoch', 'final_epoch', 'additional_updates',
                                               'dataset_augmentations_after_resume')}, indent=1, default=str))
        print(json.dumps(report.get('augmentation_rng', report.get('d4_equivariance')), default=str))
        print('DRY RUN PASSED:', receipt_path, '(no training performed; nothing written under saved_runs)')
        return

    # ---- execute ----
    if not os.environ.get('SLURM_JOB_ID') or not torch.cuda.is_available():
        raise RuntimeError('Existing GPU allocation required')
    if int(os.environ.get('WORLD_SIZE', '1')) != 1:
        raise RuntimeError('Single GPU only')
    dry = json.loads(receipt_for(row).read_text())
    assert dry['status'] == 'dry_run_passed' and dry['code_hashes'] == hashes and dry['extra_hashes'] == extra, \
        'Code changed since the dry run: rerun the dry run'
    assert dry['dataset_sha256'] == data_sha and dry['seed_weights_sha256'] == seed_receipt['weights_sha256']
    out = Path(row['output_dir'])
    config = dict(base_train_cfg)
    config['io'] = dict(base_train_cfg['io'], output_dir=str(out))
    with (out / 'training_config.yaml').open('x') as f:
        yaml.safe_dump(config, f)
    with (out / 'data_identity.json').open('x') as f:
        json.dump(dict(N=len(dataset), dataset_sha256=data_sha, map_sha256=map_hashes, normalization=normalization,
                       source_config=row['config']), f, indent=2)
    with (out / 'experiment.json').open('x') as f:
        json.dump(dict(arm=row['arm'], N=len(dataset), seed=row['training_seed'], source_config=row['config'],
                       source_sha256=row['config_sha256'], dataset_sha256=data_sha, continuation_of=row['base_run'],
                       resume_from=row['seed_checkpoint'], base_checkpoint=row['base_checkpoint'],
                       base_updates=row['base_updates'], target_updates=row['target_updates'],
                       start_epoch=row['resume_start_epoch'], final_epoch=row['final_epoch'],
                       source_hashes=hashes, extra_hashes=extra, code_status=code_status,
                       augmentation=row['augmentation'], global_rng=random_seed_note,
                       ema='restarted at the resume point; snapshots cover updates 200k..400k only',
                       metrics='metrics.json covers the continuation leg only',
                       slurm_job=os.environ.get('SLURM_JOB_ID'), slurm_array=os.environ.get('SLURM_ARRAY_JOB_ID'),
                       slurm_task=os.environ.get('SLURM_ARRAY_TASK_ID')), f, indent=2)
    result = optim.train(dataset, None, output_dir=str(out), resume_from_checkpoint=row['seed_checkpoint'], **train_kwargs)
    metrics_path = out / 'metrics.json'
    if metrics_path.exists():
        raise FileExistsError(metrics_path)
    utils.write_metrics(result['metrics'], str(metrics_path))
    sys.path.insert(0, str(ROOT / 'scripts'))
    from sample_cosmodiff import _load_for_sampling, _find_latest_checkpoint
    final = _find_latest_checkpoint(out)
    assert final.name == f"checkpoint-epoch-{row['final_epoch']:04d}", final
    sched = torch.load(str(final / 'scheduler.bin'), map_location='cpu')
    opt = torch.load(str(final / 'optimizer.bin'), map_location='cpu')
    steps = [float(s['step']) for s in opt['state'].values() if 'step' in s]
    reloaded, _ = _load_for_sampling(final, out / 'training_config.yaml')
    assert type(reloaded).__name__ == row['model_class'], type(reloaded)
    trained = captured['model']
    trained = trained.module if hasattr(trained, 'module') else trained
    tstate = {k: v.detach().cpu() for k, v in trained.state_dict().items()}
    rstate = reloaded.state_dict()
    assert set(tstate) == set(rstate)
    max_abs = max(float((tstate[k].float() - v.cpu().float()).abs().max()) for k, v in rstate.items())
    # "Updates" in this sweep count batches (200k = 100000 epochs x 2 batches at N=64). Under fp16,
    # accelerate skips optimizer and LR-scheduler steps on grad-scaler overflow, so the scheduler /
    # Adam counters run slightly below the batch count (199925 at the 200k d2p06_d4equiv seed).
    seed_sched = torch.load(str(Path(row['seed_checkpoint']) / 'scheduler.bin'), map_location='cpu')
    n_batches = len(result['metrics']['loss'])
    with (out / 'reload_check.json').open('x') as f:
        json.dump(dict(status='sampler_loader_reload_passed', checkpoint=str(final), model_class=type(reloaded).__name__,
                       max_abs_weight_diff_vs_in_memory=max_abs, lr_scheduler_last_epoch=sched.get('last_epoch'),
                       seed_lr_scheduler_last_epoch=seed_sched.get('last_epoch'), adam_step_min=min(steps),
                       adam_step_max=max(steps), n_batches_this_leg=n_batches,
                       expected_batches_this_leg=row['additional_epochs'] * row['updates_per_epoch']), f, indent=2)
    assert n_batches == row['additional_epochs'] * row['updates_per_epoch'], n_batches
    assert seed_sched.get('last_epoch') < sched.get('last_epoch') <= seed_sched.get('last_epoch') + n_batches
    print('CONTINUATION DONE:', final)


if __name__ == '__main__':
    main()
