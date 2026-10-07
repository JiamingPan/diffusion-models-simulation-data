"""Print the complete remaining-sweep submission; --submit executes it once.

Requires the same pinned environment variables as launch_dit_adaln_zero_screen.sh.
No submission is made without --submit. The immutable plan must already exist.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

def commands(env):
    root, code = env['SCREEN_ROOT'], env['CODE_ROOT']
    common = ['sbatch', '--parsable', '--account='+env['SLURM_ACCOUNT'],
              '--nodes=1', '--ntasks=1', '--no-requeue', '--export=ALL', '--chdir='+code]
    jobs = []
    def add(key, script, resource, dependency=None, array=None):
        cmd = common + ['--job-name=dit_z_'+key] + resource
        if dependency:
            cmd += ['--dependency='+dependency, '--kill-on-invalid-dep=yes']
        if array:
            cmd += ['--array='+array]
        cmd += [f'--output={root}/logs/{key}_%A_%a.out',
                f'--error={root}/logs/{key}_%A_%a.err', code+'/scripts/slurm/'+script]
        jobs.append((key, cmd))
    gpu = ['--partition=spgpu', '--gres=gpu:1', '--cpus-per-task=4', '--mem=80G']
    add('data', 'preflight_dit_adaln_zero_data.sbatch',
        ['--partition=standard', '--cpus-per-task=8', '--mem=128G', '--time=02:00:00'])
    add('smoke', 'smoke_dit_zero_remaining.sbatch', gpu+['--time=01:00:00'], 'afterok:{data}')
    add('l16', 'train_dit_adaln_zero_screen.sbatch', gpu+['--time=2-00:00:00'], 'afterok:{smoke}', '0-4%4')
    # afterany permits the remaining depths even if one L16 run fails. No reruns are automatic.
    add('l12_l8', 'train_dit_adaln_zero_screen.sbatch', gpu+['--time=2-00:00:00'],
        'afterok:{smoke},afterany:{l16}', '5-24%4')
    for label, indices in [('l16', '0-4%1'), ('l12_l8', '5-24%1')]:
        add('sample_'+label, 'sample_dit_adaln_zero_screen.sbatch', gpu+['--time=01:00:00'],
            'aftercorr:{'+label+'}', indices)
        add('eval_'+label, 'evaluate_dit_adaln_zero_screen.sbatch', gpu+['--time=01:00:00'],
            'aftercorr:{sample_'+label+'}', indices)
    return jobs

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--submit', action='store_true')
    a = p.parse_args()
    required = ['CODE_ROOT','EXPECTED_COMMIT','PLAN_PATH','PLAN_SHA256','COSMODIFF_ROOT',
                'COSMODIFF_COMMIT','COSMODIFF_MANIFEST','COSMODIFF_MANIFEST_SHA256',
                'PYTHON_BIN','SCREEN_ROOT','SLURM_ACCOUNT','SCRATCH_ACCOUNT']
    env = {k: os.environ[k] for k in required}
    root, code = Path(env['SCREEN_ROOT']), Path(env['CODE_ROOT'])
    plan_path = Path(env['PLAN_PATH'])
    if plan_path.resolve() != (root/'plan.json').resolve():
        raise ValueError('Plan must belong to SCREEN_ROOT')
    plan = json.loads(plan_path.read_text())
    pairs = [(r['num_layers'], r['dataset_size']) for r in plan['runs']]
    expected = [(16,n) for n in [64,128,1024,4096,16384]]
    expected += [(d,2**k) for d in [12,8] for k in range(6,16)]
    if pairs != expected:
        raise ValueError('Expected exactly the approved ordered 25-run grid')
    actual = subprocess.check_output(['git','-C',str(code),'rev-parse','HEAD'], text=True).strip()
    if actual != env['EXPECTED_COMMIT']:
        raise ValueError('Code commit mismatch')
    if subprocess.check_output(['git','-C',str(code),'status','--porcelain'], text=True).strip():
        raise ValueError('Code checkout is dirty')
    subprocess.run([env['PYTHON_BIN'], str(code/'scripts/verify_dit_adaln_zero_screen.py'),
        '--plan',str(plan_path),'--plan-sha256',env['PLAN_SHA256'],
        '--cosmodiff-root',env['COSMODIFF_ROOT'],'--cosmodiff-commit',env['COSMODIFF_COMMIT'],
        '--cosmodiff-manifest',env['COSMODIFF_MANIFEST'],
        '--cosmodiff-manifest-sha256',env['COSMODIFF_MANIFEST_SHA256']], check=True)
    for key, cmd in commands(env):
        print(key+': '+shlex.join(cmd), flush=True)
    if not a.submit:
        print('PREVIEW ONLY; NO JOBS SUBMITTED')
        return
    quota = subprocess.check_output(['scratch-quota',env['SCRATCH_ACCOUNT']], text=True)
    subprocess.run([env['PYTHON_BIN'],str(code/'scripts/check_gl_scratch_quota.py'),
                    '--min-gib','1024','--min-files','5000'], input=quota, text=True, check=True)
    if (root/'data_preflight.json').exists():
        raise FileExistsError('Existing preflight receipt: inspect previous launch; do not duplicate')
    marker = root/'launch_remaining_v1'
    marker.mkdir()  # Exclusive marker; never auto-remove, even after partial submission.
    (root/'logs').mkdir(exist_ok=True)
    ids = {}
    for key, template in commands(env):
        cmd = [arg.format(**ids) for arg in template]
        job = subprocess.check_output(cmd, text=True).strip().split(';')[0]
        if not job.isdigit():
            raise RuntimeError('Unexpected sbatch output; inspect queue before retrying: '+job)
        ids[key] = job
        (marker/(key+'.json')).write_text(json.dumps({'job_id':job,'command':cmd},indent=2)+'\n')
        print(key.upper()+': '+job, flush=True)
    print('SUBMITTED; IDs recorded in '+str(marker))

if __name__ == '__main__':
    main()
