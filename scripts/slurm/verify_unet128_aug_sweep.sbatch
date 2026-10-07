#!/bin/bash
#SBATCH --job-name=u128aug_verify
#SBATCH --account=huterer2
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64gb
#SBATCH --time=02:00:00
#SBATCH --no-requeue
#SBATCH -o /home/jiamingp/diffusion_models_repo/logs/unet128_aug_sweep/verify_%j.out
#SBATCH -e /home/jiamingp/diffusion_models_repo/logs/unet128_aug_sweep/verify_%j.err
# CPU-only pre-submission checks (no training): runner preflight receipt, dataset/normalization
# reference per N, width-128 model checks, D4 equivariance at width 128, sampler-loader round trip,
# and scripts/sample_cosmodiff.py on the saved width-128 timing checkpoints.
source /home/jiamingp/diffusion_models_repo/scripts/slurm/unet128_aug_sweep_env.sh
# Scratch checkpoints on node-local /tmp: NFS (turbo) cleanup fails with "Directory not empty".
export TMPDIR=$(mktemp -d /tmp/u128aug_verify.XXXXXX)
trap 'rm -rf "$TMPDIR"' EXIT
"${PYTHON_BIN}" scripts/verify_unet128_aug_sweep.py --runtime-root "${COSMODIFF_DIR}"
