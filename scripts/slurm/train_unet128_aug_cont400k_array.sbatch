#!/bin/bash
#SBATCH --job-name=u128aug_cont400k
#SBATCH --account=huterer2
#SBATCH --partition=spgpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16gb
#SBATCH --time=16:00:00
#SBATCH --no-requeue
#SBATCH --array=0-3%2
#SBATCH -o /home/jiamingp/diffusion_models_repo/logs/unet128_aug_sweep/train_cont400k_%A_%a.out
#SBATCH -e /home/jiamingp/diffusion_models_repo/logs/unet128_aug_sweep/train_cont400k_%A_%a.err
# Continue 4 width-128 augmentation-sweep runs from 200k to 400k updates (one leg, +200k updates),
# resuming from a sha256-verified copy of the 200k checkpoint in a NEW dir; originals untouched.
# Tasks (local/unet128_aug_sweep/cont400k_runs.json): 0 d2p06_d4shift, 1 d2p06_d4equiv,
# 2 d2p07_d4shift, 3 d2p07_d4equiv. Same train block as the original (fp16, batch 32, AdamW +
# CosineAnnealingWarmRestarts restored from the checkpoint), checkpoints every 20k updates.
# Time: original 200k legs took 11:33:13-12:05:46 on spgpu (sacct 63205244_0/1, 63205250_0/1);
# 16 h gives >=32% margin. Memory: max MaxRSS of those four tasks was 1.48 GB (63205250_1); 16 GB.
# CPUs: 4, as the original jobs; OMP_NUM_THREADS=8 as the original task script (dataset hash).
# Requires: seeded dirs (--mode seed) and dry-run receipts (local/unet128_aug_sweep/cont400k_dryrun/).
set -euo pipefail
: "${RUN_NAME:?submit with --export=ALL,RUN_NAME=unet128_aug_cont400k}"
source /home/jiamingp/diffusion_models_repo/scripts/slurm/unet128_aug_sweep_env.sh
export OMP_NUM_THREADS=8
TASK=${SLURM_ARRAY_TASK_ID}
echo "RUN_NAME=${RUN_NAME} task ${TASK} job ${SLURM_ARRAY_JOB_ID:-}_${TASK} host $(hostname)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
# The runner refuses unless the new output dir holds exactly the seeded checkpoint + seed_receipt.json.
"${PYTHON_BIN}" scripts/continue_unet128_aug_sweep.py --mode execute --index "${TASK}" \
  --runtime-root "${COSMODIFF_DIR}"
