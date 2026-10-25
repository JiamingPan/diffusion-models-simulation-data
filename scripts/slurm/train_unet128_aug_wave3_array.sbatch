#!/bin/bash
#SBATCH --job-name=u128aug_wave3
#SBATCH --account=huterer2
#SBATCH --partition=spgpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=80gb
#SBATCH --time=24:00:00
#SBATCH --no-requeue
#SBATCH --array=0-17%2
#SBATCH -o /scratch/huterer_root/huterer0/jiamingp/saved_runs/unet128_aug_wave3/logs/train_%A_%a.out
#SBATCH -e /scratch/huterer_root/huterer0/jiamingp/saved_runs/unet128_aug_wave3/logs/train_%A_%a.err
# Augmentation wave 3 (ledger unet128_aug_wave3): UNet-128 unconditional, patch-orbit sigma_c test.
# 18 tasks, task i = local/unet128_aug_wave3/runs.json[i]. First: 0 jitter4 N=16, 1 stride32 N=16, 2 shiftonly N=4,
# 3 d4shift N=4, 4 jitter4 N=32, 5 jitter4 N=64; then jitter4 128/256, shiftonly 8/16/32, d4shift 8/16/32,
# d4only 16/32, stride32 64/256.
# 200k optimizer updates each (num_epochs set per N; effective batch min(32, N)); seed 123, augmentation seed 1123.
# Training logs go to scratch (tqdm stderr was ~56 MB per wave-2 run at 100k epochs; these runs have up to 200k).
exec bash /home/jiamingp/diffusion_models_repo/scripts/slurm/unet128_aug_wave3_train_task.sh
