#!/bin/bash
#SBATCH --job-name=u128aug_wave2
#SBATCH --account=huterer2
#SBATCH --partition=spgpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=80gb
#SBATCH --time=24:00:00
#SBATCH --no-requeue
#SBATCH --array=0-8%2
#SBATCH -o /home/jiamingp/diffusion_models_repo/logs/unet128_aug_wave2/train_%A_%a.out
#SBATCH -e /home/jiamingp/diffusion_models_repo/logs/unet128_aug_wave2/train_%A_%a.err
# Augmentation wave 2 (ledger unet128_aug_wave2): UNet-128, arms shiftonly / d4only / fliponly.
# Tasks 0-2 N=64, 3-5 N=128, 6-8 N=256 (arm order shiftonly, d4only, fliponly); see
# local/unet128_aug_wave2/runs.json. 200k updates each from the unchanged
# nf_fig2_u128_<tag>_noaug_200k config; seed 123, augmentation seed 1123.
exec bash /home/jiamingp/diffusion_models_repo/scripts/slurm/unet128_aug_wave2_train_task.sh
