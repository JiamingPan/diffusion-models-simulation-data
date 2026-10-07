#!/bin/bash
#SBATCH --job-name=u128aug_warp
#SBATCH --account=huterer2
#SBATCH --partition=spgpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=80gb
#SBATCH --time=24:00:00
#SBATCH --no-requeue
#SBATCH --array=0-5%2
#SBATCH -o /home/jiamingp/diffusion_models_repo/logs/unet128_aug_sweep/train_warp_%A_%a.out
#SBATCH -e /home/jiamingp/diffusion_models_repo/logs/unet128_aug_sweep/train_warp_%A_%a.err
# Width-128 augmentation sweep, arm warp: smooth periodic warp, novelty-only distortion control (UNet2DModel).
# Tasks 0-5 = N 64,128,256,512,1024,2048; 200k updates each from the unchanged
# nf_fig2_u128_<tag>_noaug_200k config; seed 123. See local/unet128_aug_sweep/runs.json.
ARM_TAG=warp exec bash /home/jiamingp/diffusion_models_repo/scripts/slurm/unet128_aug_sweep_train_task.sh
