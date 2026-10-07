#!/bin/bash
#SBATCH --job-name=u128aug_orbit
#SBATCH --account=huterer2
#SBATCH --partition=spgpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=80gb
#SBATCH --time=04:00:00
#SBATCH --no-requeue
#SBATCH -o /home/jiamingp/diffusion_models_repo/logs/unet128_aug_sweep/orbit_%j.out
#SBATCH -e /home/jiamingp/diffusion_models_repo/logs/unet128_aug_sweep/orbit_%j.err
# Orbit-aware (D4 x periodic shift) nearest-training copies with orbit-excluded
# train-to-train thresholds, P(k) and one-point vs held-out real maps; baseline + 3 arms.
source /home/jiamingp/diffusion_models_repo/scripts/slurm/unet128_aug_sweep_env.sh
OUT=${PROJECT_DIR}/results/unet128_aug_sweep/orbit_eval
[[ ! -e "${OUT}/orbit_pk_onepoint_summary.csv" ]] || { echo "Refusing to overwrite ${OUT}" >&2; exit 1; }
"${PYTHON_BIN}" scripts/evaluate_unet128_aug_sweep.py --runtime-root "${COSMODIFF_DIR}" --out-dir "${OUT}"
