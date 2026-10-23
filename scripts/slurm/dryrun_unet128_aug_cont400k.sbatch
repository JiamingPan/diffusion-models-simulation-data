#!/bin/bash
#SBATCH --job-name=u128aug_cont400k_dry
#SBATCH --account=huterer2
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16gb
#SBATCH --time=01:30:00
#SBATCH --no-requeue
#SBATCH --array=0-3%2
#SBATCH -o /home/jiamingp/diffusion_models_repo/logs/unet128_aug_sweep/dryrun_cont400k_%A_%a.out
#SBATCH -e /home/jiamingp/diffusion_models_repo/logs/unet128_aug_sweep/dryrun_cont400k_%A_%a.err
# CPU dry run for the 200k -> 400k continuation: rebuild + hash the dataset, resume the seeded
# checkpoint through cosmodiff optim.train with num_epochs=0 (zero updates), report restored
# step counter / LR / optimizer state / augmentation RNG. Writes only
# local/unet128_aug_sweep/cont400k_dryrun/<run>.json and the throwaway WORK_DIR.
set -euo pipefail
: "${WORK_DIR:?set WORK_DIR to a throwaway directory outside /scratch}"
source /home/jiamingp/diffusion_models_repo/scripts/slurm/unet128_aug_sweep_env.sh
export OMP_NUM_THREADS=8
"${PYTHON_BIN}" scripts/continue_unet128_aug_sweep.py --mode dry-run --index "${SLURM_ARRAY_TASK_ID}" \
  --runtime-root "${COSMODIFF_DIR}" --work-dir "${WORK_DIR}"
