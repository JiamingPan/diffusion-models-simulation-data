#!/bin/bash
#SBATCH --job-name=u128aug_pca95
#SBATCH --account=huterer2
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80gb
#SBATCH --time=03:00:00
#SBATCH --no-requeue
#SBATCH -o /home/jiamingp/diffusion_models_repo/logs/unet128_aug_sweep/pca95_%j.out
#SBATCH -e /home/jiamingp/diffusion_models_repo/logs/unet128_aug_sweep/pca95_%j.err
# Existing PCA95 score, identical settings to analyze_nf_generalize_fig2_dit_pca.sbatch
# (8192 components, full fit, 0.98 variance, 512 generated, 8 held-out sims/source, 512 val slices).
# The manifest keeps the 10 baseline u128 rows, so the basis is fitted on u128 d2p15 as in the fig2 table.
set -euo pipefail
PROJECT_DIR=/home/jiamingp/diffusion_models_repo
cd "${PROJECT_DIR}"
source /home/jiamingp/venvs/cosmodiff_nf/bin/activate
export PYTHONPATH="${PROJECT_DIR}:${PYTHONPATH:-}"
export MPLCONFIGDIR="${PROJECT_DIR}/results/cache/matplotlib"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}" MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
mkdir -p results/unet128_aug_sweep/quickcheck results/unet128_aug_sweep/tables
python scripts/compute_nf_generalize_pca_full_nn.py \
  --project-dir "${PROJECT_DIR}" \
  --manifest "${PROJECT_DIR}/local/unet128_aug_sweep/manifest.json" \
  --output-dir "${PROJECT_DIR}/results/unet128_aug_sweep/quickcheck" \
  --table-dir "${PROJECT_DIR}/results/unet128_aug_sweep/tables" \
  --out-prefix unet128_aug_sweep_pca_full_nn \
  --pca-components 8192 --pca-fit-max-slices 0 --pca-target-variance 0.98 \
  --sample-label dpm50 --max-generated 512 --val-raw-per-source 8 --max-val-slices 512 \
  --similarity-batch-size 512 --fixed-similarity-thresholds 0.5,0.6,0.7,0.8,0.9,0.95 \
  --skip-missing-samples
