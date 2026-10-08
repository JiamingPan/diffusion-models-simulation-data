#!/bin/bash
#SBATCH --job-name=evalrun_patchmosaic
#SBATCH --account=huterer2
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64gb
#SBATCH --time=16:00:00
#SBATCH --no-requeue
#SBATCH -o /home/jiamingp/diffusion_models_repo/logs/evaluate_run/unet128_aug_patch_mosaic_%j.out
#SBATCH -e /home/jiamingp/diffusion_models_repo/logs/evaluate_run/unet128_aug_patch_mosaic_%j.err
# Ledger unet128_aug_patch_mosaic. CPU only, existing samples, no GPU.
# scripts/evaluate_run.py --mode unconditional on the u128 augmentation sweep + noaug baseline at
# N = 64, 128, 256, 1024 (16 runs), with the patch-mosaic test at p = 8, 16, 32, 64
# (64 maps x 16 grid locations per set; train-train threshold and held-out once per N).
# Sizing (pilot 2026-10-08, login node, timing only): 3.3 ms single-core per (query, reference map)
# pair, independent of p. Patch: 6 sets x 1024 patches x 4 p x (64+128+256+1024) refs = 36.2M pairs;
# full-map D4 x shift orbit (existing metric, recomputed on CPU): 10.6M pairs. ~43 core-h total,
# ~1.4 h at perfect 32-core scaling; 16 h limit covers 4x worse thread scaling. Pilot RSS 1.3-1.7 GB;
# full-map orbit batches (4 queries x 512 refs) peak ~3-4 GB.
set -euo pipefail
export LD_LIBRARY_PATH=/sw/pkgs/arc/python3.10-anaconda/2023.03/lib:${LD_LIBRARY_PATH:-}
export LD_PRELOAD=/sw/pkgs/arc/python3.10-anaconda/2023.03/lib/libstdc++.so.6
PROJECT_DIR=${PROJECT_DIR:-/home/jiamingp/diffusion_models_repo}
VENV_PATH=${VENV_PATH:-/home/jiamingp/venvs/cosmodiff_nf_class}
OUTPUT_DIR=results/unet128_aug_sweep/patch_mosaic_v1

[[ "${RUN_NAME:-}" == "unet128_aug_patch_mosaic" ]] || { echo "RUN_NAME must be unet128_aug_patch_mosaic" >&2; exit 1; }
cd "${PROJECT_DIR}"
mkdir -p logs/evaluate_run results/cache/matplotlib
[[ ! -e "${OUTPUT_DIR}" ]] || { echo "Refusing to overwrite ${OUTPUT_DIR}" >&2; exit 1; }
source "${VENV_PATH}/bin/activate"
export PYTHONNOUSERSITE=1
RUNTIME=$(mktemp -d "${TMPDIR:-/tmp}/evalrun-runtime.XXXXXX")
python scripts/write_diffusers_runtime_sitecustomize.py "${RUNTIME}/sitecustomize.py"
export PYTHONPATH="${RUNTIME}:${PROJECT_DIR}:${PROJECT_DIR}/scripts"
export TORCH_HOME=${TORCH_HOME:-/nfs/turbo/lsa-huterer/jiamingp/torch_cache}
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export MPLCONFIGDIR="${PROJECT_DIR}/results/cache/matplotlib"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-32}" MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-32}"
echo "git $(git -C "${PROJECT_DIR}" rev-parse --short HEAD) dirty_files=$(git -C "${PROJECT_DIR}" status --porcelain simdiff_eval scripts/evaluate_run.py | wc -l)"

python scripts/evaluate_run.py --mode unconditional --preset unet128_aug_sweep \
  --dataset-size 64 --dataset-size 128 --dataset-size 256 --dataset-size 1024 \
  --patch-scales 8,16,32,64 --no-probe --device cpu \
  --out-dir "${OUTPUT_DIR}"
