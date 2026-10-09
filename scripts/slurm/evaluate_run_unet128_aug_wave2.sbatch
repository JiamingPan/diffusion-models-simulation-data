#!/bin/bash
#SBATCH --job-name=evalrun_u128aug_w2
#SBATCH --account=huterer2
#SBATCH --partition=spgpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80gb
#SBATCH --time=03:00:00
#SBATCH --no-requeue
#SBATCH -o /home/jiamingp/diffusion_models_repo/logs/evaluate_run/unet128_aug_wave2_%j.out
#SBATCH -e /home/jiamingp/diffusion_models_repo/logs/evaluate_run/unet128_aug_wave2_%j.err
# scripts/evaluate_run.py --mode unconditional on local/unet128_aug_wave2/eval_runs.json:
# 9 wave-2 runs (shiftonly, d4only, fliponly) + existing u128 noaug and d4shift rows at N = 64, 128, 256,
# one output. Same settings as evaluate_run_unet128_aug_sweep.sbatch (PCA fit on u128 d2p15, PCA32 q95,
# D4 x shift orbit copies, P(k), one-point, probe). Extra args pass through ("$@"), e.g.
# --orbit-group shift --orbit-group d4 once that option exists.
set -euo pipefail
export LD_LIBRARY_PATH=/sw/pkgs/arc/python3.10-anaconda/2023.03/lib:${LD_LIBRARY_PATH:-}
export LD_PRELOAD=/sw/pkgs/arc/python3.10-anaconda/2023.03/lib/libstdc++.so.6
PROJECT_DIR=${PROJECT_DIR:-/home/jiamingp/diffusion_models_repo}
VENV_PATH=${VENV_PATH:-/home/jiamingp/venvs/cosmodiff_nf_class}
OUTPUT_DIR=${OUTPUT_DIR:-results/unet128_aug_wave2/evaluate_run_v1}

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
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}" MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

python scripts/evaluate_run.py --mode unconditional --runs-json local/unet128_aug_wave2/eval_runs.json --device auto \
  --pca-fit-config local/nf_generalize_fig2/configs/nf_fig2_u128_d2p15_noaug_200k.yaml \
  --out-dir "${OUTPUT_DIR}" "$@"
