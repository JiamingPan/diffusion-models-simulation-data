#!/bin/bash
#SBATCH --job-name=evalrun_u128aug_w3
#SBATCH --account=huterer2
#SBATCH --partition=spgpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80gb
#SBATCH --time=03:00:00
#SBATCH --no-requeue
#SBATCH -o /home/jiamingp/diffusion_models_repo/logs/evaluate_run/unet128_aug_wave3_%j.out
#SBATCH -e /home/jiamingp/diffusion_models_repo/logs/evaluate_run/unet128_aug_wave3_%j.err
# scripts/evaluate_run.py --mode unconditional on local/unet128_aug_wave3/eval_runs.json (the 18 wave-3 runs).
# Same settings as evaluate_run_unet128_aug_wave2.sbatch (PCA fit on u128 d2p15, PCA32 q95, D4 x shift orbit
# copies plus --orbit-group shift --orbit-group d4, P(k), one-point, probe) and --heldout-zthin 8: N <= 64 runs
# are scored on exactly the N = 64 held-out set (8 unused sims x 4 sources x 16 slices = 512); N = 128 / 256 on
# the held-out set of the d2p07 / d2p08 baseline config (flag is a no-op there, config zthin 8), as in wave 2.
# --heldout-zthin is in scripts/evaluate_run.py (applied with approval; evaluate_run_heldout_zthin.diff);
# the job refuses if it is missing. Submit with --dependency=afterok:<sample array job>. Extra args pass through ("$@").
set -euo pipefail
export LD_LIBRARY_PATH=/sw/pkgs/arc/python3.10-anaconda/2023.03/lib:${LD_LIBRARY_PATH:-}
export LD_PRELOAD=/sw/pkgs/arc/python3.10-anaconda/2023.03/lib/libstdc++.so.6
PROJECT_DIR=${PROJECT_DIR:-/home/jiamingp/diffusion_models_repo}
VENV_PATH=${VENV_PATH:-/home/jiamingp/venvs/cosmodiff_nf_class}
OUTPUT_DIR=${OUTPUT_DIR:-results/unet128_aug_wave3/evaluate_run_v1}

cd "${PROJECT_DIR}"
mkdir -p logs/evaluate_run results/cache/matplotlib
[[ ! -e "${OUTPUT_DIR}" ]] || { echo "Refusing to overwrite ${OUTPUT_DIR}" >&2; exit 1; }
grep -q -- '"--heldout-zthin"' scripts/evaluate_run.py \
  || { echo "scripts/evaluate_run.py has no --heldout-zthin: apply local/unet128_aug_wave3/evaluate_run_heldout_zthin.diff first" >&2; exit 1; }
source "${VENV_PATH}/bin/activate"
export PYTHONNOUSERSITE=1
RUNTIME=$(mktemp -d "${TMPDIR:-/tmp}/evalrun-runtime.XXXXXX")
python scripts/write_diffusers_runtime_sitecustomize.py "${RUNTIME}/sitecustomize.py"
export PYTHONPATH="${RUNTIME}:${PROJECT_DIR}:${PROJECT_DIR}/scripts"
export TORCH_HOME=${TORCH_HOME:-/nfs/turbo/lsa-huterer/jiamingp/torch_cache}
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export MPLCONFIGDIR="${PROJECT_DIR}/results/cache/matplotlib"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}" MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

python scripts/evaluate_run.py --mode unconditional --runs-json local/unet128_aug_wave3/eval_runs.json --device auto \
  --pca-fit-config local/nf_generalize_fig2/configs/nf_fig2_u128_d2p15_noaug_200k.yaml \
  --orbit-group shift --orbit-group d4 --heldout-zthin 8 \
  --out-dir "${OUTPUT_DIR}" "$@"
