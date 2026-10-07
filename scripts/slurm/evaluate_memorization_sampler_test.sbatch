#!/bin/bash
#SBATCH --job-name=memo_sampler_eval
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48gb
#SBATCH --time=02:00:00
#SBATCH -o /home/jiamingp/diffusion_models_repo/logs/memorization_sampler_test/evaluate_%j.out
#SBATCH -e /home/jiamingp/diffusion_models_repo/logs/memorization_sampler_test/evaluate_%j.err
#SBATCH -A huterer2
# Frozen VGG probe on the sampler-test samples (same probe/settings as the omsig calibration_vgg runs). CPU.
set -euo pipefail
export LD_LIBRARY_PATH=/sw/pkgs/arc/python3.10-anaconda/2023.03/lib:${LD_LIBRARY_PATH:-}
export LD_PRELOAD=/sw/pkgs/arc/python3.10-anaconda/2023.03/lib/libstdc++.so.6
PROJECT_DIR=/home/jiamingp/diffusion_models_repo
cd "${PROJECT_DIR}"
source /home/jiamingp/venvs/cosmodiff_nf_class/bin/activate
export PYTHONNOUSERSITE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export TORCH_HOME=${TORCH_HOME:-/nfs/turbo/lsa-huterer/jiamingp/torch_cache}
PROBE_RUNTIME=$(mktemp -d "${TMPDIR:-/tmp}/memo-sampler-eval.XXXXXX")
python scripts/write_diffusers_runtime_sitecustomize.py "${PROBE_RUNTIME}/sitecustomize.py"
export PYTHONPATH="${PROBE_RUNTIME}:${PROJECT_DIR}:${PROJECT_DIR}/scripts"
export MPLCONFIGDIR="${PROJECT_DIR}/results/cache/matplotlib"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}" MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
for TAG in ddpm500 dpm200; do
  OUT="results/memorization_regime_diagnostics/sampler_test/calibration_vgg_${TAG}"
  [[ -e "${OUT}" ]] && { echo "${OUT} exists; refusing to overwrite" >&2; exit 1; }
  python scripts/evaluate_nf_conditional_bias_probe.py \
    --project-dir "${PROJECT_DIR}" \
    --manifest "${PROJECT_DIR}/local/memorization_sampler_test/manifest_${TAG}.json" \
    --encoder-type vgg \
    --vgg-encoder results/nf_conditional_bias_probe/encoder/vgg_mlp_encoder.npz \
    --vgg-device cpu \
    --seed 123 \
    --samples-per-cosmology 64 \
    --bootstrap 2000 \
    --output-dir "${PROJECT_DIR}/${OUT}"
done
