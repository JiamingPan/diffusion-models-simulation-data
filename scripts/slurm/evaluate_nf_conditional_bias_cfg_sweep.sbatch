#!/bin/bash
#SBATCH --job-name=nf_bias_cfg_eval
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32gb
#SBATCH --time=02:00:00
#SBATCH -o /home/jiamingp/diffusion_models_repo/logs/nf_conditional_bias_probe_cfg_sweep/eval_%j.out
#SBATCH -e /home/jiamingp/diffusion_models_repo/logs/nf_conditional_bias_probe_cfg_sweep/eval_%j.err
#SBATCH -A huterer2

set -euo pipefail

PROJECT_DIR=${PROJECT_DIR:-/home/jiamingp/diffusion_models_repo}
VENV_PATH=${VENV_PATH:-/home/jiamingp/venvs/cosmodiff_nf_class}
PYTHON_BIN=${PYTHON_BIN:-${VENV_PATH}/bin/python}
SAMPLES_PER_COSMOLOGY=${SAMPLES_PER_COSMOLOGY:-64}
SEED=${SEED:-123}
GUIDANCE_SCALES=${GUIDANCE_SCALES:-none,1.0,1.5,2.0}
BOOTSTRAP=${BOOTSTRAP:-1000}
ENCODER_TYPE=${ENCODER_TYPE:-vgg}
VGG_ENCODER=${VGG_ENCODER:-results/nf_conditional_bias_probe/encoder/vgg_mlp_big_avgmax.npz}
VGG_DEVICE=${VGG_DEVICE:-auto}
TORCH_HOME=${TORCH_HOME:-/nfs/turbo/lsa-huterer/jiamingp/torch_cache}

cd "${PROJECT_DIR}"
mkdir -p logs/nf_conditional_bias_probe_cfg_sweep \
  results/nf_conditional_bias_probe_cfg_sweep/calibration_vgg \
  results/cache/matplotlib
if [[ -f "${VENV_PATH}/bin/activate" ]]; then
  source "${VENV_PATH}/bin/activate"
fi
export PYTHONNOUSERSITE=1
export PYTHONPATH="${PROJECT_DIR}:${PROJECT_DIR}/scripts"
export MPLCONFIGDIR="${PROJECT_DIR}/results/cache/matplotlib"
export TORCH_HOME
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"

IFS=',' read -r -a GUIDANCE_ARRAY <<< "${GUIDANCE_SCALES}"
GUIDANCE_ARGS=()
for scale in "${GUIDANCE_ARRAY[@]}"; do
  GUIDANCE_ARGS+=(--guidance-scale "${scale}")
done

"${PYTHON_BIN}" scripts/evaluate_nf_conditional_bias_probe.py \
  --project-dir "${PROJECT_DIR}" \
  --manifest "${PROJECT_DIR}/local/nf_conditional_bias_probe_cfg_sweep/manifest.json" \
  --encoder-type "${ENCODER_TYPE}" \
  --vgg-encoder "${VGG_ENCODER}" \
  --vgg-device "${VGG_DEVICE}" \
  --seed "${SEED}" \
  --samples-per-cosmology "${SAMPLES_PER_COSMOLOGY}" \
  --bootstrap "${BOOTSTRAP}" \
  --output-dir "${PROJECT_DIR}/results/nf_conditional_bias_probe_cfg_sweep/calibration_${ENCODER_TYPE}" \
  "${GUIDANCE_ARGS[@]}"
