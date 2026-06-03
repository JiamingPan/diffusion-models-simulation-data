#!/bin/bash
#SBATCH --job-name=nf_bias_cfg_prep
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8gb
#SBATCH --time=00:20:00
#SBATCH -o /home/jiamingp/diffusion_models_repo/logs/nf_conditional_bias_probe_cfg_sweep/prepare_%j.out
#SBATCH -e /home/jiamingp/diffusion_models_repo/logs/nf_conditional_bias_probe_cfg_sweep/prepare_%j.err
#SBATCH -A huterer2

set -euo pipefail

PROJECT_DIR=${PROJECT_DIR:-/home/jiamingp/diffusion_models_repo}
VENV_PATH=${VENV_PATH:-/home/jiamingp/venvs/cosmodiff_nf_class}
PYTHON_BIN=${PYTHON_BIN:-python}
BIAS_DATASET_SIZES=${BIAS_DATASET_SIZES:-128,16384}
CFG_DROPOUTS=${CFG_DROPOUTS:-0.05,0.1}
GUIDANCE_SCALES=${GUIDANCE_SCALES:-none,1.0,1.5,2.0}
SAMPLES_PER_COSMOLOGY=${SAMPLES_PER_COSMOLOGY:-64}

cd "${PROJECT_DIR}"
mkdir -p logs/nf_conditional_bias_probe_cfg_sweep
if [[ -f "${VENV_PATH}/bin/activate" ]]; then
  source "${VENV_PATH}/bin/activate"
fi

"${PYTHON_BIN}" scripts/prepare_nf_conditional_bias_cfg_sweep_configs.py \
  --project-dir "${PROJECT_DIR}" \
  --dataset-sizes "${BIAS_DATASET_SIZES}" \
  --cfg-dropouts "${CFG_DROPOUTS}" \
  --guidance-scales "${GUIDANCE_SCALES}" \
  --sample-k-per-cosmology "${SAMPLES_PER_COSMOLOGY}" \
  --print-table
