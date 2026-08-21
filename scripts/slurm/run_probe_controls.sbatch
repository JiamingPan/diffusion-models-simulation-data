#!/bin/bash
# Frozen C0/C1 transform controls followed by C4 degradation control.
#SBATCH --job-name=nf_probe_controls
#SBATCH --partition=spgpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=80gb
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:1
#SBATCH -A huterer2
#SBATCH -o /home/jiamingp/diffusion_models_repo/logs/nf_conditional_bias_probe/controls_%j.out
#SBATCH -e /home/jiamingp/diffusion_models_repo/logs/nf_conditional_bias_probe/controls_%j.err

set -euo pipefail

PROJECT_DIR=${PROJECT_DIR:-/home/jiamingp/diffusion_models_repo}
CODE_ROOT=${CODE_ROOT:-/scratch/huterer_root/huterer0/jiamingp/probe_controls_code_456f01a}
PYTHON_BIN=${PYTHON_BIN:-/home/jiamingp/venvs/cosmodiff_nf_class/bin/python}
TORCH_HOME=${TORCH_HOME:-/scratch/huterer_root/huterer0/jiamingp/torch_cache}
DATA_ROOT=${DATA_ROOT:-/scratch/huterer_root/huterer0/CAMELS/CMD/3d_grids/IllustrisTNG}
INCOMPATIBLE_PYTHON_PATHS=${INCOMPATIBLE_PYTHON_PATHS:-/home/jiamingp/venvs/cosmodiff_nf/lib/python3.10/site-packages}
EXPECTED_COMMIT=${EXPECTED_COMMIT:-}

export PROJECT_DIR CODE_ROOT PYTHON_BIN TORCH_HOME DATA_ROOT INCOMPATIBLE_PYTHON_PATHS EXPECTED_COMMIT
bash "${CODE_ROOT}/scripts/preflight_probe_controls.sh"
source "${CODE_ROOT}/scripts/probe_controls_runtime.sh"
probe_controls_prepare_runtime "${CODE_ROOT}" "${PYTHON_BIN}" "${INCOMPATIBLE_PYTHON_PATHS}"
export PYTHONNOUSERSITE=1
export TORCH_HOME
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"

mkdir -p "${PROJECT_DIR}/logs/nf_conditional_bias_probe" \
  "${PROJECT_DIR}/results/nf_conditional_bias_probe/transform_controls" \
  "${PROJECT_DIR}/results/nf_conditional_bias_probe/degradation_controls"
cd "${PROJECT_DIR}"

"${PYTHON_BIN}" "${CODE_ROOT}/scripts/evaluate_probe_transform_controls.py" \
  --project-dir "${PROJECT_DIR}" \
  --source-project-dir "${CODE_ROOT}" \
  --data-root "${DATA_ROOT}" \
  --encoder "${PROJECT_DIR}/results/nf_conditional_bias_probe/encoder/vgg_mlp_encoder.npz" \
  --device cuda \
  --embedding-batch-size 32 \
  --output-dir "${PROJECT_DIR}/results/nf_conditional_bias_probe/transform_controls" \
  --bootstrap 2000 \
  --bootstrap-seed 123 \
  --roll-seed 123 \
  --control c0 \
  --control c1

"${PYTHON_BIN}" "${CODE_ROOT}/scripts/evaluate_probe_degradation_control.py" \
  --project-dir "${PROJECT_DIR}" \
  --source-project-dir "${CODE_ROOT}" \
  --data-root "${DATA_ROOT}" \
  --manifest "${PROJECT_DIR}/local/nf_conditional_bias_probe/manifest.json" \
  --run-name nf_cond_bias_hi_u128_d2p07_n128_200k \
  --run-name nf_cond_bias_hi_u128_d2p14_n16384_200k \
  --encoder "${PROJECT_DIR}/results/nf_conditional_bias_probe/encoder/vgg_mlp_encoder.npz" \
  --device cuda \
  --seed 123 \
  --samples-per-cosmology 64 \
  --embedding-batch-size 32 \
  --pk-nbins 25 \
  --bootstrap 2000 \
  --bootstrap-seed 123 \
  --split-seed 123 \
  --output-dir "${PROJECT_DIR}/results/nf_conditional_bias_probe/degradation_controls"
