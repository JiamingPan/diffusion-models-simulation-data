#!/bin/bash
#SBATCH --job-name=nf_l16_c5pre
#SBATCH --partition=spgpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=80gb
#SBATCH --time=00:45:00
#SBATCH -o /home/jiamingp/diffusion_models_repo/logs/nf_generalize_fig2_dit_l16_continue500k_v2/precheck_%j.out
#SBATCH -e /home/jiamingp/diffusion_models_repo/logs/nf_generalize_fig2_dit_l16_continue500k_v2/precheck_%j.err
#SBATCH -A huterer2

set -euo pipefail

PROJECT_DIR=${PROJECT_DIR:-/home/jiamingp/diffusion_models_repo}
COSMODIFF_DIR=${COSMODIFF_DIR_OVERRIDE:-/home/jiamingp/Diffusion_model/cosmo_diffusion_main}
VENV_PATH=${VENV_PATH:-/home/jiamingp/venvs/cosmodiff_nf_class}
BASE_VENV_PATH=${BASE_VENV_PATH:-/home/jiamingp/venvs/cosmodiff_nf}
PYTHON_BIN=${PYTHON_BIN:-python}
SWEEP=nf_generalize_fig2_dit_l16_continue500k_v2
PREPARE=${PROJECT_DIR}/scripts/prepare_nf_generalize_fig2_dit_l16_continue500k_v2_configs.py
REPORT=${PROJECT_DIR}/local/${SWEEP}/precheck_report.json

cd "${PROJECT_DIR}"
mkdir -p "${PROJECT_DIR}/logs/${SWEEP}"
if [[ -f "${VENV_PATH}/bin/activate" ]]; then
  source "${VENV_PATH}/bin/activate"
fi
source "${PROJECT_DIR}/scripts/nf_class_conditional_pythonpath.sh"
set_nf_class_conditional_pythonpath "${VENV_PATH}/bin/python" "${BASE_VENV_PATH}"

STUB_ROOT=${PROJECT_DIR}/results/cache/python_stubs
mkdir -p "${STUB_ROOT}"
"${PYTHON_BIN}" "${PROJECT_DIR}/scripts/write_diffusers_runtime_sitecustomize.py" \
  "${STUB_ROOT}/sitecustomize.py"
export TORCHDYNAMO_DISABLE=1
export PYTHONPATH="${STUB_ROOT}:${COSMODIFF_DIR}:${PROJECT_DIR}:${PYTHONPATH:-}"
export COSMODIFF_DIR

"${PYTHON_BIN}" "${PREPARE}" \
  --project-dir "${PROJECT_DIR}" \
  --use-existing-manifest \
  --check-only

"${PYTHON_BIN}" \
  "${PROJECT_DIR}/scripts/check_nf_generalize_fig2_dit_l16_continue500k_v2.py" \
  --project-dir "${PROJECT_DIR}" \
  --cosmodiff-dir "${COSMODIFF_DIR}" \
  --runtime-check \
  --report "${REPORT}"
test -s "${REPORT}"

for DATASET_TAG in d2p06 d2p15; do
  CHECKPOINT=$("${PYTHON_BIN}" "${PREPARE}" \
    --project-dir "${PROJECT_DIR}" \
    --use-existing-manifest \
    --dataset-tag "${DATASET_TAG}" \
    --stage 1 \
    --print-field source_checkpoint)
  CONFIG=$("${PYTHON_BIN}" "${PREPARE}" \
    --project-dir "${PROJECT_DIR}" \
    --use-existing-manifest \
    --dataset-tag "${DATASET_TAG}" \
    --stage 1 \
    --print-field source_config)

  echo "GPU sampling preflight: ${DATASET_TAG} ${CHECKPOINT}"
  "${PYTHON_BIN}" "${PROJECT_DIR}/scripts/sample_cosmodiff.py" \
    --checkpoint "${CHECKPOINT}" \
    --config "${CONFIG}" \
    --output "${SLURM_TMPDIR:-/tmp}/${SWEEP}_${DATASET_TAG}_preflight.npy" \
    --num-samples 1 \
    --batch-size 1 \
    --image-size 128 \
    --scheduler DPMSolverMultistepScheduler \
    --num-steps 50 \
    --seed 123 \
    --class-label 0 \
    --device cuda \
    --preflight-only
done

echo "PASS: full-state, architecture, and endpoint reconstruction prechecks"
