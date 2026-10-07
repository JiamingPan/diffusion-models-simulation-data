#!/bin/bash
#SBATCH --job-name=nf_thinterp
#SBATCH --partition=spgpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=40gb
#SBATCH --time=01:00:00
#SBATCH -o /home/jiamingp/diffusion_models_repo/logs/nf_conditional_bias_fresh_full_sweep_200k/theta_interp_%j.out
#SBATCH -e /home/jiamingp/diffusion_models_repo/logs/nf_conditional_bias_fresh_full_sweep_200k/theta_interp_%j.err
#SBATCH -A huterer2

# Samples the N=64 conditional checkpoint at (a) straight lines between mutually
# nearest training cosmologies and (b) the training cosmologies themselves.
# Same environment, patches and sampler settings as
# sample_nf_conditional_bias_full_sweep_array.sbatch. 336 images total; minutes.

set -euo pipefail
PROJECT_DIR=${PROJECT_DIR:-/home/jiamingp/diffusion_models_repo}
COSMODIFF_DIR=${COSMODIFF_DIR_OVERRIDE:-/home/jiamingp/Diffusion_model/cosmo_diffusion_main}
VENV_PATH=${VENV_PATH:-/home/jiamingp/venvs/cosmodiff_nf_class}
BASE_VENV_PATH=${BASE_VENV_PATH:-/home/jiamingp/venvs/cosmodiff_nf}
PYTHON_BIN=${PYTHON_BIN:-python}
DATASET_SIZE=${DATASET_SIZE:-64}

cd "${PROJECT_DIR}"
source "${PROJECT_DIR}/scripts/nf_class_conditional_pythonpath.sh"
mkdir -p logs/nf_conditional_bias_fresh_full_sweep_200k results/cache/python_stubs
source "${VENV_PATH}/bin/activate"
set_nf_class_conditional_pythonpath "${PYTHON_BIN}" "${BASE_VENV_PATH}"

STUB_ROOT="${PROJECT_DIR}/results/cache/python_stubs/${SLURM_JOB_ID:-manual_thinterp}"
mkdir -p "${STUB_ROOT}/sklearn/metrics"
printf 'from . import metrics\n' > "${STUB_ROOT}/sklearn/__init__.py"
printf "def roc_curve(*args, **kwargs):\n    raise RuntimeError('sklearn.metrics.roc_curve is stubbed')\n" > "${STUB_ROOT}/sklearn/metrics/__init__.py"
"${PYTHON_BIN}" scripts/write_diffusers_runtime_sitecustomize.py "${STUB_ROOT}/sitecustomize.py"
export PYTHONPATH="${STUB_ROOT}:${COSMODIFF_DIR}:${PROJECT_DIR}:${PROJECT_DIR}/scripts:${PYTHONPATH:-}"
export COSMODIFF_DIR
export TORCHDYNAMO_DISABLE=1

PATCH_LOCK="${COSMODIFF_DIR}/.codex_conditional_theta_interp_patch.lock"
while ! mkdir "${PATCH_LOCK}" 2>/dev/null; do sleep 1; done
trap 'rmdir "${PATCH_LOCK}" 2>/dev/null || true' EXIT
"${PYTHON_BIN}" scripts/patch_cosmodiff_package_metadata.py "${COSMODIFF_DIR}"
"${PYTHON_BIN}" scripts/patch_cosmodiff_continuous_labels.py "${COSMODIFF_DIR}"
"${PYTHON_BIN}" scripts/patch_cosmodiff_direct_unet_checkpoint.py "${COSMODIFF_DIR}"
"${PYTHON_BIN}" scripts/patch_cosmodiff_numpy_compat.py "${COSMODIFF_DIR}"
"${PYTHON_BIN}" scripts/patch_cosmodiff_sampling_compat.py "${COSMODIFF_DIR}"
rmdir "${PATCH_LOCK}" 2>/dev/null || true
trap - EXIT

"${PYTHON_BIN}" scripts/prepare_nf_conditional_theta_interp_labels.py \
  --project-dir "${PROJECT_DIR}" --dataset-size "${DATASET_SIZE}" --pairs 4 --seeds 4

for LABEL_SET in theta_interp train_theta; do
  "${PYTHON_BIN}" scripts/sample_nf_conditional_theta_interp.py \
    --project-dir "${PROJECT_DIR}" --dataset-size "${DATASET_SIZE}" \
    --label-set "${LABEL_SET}" --seed 123 --batch-size 8 \
    --scheduler DPMSolverMultistepScheduler --num-steps 50 --device cuda --overwrite
done
