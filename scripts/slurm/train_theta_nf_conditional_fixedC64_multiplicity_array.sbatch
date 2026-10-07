#!/bin/bash
#SBATCH --job-name=nf_multtrth
#SBATCH --partition=spgpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=40gb
#SBATCH --time=01:00:00
#SBATCH -o /home/jiamingp/diffusion_models_repo/logs/nf_conditional_fixedC64_multiplicity_200k/train_theta_%A_%a.out
#SBATCH -e /home/jiamingp/diffusion_models_repo/logs/nf_conditional_fixedC64_multiplicity_200k/train_theta_%A_%a.err
#SBATCH -A huterer2

# Per run: sample at the 64 training cosmologies (4 draws each = 256 fields) and
# run the training-theta report with the frozen VGG probe. Same env/patch preamble
# as the sweep sampling job. No --overwrite: existing samples are reused.

set -euo pipefail
export LD_LIBRARY_PATH=/sw/pkgs/arc/python3.10-anaconda/2023.03/lib:${LD_LIBRARY_PATH:-}
export LD_PRELOAD=/sw/pkgs/arc/python3.10-anaconda/2023.03/lib/libstdc++.so.6
PROJECT_DIR=${PROJECT_DIR:-/home/jiamingp/diffusion_models_repo}
COSMODIFF_DIR=${COSMODIFF_DIR_OVERRIDE:-/home/jiamingp/Diffusion_model/cosmo_diffusion_main}
VENV_PATH=${VENV_PATH:-/home/jiamingp/venvs/cosmodiff_nf_class}
BASE_VENV_PATH=${BASE_VENV_PATH:-/home/jiamingp/venvs/cosmodiff_nf}
PYTHON_BIN=${PYTHON_BIN:-python}
SWEEP=nf_conditional_fixedC64_multiplicity_200k

cd "${PROJECT_DIR}"
source "${PROJECT_DIR}/scripts/nf_class_conditional_pythonpath.sh"
mkdir -p logs/${SWEEP} results/${SWEEP}/samples_theta_interp results/cache/python_stubs
source "${VENV_PATH}/bin/activate"
set_nf_class_conditional_pythonpath "${PYTHON_BIN}" "${BASE_VENV_PATH}"

mapfile -t RUNS < <("${PYTHON_BIN}" scripts/prepare_nf_conditional_fixedC64_multiplicity_configs.py --print-runs)
TASK_ID=${SLURM_ARRAY_TASK_ID:-0}
RUN_NAME=${RUNS[$TASK_ID]}
SIZES=(64 128 256 512)
DATASET_SIZE=${SIZES[$TASK_ID]}

STUB_ROOT="${PROJECT_DIR}/results/cache/python_stubs/${SLURM_JOB_ID:-manual_multtrth}_${TASK_ID}"
mkdir -p "${STUB_ROOT}/sklearn/metrics"
printf 'from . import metrics\n' > "${STUB_ROOT}/sklearn/__init__.py"
printf "def roc_curve(*args, **kwargs):\n    raise RuntimeError('sklearn.metrics.roc_curve is stubbed')\n" > "${STUB_ROOT}/sklearn/metrics/__init__.py"
"${PYTHON_BIN}" scripts/write_diffusers_runtime_sitecustomize.py "${STUB_ROOT}/sitecustomize.py"
export PYTHONPATH="${STUB_ROOT}:${COSMODIFF_DIR}:${PROJECT_DIR}:${PROJECT_DIR}/scripts:${PYTHONPATH:-}"
export COSMODIFF_DIR
export TORCHDYNAMO_DISABLE=1

PATCH_LOCK="${COSMODIFF_DIR}/.codex_conditional_mult_trth_patch.lock"
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
  --project-dir "${PROJECT_DIR}" --sweep "${SWEEP}" --dataset-size "${DATASET_SIZE}" --pairs 4 --seeds 4
"${PYTHON_BIN}" scripts/sample_nf_conditional_theta_interp.py \
  --project-dir "${PROJECT_DIR}" --sweep "${SWEEP}" --dataset-size "${DATASET_SIZE}" \
  --label-set train_theta --seed 123 --batch-size 8 --num-steps 50 --device cuda

# Report: real scikit-learn is needed for the probe head, so drop the stub dir.
export PYTHONPATH="${PROJECT_DIR}:${PROJECT_DIR}/scripts"
PROBE_RUNTIME=$(mktemp -d "${TMPDIR:-/tmp}/nf-multtrth-runtime.XXXXXX")
"${PYTHON_BIN}" scripts/write_diffusers_runtime_sitecustomize.py "${PROBE_RUNTIME}/sitecustomize.py"
export PYTHONPATH="${PROBE_RUNTIME}:${PROJECT_DIR}:${PROJECT_DIR}/scripts"
export PYTHONNOUSERSITE=1
export TORCH_HOME=${TORCH_HOME:-/nfs/turbo/lsa-huterer/jiamingp/torch_cache}
"${PYTHON_BIN}" scripts/report_nf_conditional_train_theta_test.py \
  --project-dir "${PROJECT_DIR}" --sweep "${SWEEP}" --dataset-size "${DATASET_SIZE}" --probe-device cuda
