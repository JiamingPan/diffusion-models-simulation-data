#!/bin/bash
#SBATCH --job-name=n256ema_samp
#SBATCH --partition=spgpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16gb
#SBATCH --time=02:00:00
#SBATCH --array=0-1%2
#SBATCH --no-requeue
#SBATCH -o /home/jiamingp/diffusion_models_repo/logs/nf_cond_omsigc_n256_ema_sampling/sample_%A_%a.out
#SBATCH -e /home/jiamingp/diffusion_models_repo/logs/nf_cond_omsigc_n256_ema_sampling/sample_%A_%a.err
#SBATCH -A huterer2
# Ledger nf_cond_omsigc_n256_ema_sampling. Sampling only, existing checkpoint
# nf_cond_omsigc_hi_u128_d2p08_n256_fresh200k/checkpoint-epoch-24999 (read only), with EMA weights:
#   task 0  sigma_rel 0.10 -> sample label dpm50_ema0p10
#   task 1  sigma_rel 0.02 -> sample label dpm50_ema0p02
# EMA comes from generate.ema_sigma_rel in local/nf_cond_omsigc_n256_ema_sampling/configs/*.yaml (the only change
# from the baseline yaml); cosmodiff_sample.py synthesizes it post hoc from checkpoint-epoch-*/ema/*.pt. For targets
# 0.10 / 0.02 the synthesis weight is 1.0 on the stored final snapshot ema/1.199000.pt / ema/0.199000.pt (others
# < 1e-14), i.e. the stored EMA. Otherwise identical to 63381902: seed 123, 32 held-out cosmologies x 64 draws,
# DPM-Solver 50 steps, batch 8. Writes results/nf_cond_omsigc_n256_ema_sampling/samples/. Never --overwrite.
set -euo pipefail
PROJECT_DIR=/home/jiamingp/diffusion_models_repo
COSMODIFF_DIR=/home/jiamingp/Diffusion_model/cosmo_diffusion_main
VENV_PATH=/home/jiamingp/venvs/cosmodiff_nf_class
BASE_VENV_PATH=/home/jiamingp/venvs/cosmodiff_nf
PYTHON_BIN=python
MANIFEST="${PROJECT_DIR}/local/nf_cond_omsigc_n256_ema_sampling/manifest.json"

cd "${PROJECT_DIR}"
mkdir -p logs/nf_cond_omsigc_n256_ema_sampling
source "${PROJECT_DIR}/scripts/nf_class_conditional_pythonpath.sh"
source "${VENV_PATH}/bin/activate"
set_nf_class_conditional_pythonpath "${PYTHON_BIN}" "${BASE_VENV_PATH}"

mapfile -t RUNS < <("${PYTHON_BIN}" scripts/prepare_nf_cond_omsigc_n256_ema_sampling.py --print-runs)
TASK_ID=${SLURM_ARRAY_TASK_ID:-0}
if (( TASK_ID < 0 || TASK_ID >= ${#RUNS[@]} )); then
  echo "Array index ${TASK_ID} outside run count ${#RUNS[@]}" >&2
  exit 1
fi
MODEL_RUN=${RUNS[$TASK_ID]}
echo "task ${TASK_ID}: ${MODEL_RUN}"
grep -n "ema_sigma_rel" "${PROJECT_DIR}/local/nf_cond_omsigc_n256_ema_sampling/configs/${MODEL_RUN}.yaml"

STUB_ROOT="${PROJECT_DIR}/results/cache/python_stubs/${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID:-manual_n256ema}}_${TASK_ID}"
mkdir -p "${STUB_ROOT}/sklearn/metrics"
printf 'from . import metrics\n' > "${STUB_ROOT}/sklearn/__init__.py"
printf "def roc_curve(*args, **kwargs):\n    raise RuntimeError('sklearn.metrics.roc_curve is stubbed')\n" > "${STUB_ROOT}/sklearn/metrics/__init__.py"
"${PYTHON_BIN}" scripts/write_diffusers_runtime_sitecustomize.py "${STUB_ROOT}/sitecustomize.py"
printf '\nimport simdiff_eval.timeemb_conditioning as _timeemb\n_timeemb.install()\n' >> "${STUB_ROOT}/sitecustomize.py"
export PYTHONPATH="${STUB_ROOT}:${COSMODIFF_DIR}:${PROJECT_DIR}:${PROJECT_DIR}/scripts:${PYTHONPATH:-}"
export COSMODIFF_DIR TORCHDYNAMO_DISABLE=1

PATCH_LOCK="${COSMODIFF_DIR}/.codex_conditional_omsc_sample_patch.lock"
while ! mkdir "${PATCH_LOCK}" 2>/dev/null; do sleep 1; done
trap 'rmdir "${PATCH_LOCK}" 2>/dev/null || true' EXIT
for patch in package_metadata continuous_labels direct_unet_checkpoint numpy_compat sampling_compat; do
  "${PYTHON_BIN}" "scripts/patch_cosmodiff_${patch}.py" "${COSMODIFF_DIR}"
done
rmdir "${PATCH_LOCK}" 2>/dev/null || true
trap - EXIT

"${PYTHON_BIN}" scripts/sample_nf_conditional_bias_probe.py \
  --project-dir "${PROJECT_DIR}" \
  --manifest "${MANIFEST}" \
  --run-name "${MODEL_RUN}" \
  --seed 123 \
  --samples-per-cosmology 64 \
  --batch-size 8 \
  --scheduler DPMSolverMultistepScheduler \
  --num-steps 50 \
  --device cuda
