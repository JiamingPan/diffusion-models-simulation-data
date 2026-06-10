#!/bin/bash
#SBATCH --job-name=nf_epoch_samples
#SBATCH --partition=spgpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=80gb
#SBATCH --time=04:00:00
#SBATCH -o /home/jiamingp/diffusion_models_repo/logs/nf_generalize_fig2/epoch_samples_%j.out
#SBATCH -e /home/jiamingp/diffusion_models_repo/logs/nf_generalize_fig2/epoch_samples_%j.err
#SBATCH -A huterer2

set -euo pipefail

PROJECT_DIR=${PROJECT_DIR:-/home/jiamingp/diffusion_models_repo}
INHERITED_COSMODIFF_DIR=${COSMODIFF_DIR:-}
PREFERRED_COSMODIFF_DIR=/home/jiamingp/Diffusion_model/cosmo_diffusion_main
if [[ -n "${COSMODIFF_DIR_OVERRIDE:-}" ]]; then
  COSMODIFF_DIR=${COSMODIFF_DIR_OVERRIDE}
else
  COSMODIFF_DIR=${PREFERRED_COSMODIFF_DIR}
fi
if [[ -n "${INHERITED_COSMODIFF_DIR}" && -z "${COSMODIFF_DIR_OVERRIDE:-}" && "${INHERITED_COSMODIFF_DIR}" != "${COSMODIFF_DIR}" ]]; then
  echo "Ignoring inherited COSMODIFF_DIR=${INHERITED_COSMODIFF_DIR}; using ${COSMODIFF_DIR}" >&2
fi

VENV_PATH=${VENV_PATH:-/home/jiamingp/venvs/cosmodiff_nf_class}
BASE_VENV_PATH=${BASE_VENV_PATH:-/home/jiamingp/venvs/cosmodiff_nf}
CHECKPOINT_ROOT=${CHECKPOINT_ROOT:-/scratch/huterer_root/huterer0/jiamingp/saved_runs/nf_generalize_fig2}
CONFIG_DIR=${CONFIG_DIR:-${PROJECT_DIR}/local/nf_generalize_fig2/configs}
SNAPSHOT_ROOT=${SNAPSHOT_ROOT:-${PROJECT_DIR}/results/nf_generalize_fig2/epoch_snapshots}
PLOT_OUTPUT=${PLOT_OUTPUT:-${PROJECT_DIR}/results/nf_generalize_fig2/quickcheck/learning_process_epoch_samples.png}
POSTER_OUTPUT=${POSTER_OUTPUT:-${PROJECT_DIR}/poster/figs/learning_process_epoch_samples.png}
PYTHON_BIN=${PYTHON_BIN:-python}

RUN_NAME=${RUN_NAME:-nf_fig2_u128_d2p15_noaug_200k}
EPOCHS=${EPOCHS:-auto}
SAMPLE_LABEL=${SAMPLE_LABEL:-dpm50}
SAMPLER_CLASS=${SAMPLER_CLASS:-DPMSolverMultistepScheduler}
SAMPLER_STEPS=${SAMPLER_STEPS:-50}
NUM_SAMPLES=${NUM_SAMPLES:-6}
BATCH_SIZE=${BATCH_SIZE:-6}
IMAGE_SIZE=${IMAGE_SIZE:-128}
SEED=${SEED:-123}
OVERWRITE=${OVERWRITE:-0}
MAX_PLOT_SAMPLES=${MAX_PLOT_SAMPLES:-5}
REQUIRED_COSMODIFF_BRANCH=${REQUIRED_COSMODIFF_BRANCH:-main}

cd "${PROJECT_DIR}"
mkdir -p logs/nf_generalize_fig2 "${SNAPSHOT_ROOT}" "$(dirname "${PLOT_OUTPUT}")" "$(dirname "${POSTER_OUTPUT}")"
if [[ -f "${VENV_PATH}/bin/activate" ]]; then
  source "${VENV_PATH}/bin/activate"
fi
source "${PROJECT_DIR}/scripts/nf_class_conditional_pythonpath.sh"
set_nf_class_conditional_pythonpath "${VENV_PATH}/bin/python" "${BASE_VENV_PATH}"

"${PYTHON_BIN}" scripts/prepare_nf_generalize_fig2_configs.py --project-dir "${PROJECT_DIR}"

CONFIG_PATH=${CONFIG_DIR}/${RUN_NAME}.yaml
CHECKPOINT_DIR=${CHECKPOINT_ROOT}/${RUN_NAME}_checkpoints
if [[ ! -f "${CONFIG_PATH}" ]]; then
  echo "Missing config: ${CONFIG_PATH}" >&2
  exit 1
fi
if [[ ! -d "${CHECKPOINT_DIR}" ]]; then
  echo "Missing checkpoint directory: ${CHECKPOINT_DIR}" >&2
  exit 1
fi

STUB_ROOT="${PROJECT_DIR}/results/cache/python_stubs"
mkdir -p "${STUB_ROOT}/sklearn/metrics"
printf 'from . import metrics\n' > "${STUB_ROOT}/sklearn/__init__.py"
printf "def roc_curve(*args, **kwargs):\n    raise RuntimeError('sklearn.metrics.roc_curve is stubbed for cosmodiff sampling')\n" > "${STUB_ROOT}/sklearn/metrics/__init__.py"
export COSMODIFF_STUB_SKLEARN=1
export TORCHDYNAMO_DISABLE=1
export MPLCONFIGDIR="${PROJECT_DIR}/results/cache/matplotlib"
export PYTHONPATH="${STUB_ROOT}:${COSMODIFF_DIR}:${PROJECT_DIR}:${PYTHONPATH:-}"
export COSMODIFF_DIR
mkdir -p "${MPLCONFIGDIR}"

COSMODIFF_BRANCH=$(git -C "${COSMODIFF_DIR}" rev-parse --abbrev-ref HEAD)
COSMODIFF_COMMIT=$(git -C "${COSMODIFF_DIR}" rev-parse --short HEAD)
echo "cosmodiff git branch: ${COSMODIFF_BRANCH}"
echo "cosmodiff git commit: ${COSMODIFF_COMMIT}"
if [[ "${COSMODIFF_BRANCH}" != "${REQUIRED_COSMODIFF_BRANCH}" ]]; then
  echo "Wrong cosmo_diffusion branch: expected ${REQUIRED_COSMODIFF_BRANCH}, got ${COSMODIFF_BRANCH}" >&2
  exit 1
fi

mapfile -t ALL_CKPTS < <(find "${CHECKPOINT_DIR}" -maxdepth 1 -type d -name 'checkpoint-epoch-*' | sort -V)
if [[ "${#ALL_CKPTS[@]}" -eq 0 ]]; then
  echo "No checkpoint-epoch-* directories found under ${CHECKPOINT_DIR}" >&2
  exit 1
fi

SELECTED_CKPTS=()
if [[ "${EPOCHS}" == "auto" ]]; then
  last_idx=$((${#ALL_CKPTS[@]} - 1))
  for idx in 0 $((last_idx / 3)) $(((2 * last_idx) / 3)) "${last_idx}"; do
    ckpt=${ALL_CKPTS[$idx]}
    if [[ " ${SELECTED_CKPTS[*]} " != *" ${ckpt} "* ]]; then
      SELECTED_CKPTS+=("${ckpt}")
    fi
  done
else
  IFS=',' read -ra REQUESTED_EPOCHS <<< "${EPOCHS}"
  for epoch in "${REQUESTED_EPOCHS[@]}"; do
    epoch=${epoch## }
    epoch=${epoch%% }
    epoch_num=$((10#${epoch}))
    epoch=$(printf '%04d' "${epoch_num}")
    ckpt="${CHECKPOINT_DIR}/checkpoint-epoch-${epoch}"
    if [[ ! -d "${ckpt}" ]]; then
      echo "Requested checkpoint does not exist: ${ckpt}" >&2
      exit 1
    fi
    SELECTED_CKPTS+=("${ckpt}")
  done
fi

echo "Sampling learning-process snapshots for ${RUN_NAME}"
echo "  checkpoint root: ${CHECKPOINT_DIR}"
echo "  selected checkpoints:"
printf '    %s\n' "${SELECTED_CKPTS[@]}"

first_ckpt=${SELECTED_CKPTS[0]}
"${PYTHON_BIN}" "${PROJECT_DIR}/scripts/sample_cosmodiff.py" \
  --checkpoint "${first_ckpt}" \
  --config "${CONFIG_PATH}" \
  --output "${SNAPSHOT_ROOT}/preflight_ignore.npz" \
  --num-samples 1 \
  --batch-size 1 \
  --image-size "${IMAGE_SIZE}" \
  --scheduler "${SAMPLER_CLASS}" \
  --num-steps "${SAMPLER_STEPS}" \
  --seed "${SEED}" \
  --device cpu \
  --preflight-only

for ckpt in "${SELECTED_CKPTS[@]}"; do
  epoch=${ckpt##*-}
  output="${SNAPSHOT_ROOT}/${RUN_NAME}_epoch${epoch}_seed${SEED}_${SAMPLE_LABEL}.npz"
  if [[ -f "${output}" && "${OVERWRITE}" != "1" ]]; then
    echo "Skipping existing sample file: ${output}"
    continue
  fi
  echo "Sampling ${ckpt} -> ${output}"
  "${PYTHON_BIN}" "${PROJECT_DIR}/scripts/sample_cosmodiff.py" \
    --checkpoint "${ckpt}" \
    --config "${CONFIG_PATH}" \
    --output "${output}" \
    --num-samples "${NUM_SAMPLES}" \
    --batch-size "${BATCH_SIZE}" \
    --image-size "${IMAGE_SIZE}" \
    --scheduler "${SAMPLER_CLASS}" \
    --num-steps "${SAMPLER_STEPS}" \
    --seed "${SEED}" \
    --device cuda
done

"${PYTHON_BIN}" "${PROJECT_DIR}/scripts/plot_nf_generalize_epoch_samples.py" \
  --sample-dir "${SNAPSHOT_ROOT}" \
  --run-name "${RUN_NAME}" \
  --sample-label "${SAMPLE_LABEL}" \
  --max-samples "${MAX_PLOT_SAMPLES}" \
  --output "${PLOT_OUTPUT}" \
  --poster-output "${POSTER_OUTPUT}"
