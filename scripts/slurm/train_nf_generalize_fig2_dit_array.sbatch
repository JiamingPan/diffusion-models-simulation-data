#!/bin/bash
#SBATCH --job-name=nf_dit_train
#SBATCH --partition=spgpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=80gb
#SBATCH --time=24:00:00
#SBATCH --array=0-9%2
#SBATCH -o /home/jiamingp/diffusion_models_repo/logs/nf_generalize_fig2_dit/train_%A_%a.out
#SBATCH -e /home/jiamingp/diffusion_models_repo/logs/nf_generalize_fig2_dit/train_%A_%a.err
#SBATCH -A huterer0

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
CONFIG_DIR=${CONFIG_DIR:-${PROJECT_DIR}/local/nf_generalize_fig2_dit/configs}
CHECKPOINT_ROOT=${CHECKPOINT_ROOT:-/scratch/huterer_root/huterer0/jiamingp/saved_runs/nf_generalize_fig2_dit}
PYTHON_BIN=${PYTHON_BIN:-python}
SKIP_IF_ANY_CHECKPOINT=${SKIP_IF_ANY_CHECKPOINT:-0}
REQUIRED_COSMODIFF_BRANCH=${REQUIRED_COSMODIFF_BRANCH:-main}
DIT_VARIANTS=${DIT_VARIANTS:-dit_base}

cd "${PROJECT_DIR}"
mkdir -p logs/nf_generalize_fig2_dit
if [[ -f "${VENV_PATH}/bin/activate" ]]; then
  source "${VENV_PATH}/bin/activate"
fi
source "${PROJECT_DIR}/scripts/nf_class_conditional_pythonpath.sh"
set_nf_class_conditional_pythonpath "${VENV_PATH}/bin/python" "${BASE_VENV_PATH}"

VARIANT_ARGS=()
IFS=',' read -ra VARIANT_LIST <<< "${DIT_VARIANTS}"
for variant in "${VARIANT_LIST[@]}"; do
  variant="${variant//[[:space:]]/}"
  if [[ -n "${variant}" ]]; then
    VARIANT_ARGS+=(--variant "${variant}")
  fi
done

mapfile -t RUNS < <("${PYTHON_BIN}" scripts/prepare_nf_generalize_fig2_dit_configs.py --project-dir "${PROJECT_DIR}" --print-runs "${VARIANT_ARGS[@]}")
if (( SLURM_ARRAY_TASK_ID < 0 || SLURM_ARRAY_TASK_ID >= ${#RUNS[@]} )); then
  echo "Invalid array task ${SLURM_ARRAY_TASK_ID}; valid range is 0-$(( ${#RUNS[@]} - 1 ))" >&2
  exit 1
fi

RUN_NAME=${RUNS[$SLURM_ARRAY_TASK_ID]}
CONFIG_PATH=${CONFIG_DIR}/${RUN_NAME}.yaml
CHECKPOINT_DIR=${CHECKPOINT_ROOT}/${RUN_NAME}_checkpoints
mkdir -p "${CHECKPOINT_DIR}"

STUB_ROOT="${PROJECT_DIR}/results/cache/python_stubs"
mkdir -p "${STUB_ROOT}/sklearn/metrics"
printf 'from . import metrics\n' > "${STUB_ROOT}/sklearn/__init__.py"
printf "def roc_curve(*args, **kwargs):\n    raise RuntimeError('sklearn.metrics.roc_curve is stubbed for DiT training')\n" > "${STUB_ROOT}/sklearn/metrics/__init__.py"
"${PYTHON_BIN}" scripts/write_diffusers_runtime_sitecustomize.py "${STUB_ROOT}/sitecustomize.py"
export COSMODIFF_STUB_SKLEARN=1
export TORCHDYNAMO_DISABLE=1
export PYTHONPATH="${STUB_ROOT}:${COSMODIFF_DIR}:${PROJECT_DIR}:${PYTHONPATH:-}"
export COSMODIFF_DIR

if [[ ! -f "${CONFIG_PATH}" ]]; then
  echo "Missing config: ${CONFIG_PATH}" >&2
  echo "Run first: python scripts/prepare_nf_generalize_fig2_dit_configs.py --project-dir ${PROJECT_DIR}" >&2
  exit 1
fi

if [[ ! -f "${COSMODIFF_DIR}/scripts/cosmodiff_train.py" ]]; then
  echo "Missing main cosmo_diffusion checkout: ${COSMODIFF_DIR}" >&2
  exit 1
fi

COSMODIFF_BRANCH=$(git -C "${COSMODIFF_DIR}" rev-parse --abbrev-ref HEAD)
COSMODIFF_COMMIT=$(git -C "${COSMODIFF_DIR}" rev-parse --short HEAD)
echo "cosmodiff git branch: ${COSMODIFF_BRANCH}"
echo "cosmodiff git commit: ${COSMODIFF_COMMIT}"
if [[ "${COSMODIFF_BRANCH}" != "${REQUIRED_COSMODIFF_BRANCH}" ]]; then
  echo "Wrong cosmo_diffusion branch: expected ${REQUIRED_COSMODIFF_BRANCH}, got ${COSMODIFF_BRANCH}" >&2
  exit 1
fi

PATCH_LOCK="${COSMODIFF_DIR}/.codex_dit_patches.lock"
(
  flock -x 9
  "${PYTHON_BIN}" "${PROJECT_DIR}/scripts/patch_cosmodiff_constant_label.py" "${COSMODIFF_DIR}"
  "${PYTHON_BIN}" "${PROJECT_DIR}/scripts/patch_cosmodiff_dit_class_labels.py" "${COSMODIFF_DIR}"
) 9>"${PATCH_LOCK}"

"${PYTHON_BIN}" - <<'PY'
import inspect
import os
from pathlib import Path
from simdiff_eval.torch_compat import install_torch_backend_compat
install_torch_backend_compat(entry_point="train_nf_generalize_fig2_dit_array")
from cosmodiff import optim, utils
from cosmodiff.transform import Transform

expected_root = Path(os.environ["COSMODIFF_DIR"]).resolve()
optim_path = Path(inspect.getsourcefile(optim)).resolve()
utils_path = Path(inspect.getsourcefile(utils)).resolve()
if expected_root not in optim_path.parents:
    raise SystemExit(f"Imported cosmodiff from {optim_path}, expected under {expected_root}")

required_train = {
    "ema_sigma_rels",
    "ema_update_every",
    "ema_burn_in",
    "min_snr_gamma",
    "sigma_log_normal",
}
missing = required_train.difference(inspect.signature(optim.train).parameters)
if missing:
    raise SystemExit(f"cosmodiff.optim.train is too old; missing {sorted(missing)}")

source = optim_path.read_text()
if "codex DiT class-label patch" not in source and "class_labels=batch_labels" not in source:
    raise SystemExit(
        "cosmodiff.optim.train does not pass labels into DiT forward(); "
        "the DiT class-label patch did not apply."
    )

utils_source = utils_path.read_text()
if "codex constant-label patch" not in utils_source and "constant_label = data_cfg.get" not in utils_source:
    raise SystemExit(
        "cosmodiff.utils.parse_config_data does not create labels from data.constant_label; "
        "the constant-label data patch did not apply."
    )

print("cosmodiff main support:", optim_path)
print("cosmodiff data support:", utils_path)
print("transform support:", Transform)
PY

"${PYTHON_BIN}" scripts/prepare_nf_generalize_fig2_dit_configs.py \
  --project-dir "${PROJECT_DIR}" \
  --check-only \
  --run-name "${RUN_NAME}"

if [[ "${SKIP_IF_ANY_CHECKPOINT}" == "1" ]] && compgen -G "${CHECKPOINT_DIR}/checkpoint-*" > /dev/null; then
  echo "Skipping ${RUN_NAME}; checkpoint already exists under ${CHECKPOINT_DIR}"
  exit 0
fi

echo "Training ${RUN_NAME}"
echo "  config:      ${CONFIG_PATH}"
echo "  checkpoints: ${CHECKPOINT_DIR}"
echo "  target:      about 200k optimizer updates, no augmentation"
echo "  DiT labels:  unconditional null class label 0"
echo "  cosmodiff:   ${COSMODIFF_DIR}"
echo "  python:      ${PYTHON_BIN}"

"${PYTHON_BIN}" scripts/train_cosmodiff.py \
  --config "${CONFIG_PATH}" \
  --cosmodiff-train "${COSMODIFF_DIR}/scripts/cosmodiff_train.py"
