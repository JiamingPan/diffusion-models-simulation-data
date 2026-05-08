#!/bin/bash
set -euo pipefail

TARGET_DIR=${TARGET_DIR:-/home/jiamingp/Diffusion_model/cosmo_diffusion_normalization_fixes_git}
VENV_PATH=${VENV_PATH:-/home/jiamingp/venvs/cosmodiff_nf}
REPO_URL=${REPO_URL:-https://github.com/nkern/cosmo_diffusion.git}
BRANCH=${BRANCH:-normalization_fixes}
INSTALL_MISSING_DEPS=${INSTALL_MISSING_DEPS:-1}

mkdir -p "$(dirname "${TARGET_DIR}")"

if [[ -d "${TARGET_DIR}/.git" ]]; then
  git -C "${TARGET_DIR}" fetch origin
  git -C "${TARGET_DIR}" checkout -B "${BRANCH}" "origin/${BRANCH}"
elif [[ -e "${TARGET_DIR}" ]]; then
  echo "Target exists but is not a git checkout: ${TARGET_DIR}" >&2
  echo "Choose another TARGET_DIR or move the existing directory." >&2
  exit 1
else
  git clone --branch "${BRANCH}" "${REPO_URL}" "${TARGET_DIR}"
fi

if [[ -f "${VENV_PATH}/bin/activate" ]]; then
  source "${VENV_PATH}/bin/activate"
fi

if ! python - <<'PY'
import ema_pytorch  # noqa: F401
PY
then
  if [[ "${INSTALL_MISSING_DEPS}" == "1" ]]; then
    python -m pip install ema-pytorch
  else
    echo "Missing dependency: ema-pytorch. Install it with: python -m pip install ema-pytorch" >&2
    exit 1
  fi
fi

export COSMODIFF_DIR="${TARGET_DIR}"
export PYTHONPATH="${COSMODIFF_DIR}:${PYTHONPATH:-}"

python - <<'PY'
import inspect
import os
from pathlib import Path
from diffusers import DDPMScheduler
from cosmodiff import optim
from cosmodiff.transform import Normalization
import ema_pytorch  # noqa: F401

expected_root = Path(os.environ["COSMODIFF_DIR"]).resolve()
optim_path = Path(inspect.getsourcefile(optim)).resolve()
if expected_root not in optim_path.parents:
    raise SystemExit(
        f"Imported cosmodiff from {optim_path}, expected it under {expected_root}"
    )

required = {"ema_sigma_rels", "ema_update_every", "ema_burn_in", "min_snr_gamma", "sigma_log_normal"}
missing = required.difference(inspect.signature(optim.train).parameters)
if missing:
    raise SystemExit(f"cosmodiff.optim.train is too old; missing {sorted(missing)}")

DDPMScheduler(
    num_train_timesteps=500,
    beta_schedule="sigmoid",
    prediction_type="v_prediction",
    rescale_betas_zero_snr=True,
    timestep_spacing="trailing",
)

print("OK")
print("cosmodiff:", optim_path)
print("Normalization:", Normalization)
PY

echo
echo "Use this checkout for nf_sweep jobs:"
echo "  export COSMODIFF_DIR_OVERRIDE=${TARGET_DIR}"
