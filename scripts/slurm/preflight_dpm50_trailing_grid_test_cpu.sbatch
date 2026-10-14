#!/bin/bash
#SBATCH --job-name=preflight_dpm50_trailing
#SBATCH -A huterer2
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32gb
#SBATCH --time=01:00:00
#SBATCH --no-requeue
#SBATCH -o /home/jiamingp/diffusion_models_repo/logs/dpm50_trailing_grid_test/preflight_%j.out
#SBATCH -e /home/jiamingp/diffusion_models_repo/logs/dpm50_trailing_grid_test/preflight_%j.err
# CPU only (RUN_NAME=analysis). For ledger run dpm50_trailing_grid_test:
#   1. python -m pytest -q tests                                  (full suite)
#   2. scripts/preflight_dpm50_trailing_grid_test.py run: static checks, private runtimes (reference and
#      timestep_spacing-patched) hash checks, grid checks on the three real checkpoint schedulers
#      -> local/dpm50_trailing_grid_test/preflight_receipt.json (written only on PASS, never overwritten).
# No GPU, no samples, no writes outside the receipt and $TMPDIR.
set -uo pipefail
[[ "${RUN_NAME:-}" == "analysis" || "${RUN_NAME:-}" == "dpm50_trailing_grid_test" ]] \
  || { echo "Submit with --export=ALL,RUN_NAME=analysis" >&2; exit 1; }
export LD_LIBRARY_PATH=/sw/pkgs/arc/python3.10-anaconda/2023.03/lib:${LD_LIBRARY_PATH:-}
PROJECT_DIR=/home/jiamingp/diffusion_models_repo
VENV_PATH=/home/jiamingp/venvs/cosmodiff_nf_class
RECEIPT=${PROJECT_DIR}/local/dpm50_trailing_grid_test/preflight_receipt.json
cd "${PROJECT_DIR}"
[[ ! -e "${RECEIPT}" ]] || { echo "Refusing: ${RECEIPT} exists" >&2; exit 1; }
source "${VENV_PATH}/bin/activate"
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 TORCHDYNAMO_DISABLE=1 MPLCONFIGDIR="${PROJECT_DIR}/results/cache/matplotlib"
export LD_PRELOAD=/sw/pkgs/arc/python3.10-anaconda/2023.03/lib/libstdc++.so.6 PYTHONNOUSERSITE=1
export TORCH_HOME=/nfs/turbo/lsa-huterer/jiamingp/torch_cache HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
RT=$(mktemp -d "${TMPDIR:-/tmp}/pytest-rt.XXXXXX")  # torch-compat sitecustomize so diffusers imports
python scripts/write_diffusers_runtime_sitecustomize.py "${RT}/sitecustomize.py" >/dev/null
export PYTHONPATH="${RT}:${PROJECT_DIR}:${PROJECT_DIR}/scripts"

echo "=== pytest"
python -m pytest -q -p no:cacheprovider --basetemp="${TMPDIR:-/tmp}/pytest_dpm50_trailing" tests 2>&1 | tail -60
echo "pytest exit ${PIPESTATUS[0]}"
echo "=== preflight dpm50_trailing_grid_test"
OUTJ="${TMPDIR:-/tmp}/dpm50_trailing_preflight.json"
python scripts/preflight_dpm50_trailing_grid_test.py run > "${OUTJ}"
echo "preflight exit $?"; python - "${OUTJ}" <<'PY'
import json, sys
try:
    d = json.load(open(sys.argv[1]))
    print(json.dumps({"status": d["status"], "fails": d["fails"], "grid_checks": d["grid_checks"],
                      "differs_from_reference_runtime": {k: v.get("differs_from_reference_runtime")
                                                         for k, v in d["runtime"].items()}}, indent=1))
except Exception as e:
    print("could not parse", e); print(open(sys.argv[1]).read()[-4000:])
PY
