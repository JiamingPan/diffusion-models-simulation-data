#!/bin/bash
#SBATCH --job-name=preflight_omscft_n1k4k
#SBATCH -A huterer2
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64gb
#SBATCH --time=02:00:00
#SBATCH --no-requeue
#SBATCH -o /home/jiamingp/diffusion_models_repo/logs/nf_cond_omsigc_n1024_n4096_aug_finetune/preflight_%j.out
#SBATCH -e /home/jiamingp/diffusion_models_repo/logs/nf_cond_omsigc_n1024_n4096_aug_finetune/preflight_%j.err
# CPU only (RUN_NAME=analysis). For ledger run nf_cond_omsigc_n1024_n4096_aug_finetune:
#   1. python -m pytest -q tests                       (full suite)
#   2. scripts/preflight_nf_cond_omsigc_n1024_n4096_aug_finetune.py run: static checks + a 1-epoch CPU resume check
#      per training run (32 updates at N = 1024, 128 at N = 4096) from the real baseline checkpoints (read only) into
#      $TMPDIR  -> local/nf_cond_omsigc_n1024_n4096_aug_finetune/preflight_receipt.json (written only on PASS,
#      never overwritten). No GPU, no training output under saved_runs, no samples.
set -uo pipefail
[[ "${RUN_NAME:-}" == "analysis" || "${RUN_NAME:-}" == "nf_cond_omsigc_n1024_n4096_aug_finetune" ]] \
  || { echo "Submit with --export=ALL,RUN_NAME=analysis" >&2; exit 1; }
export LD_LIBRARY_PATH=/sw/pkgs/arc/python3.10-anaconda/2023.03/lib:${LD_LIBRARY_PATH:-}
PROJECT_DIR=/home/jiamingp/diffusion_models_repo
VENV_PATH=/home/jiamingp/venvs/cosmodiff_nf_class
RECEIPT=${PROJECT_DIR}/local/nf_cond_omsigc_n1024_n4096_aug_finetune/preflight_receipt.json
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
python -m pytest -q tests 2>&1 | tail -60
echo "pytest exit ${PIPESTATUS[0]}"
echo "=== preflight nf_cond_omsigc_n1024_n4096_aug_finetune"
OUTJ="${TMPDIR:-/tmp}/ft_n1k4k_preflight.json"
python scripts/preflight_nf_cond_omsigc_n1024_n4096_aug_finetune.py run > "${OUTJ}"
echo "finetune preflight exit $?"; python - "${OUTJ}" <<'PY'
import json, sys
try:
    d = json.load(open(sys.argv[1]))
    print(json.dumps({"status": d["status"], "fails": d["fails"], "resume_check": d["resume_check"]}, indent=1))
except Exception as e:
    print("could not parse", e); print(open(sys.argv[1]).read()[-4000:])
PY
