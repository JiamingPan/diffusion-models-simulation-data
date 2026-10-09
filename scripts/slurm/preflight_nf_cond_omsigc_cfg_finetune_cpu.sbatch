#!/bin/bash
#SBATCH --job-name=preflight_cfg_ft
#SBATCH -A huterer2
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64gb
#SBATCH --time=02:00:00
#SBATCH --no-requeue
#SBATCH -o /home/jiamingp/diffusion_models_repo/logs/nf_cond_omsigc_cfg_200k/preflight_%j.out
#SBATCH -e /home/jiamingp/diffusion_models_repo/logs/nf_cond_omsigc_cfg_200k/preflight_%j.err
# CPU only (RUN_NAME=analysis). For ledger runs nf_cond_omsigc_cfg and nf_cond_omsigc_n256_aug_finetune:
#   1. python -m pytest -q tests                       (full suite, includes tests/test_cfg_null_flag.py)
#   2. scripts/preflight_nf_cond_omsigc_cfg.py run      -> local/nf_cond_omsigc_cfg_200k/preflight_receipt.json
#   3. scripts/preflight_nf_cond_omsigc_n256_aug_finetune.py run (1-epoch CPU resume check from the real baseline
#      checkpoint into $TMPDIR)                        -> local/nf_cond_omsigc_n256_aug_finetune/preflight_receipt.json
# Receipts are written only on PASS and never overwritten. No GPU, no training output, no samples.
set -uo pipefail
export LD_LIBRARY_PATH=/sw/pkgs/arc/python3.10-anaconda/2023.03/lib:${LD_LIBRARY_PATH:-}
PROJECT_DIR=/home/jiamingp/diffusion_models_repo
VENV_PATH=/home/jiamingp/venvs/cosmodiff_nf_class
cd "${PROJECT_DIR}"
source "${VENV_PATH}/bin/activate"
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 TORCHDYNAMO_DISABLE=1 MPLCONFIGDIR="${PROJECT_DIR}/results/cache/matplotlib"
export LD_PRELOAD=/sw/pkgs/arc/python3.10-anaconda/2023.03/lib/libstdc++.so.6 PYTHONNOUSERSITE=1
export TORCH_HOME=/nfs/turbo/lsa-huterer/jiamingp/torch_cache HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
RT=$(mktemp -d "${TMPDIR:-/tmp}/pytest-rt.XXXXXX")  # torch-compat sitecustomize so diffusers imports
python scripts/write_diffusers_runtime_sitecustomize.py "${RT}/sitecustomize.py" >/dev/null
export PYTHONPATH="${RT}:${PROJECT_DIR}:${PROJECT_DIR}/scripts"

echo "=== pytest"
python -m pytest -q tests 2>&1 | tail -40
echo "pytest exit ${PIPESTATUS[0]}"
echo "=== preflight nf_cond_omsigc_cfg"
python scripts/preflight_nf_cond_omsigc_cfg.py run > "${TMPDIR:-/tmp}/cfg_preflight.json"
echo "cfg preflight exit $?"; tail -c 6000 "${TMPDIR:-/tmp}/cfg_preflight.json"
echo "=== preflight nf_cond_omsigc_n256_aug_finetune"
python scripts/preflight_nf_cond_omsigc_n256_aug_finetune.py run > "${TMPDIR:-/tmp}/ft_preflight.json"
echo "finetune preflight exit $?"; python - "${TMPDIR:-/tmp}/ft_preflight.json" <<'PY'
import json, sys
try:
    d = json.load(open(sys.argv[1]))
    print(json.dumps({"status": d["status"], "fails": d["fails"], "resume_check": d["resume_check"]}, indent=1))
except Exception as e:
    print("could not parse", e); print(open(sys.argv[1]).read()[-4000:])
PY
