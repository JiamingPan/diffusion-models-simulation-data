#!/bin/bash
#SBATCH --job-name=evalrun_omscft
#SBATCH -A huterer2
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80gb
#SBATCH --time=04:00:00
#SBATCH --no-requeue
#SBATCH -o /home/jiamingp/diffusion_models_repo/logs/nf_cond_omsigc_n256_aug_finetune/evaluate_run_%j.out
#SBATCH -e /home/jiamingp/diffusion_models_repo/logs/nf_cond_omsigc_n256_aug_finetune/evaluate_run_%j.err
# Ledger nf_cond_omsigc_n256_aug_finetune. scripts/evaluate_run.py --mode conditional on the three fine-tune
# checkpoints (+25k, +50k, +100k), CPU only -> results/nf_cond_omsigc_n256_aug_finetune/evaluate_run_v1
# (evaluate_run.py creates it; refuses if it exists). Note: nearest_train_cosine here is the plain (no D4 / shift)
# cosine. Extra arguments ("$@") pass through.
set -euo pipefail
export LD_LIBRARY_PATH=/sw/pkgs/arc/python3.10-anaconda/2023.03/lib:${LD_LIBRARY_PATH:-}
export LD_PRELOAD=/sw/pkgs/arc/python3.10-anaconda/2023.03/lib/libstdc++.so.6
[[ "${RUN_NAME:-}" == "nf_cond_omsigc_n256_aug_finetune" ]] \
  || { echo "Submit with --export=ALL,RUN_NAME=nf_cond_omsigc_n256_aug_finetune" >&2; exit 1; }
PROJECT_DIR=/home/jiamingp/diffusion_models_repo
VENV_PATH=/home/jiamingp/venvs/cosmodiff_nf_class
MANIFEST=local/nf_cond_omsigc_n256_aug_finetune/manifest.json
OUT_DIR=results/nf_cond_omsigc_n256_aug_finetune/evaluate_run_v1

cd "${PROJECT_DIR}"
[[ -f "${MANIFEST}" ]] || { echo "Missing manifest ${MANIFEST}" >&2; exit 1; }
[[ ! -e "${OUT_DIR}" ]] || { echo "Refusing to overwrite ${OUT_DIR}" >&2; exit 1; }
mkdir -p results/cache/matplotlib
source "${VENV_PATH}/bin/activate"
export PYTHONNOUSERSITE=1
RUNTIME=$(mktemp -d "${TMPDIR:-/tmp}/evalrun-runtime.XXXXXX")
python scripts/write_diffusers_runtime_sitecustomize.py "${RUNTIME}/sitecustomize.py"
export PYTHONPATH="${RUNTIME}:${PROJECT_DIR}:${PROJECT_DIR}/scripts"
export TORCH_HOME=${TORCH_HOME:-/nfs/turbo/lsa-huterer/jiamingp/torch_cache}
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export MPLCONFIGDIR="${PROJECT_DIR}/results/cache/matplotlib"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}" MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

python - "${MANIFEST}" <<'PY'
import json, sys
from pathlib import Path
from evaluate_nf_conditional_bias_probe import output_path_for
rows = json.load(open(sys.argv[1]))
root = Path("/home/jiamingp/diffusion_models_repo")
missing = [str(output_path_for(root, r, 123, 64, None)) for r in rows if not output_path_for(root, r, 123, 64, None).exists()]
if missing:
    sys.exit("Missing sample files:\n  " + "\n  ".join(missing))
print(f"{len(rows)} sample files present")
PY

python scripts/evaluate_run.py --mode conditional --manifest "${MANIFEST}" --out-dir "${OUT_DIR}" --device cpu "$@"
