#!/bin/bash
#SBATCH --job-name=evalrun_omscft_plus0
#SBATCH -A huterer2
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80gb
#SBATCH --time=03:00:00
#SBATCH --no-requeue
#SBATCH -o /home/jiamingp/diffusion_models_repo/logs/nf_cond_omsigc_n1024_n4096_aug_finetune/evaluate_run_plus0_%j.out
#SBATCH -e /home/jiamingp/diffusion_models_repo/logs/nf_cond_omsigc_n1024_n4096_aug_finetune/evaluate_run_plus0_%j.err
# CPU only. Ledger nf_cond_omsigc_n1024_n4096_aug_finetune, +0 baselines: scripts/evaluate_run.py --mode conditional on
# the EXISTING samples of nf_cond_omsigc_hi_u128_d2p10_n1024_fresh200k and nf_cond_omsigc_hi_u128_d2p12_n4096_fresh200k
# (results/nf_conditional_omsig_continuous_200k/samples/*_seed123_dpm50_heldout_k64.npz, read only)
# -> NEW dir results/nf_cond_omsigc_n1024_n4096_aug_finetune/evaluate_run_plus0 (evaluate_run.py refuses an existing
# --out-dir). Same flags as the N = 256 fine-tune evaluation (evaluate_run_v1). Extra arguments ("$@") pass through.
set -euo pipefail
export LD_LIBRARY_PATH=/sw/pkgs/arc/python3.10-anaconda/2023.03/lib:${LD_LIBRARY_PATH:-}
export LD_PRELOAD=/sw/pkgs/arc/python3.10-anaconda/2023.03/lib/libstdc++.so.6
[[ "${RUN_NAME:-}" == "analysis" || "${RUN_NAME:-}" == "nf_cond_omsigc_n1024_n4096_aug_finetune" ]] \
  || { echo "Submit with --export=ALL,RUN_NAME=analysis" >&2; exit 1; }
PROJECT_DIR=/home/jiamingp/diffusion_models_repo
VENV_PATH=/home/jiamingp/venvs/cosmodiff_nf_class
MANIFEST=local/nf_conditional_omsig_continuous_200k/manifest.json
OUT_DIR=results/nf_cond_omsigc_n1024_n4096_aug_finetune/evaluate_run_plus0
RUNS=(nf_cond_omsigc_hi_u128_d2p10_n1024_fresh200k nf_cond_omsigc_hi_u128_d2p12_n4096_fresh200k)

cd "${PROJECT_DIR}"
[[ -f "${MANIFEST}" ]] || { echo "Missing manifest ${MANIFEST}" >&2; exit 1; }
[[ ! -e "${OUT_DIR}" ]] || { echo "Refusing to overwrite ${OUT_DIR}" >&2; exit 1; }
mkdir -p results/cache/matplotlib "$(dirname "${OUT_DIR}")"
source "${VENV_PATH}/bin/activate"
export PYTHONNOUSERSITE=1
RUNTIME=$(mktemp -d "${TMPDIR:-/tmp}/evalrun-runtime.XXXXXX")
python scripts/write_diffusers_runtime_sitecustomize.py "${RUNTIME}/sitecustomize.py"
export PYTHONPATH="${RUNTIME}:${PROJECT_DIR}:${PROJECT_DIR}/scripts"
export TORCH_HOME=${TORCH_HOME:-/nfs/turbo/lsa-huterer/jiamingp/torch_cache}
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export MPLCONFIGDIR="${PROJECT_DIR}/results/cache/matplotlib"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}" MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

python - "${MANIFEST}" "${RUNS[@]}" <<'PY'
import json, sys
from pathlib import Path
from evaluate_nf_conditional_bias_probe import output_path_for
rows = {r["run_name"]: r for r in json.load(open(sys.argv[1]))}
root = Path("/home/jiamingp/diffusion_models_repo")
missing = [str(output_path_for(root, rows[n], 123, 64, None)) for n in sys.argv[2:]
           if not output_path_for(root, rows[n], 123, 64, None).exists()]
if missing:
    sys.exit("Missing sample files:\n  " + "\n  ".join(missing))
print(f"{len(sys.argv) - 2} sample files present")
PY

ARGS=()
for r in "${RUNS[@]}"; do ARGS+=(--run-name "${r}"); done
python scripts/evaluate_run.py --mode conditional --manifest "${MANIFEST}" "${ARGS[@]}" --out-dir "${OUT_DIR}" --device cpu "$@"
