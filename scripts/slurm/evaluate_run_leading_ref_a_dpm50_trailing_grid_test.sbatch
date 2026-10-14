#!/bin/bash
#SBATCH --job-name=evalrun_dpm50_lead_ref_a
#SBATCH -A huterer2
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80gb
#SBATCH --time=02:00:00
#SBATCH --no-requeue
#SBATCH -o /home/jiamingp/diffusion_models_repo/logs/dpm50_trailing_grid_test/evaluate_run_leading_ref_a_%j.out
#SBATCH -e /home/jiamingp/diffusion_models_repo/logs/dpm50_trailing_grid_test/evaluate_run_leading_ref_a_%j.err
# CPU only (RUN_NAME=analysis). Ledger dpm50_trailing_grid_test, leading-grid reference for arm a (no evaluate_run
# output existed for it): scripts/evaluate_run.py --mode conditional on the EXISTING sample file
#   results/nf_conditional_omsig_continuous_200k/samples/nf_cond_omsigc_hi_u128_d2p15_n32768_fresh200k_seed123_dpm50_heldout_k64.npz
# (read only; job 62232390_4, leading grid) -> NEW dir results/dpm50_trailing_grid_test/evaluate_run_leading_ref_a
# (evaluate_run.py refuses an existing --out-dir). Same flags as evaluate_run_plus0 / evaluate_run_v1 (no extra flags).
# Extra arguments ("$@") pass through to evaluate_run.py.
set -euo pipefail
export LD_LIBRARY_PATH=/sw/pkgs/arc/python3.10-anaconda/2023.03/lib:${LD_LIBRARY_PATH:-}
export LD_PRELOAD=/sw/pkgs/arc/python3.10-anaconda/2023.03/lib/libstdc++.so.6
[[ "${RUN_NAME:-}" == "analysis" || "${RUN_NAME:-}" == "dpm50_trailing_grid_test" ]] \
  || { echo "Submit with --export=ALL,RUN_NAME=analysis" >&2; exit 1; }
PROJECT_DIR=/home/jiamingp/diffusion_models_repo
VENV_PATH=/home/jiamingp/venvs/cosmodiff_nf_class
MANIFEST=local/nf_conditional_omsig_continuous_200k/manifest.json
OUT_DIR=results/dpm50_trailing_grid_test/evaluate_run_leading_ref_a
RUN=nf_cond_omsigc_hi_u128_d2p15_n32768_fresh200k
SAMPLE=results/nf_conditional_omsig_continuous_200k/samples/nf_cond_omsigc_hi_u128_d2p15_n32768_fresh200k_seed123_dpm50_heldout_k64.npz

cd "${PROJECT_DIR}"
[[ -f "${MANIFEST}" ]] || { echo "Missing manifest ${MANIFEST}" >&2; exit 1; }
[[ -f "${SAMPLE}" ]] || { echo "Missing sample file ${SAMPLE}" >&2; exit 1; }
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

python - "${MANIFEST}" "${RUN}" "${SAMPLE}" <<'PY'
import json, sys
from pathlib import Path
from evaluate_nf_conditional_bias_probe import output_path_for
rows = {r["run_name"]: r for r in json.load(open(sys.argv[1]))}
root = Path("/home/jiamingp/diffusion_models_repo")
p = output_path_for(root, rows[sys.argv[2]], 123, 64, None)
if p != root / sys.argv[3]:
    sys.exit(f"manifest resolves to {p}, expected {root / sys.argv[3]}")
print(f"scoring existing leading-grid sample {p}")
PY

python scripts/evaluate_run.py --mode conditional --manifest "${MANIFEST}" --run-name "${RUN}" --out-dir "${OUT_DIR}" --device cpu "$@"
