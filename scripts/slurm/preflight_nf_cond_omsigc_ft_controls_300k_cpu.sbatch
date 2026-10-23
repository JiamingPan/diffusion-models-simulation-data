#!/bin/bash
#SBATCH --job-name=preflight_ftctl300k
#SBATCH -A huterer2
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64gb
#SBATCH --time=04:00:00
#SBATCH --no-requeue
#SBATCH -o /home/jiamingp/diffusion_models_repo/logs/nf_cond_omsigc_ft_controls_300k/preflight_%j.out
#SBATCH -e /home/jiamingp/diffusion_models_repo/logs/nf_cond_omsigc_ft_controls_300k/preflight_%j.err
# CPU only (RUN_NAME=analysis). For ledger run nf_cond_omsigc_ft_controls_300k. Run AFTER the opt-in diff to
# scripts/finetune_cosmodiff_resume.py is applied (APPROVE EDIT); before that the preflight FAILs by design.
#   1. python -m pytest -q tests                       (full suite)
#   2. scripts/preflight_nf_cond_omsigc_ft_controls_300k.py run: static checks; per-config pipeline check on every
#      training item (A none, B bitwise identity, C D4 x roll); a 1-epoch CPU resume check per training segment
#      (8: seg1 A/B and C from the real sources, read only; seg2 A/B chained from the seg1 check checkpoint) into
#      $TMPDIR; three negative controls (wrapper must still refuse without the flags)
#      -> local/nf_cond_omsigc_ft_controls_300k/preflight_receipt.json (written only on PASS, never overwritten).
#   No GPU, no output under saved_runs, no samples.
set -uo pipefail
[[ "${RUN_NAME:-}" == "analysis" || "${RUN_NAME:-}" == "nf_cond_omsigc_ft_controls_300k" ]] \
  || { echo "Submit with --export=ALL,RUN_NAME=analysis" >&2; exit 1; }
export LD_LIBRARY_PATH=/sw/pkgs/arc/python3.10-anaconda/2023.03/lib:${LD_LIBRARY_PATH:-}
PROJECT_DIR=/home/jiamingp/diffusion_models_repo
VENV_PATH=/home/jiamingp/venvs/cosmodiff_nf_class
RECEIPT=${PROJECT_DIR}/local/nf_cond_omsigc_ft_controls_300k/preflight_receipt.json
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
echo "=== preflight nf_cond_omsigc_ft_controls_300k"
OUTJ="${TMPDIR:-/tmp}/ftctl300k_preflight.json"
python scripts/preflight_nf_cond_omsigc_ft_controls_300k.py run > "${OUTJ}"
echo "ft-controls preflight exit $?"; python - "${OUTJ}" <<'PY'
import json, sys
try:
    d = json.load(open(sys.argv[1]))
    print(json.dumps({"status": d.get("status"), "fails": d.get("fails"), "pipeline_check": d.get("pipeline_check"),
                      "negative_controls": d.get("negative_controls"),
                      "resume_check_summary": {k: {kk: v.get(kk) for kk in (
                          "mode", "stock_load_checkpoint_augmentations_description", "train_fetches_recorded",
                          "train_fetches_equal_stored", "train_fetches_in_d4_shift_orbit",
                          "post_fetch_index0_twice_differ", "source_checkpoint", "new_checkpoint",
                          "new_checkpoint_name", "chained_from_check_checkpoint")}
                          for k, v in (d.get("resume_check") or {}).items()}}, indent=1))
except Exception as e:
    print("could not parse", e); print(open(sys.argv[1]).read()[-4000:])
PY
