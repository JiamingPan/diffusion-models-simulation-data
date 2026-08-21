#!/bin/bash
#SBATCH --job-name=nf_c4_umap
#SBATCH --partition=spgpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=80gb
#SBATCH --time=02:00:00
#SBATCH -A huterer2
#SBATCH -o /home/jiamingp/diffusion_models_repo/logs/nf_conditional_bias_probe/c4_umap_%j.out
#SBATCH -e /home/jiamingp/diffusion_models_repo/logs/nf_conditional_bias_probe/c4_umap_%j.err

set -euo pipefail
trap 'rc=$?; echo "[c4-umap] failed at line ${LINENO}: ${BASH_COMMAND} (exit ${rc})" >&2' ERR

PROJECT_DIR=${PROJECT_DIR:-/home/jiamingp/diffusion_models_repo}
CODE_ROOT=${CODE_ROOT:?Set CODE_ROOT to the frozen C4 UMAP code worktree}
EXPECTED_COMMIT=${EXPECTED_COMMIT:?Set EXPECTED_COMMIT to the frozen code revision}
EXPECTED_RESULTS_COMMIT=${EXPECTED_RESULTS_COMMIT:-dced4f8928efe248d819a72560ef61a099d0c4a3}
RESULTS_ROOT=${RESULTS_ROOT:-/nfs/turbo/lsa-huterer/jiamingp/diffusion_models_repo/results/nf_conditional_bias_probe}
DATA_ROOT=${DATA_ROOT:-/gpfs/accounts/huterer_root/huterer0/CAMELS/CMD/3d_grids/IllustrisTNG}
PYTHON_BIN=${PYTHON_BIN:-/home/jiamingp/venvs/cosmodiff_nf_class/bin/python}
TORCH_HOME=${TORCH_HOME:-/scratch/huterer_root/huterer0/jiamingp/torch_cache}
INCOMPATIBLE_PYTHON_PATHS=${INCOMPATIBLE_PYTHON_PATHS:-/home/jiamingp/venvs/cosmodiff_nf/lib/python3.10/site-packages}
OUTPUT_DIR=${RESULTS_ROOT}/c4_frozen_vgg_umap_seed123_v1
VGG_WEIGHTS=${TORCH_HOME}/hub/checkpoints/vgg16-397923af.pth

actual=$(git -C "${CODE_ROOT}" rev-parse HEAD)
test "${actual}" = "${EXPECTED_COMMIT}"
test -z "$(git -C "${CODE_ROOT}" status --porcelain)"
test -x "${PYTHON_BIN}"
test ! -e "${OUTPUT_DIR}"

required_files=(
  "${CODE_ROOT}/scripts/evaluate_probe_c4_umap.py"
  "${CODE_ROOT}/simdiff_eval/probe_c4_umap.py"
  "${PROJECT_DIR}/local/nf_conditional_bias_probe/manifest.json"
  "${RESULTS_ROOT}/encoder/vgg_mlp_encoder.npz"
  "${RESULTS_ROOT}/encoder/vgg_mlp_encoder.pkl"
  "${RESULTS_ROOT}/degradation_controls/manifest.json"
  "${RESULTS_ROOT}/degradation_controls/power_transfer_curves.json"
  "${RESULTS_ROOT}/samples/nf_cond_bias_hi_u128_d2p07_n128_200k_seed123_dpm50_heldout_k64.npz"
  "${RESULTS_ROOT}/samples/nf_cond_bias_hi_u128_d2p14_n16384_200k_seed123_dpm50_heldout_k64.npz"
  "${VGG_WEIGHTS}"
)
for path in "${required_files[@]}"; do
  test -f "${path}"
done

export PYTHONNOUSERSITE=1
export TORCH_HOME
source "${CODE_ROOT}/scripts/probe_controls_runtime.sh"
probe_controls_prepare_runtime "${CODE_ROOT}" "${PYTHON_BIN}" "${INCOMPATIBLE_PYTHON_PATHS}"
export MPLCONFIGDIR="${PROJECT_DIR}/results/cache/matplotlib_c4_umap_${SLURM_JOB_ID}"
export NUMBA_CACHE_DIR="${PROJECT_DIR}/results/cache/numba_c4_umap_${SLURM_JOB_ID}"
mkdir -p "${MPLCONFIGDIR}" "${NUMBA_CACHE_DIR}" "${PROJECT_DIR}/logs/nf_conditional_bias_probe"

cd "${CODE_ROOT}"
"${PYTHON_BIN}" -c '
import pandas
import sklearn
import torch
import torchvision
import umap
from simdiff_eval import probe_c4_umap
print("[c4-umap] imports passed")
print("[c4-umap] torch", torch.__version__, "torchvision", torchvision.__version__)
print("[c4-umap] sklearn", sklearn.__version__, "umap", umap.__version__)
print("[c4-umap] module", probe_c4_umap.__file__)
'

"${PYTHON_BIN}" "${CODE_ROOT}/scripts/evaluate_probe_c4_umap.py" \
  --project-dir "${PROJECT_DIR}" \
  --code-root "${CODE_ROOT}" \
  --expected-code-revision "${EXPECTED_COMMIT}" \
  --expected-results-revision "${EXPECTED_RESULTS_COMMIT}" \
  --data-root "${DATA_ROOT}" \
  --results-root "${RESULTS_ROOT}" \
  --head "${RESULTS_ROOT}/encoder/vgg_mlp_encoder.pkl" \
  --vgg-weights-path "${VGG_WEIGHTS}" \
  --output-dir "${OUTPUT_DIR}" \
  --device cuda \
  --embedding-batch-size 32 \
  --seed 123 \
  --umap-neighbors 30 \
  --umap-min-dist 0.1 \
  --knn-k 15 \
  --bootstrap 2000

test -s "${OUTPUT_DIR}/manifest.json"
test -s "${OUTPUT_DIR}/c4_umap_metrics.csv"
test -s "${OUTPUT_DIR}/pooled_umap_reducer.joblib"
echo "PASS: frozen-probe C4 UMAP outputs are in ${OUTPUT_DIR}"
