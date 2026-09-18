#!/bin/bash
#SBATCH --job-name=dit_a40_early
#SBATCH --partition=spgpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=80G
#SBATCH --time=01:00:00
#SBATCH --no-requeue

# This file is a payload only: it never submits/cancels another job.
set -euo pipefail
: "${EARLY_CODE_ROOT:?}" "${EARLY_EXPECTED_COMMIT:?}"
export PROJECT_DIR=/home/jiamingp/diffusion_models_repo
export EXPECTED_COMMIT=50b685f5999b90a212f696ab38b172dca5d968ab
export CODE_ROOT=/scratch/huterer_root/huterer0/jiamingp/dit_a40_50b685f
export RUNTIME_CODE_ROOT=/scratch/huterer_root/huterer0/jiamingp/iaifi_poster_code_555f350
export PYTHON_BIN=/home/jiamingp/venvs/cosmodiff_nf_class/bin/python
export COSMODIFF_PIN_ROOT=/scratch/huterer_root/huterer0/jiamingp/cosmodiff_seed_restart_pin_58c77eb_555f350
export COSMODIFF_PIN_MANIFEST="$COSMODIFF_PIN_ROOT/seed_restart_pin_manifest.json"
export EXPECTED_COSMODIFF_BASE_REVISION=58c77eb45de6e4d135697ba83ffee93ae54d918c
export ABLATION_OUT_DIR=/scratch/huterer_root/huterer0/jiamingp/dit_l16_a40_init_patch_v2_datafix
export ABLATION_PLAN_SHA256=e814a42ad2026c2b9963a43b017220ff0326800c21d920c4234523e16b06b21f
export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 PYTHONHASHSEED=123
export PYTHONPATH="$COSMODIFF_PIN_ROOT/seed_restart_runtime:$RUNTIME_CODE_ROOT:$COSMODIFF_PIN_ROOT"
export COSMODIFF_DIR="$COSMODIFF_PIN_ROOT"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
cd "$RUNTIME_CODE_ROOT"
case "${1:-}" in
  --preflight) mode=--preflight ;;
  '') mode=--run ;;
  *) echo "Unknown argument; no sampling" >&2; exit 1 ;;
esac
"$PYTHON_BIN" "$EARLY_CODE_ROOT/scripts/sample_dit_a40_early.py" \
  "$mode" --out-dir "$ABLATION_OUT_DIR"
