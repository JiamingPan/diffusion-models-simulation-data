#!/bin/bash
#SBATCH --job-name=nf_l16_nb
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=08:00:00
#SBATCH --output=/home/jiamingp/diffusion_models_repo/logs/nf_generalize_fig2_dit_l16_continue500k_v2/notebook_%j.out
#SBATCH --error=/home/jiamingp/diffusion_models_repo/logs/nf_generalize_fig2_dit_l16_continue500k_v2/notebook_%j.err

set -euo pipefail

PROJECT_DIR=${PROJECT_DIR:-/home/jiamingp/diffusion_models_repo}
JUPYTER_BIN=${JUPYTER_BIN:-/sw/pkgs/arc/python3.10-anaconda/2023.03/bin/jupyter}
NOTEBOOK=${NOTEBOOK:-notebooks/nf_generalize_fig2_dit_l16_300k_500k_analysis.ipynb}

cd "${PROJECT_DIR}"
mkdir -p "logs/nf_generalize_fig2_dit_l16_continue500k_v2"

export JUPYTER_RUNTIME_DIR="${SLURM_TMPDIR:-/tmp}/${USER}_jupyter_${SLURM_JOB_ID:-$$}"
mkdir -p "$JUPYTER_RUNTIME_DIR"
chmod 700 "$JUPYTER_RUNTIME_DIR"
export TMPDIR="$JUPYTER_RUNTIME_DIR"
export XDG_RUNTIME_DIR="$JUPYTER_RUNTIME_DIR"

RUNTIME_PROBE="$JUPYTER_RUNTIME_DIR/.secure-write-probe"
: > "$RUNTIME_PROBE"
chmod 600 "$RUNTIME_PROBE"
RUNTIME_MODE=$(stat -c "%a" "$RUNTIME_PROBE")
if [[ "$RUNTIME_MODE" != "600" ]]; then
  echo "secure runtime preflight failed: $RUNTIME_PROBE has mode $RUNTIME_MODE, expected 600" >&2
  exit 1
fi
rm -f "$RUNTIME_PROBE"

export PYTHONPATH="${PROJECT_DIR}:${PROJECT_DIR}/scripts${PYTHONPATH:+:${PYTHONPATH}}"

exec "$JUPYTER_BIN" nbconvert \
  --execute \
  --to notebook \
  --inplace \
  --ExecutePreprocessor.timeout=-1 \
  "$NOTEBOOK"
