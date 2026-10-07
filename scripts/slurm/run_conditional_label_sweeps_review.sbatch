#!/bin/bash
#SBATCH --job-name=label_sweeps_review
#SBATCH --account=huterer2
#SBATCH --partition=standard
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --no-requeue
#SBATCH --output=/home/jiamingp/diffusion_models_repo/logs/label_sweeps_review-%j.out
#SBATCH --error=/home/jiamingp/diffusion_models_repo/logs/label_sweeps_review-%j.err
# Execute notebooks/conditional_label_sweeps_review.ipynb on a CPU node (reads saved CSVs and
# sample NPZs only). Writes a new executed copy; never overwrites an existing one.
set -euo pipefail
cd /home/jiamingp/diffusion_models_repo
module load python3.10-anaconda/2023.03
# Jupyter needs 0600 connection files; turbo ACLs force 0670, so use node-local /tmp.
export TMPDIR=$(mktemp -d /tmp/label_sweeps_review.XXXXXX)
export JUPYTER_RUNTIME_DIR=$TMPDIR
trap 'rm -rf "$TMPDIR"' EXIT

export REVIEW_RUN_STAMP=$(date +%Y%m%d_%H%M%S)
out_dir=notebooks/executed
out_name=conditional_label_sweeps_review_$(date +%F)
if [[ -e "$out_dir/$out_name.ipynb" ]]; then
    out_name=${out_name}_${REVIEW_RUN_STAMP}
fi
[[ ! -e "$out_dir/$out_name.ipynb" ]] || { echo "Refusing to overwrite $out_dir/$out_name.ipynb"; exit 1; }

python -m jupyter nbconvert --to notebook --execute \
    --ExecutePreprocessor.kernel_name=python3 --ExecutePreprocessor.timeout=3000 \
    --output-dir "$out_dir" --output "$out_name" \
    notebooks/conditional_label_sweeps_review.ipynb
echo "EXECUTED: $out_dir/$out_name.ipynb"
echo "FIGURES:  results/nf_conditional_omsig_{continuous,class36}_200k/figures/label_sweeps_review_${REVIEW_RUN_STAMP}/"
