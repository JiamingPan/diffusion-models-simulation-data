#!/bin/bash
# Submit one label-variant conditional sweep: omsig_continuous | omsig_class36 (five sizes each).
# Chain: prepare (CPU) -> train x5 (GPU, 2 concurrent) -> sample x5 -> VGG evaluate.
set -euo pipefail
VARIANT=${1:?usage: submit_nf_conditional_label_sweep.sh omsig_continuous|omsig_class36}
case "$VARIANT" in omsig_continuous) SWEEP=nf_conditional_omsig_continuous_200k;; omsig_class36) SWEEP=nf_conditional_omsig_class36_200k;; *) echo "unknown variant $VARIANT" >&2; exit 1;; esac
PROJECT_DIR=${PROJECT_DIR:-/home/jiamingp/diffusion_models_repo}
ACCOUNT=${ACCOUNT:-huterer2}
cd "${PROJECT_DIR}"
mkdir -p "logs/${SWEEP}"
N=$(python scripts/prepare_nf_conditional_label_sweep_configs.py --variant "$VARIANT" --print-runs | wc -l)
LAST=$((N-1))
prepare=$(sbatch -A "${ACCOUNT}" --parsable scripts/slurm/prepare_nf_conditional_${VARIANT}.sbatch)
train=$(sbatch -A "${ACCOUNT}" --dependency=afterok:${prepare} --array=0-${LAST}%2 --parsable scripts/slurm/train_nf_conditional_${VARIANT}_array.sbatch)
sample=$(sbatch -A "${ACCOUNT}" --dependency=afterok:${train} --array=0-${LAST}%2 --parsable scripts/slurm/sample_nf_conditional_${VARIANT}_array.sbatch)
evaluate=$(sbatch -A "${ACCOUNT}" --dependency=afterok:${sample} --parsable scripts/slurm/evaluate_nf_conditional_${VARIANT}.sbatch)
echo "variant=${VARIANT} sweep=${SWEEP} runs=${N}"
echo "prepare=${prepare}"; echo "train=${train}"; echo "sample=${sample}"; echo "evaluate=${evaluate}"
echo "At most two one-GPU tasks run concurrently; training tasks have 48 h walltime each."
