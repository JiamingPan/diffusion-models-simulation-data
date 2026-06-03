#!/bin/bash
# Submit the continuous HI cosmology bias-probe pipeline on Great Lakes.

set -euo pipefail

PROJECT_DIR=${PROJECT_DIR:-/home/jiamingp/diffusion_models_repo}
ACCOUNT=${ACCOUNT:-huterer2}

cd "${PROJECT_DIR}"

prepare=$(sbatch -A "${ACCOUNT}" --parsable scripts/slurm/prepare_nf_conditional_bias_probe.sbatch)
train128=$(sbatch -A "${ACCOUNT}" --dependency=afterok:${prepare} --export=ALL,BIAS_PREPARE_IN_TRAIN=0 --parsable scripts/slurm/train_nf_conditional_bias_probe_n128.sbatch)
train16k=$(sbatch -A "${ACCOUNT}" --dependency=afterok:${prepare} --export=ALL,BIAS_PREPARE_IN_TRAIN=0 --parsable scripts/slurm/train_nf_conditional_bias_probe_n16384.sbatch)
encoder=$(sbatch -A "${ACCOUNT}" --dependency=afterok:${prepare} --parsable scripts/slurm/train_nf_conditional_bias_encoder.sbatch)
sample=$(sbatch -A "${ACCOUNT}" --dependency=afterok:${train128}:${train16k} --parsable scripts/slurm/sample_nf_conditional_bias_probe.sbatch)
eval_job=$(sbatch -A "${ACCOUNT}" --dependency=afterok:${sample}:${encoder} --parsable scripts/slurm/evaluate_nf_conditional_bias_probe.sbatch)

echo "prepare:         ${prepare}"
echo "train N=128:      ${train128}"
echo "train N=16384:    ${train16k}"
echo "encoder:          ${encoder}"
echo "sample after train:${sample}"
echo "eval after sample+encoder: ${eval_job}"
