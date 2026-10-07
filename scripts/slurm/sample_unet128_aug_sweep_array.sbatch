#!/bin/bash
#SBATCH --job-name=u128aug_sample
#SBATCH --account=huterer2
#SBATCH --partition=spgpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=80gb
#SBATCH --time=04:00:00
#SBATCH --no-requeue
#SBATCH --array=0-17%6
#SBATCH -o /home/jiamingp/diffusion_models_repo/logs/unet128_aug_sweep/sample_%A_%a.out
#SBATCH -e /home/jiamingp/diffusion_models_repo/logs/unet128_aug_sweep/sample_%A_%a.err
# Same sampler as the UNet-128 noaug baseline (sample_nf_generalize_fig2_array.sbatch):
# DPMSolverMultistep, 50 steps, 512 samples, batch 8, seed 123, final checkpoint, no EMA.
# Tasks 0-5 d4shift, 6-11 warp, 12-17 d4equiv. Distinct sample_label per arm:
# dpm50_d4shift / dpm50_warp / dpm50_d4equiv under results/unet128_aug_sweep/samples/.
source /home/jiamingp/diffusion_models_repo/scripts/slurm/unet128_aug_sweep_env.sh
ARMS=(d4shift warp d4equiv)
ARM_TAG=${ARMS[$(( SLURM_ARRAY_TASK_ID / 6 ))]}
TASK=$(( SLURM_ARRAY_TASK_ID % 6 ))
RUN_NAME=$(run_field "${ARM_TAG}" "${TASK}" run_name)
OUTPUT_DIR=$(run_field "${ARM_TAG}" "${TASK}" output_dir)
SAMPLE=$(run_field "${ARM_TAG}" "${TASK}" sample_path)
SAMPLE=${PROJECT_DIR}/${SAMPLE//\{seed\}/123}
[[ -f "${OUTPUT_DIR}/reload_check.json" ]] || { echo "Training not finished/verified: ${OUTPUT_DIR}" >&2; exit 1; }
[[ ! -e "${SAMPLE}" ]] || { echo "Refusing to overwrite ${SAMPLE}" >&2; exit 1; }
mkdir -p "$(dirname "${SAMPLE}")"
# The run's own training_config.yaml names the model class (D4ScalarUNet2DModel for d4equiv).
COMMON=(--checkpoint "${OUTPUT_DIR}" --config "${OUTPUT_DIR}/training_config.yaml" --output "${SAMPLE}"
        --image-size 128 --scheduler DPMSolverMultistepScheduler --num-steps 50 --seed 123)
echo "Sampling ${RUN_NAME} -> ${SAMPLE}"
"${PYTHON_BIN}" scripts/sample_cosmodiff.py "${COMMON[@]}" --num-samples 1 --batch-size 1 --device cpu --preflight-only
"${PYTHON_BIN}" scripts/sample_cosmodiff.py "${COMMON[@]}" --num-samples 512 --batch-size 8 --device cuda
