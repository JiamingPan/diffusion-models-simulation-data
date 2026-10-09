#!/bin/bash
#SBATCH --job-name=u128aug_w2_sample
#SBATCH --account=huterer2
#SBATCH --partition=spgpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=80gb
#SBATCH --time=04:00:00
#SBATCH --no-requeue
#SBATCH --array=0-8%2
#SBATCH -o /home/jiamingp/diffusion_models_repo/logs/unet128_aug_wave2/sample_%A_%a.out
#SBATCH -e /home/jiamingp/diffusion_models_repo/logs/unet128_aug_wave2/sample_%A_%a.err
# Same sampler as the UNet-128 noaug baseline and wave 1 (sample_unet128_aug_sweep_array.sbatch):
# DPMSolverMultistep, 50 steps, 512 samples, batch 8, seed 123, final checkpoint, no EMA.
# Task i = local/unet128_aug_wave2/runs.json[i]; sample_label dpm50_<arm_tag>, results/unet128_aug_wave2/samples/.
source /home/jiamingp/diffusion_models_repo/scripts/slurm/unet128_aug_wave2_env.sh
TASK=${SLURM_ARRAY_TASK_ID}
RUN_NAME=$(wave_field "${TASK}" run_name)
OUTPUT_DIR=$(wave_field "${TASK}" output_dir)
SAMPLE=$(wave_field "${TASK}" sample_path)
SAMPLE=${PROJECT_DIR}/${SAMPLE//\{seed\}/123}
[[ -f "${OUTPUT_DIR}/reload_check.json" ]] || { echo "Training not finished/verified: ${OUTPUT_DIR}" >&2; exit 1; }
[[ ! -e "${SAMPLE}" ]] || { echo "Refusing to overwrite ${SAMPLE}" >&2; exit 1; }
mkdir -p "$(dirname "${SAMPLE}")"
COMMON=(--checkpoint "${OUTPUT_DIR}" --config "${OUTPUT_DIR}/training_config.yaml" --output "${SAMPLE}"
        --image-size 128 --scheduler DPMSolverMultistepScheduler --num-steps 50 --seed 123)
echo "Sampling ${RUN_NAME} -> ${SAMPLE}"
"${PYTHON_BIN}" scripts/sample_cosmodiff.py "${COMMON[@]}" --num-samples 1 --batch-size 1 --device cpu --preflight-only
"${PYTHON_BIN}" scripts/sample_cosmodiff.py "${COMMON[@]}" --num-samples 512 --batch-size 8 --device cuda
