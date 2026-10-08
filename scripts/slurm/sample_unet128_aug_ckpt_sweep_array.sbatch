#!/bin/bash
#SBATCH --job-name=u128aug_ckpt
#SBATCH --account=huterer2
#SBATCH --partition=spgpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16gb
#SBATCH --time=01:00:00
#SBATCH --no-requeue
#SBATCH --array=0-35%2
#SBATCH -o /home/jiamingp/diffusion_models_repo/logs/unet128_aug_sweep/ckpt_sample_%A_%a.out
#SBATCH -e /home/jiamingp/diffusion_models_repo/logs/unet128_aug_sweep/ckpt_sample_%A_%a.err
# Intermediate-checkpoint samples for d4shift / d4equiv at N = 64, 128 (every 20k updates, 20k..180k;
# the 200k samples already exist). Same sampler as the sweep: DPMSolverMultistep, 50 steps, 512 samples,
# batch 8, seed 123, no EMA. Task k = k-th row of local/unet128_aug_sweep/ckpt_sweep_runs.json with
# needs_sampling == true. Writes results/unet128_aug_sweep/samples_ckpt/; refuses to overwrite.
source /home/jiamingp/diffusion_models_repo/scripts/slurm/unet128_aug_sweep_env.sh
read -r RUN_NAME CHECKPOINT BASE SAMPLE < <("${PYTHON_BIN}" - "${SLURM_ARRAY_TASK_ID}" <<'PY'
import json, sys
rows = [r for r in json.load(open("local/unet128_aug_sweep/ckpt_sweep_runs.json")) if r["needs_sampling"]]
r = rows[int(sys.argv[1])]
print(r["run_name"], r["checkpoint"], r["base_run"], r["sample_path"])
PY
)
OUTPUT_DIR=$(dirname "${CHECKPOINT}")
SAMPLE=${PROJECT_DIR}/${SAMPLE}
[[ -d "${CHECKPOINT}" ]] || { echo "Missing checkpoint ${CHECKPOINT}" >&2; exit 1; }
[[ ! -e "${SAMPLE}" ]] || { echo "Refusing to overwrite ${SAMPLE}" >&2; exit 1; }
mkdir -p "$(dirname "${SAMPLE}")"
COMMON=(--checkpoint "${CHECKPOINT}" --config "${OUTPUT_DIR}/training_config.yaml" --output "${SAMPLE}"
        --image-size 128 --scheduler DPMSolverMultistepScheduler --num-steps 50 --seed 123)
echo "Sampling ${RUN_NAME} (${CHECKPOINT}) -> ${SAMPLE}"
"${PYTHON_BIN}" scripts/sample_cosmodiff.py "${COMMON[@]}" --num-samples 1 --batch-size 1 --device cpu --preflight-only
"${PYTHON_BIN}" scripts/sample_cosmodiff.py "${COMMON[@]}" --num-samples 512 --batch-size 8 --device cuda
