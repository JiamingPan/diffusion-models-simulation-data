#!/bin/bash
# Submit the four-stage DiT-L16 continuation and exact-checkpoint sampling chain.

set -euo pipefail

PROJECT_DIR=${PROJECT_DIR:-/home/jiamingp/diffusion_models_repo}
ACCOUNT=${ACCOUNT:-huterer2}
OVERWRITE=${OVERWRITE:-0}
SUBMIT_ANALYSIS=${SUBMIT_ANALYSIS:-1}
START_STAGE=${START_STAGE:-1}
REUSE_EXISTING_MANIFEST=${REUSE_EXISTING_MANIFEST:-0}

cd "${PROJECT_DIR}"

if (( START_STAGE < 1 || START_STAGE > 4 )); then
  echo "START_STAGE must be between 1 and 4; got ${START_STAGE}" >&2
  exit 1
fi

if [[ "${REUSE_EXISTING_MANIFEST}" == "1" ]]; then
  test -f "${PROJECT_DIR}/local/nf_generalize_fig2_dit_l16_continue/manifest.json"
  test -f "${PROJECT_DIR}/local/nf_generalize_fig2_dit_l16_continue/analysis_manifest.json"
  echo "Reusing the frozen continuation manifest from its original preparation."
else
  if [[ "${START_STAGE}" != "1" ]]; then
    echo "START_STAGE>1 requires REUSE_EXISTING_MANIFEST=1 to prevent target drift." >&2
    exit 1
  fi
  python scripts/prepare_nf_generalize_fig2_dit_l16_continue_configs.py \
    --project-dir "${PROJECT_DIR}"
fi

python scripts/prepare_nf_generalize_fig2_dit_l16_continue_configs.py \
  --project-dir "${PROJECT_DIR}" \
  --use-existing-manifest \
  --check-only

python scripts/prepare_nf_generalize_fig2_dit_l16_continue_configs.py \
  --project-dir "${PROJECT_DIR}" \
  --use-existing-manifest \
  --seed-checkpoints

resume_precheck=$(
  CONTINUE_STAGE="${START_STAGE}" sbatch -A "${ACCOUNT}" --parsable \
    scripts/slurm/precheck_nf_generalize_fig2_dit_l16_resume.sbatch
)
resume_precheck=${resume_precheck%%;*}
echo "checkpoint-resume precheck: ${resume_precheck}"

ANALYSIS_MANIFEST=${PROJECT_DIR}/local/nf_generalize_fig2_dit_l16_continue/analysis_manifest.json
if [[ "${SUBMIT_ANALYSIS}" == "1" && "${START_STAGE}" == "1" ]]; then
  baseline_pca=$(
    MANIFEST_PATH="${ANALYSIS_MANIFEST}" SAMPLE_LABEL="dpm50" \
    OUT_PREFIX="nf_generalize_fig2_dit_l16_cont_200k_pca_full_nn" \
      sbatch -A "${ACCOUNT}" --parsable \
      scripts/slurm/analyze_nf_generalize_fig2_dit_pca.sbatch
  )
  baseline_sscd=$(
    MANIFEST_PATH="${ANALYSIS_MANIFEST}" SAMPLE_LABEL="dpm50" \
    OUT_PREFIX="nf_generalize_fig2_dit_l16_cont_200k_sscd_full_nn" \
      sbatch -A "${ACCOUNT}" --parsable \
      scripts/slurm/analyze_nf_generalize_fig2_dit_sscd.sbatch
  )
  echo "baseline 200k analysis: pca=${baseline_pca%%;*} sscd=${baseline_sscd%%;*}"
fi

previous_job=${resume_precheck}
echo "Submitting four sequential 25k-update stages; each array is limited to two GPUs."
for stage in 1 2 3 4; do
  if (( stage < START_STAGE )); then
    continue
  fi
  dependency_args=()
  if [[ -n "${previous_job}" ]]; then
    dependency_args+=(--dependency="afterok:${previous_job}")
  fi

  train_job=$(
    CONTINUE_STAGE="${stage}" sbatch -A "${ACCOUNT}" --array=0-4%2 --parsable \
      "${dependency_args[@]}" \
      scripts/slurm/train_nf_generalize_fig2_dit_l16_continue_array.sbatch
  )
  train_job=${train_job%%;*}

  sample_job=$(
    CONTINUE_STAGE="${stage}" OVERWRITE="${OVERWRITE}" \
      sbatch -A "${ACCOUNT}" --array=0-4%2 --parsable \
      --dependency="afterok:${train_job}" \
      scripts/slurm/sample_nf_generalize_fig2_dit_l16_continue_array.sbatch
  )
  sample_job=${sample_job%%;*}

  echo "stage ${stage}: train=${train_job} sample=${sample_job}"

  if [[ "${SUBMIT_ANALYSIS}" == "1" ]]; then
    total_k=$((200 + 25 * stage))
    sample_label="dpm50_cont_${total_k}k"
    pca_job=$(
      SAMPLE_LABEL="${sample_label}" \
      MANIFEST_PATH="${ANALYSIS_MANIFEST}" \
      OUT_PREFIX="nf_generalize_fig2_dit_l16_cont_${total_k}k_pca_full_nn" \
        sbatch -A "${ACCOUNT}" --parsable \
        --dependency="afterok:${sample_job}" \
        scripts/slurm/analyze_nf_generalize_fig2_dit_pca.sbatch
    )
    pca_job=${pca_job%%;*}
    sscd_job=$(
      SAMPLE_LABEL="${sample_label}" \
      MANIFEST_PATH="${ANALYSIS_MANIFEST}" \
      OUT_PREFIX="nf_generalize_fig2_dit_l16_cont_${total_k}k_sscd_full_nn" \
        sbatch -A "${ACCOUNT}" --parsable \
        --dependency="afterok:${sample_job}" \
        scripts/slurm/analyze_nf_generalize_fig2_dit_sscd.sbatch
    )
    sscd_job=${sscd_job%%;*}
    echo "         analysis: pca=${pca_job} sscd=${sscd_job}"
  fi
  previous_job=${sample_job}
done

echo "Continuation stages are sequential; each continuation array uses at most two GPUs."
echo "Other jobs, including Jupyter and CPU analyses, are not included in that two-GPU limit."
echo "If a stage times out, restart from it with:"
echo "  START_STAGE=<stage> REUSE_EXISTING_MANIFEST=1 bash scripts/slurm/submit_nf_generalize_fig2_dit_l16_continue.sh"
