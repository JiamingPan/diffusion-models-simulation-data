#!/bin/bash
# Build/verify the immutable pin and inspect source checkpoints without compute.

set -euo pipefail

PROJECT_DIR=${PROJECT_DIR:-/home/jiamingp/diffusion_models_repo}
CODE_ROOT=${CODE_ROOT:?Set CODE_ROOT to the frozen repository worktree}
EXPECTED_COMMIT=${EXPECTED_COMMIT:?Set the exact repository commit}
BASE_COSMODIFF_DIR=${BASE_COSMODIFF_DIR:?Set the clean base cosmodiff checkout}
EXPECTED_COSMODIFF_BASE_REVISION=${EXPECTED_COSMODIFF_BASE_REVISION:?Set the exact cosmodiff revision}
COSMODIFF_PIN_ROOT=${COSMODIFF_PIN_ROOT:?Set the new immutable pin destination}
PYTHON_BIN=${PYTHON_BIN:-/home/jiamingp/venvs/cosmodiff_nf_class/bin/python}
VENV_PATH=${VENV_PATH:-/home/jiamingp/venvs/cosmodiff_nf_class}
EXPECTED_TORCH_PREFIX=${EXPECTED_TORCH_PREFIX:-${VENV_PATH}}
BASE_VENV_PATH=${BASE_VENV_PATH:-/home/jiamingp/venvs/cosmodiff_nf}
BASE_VENV_SITE=${BASE_VENV_PATH}/lib/python3.10/site-packages
SYSTEM_ANACONDA_SITE=/sw/pkgs/arc/python3.10-anaconda/2023.03/lib/python3.10/site-packages
PIN_MANIFEST=${COSMODIFF_PIN_ROOT}/seed_restart_pin_manifest.json
PREPARE=${CODE_ROOT}/scripts/prepare_nf_generalize_fig2_dit_l16_seed_restart500k_configs.py
MANIFEST=${PROJECT_DIR}/local/nf_generalize_fig2_dit_l16_seed_restart500k_v1/manifest.json

test "$(git -C "${CODE_ROOT}" rev-parse HEAD)" = "${EXPECTED_COMMIT}"
test -z "$(git -C "${CODE_ROOT}" status --porcelain)"
test "$(git -C "${BASE_COSMODIFF_DIR}" rev-parse HEAD)" = \
  "${EXPECTED_COSMODIFF_BASE_REVISION}"
test -z "$(git -C "${BASE_COSMODIFF_DIR}" status --porcelain)"
test -x "${PYTHON_BIN}"
test -s "${MANIFEST}"

PATCH_ARGS=(
  --patch-script "${CODE_ROOT}/scripts/patch_cosmodiff_package_metadata.py"
  --patch-script "${CODE_ROOT}/scripts/patch_cosmodiff_constant_label.py"
  --patch-script "${CODE_ROOT}/scripts/patch_cosmodiff_dit_class_labels.py"
  --patch-script "${CODE_ROOT}/scripts/patch_cosmodiff_checkpoint_state.py"
)
RUNTIME_ARGS=(
  --code-root "${CODE_ROOT}"
  --expected-torch-prefix "${EXPECTED_TORCH_PREFIX}"
  --incompatible-python-path "${BASE_VENV_SITE}"
  --incompatible-python-path "${SYSTEM_ANACONDA_SITE}"
)

if [[ ! -e "${COSMODIFF_PIN_ROOT}" ]]; then
  "${PYTHON_BIN}" "${CODE_ROOT}/scripts/build_cosmodiff_seed_restart_pin.py" \
    --source-repo "${BASE_COSMODIFF_DIR}" \
    --base-revision "${EXPECTED_COSMODIFF_BASE_REVISION}" \
    --destination "${COSMODIFF_PIN_ROOT}" \
    --python-bin "${PYTHON_BIN}" \
    "${PATCH_ARGS[@]}" \
    "${RUNTIME_ARGS[@]}" >/dev/null
fi

RUNTIME_ROOT=${COSMODIFF_PIN_ROOT}/seed_restart_runtime
export PYTHONNOUSERSITE=1
export PYTHONPATH="${RUNTIME_ROOT}:${CODE_ROOT}:${COSMODIFF_PIN_ROOT}"

"${PYTHON_BIN}" "${CODE_ROOT}/scripts/verify_cosmodiff_seed_restart_runtime.py" \
  "${COSMODIFF_PIN_ROOT}" \
  --manifest "${PIN_MANIFEST}" \
  --expected-base-revision "${EXPECTED_COSMODIFF_BASE_REVISION}" \
  --python-bin "${PYTHON_BIN}" \
  "${PATCH_ARGS[@]}" \
  "${RUNTIME_ARGS[@]}" >/dev/null

"${PYTHON_BIN}" "${PREPARE}" \
  --project-dir "${PROJECT_DIR}" \
  --use-existing-manifest \
  --check-only

"${PYTHON_BIN}" - "${PREPARE}" "${MANIFEST}" <<'PY'
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

prepare_path = Path(sys.argv[1]).resolve()
manifest_path = Path(sys.argv[2]).resolve()
spec = importlib.util.spec_from_file_location("seed_restart_prepare", prepare_path)
module = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(module)
rows = json.loads(manifest_path.read_text())

stage_one = {
    str(row["dataset_tag"]): row
    for row in rows
    if int(row["continue_stage"]) == 1
}
assert set(stage_one) == {"d2p08", "d2p10"}

for dataset_tag in ("d2p08", "d2p10"):
    row = stage_one[dataset_tag]
    assert int(row["resume_seed"]) == 456
    source_checkpoint = Path(row["source_checkpoint"])
    inventory = module.checkpoint_inventory(source_checkpoint)
    assert inventory
    encoded = json.dumps(inventory, sort_keys=True).encode()
    inventory_sha256 = hashlib.sha256(encoded).hexdigest()
    targets = sorted(
        int(candidate["target_total_updates"])
        for candidate in rows
        if str(candidate["dataset_tag"]) == dataset_tag
    )
    assert targets == [340000, 380000, 420000, 460000, 500000]
    print(json.dumps({
        "dataset_tag": dataset_tag,
        "source_checkpoint": str(source_checkpoint),
        "source_file_count": len(inventory),
        "source_inventory_sha256": inventory_sha256,
        "resume_seed": int(row["resume_seed"]),
        "target_total_updates": targets,
    }, sort_keys=True))

print("NO JOB SUBMITTED")
PY
