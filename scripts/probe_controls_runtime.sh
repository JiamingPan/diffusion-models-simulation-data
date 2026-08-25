#!/bin/bash

probe_controls_prepare_runtime() {
  local code_root=${1:?code root is required}
  local python_bin=${2:?python executable is required}
  local incompatible_paths=${3:?incompatible path list is required}
  local runtime_root

  runtime_root="${TMPDIR:-/tmp}/probe_controls_runtime_${USER:-user}_$$"
  mkdir -p "${runtime_root}"
  "${python_bin}" -S -c '
import sys
from pathlib import Path
sys.path.insert(0, sys.argv[1])
from probe_controls_runtime import write_sitecustomize
write_sitecustomize(Path(sys.argv[2]) / "sitecustomize.py", sys.argv[3].split(":") if sys.argv[3] else ())
' "${code_root}/scripts" "${runtime_root}" "${incompatible_paths}"

  export PROBE_CONTROLS_INCOMPATIBLE_PATHS="${incompatible_paths}"
  export PROBE_CONTROLS_RUNTIME_ROOT="${runtime_root}"
  export PYTHONPATH="${runtime_root}:${code_root}:${code_root}/scripts${PYTHONPATH:+:${PYTHONPATH}}"
}
