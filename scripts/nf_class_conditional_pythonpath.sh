#!/bin/bash

set_nf_class_conditional_pythonpath() {
  local class_python=${1:-python}
  local base_venv_path=${2:-/home/jiamingp/venvs/cosmodiff_nf}

  if [[ ! -x "${base_venv_path}/bin/python" ]]; then
    echo "Missing base package fallback env: ${base_venv_path}" >&2
    return 1
  fi

  local class_sites
  class_sites=$("${class_python}" - <<'PY'
import os
import sysconfig

seen = []
paths = sysconfig.get_paths()
for key in ("purelib", "platlib"):
    path = paths.get(key)
    if path and path not in seen:
        seen.append(path)
print(os.pathsep.join(seen))
PY
)

  local base_sites
  base_sites=$("${base_venv_path}/bin/python" - <<'PY'
import os
import site
import sys
import sysconfig
from pathlib import Path

seen = []

def add(path):
    if not path:
        return
    path = str(path)
    if path in seen or not Path(path).exists():
        return
    if "site-packages" not in path and "dist-packages" not in path:
        return
    seen.append(path)

paths = sysconfig.get_paths()
for key in ("purelib", "platlib"):
    add(paths.get(key))

try:
    for path in site.getsitepackages():
        add(path)
except Exception:
    pass

try:
    add(site.getusersitepackages())
except Exception:
    pass

for path in sys.path:
    add(path)

if not seen:
    raise SystemExit("Could not discover base Python package paths")
print(os.pathsep.join(seen))
PY
)

  local pth_dir="${class_sites%%:*}"
  mkdir -p "${pth_dir}"
  printf '%s' "${base_sites}" | tr ':' '\n' > "${pth_dir}/00-cosmodiff-base-venv.pth"

  export PYTHONPATH="${class_sites}:${base_sites}${PYTHONPATH:+:${PYTHONPATH}}"
  export NF_CLASS_CONDITIONAL_CLASS_SITES="${class_sites}"
  export NF_CLASS_CONDITIONAL_BASE_SITES="${base_sites}"
}
