#!/bin/bash

_nf_class_filter_pythonpath() {
  local value=${1:-}
  local filtered=""
  local entry

  IFS=':' read -r -a _nf_class_path_entries <<< "${value}"
  for entry in "${_nf_class_path_entries[@]}"; do
    if [[ -z "${entry}" ]]; then
      continue
    fi
    case "${entry}" in
      *site-packages*|*dist-packages*)
        continue
        ;;
    esac
    filtered="${filtered:+${filtered}:}${entry}"
  done
  printf '%s\n' "${filtered}"
}

_nf_class_enable_system_site_packages() {
  local class_python=${1:-python}
  local pyvenv_cfg
  pyvenv_cfg=$("${class_python}" - <<'PY'
import sys
from pathlib import Path

prefix = Path(sys.prefix)
cfg = prefix / "pyvenv.cfg"
print(cfg if cfg.exists() else "")
PY
)
  if [[ -z "${pyvenv_cfg}" ]]; then
    return 0
  fi

  "${class_python}" - "${pyvenv_cfg}" <<'PY'
from pathlib import Path
import sys

path = Path(sys.argv[1])
text = path.read_text()
needle = "include-system-site-packages = false"
if needle in text:
    text = text.replace(needle, "include-system-site-packages = true")
elif "include-system-site-packages = true" not in text:
    text = text.rstrip() + "\ninclude-system-site-packages = true\n"
path.write_text(text)
PY
}

set_nf_class_conditional_pythonpath() {
  local class_python=${1:-python}
  local base_venv_path=${2:-/home/jiamingp/venvs/cosmodiff_nf}

  if [[ ! -x "${base_venv_path}/bin/python" ]]; then
    echo "Missing base package fallback env: ${base_venv_path}" >&2
    return 1
  fi

  PYTHONPATH=$(_nf_class_filter_pythonpath "${PYTHONPATH:-}")
  export PYTHONPATH

  _nf_class_enable_system_site_packages "${class_python}"

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

if not seen:
    raise SystemExit("Could not discover base Python package paths")
print(os.pathsep.join(seen))
PY
)

  local pth_dir="${class_sites%%:*}"
  mkdir -p "${pth_dir}"
  printf '%s' "${base_sites}" | tr ':' '\n' > "${pth_dir}/00-cosmodiff-base-venv.pth"

  # Do not add site-packages to PYTHONPATH: that can shadow Python's standard
  # library on Great Lakes (for example, a stale pathlib backport).
  export NF_CLASS_CONDITIONAL_CLASS_SITES="${class_sites}"
  export NF_CLASS_CONDITIONAL_BASE_SITES="${base_sites}"
}
