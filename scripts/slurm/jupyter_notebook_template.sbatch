#!/bin/bash
#SBATCH --job-name=jupyter_notebook
#SBATCH --partition=<partition>
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --mem=32gb
#SBATCH --time=03:00:00
#SBATCH -o /path/to/project/notebook.out
#SBATCH -e /path/to/project/notebook.err
#SBATCH -A huterer2
#SBATCH --mail-user=jiamingp@umich.edu

set -euo pipefail

# Copy this file into local/ and edit these paths/settings there.
# local/ is ignored by git, so cluster-specific paths and accounts stay private.
PROJECT_DIR=${PROJECT_DIR:-/path/to/diffusion-models-simulation-data}
COSMODIFF_DIR=${COSMODIFF_DIR:-/path/to/cosmo_diffusion}
NOTEBOOK_DIR=${NOTEBOOK_DIR:-${PROJECT_DIR}/notebooks}
INFO_FILE=${INFO_FILE:-${PROJECT_DIR}/jupyter_info_gpu.txt}
JUPYTER_BIN=${JUPYTER_BIN:-jupyter}

cd "${PROJECT_DIR}"
unset XDG_RUNTIME_DIR

export PYTHONPATH="${COSMODIFF_DIR}:${PROJECT_DIR}:${PYTHONPATH:-}"

nport=$(shuf -i 8000-9000 -n1)
ip=$(hostname -i)
hn=$(hostname)
me=$(whoami)

cat > "${INFO_FILE}" <<EOF
Jupyter notebook running on host ${hn}.

Copy this command into your local terminal:
ssh -N -L ${nport}:${ip}:${nport} ${me}@<cluster-login-host>

Then open:
http://localhost:${nport}

Notebook directory:
${NOTEBOOK_DIR}
EOF

echo "Wrote connection info to ${INFO_FILE}"
cat "${INFO_FILE}"

cd "${NOTEBOOK_DIR}"

"${JUPYTER_BIN}" notebook \
  --no-browser \
  --port="${nport}" \
  --ip=0.0.0.0
