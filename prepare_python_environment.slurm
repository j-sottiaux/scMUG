#!/bin/bash
#SBATCH --job-name=scmug_env_setup
#SBATCH --output=logs/python_environment_%j.out
#SBATCH --error=logs/python_environment_%j.err
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=julien.sottiaux@univ-lille.fr

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/home/julien.sottiaux/scMUG}"
SCMUG_PYTHON_MODULE="${SCMUG_PYTHON_MODULE:-pytorch/2.0.1/gpu}"
PIP_SETUP_LOCK="${PROJECT_DIR}/.slurm_pip_setup.lock"
PIP_SETUP_STAMP="${PROJECT_DIR}/.slurm_pip_setup.ok"

module purge
module load "${SCMUG_PYTHON_MODULE}"
cd "${PROJECT_DIR}"
source "${PROJECT_DIR}/slurm_helpers.sh"
require_cpu_partition

(
    flock -x 201
    if [ ! -f "${PIP_SETUP_STAMP}" ] \
        || [ "${PROJECT_DIR}/requirements.txt" -nt "${PIP_SETUP_STAMP}" ] \
        || [ "${PROJECT_DIR}/prepare_python_environment.slurm" -nt "${PIP_SETUP_STAMP}" ]; then
        python3 -m pip install --user --upgrade pip setuptools wheel
        python3 -m pip install --user --force-reinstall \
            "numpy==1.26.4" \
            "scipy==1.11.4" \
            "zarr<3"
        python3 -m pip install --user -r requirements.txt
        python3 - <<'PY'
import anndata
import h5py
import joblib
import numpy
import pandas
import scanpy
import scipy
import sklearn
import torch
import umap
import yaml

print("Installed Python dependency imports: OK")
PY
        date >"${PIP_SETUP_STAMP}"
    else
        echo "Python dependency stamp is current: ${PIP_SETUP_STAMP}"
    fi
) 201>"${PIP_SETUP_LOCK}"
