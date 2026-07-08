#!/bin/bash
#SBATCH --job-name=scmug
#SBATCH --output=logs/result_%A_%a.out
#SBATCH --error=logs/error_%A_%a.err
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --gpus=L40S:1
#SBATCH --partition=gpuai
#SBATCH --mail-type=ALL
#SBATCH --mail-user=julien.sottiaux@univ-lille.fr
#SBATCH --array=1-6%2

set -euo pipefail

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-16}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-16}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK:-16}
export NUMEXPR_NUM_THREADS=${SLURM_CPUS_PER_TASK:-16}

PROJECT_DIR="/home/julien.sottiaux/scMUG"
GCORR_DIR="${PROJECT_DIR}/gcorr"
CONFIG_FILE="${PROJECT_DIR}/configs/data.yaml"

echo "=== Job started on $(hostname) at $(date) ==="
echo "=== SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID} ==="

module purge
module load pytorch/2.0.1/gpu

cd "${PROJECT_DIR}"

echo "=== Load dataset configuration ==="
eval "$(
python3 "${PROJECT_DIR}/read_configs.py" \
  --config "${CONFIG_FILE}" \
  --index "${SLURM_ARRAY_TASK_ID}"
)"

echo "Dataset: ${DATASET}"
echo "Cluster number: ${CLUSTER_NUMBER}"
echo "N_GFM: ${N_GFM}"
echo "Cutoffs: ${CUTOFFS}"
echo "Full alpha: ${FULL_ALPHA}"
echo "Full beta: ${FULL_BETA}"
echo "N_NEIGHBOUR: ${N_NEIGHBOUR}"
echo "RED_GLOBAL: ${RED_GLOBAL}"
echo "RED_LOCAL: ${RED_LOCAL}"

SEEDS="1111,2222,3333,4444,5555,6666,7777,8888,9999,10000"
REPEAT=3
KMEANS_TIMES=20
EPOCH=100
echo "EPOCH: ${EPOCH}"

DATASET_OUTDIR="./outputs/${DATASET}"
mkdir -p "${DATASET_OUTDIR}"

LATENTS_AE="${DATASET_OUTDIR}/latents_ae.joblib"
LATENTS_DMKCN="${DATASET_OUTDIR}/latents_dmkcn.joblib"
OUTFILE="${DATASET_OUTDIR}/ablation_tests.tsv"
SUMMARY_OUTFILE="${DATASET_OUTDIR}/comparison_summary.csv"
SIGNIFICANCE_OUTFILE="${DATASET_OUTDIR}/significance.csv"

echo "=== Python environment ==="
which python3
python3 --version
python3 -m pip --version

echo "=== CUDA check ==="
nvidia-smi || true
which nvcc || true
nvcc --version || true

echo "=== Install/fix Python dependencies if needed ==="
PIP_SETUP_LOCK="${PROJECT_DIR}/.slurm_pip_setup.lock"
PIP_SETUP_STAMP="${PROJECT_DIR}/.slurm_pip_setup.ok"

(
    flock -x 201

    if [ ! -f "${PIP_SETUP_STAMP}" ] || [ "${PROJECT_DIR}/requirements.txt" -nt "${PIP_SETUP_STAMP}" ] || [ "${PROJECT_DIR}/run_experiment.slurm" -nt "${PIP_SETUP_STAMP}" ]; then
        echo "Task ${SLURM_ARRAY_TASK_ID}: installing Python dependencies"

        python3 -m pip install --user --upgrade pip setuptools wheel

        python3 -m pip install --user --force-reinstall \
          "numpy==1.26.4" \
          "scipy==1.11.4" \
          "zarr<3"

        python3 -m pip install --user -r requirements.txt

        date > "${PIP_SETUP_STAMP}"
    else
        echo "Task ${SLURM_ARRAY_TASK_ID}: Python dependency setup already done"
    fi
) 201>"${PIP_SETUP_LOCK}"

echo "=== Dependency check ==="
python3 -m pip check || true
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
import yaml

print("dependency import check ok")
print("numpy", numpy.__version__)
print("scipy", scipy.__version__)
print("scanpy", scanpy.__version__)
print("torch", torch.__version__)
PY

echo "=== Compile gcorr if needed ==="
GCORR_LOCK="${GCORR_DIR}/.gcorr_build.lock"

(
    flock -x 200

    cd "${GCORR_DIR}"

    if [ ! -f "libgcorr.so" ] || [ "gcorr.cu" -nt "libgcorr.so" ]; then
        echo "Task ${SLURM_ARRAY_TASK_ID}: building libgcorr.so"
        rm -f gcorr libgcorr.so gcorr.so
        nvcc -shared -Xcompiler -fPIC -arch=sm_89 -o libgcorr.so gcorr.cu
        ln -sf libgcorr.so gcorr
    else
        echo "Task ${SLURM_ARRAY_TASK_ID}: libgcorr.so up to date"
        ln -sf libgcorr.so gcorr
    fi
) 200>"${GCORR_LOCK}"

export LD_LIBRARY_PATH="${GCORR_DIR}:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="${PROJECT_DIR}:${PYTHONPATH:-}"
cd "${PROJECT_DIR}"

echo "=== gcorr dynamic library check ==="
ldd "${GCORR_DIR}/libgcorr.so" || true

echo "=== Test gcorr loading ==="
python3 - <<'PY'
from ctypes import cdll
cdll.LoadLibrary("gcorr")
print("gcorr loaded successfully")
PY

echo "=== Run original scMUG (B = autoencoder) ==="

python3 scMUG.py \
    --dataset "${DATASET}" \
    --cluster_number "${CLUSTER_NUMBER}" \
    --n_gfm "${N_GFM}" \
    --cutoffs "${CUTOFFS}" \
    --seeds "${SEEDS}" \
    --repeat "${REPEAT}" \
    --epoch "${EPOCH}" \
    --kmeans_times "${KMEANS_TIMES}" \
    --n_neighbour "${N_NEIGHBOUR}" \
    --thread-num "${SLURM_CPUS_PER_TASK}" \
    --red_global "${RED_GLOBAL}" \
    --red_local "${RED_LOCAL}" \
    --block-b autoencoder \
    --output-tag ae \
    --output-dir "${DATASET_OUTDIR}"


echo "=== Job finished for ${DATASET} at $(date) ==="
