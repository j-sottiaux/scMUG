#!/bin/bash
#SBATCH --job-name=xp_all
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
#SBATCH --array=1-9%2

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

SEEDS="1111,2222,3333,4444,5555"
REPEAT=3
KMEANS_TIMES=20

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

echo "=== Install/fix Python dependencies ==="
python3 -m pip install --user --upgrade pip setuptools wheel

python3 -m pip install --user --force-reinstall \
  "numpy==1.26.4" \
  "scipy==1.11.4" \
  "zarr<3"

python3 -m pip install --user -r requirements.txt

echo "=== Dependency check ==="
python3 -m pip check || true

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

echo "=== Run scMUG autoencoder block B ==="

python3 scMUG.py \
    --dataset "${DATASET}" \
    --cluster_number "${CLUSTER_NUMBER}" \
    --n_gfm "${N_GFM}" \
    --cutoffs "${CUTOFFS}" \
    --seeds "${SEEDS}" \
    --repeat "${REPEAT}" \
    --kmeans_times "${KMEANS_TIMES}" \
    --n_neighbour "${N_NEIGHBOUR}" \
    --thread-num "${SLURM_CPUS_PER_TASK}" \
    --red_global "${RED_GLOBAL}" \
    --red_local "${RED_LOCAL}" \
    --block-b autoencoder \
    --output-tag ae \
    --output-dir "${DATASET_OUTDIR}"

echo "=== Run scMUG DMKCN block B ==="

python3 scMUG.py \
    --dataset "${DATASET}" \
    --cluster_number "${CLUSTER_NUMBER}" \
    --n_gfm "${N_GFM}" \
    --cutoffs "${CUTOFFS}" \
    --seeds "${SEEDS}" \
    --repeat "${REPEAT}" \
    --kmeans_times "${KMEANS_TIMES}" \
    --n_neighbour "${N_NEIGHBOUR}" \
    --thread-num "${SLURM_CPUS_PER_TASK}" \
    --red_global "${RED_GLOBAL}" \
    --red_local "${RED_LOCAL}" \
    --block-b dmkcn \
    --output-tag dmkcn \
    --dmkcn-zinb-on-counts \
    --output-dir "${DATASET_OUTDIR}"

echo "=== Run C/D ablation from saved latents ==="

python3 ablation_block_c_d.py \
    --dataset "${DATASET}" \
    --cluster_number "${CLUSTER_NUMBER}" \
    --seeds "${SEEDS}" \
    --repeat "${REPEAT}" \
    --kmeans_times "${KMEANS_TIMES}" \
    --n_neighbour "${N_NEIGHBOUR}" \
    --latents-autoencoder "${LATENTS_AE}" \
    --latents-dmkcn "${LATENTS_DMKCN}" \
    --outfile "${OUTFILE}" \
    --thread-num "${SLURM_CPUS_PER_TASK}" \
    --red_global "${RED_GLOBAL}" \
    --red_local "${RED_LOCAL}"

echo "=== Generate executive summary ==="

python3 summarize_xp.py \
  --dataset "${DATASET}" \
  --full-autoencoder "${DATASET_OUTDIR}/${DATASET}s_ae.txt" \
  --full-dmkcn "${DATASET_OUTDIR}/${DATASET}s_dmkcn.txt" \
  --ablation "${OUTFILE}" \
  --full-alpha "${FULL_ALPHA}" \
  --full-beta "${FULL_BETA}" \
  --outfile "${SUMMARY_OUTFILE}" \
  --significance-outfile "${SIGNIFICANCE_OUTFILE}"

echo "=== Job finished for ${DATASET} at $(date) ==="