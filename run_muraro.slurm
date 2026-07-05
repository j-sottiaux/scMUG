#!/bin/bash
#SBATCH --job-name=muraro
#SBATCH --output=result_%j.out
#SBATCH --error=error_%j.err
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=64G
#SBATCH --gpus=L40S:1
#SBATCH --partition=gpuai
#SBATCH --mail-type=ALL
#SBATCH --mail-user=julien.sottiaux@univ-lille.fr

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-16}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-16}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK:-16}
export NUMEXPR_NUM_THREADS=${SLURM_CPUS_PER_TASK:-16}

set -euo pipefail

echo "=== Job started on $(hostname) at $(date) ==="

module purge
module load pytorch/2.0.1/gpu

PROJECT_DIR="/home/julien.sottiaux/scMUG"
GCORR_DIR="${PROJECT_DIR}/gcorr"

cd "${PROJECT_DIR}"

echo "=== Python environment ==="
which python3
python3 --version
python3 -m pip --version

echo "=== CUDA check ==="
nvidia-smi || true
which nvcc || true
nvcc --version || true

echo "=== Prepare folders ==="
mkdir -p "${PROJECT_DIR}/outputs"

echo "=== Install/fix Python dependencies ==="
python3 -m pip install --user --upgrade pip setuptools wheel

python3 -m pip install --user --force-reinstall \
  "numpy==1.26.4" \
  "scipy==1.11.4" \
  "zarr<3"

python3 -m pip install --user -r requirements.txt

echo "=== Dependency check ==="
python3 -m pip check || true

echo "=== Version check ==="
python3 - <<'PY'
import sys
print("python", sys.executable)

for pkg in ["numpy", "scipy", "zarr", "anndata", "torch"]:
    try:
        mod = __import__(pkg)
        print(pkg, getattr(mod, "__version__", "unknown"))
    except Exception as e:
        print(pkg, "IMPORT_FAILED:", repr(e))
PY

echo "=== Compile gcorr ==="
cd "${GCORR_DIR}"

rm -f gcorr libgcorr.so gcorr.so

nvcc -shared -Xcompiler -fPIC -o libgcorr.so gcorr.cu

# scMUG currently calls cdll.LoadLibrary("gcorr"),
# so provide a filename matching that exact lookup.
ln -sf libgcorr.so gcorr

echo "=== gcorr files ==="
ls -lh "${GCORR_DIR}"

echo "=== gcorr dynamic library check ==="
ldd "${GCORR_DIR}/libgcorr.so" || true

export LD_LIBRARY_PATH="${GCORR_DIR}:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="${PROJECT_DIR}:${PYTHONPATH:-}"

echo "=== Test gcorr loading ==="
cd "${PROJECT_DIR}"
python3 - <<'PY'
from ctypes import cdll
cdll.LoadLibrary("gcorr")
print("gcorr loaded successfully")
PY

DATASET="muraro"
CLUSTER_NUMBER=7
N_GFM=5
CUTOFFS="0.14,0.14,0.15,0.14,0.14"
SEEDS="1111,2222,3333,4444,5555,6666,7777,8888,9999,10000"

REPEAT=3
KMEANS_TIMES=20
N_NEIGHBOUR=3

LATENTS_AE="./outputs/${DATASET}_latents_autoencoder.joblib"
LATENTS_DMKCN="./outputs/${DATASET}_latents_dmkcn.joblib"
OUTFILE="./outputs/ablation_tests_${DATASET}.tsv"
SUMMARY_OUTFILE="./outputs/summary_comparison_${DATASET}.csv"

echo "=== Run scMUG autoencoder block B ==="
cd "${PROJECT_DIR}"

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
    --block-b autoencoder \
    --output-tag autoencoder

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
    --block-b dmkcn \
    --dmkcn-zinb-on-counts \
    --output-tag dmkcn
    
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
    --red_global umap \
    --red_local umap \

echo "=== Generate executive summary ==="
python3 summarize_xp.py \
  --dataset "${DATASET}" \
  --full-autoencoder "./outputs/${DATASET}s_autoencoder.txt" \
  --full-dmkcn "./outputs/${DATASET}s_dmkcn.txt" \
  --ablation "${OUTFILE}" \
  --full-alpha 1 \
  --full-beta 1 \
  --outfile "${SUMMARY_OUTFILE}" \
  --significance-outfile "./outputs/significance_${DATASET}.csv"

echo "=== Job finished at $(date) ==="
