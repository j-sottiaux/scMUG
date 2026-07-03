#!/bin/bash
#SBATCH --job-name=scmug_dmkcn_ablation_c_d_test1
#SBATCH --output=result_%j.out
#SBATCH --error=error_%j.err
#SBATCH --time=06:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --gpus=L40S:1
#SBATCH --partition=gpuai
#SBATCH --mail-type=ALL
#SBATCH --mail-user=julien.sottiaux@univ-lille.fr

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

echo "=== Run scMUG ==="
cd "${PROJECT_DIR}"

python ablation_block_c_d_compare.py \
    --dataset muraro \
    --latents-scmug ./outputs/muraro_latents_scmug.joblib \
    --latents-dmkcn ./outputs/muraro_latents_scmug_dmkcn.joblib \
    --outfile ./outputs/ablation_c_d_compare_muraro.tsv \
    --cluste-number 7
    
echo "=== Job finished at $(date) ==="
