#!/bin/bash
#SBATCH --job-name=scmug_test
#SBATCH --output=result_%j.out
#SBATCH --error=error_%j.err
#SBATCH --time=2:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gpus=L40S:1
#SBATCH --partition=gpuai
#SBATCH --mail-type=ALL
#SBATCH --mail-user=julien.sottiaux@univ-lille.fr

set -euo pipefail

echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "Working directory at start: $(pwd)"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-not_set}"

module purge
module load pytorch/2.0.1/gpu

cd /home/julien.sottiaux/scMUG

echo "Python:"
which python3
python3 --version

echo "Pip:"
which pip
pip --version

echo "CUDA check:"
nvidia-smi || true

echo "Installing requirements"
python3 -m pip install git+https://github.com/ChangSuBiostats/CS-CORE_python.git
python3 -m pip install --upgrade pip setuptools wheel
python3 -m pip install -r requirements.txt --user

echo "Compiling gcorr"
cd /home/julien.sottiaux/scMUG/gcorr
nvcc -o gcorr gcorr.cu --shared

export LD_LIBRARY_PATH=/home/julien.sottiaux/scMUG/gcorr:$LD_LIBRARY_PATH
export PYTHONPATH=/home/julien.sottiaux/scMUG/gcorr:$PYTHONPATH

echo "Running scMUG"
cd /home/julien.sottiaux/scMUG

python3 scMUG.py \
  --dataset muraro \
  --cluster_number 7 \
  --n_gfm 5 \
  --cutoffs 0.14,0.14,0.15,0.14,0.14 \
  --seeds 1111,2222,3333,4444,5555,6666,7777,8888,9999,10000
