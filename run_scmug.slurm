#!/bin/bash

#SBATCH --job-name=26_06_05_SCMUG_TEST
#SBATCH --output=result_%j.out
#SBATCH --error=error_%j.er
#SBATCH --time=00:45:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=14
#SBATCH --gpus=1
#SBATCH --mem=64G
#SBATCH --mail-type=ALL
#SBATCH --mail-user=julien.sottiaux@univ-lille.fr

module load pytorch/2.10 
   
# install all required packages
pip install -r requirements.txt

# Compile gcorr
cd ./gcorr 
nvcc -o gcorr gcorr.cu --shared

# Add ./gcorr to system LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/home/julien.sottiaux/scMUG/gcorr:$LD_LIBRARY_PATH

# Add ./gcorr to PYTHONPATH
export PYTHONPATH=/home/julien.sottiaux/scMUG/gcorr:$PYTHONPATH

# run scMUG
python3 scMUG.py --dataset muraro --cluster_number 7 --n_gfm 5 --cutoffs 0.14,0.14,0.15,0.14,0.14 --seeds 1111,2222,3333,4444,5555,6666,7777,8888,9999,10000