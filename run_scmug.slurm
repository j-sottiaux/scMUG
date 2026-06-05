#!/bin/bash

#SBATCH --job-name=26_06_05_SCMUG_TEST
#SBATCH --output=result_%j.out
#SBATCH --error=error_%j.er
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:A100-HGX-80G:1
#SBATCH --partition=gpu64
#SBATCH --qos=gpu-default
#SBATCH --mem=64G
#SBATCH --mail-type=ALL
#SBATCH --mail-user=julien.sottiaux@univ-lille.fr

module load pytorch/2.10 
pip install -r requirements.txt


cd ./gcorr 
nvcc -o gcorr gcorr.cu --shared
export LD_LIBRARY_PATH=/home/julien.sottiaux/scMUG/gcorr:$LD_LIBRARY_PATH
export PYTHONPATH=/home/julien.sottiaux/scMUG/gcorr:$PYTHONPATH

python3 scMUG.py --dataset muraro --cluster_number 7 --n_gfm 5 --cutoffs 0.14,0.14,0.15,0.14,0.14 --seeds 1111,2222,3333,4444,5555,6666,7777,8888,9999,10000