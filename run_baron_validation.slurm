#!/bin/bash
#SBATCH --job-name=baron_validation
#SBATCH --output=logs/baron_validation_%j.out
#SBATCH --error=logs/baron_validation_%j.err
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --gpus=L40S:1
#SBATCH --partition=gpuai
#SBATCH --mail-type=ALL
#SBATCH --mail-user=julien.sottiaux@univ-lille.fr
#SBATCH --array=1

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/home/julien.sottiaux/scMUG}"
export PROJECT_DIR

# Keep this validation identifiable without creating a second output root.
export OUTPUT_ROOT="${OUTPUT_ROOT:-./outputs}"
export EXPERIMENT_ID="${EXPERIMENT_ID:-baron_validation}"

# Frozen configuration selected on Darmanis and Li, plus the canonical Baron
# C weight used as a matched downstream control.
export CD_VALIDATION_ALPHA_BETA_GRID="0.001:1,0.1:1"
export DMKCN_PROJECTIONS="spectral_dense"
export DMKCN_EMBEDDING_DIMS="32"
export DMKCN_PRIMARY_PROJECTION="spectral_dense"
export DMKCN_PRIMARY_D="32"

export SEEDS="1111,2222,3333,4444,5555,6666,7777,8888,9999,10000"
export REPEAT="3"
export KMEANS_TIMES="20"
export EPOCH="50"
export DMKCN_ADAPTER_PILOT="false"
export DMKCN_EVALUATE_EXITS="false"
export DMKCN_EVALUATE_VARIANTS="false"

mkdir -p "${PROJECT_DIR}/logs"
exec bash "${PROJECT_DIR}/run_xp.slurm"
