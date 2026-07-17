#!/bin/bash
#SBATCH --job-name=scmug_ksweep
#SBATCH --output=logs/ksweep_%A_%a.out
#SBATCH --error=logs/ksweep_%A_%a.err
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --gpus=L40S:1
#SBATCH --partition=gpuai
#SBATCH --array=0-79%2
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=julien.sottiaux@univ-lille.fr

set -euo pipefail

export PROJECT_DIR="${PROJECT_DIR:-/home/julien.sottiaux/scMUG}"

# Flattened mapping, validated by read_configs.py:
# task 0..15 = Baron k=5..20, 16..31 = Darmanis, then Li, Manno, Muraro.
export K_SWEEP=true
export K_MIN=5
export K_MAX=20
export EXPERIMENT_ID="${EXPERIMENT_ID:-ksweep}"
export EXPERIMENT_PHASE=k_sweep
export ALPHA_BETA_MODE=canonical

# The k sweep only needs the two full pipelines and their saved representations.
export RUN_ABLATIONS=false
export RUN_ALL_ALPHA_BETA_SUMMARY=false
export RUN_STATISTICS=false
export DMKCN_ADAPTER_PILOT=false
export DMKCN_EVALUATE_EXITS=false
export DMKCN_EVALUATE_VARIANTS=false
export CD_VALIDATION_ALPHA_BETA_GRID=""

exec bash "${PROJECT_DIR}/run_xp.slurm"
