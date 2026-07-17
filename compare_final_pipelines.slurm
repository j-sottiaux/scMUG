#!/bin/bash
#SBATCH --job-name=scmug_compare
#SBATCH --output=logs/final_comparison_%j.out
#SBATCH --error=logs/final_comparison_%j.err
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=julien.sottiaux@univ-lille.fr

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/home/julien.sottiaux/scMUG}"
SCMUG_PYTHON_MODULE="${SCMUG_PYTHON_MODULE:-pytorch/2.0.1/gpu}"

module purge
module load "${SCMUG_PYTHON_MODULE}"
cd "${PROJECT_DIR}"
source "${PROJECT_DIR}/slurm_helpers.sh"
require_cpu_partition
validate_cpu_python_environment

python3 compare_final_pipelines.py \
    --config configs/data.yaml \
    --best-k publication/k_sweep/ksweep_best_k_by_nmi.csv \
    --k-scores publication/k_sweep/ksweep_full_pipeline_scores.csv \
    --k-seed-scores publication/k_sweep/ksweep_seed_scores.csv \
    --alpha-summary publication/alpha_beta/alpha_beta_summary.csv \
    --alpha-seed-scores publication/alpha_beta/alpha_beta_seed_means.csv \
    --best-alpha-beta publication/alpha_beta/alpha_beta_best_by_nmi.csv \
    --max-canonical-nmi-delta 0.01 \
    --outdir publication/final_comparison
