#!/bin/bash
#SBATCH --job-name=ab_grid_summary
#SBATCH --output=logs/alpha_beta_summary_%j.out
#SBATCH --error=logs/alpha_beta_summary_%j.err
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
OUTPUT_ROOT="${OUTPUT_ROOT:-${PROJECT_DIR}/outputs}"
BEST_K_FILE="${BEST_K_FILE:-${PROJECT_DIR}/publication/k_sweep/ksweep_best_k_by_nmi.csv}"
ALPHA_BETA_EXPERIMENT_ID="${ALPHA_BETA_EXPERIMENT_ID:-alpha_beta_grid}"
OUTDIR="${ALPHA_BETA_SUMMARY_DIR:-${PROJECT_DIR}/publication/alpha_beta}"

module purge
module load "${SCMUG_PYTHON_MODULE}"
cd "${PROJECT_DIR}"
source "${PROJECT_DIR}/slurm_helpers.sh"
require_cpu_partition
validate_cpu_python_environment

python3 summarize_alpha_beta.py \
    --input-root "${OUTPUT_ROOT}" \
    --best-k "${BEST_K_FILE}" \
    --experiment-id "${ALPHA_BETA_EXPERIMENT_ID}" \
    --outdir "${OUTDIR}"
