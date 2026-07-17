#!/bin/bash
#SBATCH --job-name=compute_summary
#SBATCH --output=logs/computation_summary_%j.out
#SBATCH --error=logs/computation_summary_%j.err
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
KSWEEP_EXPERIMENT_ID="${KSWEEP_EXPERIMENT_ID:-ksweep}"
OUTDIR="${COMPUTATION_SUMMARY_DIR:-${PROJECT_DIR}/publication/computation_metrics}"

module purge
module load "${SCMUG_PYTHON_MODULE}"
cd "${PROJECT_DIR}"
source "${PROJECT_DIR}/slurm_helpers.sh"
require_cpu_partition
validate_cpu_python_environment

python3 summarize_computation.py \
    --input-root "${OUTPUT_ROOT}" \
    --experiment-id "${KSWEEP_EXPERIMENT_ID}" \
    --outdir "${OUTDIR}"
