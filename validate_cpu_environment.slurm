#!/bin/bash
#SBATCH --job-name=scmug_cpu_check
#SBATCH --output=logs/cpu_environment_%j.out
#SBATCH --error=logs/cpu_environment_%j.err
#SBATCH --time=00:10:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --mail-type=FAIL
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

python3 - <<'PY'
import ablation_block_c_d
import compare_final_pipelines
import summarize_alpha_beta
import summarize_computation
import summarize_k_sweep

print("scMUG CPU protocol imports: OK")
PY
