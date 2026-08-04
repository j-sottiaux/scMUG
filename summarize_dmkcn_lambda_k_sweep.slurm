#!/bin/bash
#SBATCH --job-name=dmkcn_c2_ksel
#SBATCH --output=logs/dmkcn_c2_ksel_%j.out
#SBATCH --error=logs/dmkcn_c2_ksel_%j.err
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --mail-type=END,FAIL

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/home/julien.sottiaux/scMUG}"
SCMUG_PYTHON_MODULE="${SCMUG_PYTHON_MODULE:-pytorch/2.0.1/gpu}"
SCMUG_CPU_PARTITION="${SCMUG_CPU_PARTITION:-}"
CONFIG_FILE="${CONFIG_FILE:-${PROJECT_DIR}/configs/data.yaml}"
LAMBDA_CONFIG="${LAMBDA_CONFIG:-${PROJECT_DIR}/configs/dmkcn_lambdas_triplets}"
INPUT_ROOT="${INPUT_ROOT:-${PROJECT_DIR}/outputs/dmkcn_lambda_campaign2/k_sweep}"
OUTDIR="${OUTDIR:-${PROJECT_DIR}/reporting/dmkcn_lambda_campaign2/k_sweep}"
DRY_RUN="${DRY_RUN:-false}"

COMMAND=(
    python3 summarize_dmkcn_lambda_campaign2.py k-sweep
    --input-root "${INPUT_ROOT}"
    --config "${CONFIG_FILE}"
    --lambda-config "${LAMBDA_CONFIG}"
    --k-min 5
    --k-max 20
    --outdir "${OUTDIR}"
)
printf 'Command:'
printf ' %q' "${COMMAND[@]}"
printf '\n'
if [ "${DRY_RUN}" = "true" ]; then
    exit 0
fi

module purge
module load "${SCMUG_PYTHON_MODULE}"
cd "${PROJECT_DIR}"
source "${PROJECT_DIR}/slurm_helpers.sh"
require_cpu_partition
validate_cpu_python_environment
"${COMMAND[@]}"
