#!/bin/bash
#SBATCH --job-name=dmkcn_c3_summary
#SBATCH --output=logs/dmkcn_kfusion_campaign3/slurm/summary_%j.out
#SBATCH --error=logs/dmkcn_kfusion_campaign3/slurm/summary_%j.err
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --mail-type=END,FAIL

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/home/julien.sottiaux/scMUG}"
CAMPAIGN_INSTANCE_ID="${CAMPAIGN_INSTANCE_ID:?Set CAMPAIGN_INSTANCE_ID}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${PROJECT_DIR}/outputs/dmkcn_kfusion_campaign3/${CAMPAIGN_INSTANCE_ID}}"
REPORTING_ROOT="${REPORTING_ROOT:-${PROJECT_DIR}/reporting/dmkcn_kfusion_campaign3/${CAMPAIGN_INSTANCE_ID}}"
SCMUG_PYTHON_MODULE="${SCMUG_PYTHON_MODULE:-pytorch/2.0.1/gpu}"
SCMUG_CPU_PARTITION="${SCMUG_CPU_PARTITION:?Set SCMUG_CPU_PARTITION}"
RESOURCE_USAGE="${OUTPUT_ROOT}/consolidation/computation/resource_usage.txt"
if [[ ! "${CAMPAIGN_INSTANCE_ID}" =~ ^[A-Za-z0-9][A-Za-z0-9_.-]*$ ]]; then
    echo "Invalid CAMPAIGN_INSTANCE_ID: ${CAMPAIGN_INSTANCE_ID}" >&2
    exit 2
fi

cd "${PROJECT_DIR}"
source "${PROJECT_DIR}/slurm_helpers.sh"
module purge
module load "${SCMUG_PYTHON_MODULE}"
require_cpu_partition
validate_cpu_python_environment

if [ -e "${REPORTING_ROOT}" ]; then
    echo "Refusing to overwrite existing campaign-3 reporting root: ${REPORTING_ROOT}" >&2
    exit 2
fi

COMMAND=(
    python3 summarize_dmkcn_kfusion_campaign3.py
    --input-root "${OUTPUT_ROOT}"
    --campaign-instance-id "${CAMPAIGN_INSTANCE_ID}"
    --outdir "${REPORTING_ROOT}"
)
printf 'Command:'
printf ' %q' "${COMMAND[@]}"
printf '\n'
python3 measure_command_resources.py --output "${RESOURCE_USAGE}" -- "${COMMAND[@]}"

test -s "${REPORTING_ROOT}/campaign_summary.json"
test -s "${REPORTING_ROOT}/statistical_tests.csv"
test -s "${REPORTING_ROOT}/condition_summary.csv"
test -s "${RESOURCE_USAGE}"
