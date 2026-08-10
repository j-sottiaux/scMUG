#!/bin/bash
# Submit the campaign-3 GPU training, CPU analysis and consolidation chain.
#
# Dry-run:
#   SCMUG_CPU_PARTITION=24c1 DRY_RUN=true \
#     bash submit_dmkcn_kfusion_campaign3.sh
#
# Submission:
#   SCMUG_CPU_PARTITION=24c1 bash submit_dmkcn_kfusion_campaign3.sh

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/home/julien.sottiaux/scMUG}"
SCMUG_CPU_PARTITION="${SCMUG_CPU_PARTITION:?Set SCMUG_CPU_PARTITION to the validated CPU partition}"
SCMUG_PYTHON_MODULE="${SCMUG_PYTHON_MODULE:-pytorch/2.0.1/gpu}"
CAMPAIGN_CONFIG="${CAMPAIGN_CONFIG:-${PROJECT_DIR}/configs/dmkcn_kfusion_campaign3.yaml}"
CAMPAIGN_INSTANCE_ID="${CAMPAIGN_INSTANCE_ID:-c3_$(date +%Y%m%d_%H%M%S)}"
DRY_RUN="${DRY_RUN:-false}"
OUTPUT_ROOT="${PROJECT_DIR}/outputs/dmkcn_kfusion_campaign3/${CAMPAIGN_INSTANCE_ID}"
REPORTING_ROOT="${PROJECT_DIR}/reporting/dmkcn_kfusion_campaign3/${CAMPAIGN_INSTANCE_ID}"
LOG_ROOT="${PROJECT_DIR}/logs/dmkcn_kfusion_campaign3/${CAMPAIGN_INSTANCE_ID}"

case "${DRY_RUN}" in
    true|false) ;;
    *)
        echo "DRY_RUN must be true or false, got ${DRY_RUN}." >&2
        exit 2
        ;;
esac
if [[ ! "${CAMPAIGN_INSTANCE_ID}" =~ ^[A-Za-z0-9][A-Za-z0-9_.-]*$ ]]; then
    echo "CAMPAIGN_INSTANCE_ID must contain only letters, digits, '.', '_' or '-'." >&2
    exit 2
fi

cd "${PROJECT_DIR}"
if command -v module >/dev/null 2>&1; then
    module purge
    module load "${SCMUG_PYTHON_MODULE}"
fi
python3 campaign3_tasks.py --campaign-config "${CAMPAIGN_CONFIG}" --list
echo
echo "Campaign instance: ${CAMPAIGN_INSTANCE_ID}"
echo "Output root: ${OUTPUT_ROOT}"
echo "Reporting root: ${REPORTING_ROOT}"
echo "CPU partition: ${SCMUG_CPU_PARTITION}"

SBATCH_EXPORT="ALL,CAMPAIGN_INSTANCE_ID=${CAMPAIGN_INSTANCE_ID},SCMUG_CPU_PARTITION=${SCMUG_CPU_PARTITION},SCMUG_PYTHON_MODULE=${SCMUG_PYTHON_MODULE},CAMPAIGN_CONFIG=${CAMPAIGN_CONFIG}"

if [ "${DRY_RUN}" = "true" ]; then
    echo
    echo "Would validate and submit:"
    echo "  sbatch --export=${SBATCH_EXPORT} run_dmkcn_kfusion_campaign3_train.slurm"
    echo "  sbatch --partition=${SCMUG_CPU_PARTITION} --dependency=afterok:<train_job> --export=${SBATCH_EXPORT} run_dmkcn_kfusion_campaign3_analyze.slurm"
    echo "  sbatch --partition=${SCMUG_CPU_PARTITION} --dependency=afterok:<analysis_job> --export=${SBATCH_EXPORT} summarize_dmkcn_kfusion_campaign3.slurm"
    exit 0
fi

for target in "${OUTPUT_ROOT}" "${REPORTING_ROOT}" "${LOG_ROOT}"; do
    if [ -e "${target}" ]; then
        echo "Refusing to reuse an existing campaign instance target: ${target}" >&2
        exit 2
    fi
done
mkdir -p "${PROJECT_DIR}/logs/dmkcn_kfusion_campaign3/slurm" "${LOG_ROOT}"

submit_job() {
    local submission
    submission="$(sbatch --parsable "$@")"
    if [ -z "${submission}" ]; then
        echo "SLURM submission returned an empty job ID: sbatch $*" >&2
        return 1
    fi
    printf '%s' "${submission%%;*}"
}

echo "Validating campaign-3 SLURM jobs..."
sbatch --test-only --export="${SBATCH_EXPORT}" \
    run_dmkcn_kfusion_campaign3_train.slurm >/dev/null
sbatch --test-only --partition="${SCMUG_CPU_PARTITION}" \
    --export="${SBATCH_EXPORT}" \
    run_dmkcn_kfusion_campaign3_analyze.slurm >/dev/null
sbatch --test-only --partition="${SCMUG_CPU_PARTITION}" \
    --export="${SBATCH_EXPORT}" \
    summarize_dmkcn_kfusion_campaign3.slurm >/dev/null

TRAIN_JOB_ID="$(
    submit_job --export="${SBATCH_EXPORT}" run_dmkcn_kfusion_campaign3_train.slurm
)"
ANALYSIS_JOB_ID="$(
    submit_job \
        --partition="${SCMUG_CPU_PARTITION}" \
        --dependency="afterok:${TRAIN_JOB_ID}" \
        --export="${SBATCH_EXPORT}" \
        run_dmkcn_kfusion_campaign3_analyze.slurm
)"
SUMMARY_JOB_ID="$(
    submit_job \
        --partition="${SCMUG_CPU_PARTITION}" \
        --dependency="afterok:${ANALYSIS_JOB_ID}" \
        --export="${SBATCH_EXPORT}" \
        summarize_dmkcn_kfusion_campaign3.slurm
)"

CHAIN_RECORD="${LOG_ROOT}/submitted_jobs.tsv"
{
    printf 'stage\tjob_id\tdependency\n'
    printf 'training\t%s\t\n' "${TRAIN_JOB_ID}"
    printf 'analysis\t%s\tafterok:%s\n' "${ANALYSIS_JOB_ID}" "${TRAIN_JOB_ID}"
    printf 'consolidation\t%s\tafterok:%s\n' "${SUMMARY_JOB_ID}" "${ANALYSIS_JOB_ID}"
} | tee "${CHAIN_RECORD}"
