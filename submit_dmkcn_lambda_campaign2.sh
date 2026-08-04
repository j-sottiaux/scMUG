#!/bin/bash
# Submit the Baron/Darmanis/Li/Manno campaign-2 dependency chain.
#
# Dry-run:
#   DRY_RUN=true SCMUG_CPU_PARTITION=<cpu_partition> \
#     bash submit_dmkcn_lambda_campaign2.sh
#
# Submission:
#   SCMUG_CPU_PARTITION=<cpu_partition> bash submit_dmkcn_lambda_campaign2.sh

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/home/julien.sottiaux/scMUG}"
SCMUG_CPU_PARTITION="${SCMUG_CPU_PARTITION:?Set SCMUG_CPU_PARTITION to the validated Zeus CPU partition}"
SCMUG_PYTHON_MODULE="${SCMUG_PYTHON_MODULE:-pytorch/2.0.1/gpu}"
DRY_RUN="${DRY_RUN:-false}"
export PROJECT_DIR SCMUG_CPU_PARTITION SCMUG_PYTHON_MODULE

cd "${PROJECT_DIR}"
if [ "${DRY_RUN}" = "true" ]; then
    python3 campaign2_tasks.py k-sweep \
        --config configs/data.yaml \
        --lambda-config configs/dmkcn_lambdas_triplets \
        --list
    echo
    echo "Would submit:"
    echo "  sbatch run_dmkcn_lambda_k_sweep.slurm"
    echo "  sbatch --partition=${SCMUG_CPU_PARTITION} --dependency=afterok:<k_job> summarize_dmkcn_lambda_k_sweep.slurm"
    echo "  sbatch --partition=${SCMUG_CPU_PARTITION} --dependency=afterok:<k_summary_job> run_dmkcn_lambda_alpha_beta.slurm"
    echo "  sbatch --partition=${SCMUG_CPU_PARTITION} --dependency=afterok:<alpha_job> summarize_dmkcn_lambda_alpha_beta.slurm"
    exit 0
fi

for existing_root in \
    "${PROJECT_DIR}/outputs/dmkcn_lambda_campaign2" \
    "${PROJECT_DIR}/reporting/dmkcn_lambda_campaign2"; do
    if [ -e "${existing_root}" ]; then
        echo "Refusing to mix a new complete campaign with existing results: ${existing_root}" >&2
        echo "Archive that root outside the campaign path before resubmitting." >&2
        exit 2
    fi
done

mkdir -p logs

submit_job() {
    local submission
    submission="$(sbatch --parsable "$@")"
    if [ -z "${submission}" ]; then
        echo "SLURM submission returned an empty job id: sbatch $*" >&2
        return 1
    fi
    printf '%s' "${submission%%;*}"
}

validate_job() {
    local description="$1"
    shift
    echo "Validating ${description}..."
    sbatch --test-only "$@" >/dev/null
}

validate_job "DMKCN lambda K-sweep" run_dmkcn_lambda_k_sweep.slurm
validate_job "K selection" \
    --partition="${SCMUG_CPU_PARTITION}" summarize_dmkcn_lambda_k_sweep.slurm
validate_job "alpha/beta grid" \
    --partition="${SCMUG_CPU_PARTITION}" run_dmkcn_lambda_alpha_beta.slurm
validate_job "final consolidation" \
    --partition="${SCMUG_CPU_PARTITION}" summarize_dmkcn_lambda_alpha_beta.slurm

KSWEEP_JOB_ID="$(submit_job run_dmkcn_lambda_k_sweep.slurm)"
K_SUMMARY_JOB_ID="$(
    submit_job \
        --partition="${SCMUG_CPU_PARTITION}" \
        --dependency="afterok:${KSWEEP_JOB_ID}" \
        summarize_dmkcn_lambda_k_sweep.slurm
)"
ALPHA_BETA_JOB_ID="$(
    submit_job \
        --partition="${SCMUG_CPU_PARTITION}" \
        --dependency="afterok:${K_SUMMARY_JOB_ID}" \
        run_dmkcn_lambda_alpha_beta.slurm
)"
FINAL_JOB_ID="$(
    submit_job \
        --partition="${SCMUG_CPU_PARTITION}" \
        --dependency="afterok:${ALPHA_BETA_JOB_ID}" \
        summarize_dmkcn_lambda_alpha_beta.slurm
)"

CHAIN_RECORD="logs/dmkcn_lambda_campaign2_$(date +%Y%m%d_%H%M%S)_jobs.tsv"
{
    printf 'stage\tjob_id\tdependency\n'
    printf 'k_sweep\t%s\t\n' "${KSWEEP_JOB_ID}"
    printf 'k_selection\t%s\tafterok:%s\n' \
        "${K_SUMMARY_JOB_ID}" "${KSWEEP_JOB_ID}"
    printf 'alpha_beta_grid\t%s\tafterok:%s\n' \
        "${ALPHA_BETA_JOB_ID}" "${K_SUMMARY_JOB_ID}"
    printf 'final_consolidation\t%s\tafterok:%s\n' \
        "${FINAL_JOB_ID}" "${ALPHA_BETA_JOB_ID}"
} | tee "${CHAIN_RECORD}"
