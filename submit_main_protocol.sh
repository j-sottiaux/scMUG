#!/bin/bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/home/julien.sottiaux/scMUG}"
SCMUG_CPU_PARTITION="${SCMUG_CPU_PARTITION:?Set SCMUG_CPU_PARTITION to the validated Zeus CPU partition}"
SCMUG_PYTHON_MODULE="${SCMUG_PYTHON_MODULE:-pytorch/2.0.1/gpu}"
export SCMUG_CPU_PARTITION SCMUG_PYTHON_MODULE
cd "${PROJECT_DIR}"
mkdir -p logs

submit_job() {
    local submission
    if ! submission="$(sbatch --parsable "$@")"; then
        echo "SLURM submission failed: sbatch $*" >&2
        return 1
    fi
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
    if ! sbatch --test-only "$@" >/dev/null; then
        echo "SLURM preflight failed for ${description}: sbatch $*" >&2
        return 1
    fi
}

# Validate every distinct resource request before creating any dependency chain.
# This prevents a late resource error from leaving a partial protocol queued.
validate_job "environment setup" \
    --partition="${SCMUG_CPU_PARTITION}" prepare_python_environment.slurm
validate_job "CPU preflight" \
    --partition="${SCMUG_CPU_PARTITION}" validate_cpu_environment.slurm
validate_job "k sweep" run_k_sweep.slurm
validate_job "k selection" \
    --partition="${SCMUG_CPU_PARTITION}" summarize_k_sweep.slurm
validate_job "computation summary" \
    --partition="${SCMUG_CPU_PARTITION}" summarize_computation.slurm
validate_job "alpha/beta grid" \
    --partition="${SCMUG_CPU_PARTITION}" run_alpha_beta_grid.slurm
validate_job "alpha/beta summary" \
    --partition="${SCMUG_CPU_PARTITION}" summarize_alpha_beta.slurm
validate_job "final comparison" \
    --partition="${SCMUG_CPU_PARTITION}" compare_final_pipelines.slurm

ENVIRONMENT_SETUP_JOB_ID="$(
    submit_job --partition="${SCMUG_CPU_PARTITION}" prepare_python_environment.slurm
)"
CPU_PREFLIGHT_JOB_ID="$(
    submit_job --partition="${SCMUG_CPU_PARTITION}" \
        --dependency="afterok:${ENVIRONMENT_SETUP_JOB_ID}" \
        validate_cpu_environment.slurm
)"
KSWEEP_JOB_ID="$(
    submit_job --dependency="afterok:${CPU_PREFLIGHT_JOB_ID}" run_k_sweep.slurm
)"
KSWEEP_SUMMARY_JOB_ID="$(
    submit_job --partition="${SCMUG_CPU_PARTITION}" \
        --dependency="afterok:${KSWEEP_JOB_ID}" summarize_k_sweep.slurm
)"
COMPUTATION_SUMMARY_JOB_ID="$(
    submit_job --partition="${SCMUG_CPU_PARTITION}" \
        --dependency="afterok:${KSWEEP_JOB_ID}" summarize_computation.slurm
)"
ALPHA_BETA_JOB_ID="$(
    submit_job --partition="${SCMUG_CPU_PARTITION}" \
        --dependency="afterok:${KSWEEP_SUMMARY_JOB_ID}" run_alpha_beta_grid.slurm
)"
ALPHA_BETA_SUMMARY_JOB_ID="$(
    submit_job --partition="${SCMUG_CPU_PARTITION}" \
        --dependency="afterok:${ALPHA_BETA_JOB_ID}" summarize_alpha_beta.slurm
)"
FINAL_COMPARISON_JOB_ID="$(
    submit_job --partition="${SCMUG_CPU_PARTITION}" \
        --dependency="afterok:${ALPHA_BETA_SUMMARY_JOB_ID}" compare_final_pipelines.slurm
)"

CHAIN_RECORD="logs/main_protocol_$(date +%Y%m%d_%H%M%S)_jobs.tsv"
{
    printf 'stage\tjob_id\tdependency\n'
    printf 'environment_setup\t%s\t\n' "${ENVIRONMENT_SETUP_JOB_ID}"
    printf 'cpu_preflight\t%s\tafterok:%s\n' \
        "${CPU_PREFLIGHT_JOB_ID}" "${ENVIRONMENT_SETUP_JOB_ID}"
    printf 'k_sweep\t%s\tafterok:%s\n' \
        "${KSWEEP_JOB_ID}" "${CPU_PREFLIGHT_JOB_ID}"
    printf 'k_selection\t%s\tafterok:%s\n' \
        "${KSWEEP_SUMMARY_JOB_ID}" "${KSWEEP_JOB_ID}"
    printf 'computation_summary\t%s\tafterok:%s\n' \
        "${COMPUTATION_SUMMARY_JOB_ID}" "${KSWEEP_JOB_ID}"
    printf 'alpha_beta_grid\t%s\tafterok:%s\n' \
        "${ALPHA_BETA_JOB_ID}" "${KSWEEP_SUMMARY_JOB_ID}"
    printf 'alpha_beta_selection\t%s\tafterok:%s\n' \
        "${ALPHA_BETA_SUMMARY_JOB_ID}" "${ALPHA_BETA_JOB_ID}"
    printf 'final_comparison\t%s\tafterok:%s\n' \
        "${FINAL_COMPARISON_JOB_ID}" "${ALPHA_BETA_SUMMARY_JOB_ID}"
} | tee "${CHAIN_RECORD}"

printf '\nSubmission record: %s\n' "${CHAIN_RECORD}"
