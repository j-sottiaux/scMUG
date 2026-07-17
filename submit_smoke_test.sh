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

SETUP_JOB_ID="$(
    submit_job --partition="${SCMUG_CPU_PARTITION}" prepare_python_environment.slurm
)"
CPU_CHECK_JOB_ID="$(
    submit_job --partition="${SCMUG_CPU_PARTITION}" \
        --dependency="afterok:${SETUP_JOB_ID}" validate_cpu_environment.slurm
)"
SMOKE_JOB_ID="$(
    submit_job --array=3 \
        --dependency="afterok:${CPU_CHECK_JOB_ID}" \
        --export=ALL,EXPERIMENT_ID=ksweep_smoke,SEEDS=1111,REPEAT=1 \
        run_k_sweep.slurm
)"

printf 'environment_setup\t%s\n' "${SETUP_JOB_ID}"
printf 'cpu_preflight\t%s\n' "${CPU_CHECK_JOB_ID}"
printf 'baron_k8_smoke\t%s\n' "${SMOKE_JOB_ID}"
