#!/bin/bash

load_shell_exports() {
    local export_file status
    export_file="$(mktemp "${TMPDIR:-/tmp}/scmug_exports.XXXXXX")"
    if "$@" >"${export_file}"; then
        if source "${export_file}"; then
            rm -f "${export_file}"
            return 0
        else
            status=$?
        fi
    else
        status=$?
    fi
    rm -f "${export_file}"
    return "${status}"
}

require_cpu_partition() {
    if [ -z "${SCMUG_CPU_PARTITION:-}" ]; then
        echo "SCMUG_CPU_PARTITION must name the validated Zeus CPU partition." >&2
        return 2
    fi
    if [ -n "${SLURM_JOB_PARTITION:-}" ] \
        && [ "${SLURM_JOB_PARTITION}" != "${SCMUG_CPU_PARTITION}" ]; then
        echo "CPU job scheduled on ${SLURM_JOB_PARTITION}; expected ${SCMUG_CPU_PARTITION}." >&2
        return 2
    fi
}

validate_cpu_python_environment() {
    python3 - <<'PY'
import torch
import umap

print(
    "CPU Python environment:",
    f"torch={torch.__version__}",
    f"cuda_available={torch.cuda.is_available()}",
    f"umap={umap.__version__}",
)
PY
}
