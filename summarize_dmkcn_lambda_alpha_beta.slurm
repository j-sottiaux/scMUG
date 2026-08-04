#!/bin/bash
#SBATCH --job-name=dmkcn_c2_final
#SBATCH --output=logs/dmkcn_c2_final_%j.out
#SBATCH --error=logs/dmkcn_c2_final_%j.err
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --mail-type=END,FAIL

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/home/julien.sottiaux/scMUG}"
SCMUG_PYTHON_MODULE="${SCMUG_PYTHON_MODULE:-pytorch/2.0.1/gpu}"
SCMUG_CPU_PARTITION="${SCMUG_CPU_PARTITION:-}"
LAMBDA_CONFIG="${LAMBDA_CONFIG:-${PROJECT_DIR}/configs/dmkcn_lambdas_triplets}"
K_REPORTING="${K_REPORTING:-${PROJECT_DIR}/reporting/dmkcn_lambda_campaign2/k_sweep}"
ALPHA_INPUT_ROOT="${ALPHA_INPUT_ROOT:-${PROJECT_DIR}/outputs/dmkcn_lambda_campaign2/alpha_beta}"
ALPHA_REPORTING="${ALPHA_REPORTING:-${PROJECT_DIR}/reporting/dmkcn_lambda_campaign2/alpha_beta}"
COMPUTATION_REPORTING="${COMPUTATION_REPORTING:-${PROJECT_DIR}/reporting/dmkcn_lambda_campaign2/computation}"
COMPARISON_REPORTING="${COMPARISON_REPORTING:-${PROJECT_DIR}/reporting/dmkcn_lambda_campaign2/final_comparison}"
REFERENCE_BEST="${REFERENCE_BEST:-${PROJECT_DIR}/publication/alpha_beta/alpha_beta_best_by_nmi.csv}"
REFERENCE_SEEDS="${REFERENCE_SEEDS:-${PROJECT_DIR}/publication/alpha_beta/alpha_beta_seed_means.csv}"
DRY_RUN="${DRY_RUN:-false}"

ALPHA_COMMAND=(
    python3 summarize_dmkcn_lambda_campaign2.py alpha-beta
    --input-root "${ALPHA_INPUT_ROOT}"
    --best-k "${K_REPORTING}/best_k_by_nmi.csv"
    --lambda-config "${LAMBDA_CONFIG}"
    --outdir "${ALPHA_REPORTING}"
)
COMPUTATION_COMMAND=(
    python3 summarize_dmkcn_lambda_campaign2.py computation
    --k-input-root "${PROJECT_DIR}/outputs/dmkcn_lambda_campaign2/k_sweep"
    --alpha-input-root "${ALPHA_INPUT_ROOT}"
    --lambda-config "${LAMBDA_CONFIG}"
    --outdir "${COMPUTATION_REPORTING}"
)
COMPARE_COMMAND=(
    python3 summarize_dmkcn_lambda_campaign2.py compare
    --new-best "${ALPHA_REPORTING}/best_alpha_beta_by_nmi.csv"
    --new-seed-scores "${ALPHA_REPORTING}/alpha_beta_seed_means.csv"
    --reference-best "${REFERENCE_BEST}"
    --reference-seed-scores "${REFERENCE_SEEDS}"
    --outdir "${COMPARISON_REPORTING}"
)
printf 'Command:'
printf ' %q' "${ALPHA_COMMAND[@]}"
printf '\n'
printf 'Command:'
printf ' %q' "${COMPUTATION_COMMAND[@]}"
printf '\n'
printf 'Command:'
printf ' %q' "${COMPARE_COMMAND[@]}"
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
"${ALPHA_COMMAND[@]}"
"${COMPUTATION_COMMAND[@]}"
"${COMPARE_COMMAND[@]}"
