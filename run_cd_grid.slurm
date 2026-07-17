#!/bin/bash
#SBATCH --job-name=cd_grid
#SBATCH --output=logs/cd_grid_%A_%a.out
#SBATCH --error=logs/cd_grid_%A_%a.err
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --array=2,3%2

set -euo pipefail

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-16}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-16}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK:-16}
export NUMEXPR_NUM_THREADS=${SLURM_CPUS_PER_TASK:-16}

PROJECT_DIR="${PROJECT_DIR:-/home/julien.sottiaux/scMUG}"
SCMUG_PYTHON_MODULE="${SCMUG_PYTHON_MODULE:-pytorch/2.0.1/gpu}"
CONFIG_FILE="${CONFIG_FILE:-${PROJECT_DIR}/configs/data.yaml}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${PROJECT_DIR}/outputs}"
SOURCE_OUTPUT_ROOT="${SOURCE_OUTPUT_ROOT:-${PROJECT_DIR}/outputs_adapter_pilot}"
SOURCE_RUN_STEM="${SOURCE_RUN_STEM:-}"
EXPERIMENT_ID="${EXPERIMENT_ID:-cd_grid}"
ALPHA_BETA_GRID="${ALPHA_BETA_GRID:-0.01:1,0.1:1,1:1}"
SEEDS="${SEEDS:-1111,2222,3333}"
REPEAT="${REPEAT:-1}"
KMEANS_TIMES="${KMEANS_TIMES:-20}"

module purge
module load "${SCMUG_PYTHON_MODULE}"

cd "${PROJECT_DIR}"
source "${PROJECT_DIR}/slurm_helpers.sh"
require_cpu_partition
validate_cpu_python_environment

load_shell_exports python3 read_configs.py \
    --config "${CONFIG_FILE}" \
    --index "${SLURM_ARRAY_TASK_ID}"

DATASET_OUTDIR="${OUTPUT_ROOT}/${DATASET}"
RESULTS_DIR="${DATASET_OUTDIR}/results"
COMPUTATION_DIR="${DATASET_OUTDIR}/computation"
ARTIFACTS_DIR="${DATASET_OUTDIR}/artifacts"
RUN_LOG_DIR="${PROJECT_DIR}/logs/${DATASET}"
RUN_ID="$(date +%Y%m%d_%H%M%S)_job${SLURM_JOB_ID:-local}_task${SLURM_ARRAY_TASK_ID:-0}"
K_LABEL="$(printf 'k%02d' "${CLUSTER_NUMBER}")"
RUN_STEM="${EXPERIMENT_ID}_${K_LABEL}_${RUN_ID}"
OUTFILE="${RESULTS_DIR}/${RUN_STEM}_alpha_beta_grid.tsv"
mkdir -p "${RESULTS_DIR}" "${COMPUTATION_DIR}" "${ARTIFACTS_DIR}" "${RUN_LOG_DIR}"
exec > >(tee -a "${RUN_LOG_DIR}/${RUN_STEM}.out") \
     2> >(tee -a "${RUN_LOG_DIR}/${RUN_STEM}.err" >&2)

if [ -n "${SOURCE_RUN_STEM}" ]; then
    SOURCE_ARTIFACTS_DIR="${OUTPUT_ROOT}/${DATASET}/artifacts"
    LATENTS_AUTOENCODER="${SOURCE_ARTIFACTS_DIR}/${SOURCE_RUN_STEM}_ae_latents.joblib"
    LATENTS_DMKCN="${SOURCE_ARTIFACTS_DIR}/${SOURCE_RUN_STEM}_dmkcn_latents__spectral_dense_d32.joblib"
else
    # Backward-compatible input paths for the already completed adapter pilot.
    SOURCE_DATASET_DIR="${SOURCE_OUTPUT_ROOT}/${DATASET}"
    LATENTS_AUTOENCODER="${SOURCE_DATASET_DIR}/latents_ae.joblib"
    LATENTS_DMKCN="${SOURCE_DATASET_DIR}/latents_dmkcn__spectral_dense_d32.joblib"
fi

for required_file in \
    "${LATENTS_AUTOENCODER}" \
    "${LATENTS_DMKCN}"
do
    if [ ! -f "${required_file}" ]; then
        echo "Missing required pilot file: ${required_file}" >&2
        exit 1
    fi
done

echo "Dataset: ${DATASET}"
echo "Experiment: ${EXPERIMENT_ID}"
echo "Run: ${RUN_ID}"
echo "Alpha/beta grid: ${ALPHA_BETA_GRID}"
echo "Autoencoder latents: ${LATENTS_AUTOENCODER}"
echo "DMKCN latents: ${LATENTS_DMKCN}"
echo "Output: ${OUTFILE}"

python3 ablation_block_c_d.py \
    --dataset "${DATASET}" \
    --experiment-id "${EXPERIMENT_ID}" \
    --run-id "${RUN_ID}" \
    --cluster_number "${CLUSTER_NUMBER}" \
    --seeds "${SEEDS}" \
    --repeat "${REPEAT}" \
    --kmeans_times "${KMEANS_TIMES}" \
    --n_neighbour "${N_NEIGHBOUR}" \
    --latents-autoencoder "${LATENTS_AUTOENCODER}" \
    --latents-dmkcn "${LATENTS_DMKCN}" \
    --outfile "${OUTFILE}" \
    --thread-num "${SLURM_CPUS_PER_TASK}" \
    --red_global "${RED_GLOBAL}" \
    --red_local "${RED_LOCAL}" \
    --alpha-beta-grid "${ALPHA_BETA_GRID}"
