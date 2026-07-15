#!/bin/bash
#SBATCH --job-name=cd_grid
#SBATCH --output=logs/cd_grid_%A_%a.out
#SBATCH --error=logs/cd_grid_%A_%a.err
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --partition=gpuai
#SBATCH --array=2,4%2

set -euo pipefail

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-16}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-16}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK:-16}
export NUMEXPR_NUM_THREADS=${SLURM_CPUS_PER_TASK:-16}

PROJECT_DIR="${PROJECT_DIR:-/home/julien.sottiaux/scMUG}"
CONFIG_FILE="${CONFIG_FILE:-${PROJECT_DIR}/configs/data.yaml}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${PROJECT_DIR}/outputs_adapter_pilot}"
ALPHA_BETA_GRID="${ALPHA_BETA_GRID:-0.01:1,0.1:1,1:1}"
SEEDS="${SEEDS:-1111,2222,3333}"
REPEAT="${REPEAT:-1}"
KMEANS_TIMES="${KMEANS_TIMES:-20}"

module purge
module load pytorch/2.0.1/gpu

cd "${PROJECT_DIR}"
mkdir -p logs

eval "$(
python3 read_configs.py \
    --config "${CONFIG_FILE}" \
    --index "${SLURM_ARRAY_TASK_ID}"
)"

DATASET_OUTDIR="${OUTPUT_ROOT}/${DATASET}"
OUTFILE="${DATASET_OUTDIR}/cd_grid_alpha_d.tsv"

for required_file in \
    "${DATASET_OUTDIR}/latents_ae.joblib" \
    "${DATASET_OUTDIR}/latents_dmkcn__spectral_dense_d32.joblib"
do
    if [ ! -f "${required_file}" ]; then
        echo "Missing required pilot file: ${required_file}" >&2
        exit 1
    fi
done

echo "Dataset: ${DATASET}"
echo "Alpha/beta grid: ${ALPHA_BETA_GRID}"
echo "Input directory: ${DATASET_OUTDIR}"
echo "Output: ${OUTFILE}"
echo "Git commit: $(git rev-parse HEAD)"

python3 ablation_block_c_d.py \
    --dataset "${DATASET}" \
    --cluster_number "${CLUSTER_NUMBER}" \
    --seeds "${SEEDS}" \
    --repeat "${REPEAT}" \
    --kmeans_times "${KMEANS_TIMES}" \
    --n_neighbour "${N_NEIGHBOUR}" \
    --latents-autoencoder "${DATASET_OUTDIR}/latents_ae.joblib" \
    --latents-dmkcn "${DATASET_OUTDIR}/latents_dmkcn__spectral_dense_d32.joblib" \
    --outfile "${OUTFILE}" \
    --thread-num "${SLURM_CPUS_PER_TASK}" \
    --red_global "${RED_GLOBAL}" \
    --red_local "${RED_LOCAL}" \
    --alpha-beta-grid "${ALPHA_BETA_GRID}"
