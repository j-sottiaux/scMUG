#!/bin/bash
#SBATCH --job-name=scmug_ab_grid
#SBATCH --output=logs/alpha_beta_%A_%a.out
#SBATCH --error=logs/alpha_beta_%A_%a.err
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --array=0-9%2
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=julien.sottiaux@univ-lille.fr

set -euo pipefail

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-16}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-16}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK:-16}
export NUMEXPR_NUM_THREADS=${SLURM_CPUS_PER_TASK:-16}

PROJECT_DIR="${PROJECT_DIR:-/home/julien.sottiaux/scMUG}"
SCMUG_PYTHON_MODULE="${SCMUG_PYTHON_MODULE:-pytorch/2.0.1/gpu}"
CONFIG_FILE="${CONFIG_FILE:-${PROJECT_DIR}/configs/data.yaml}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${PROJECT_DIR}/outputs}"
BEST_K_FILE="${BEST_K_FILE:-${PROJECT_DIR}/publication/k_sweep/ksweep_best_k_by_nmi.csv}"
EXPERIMENT_ID="${EXPERIMENT_ID:-alpha_beta_grid}"
ALPHA_BETA_GRID="${ALPHA_BETA_GRID:-0:1,0.001:1,0.01:1,0.1:1,1:1,1:0.1,1:0.01,1:0.001,1:0}"
SEEDS="${SEEDS:-1111,2222,3333,4444,5555,6666,7777,8888,9999,10000}"
REPEAT="${REPEAT:-3}"
KMEANS_TIMES="${KMEANS_TIMES:-20}"

module purge
module load "${SCMUG_PYTHON_MODULE}"
cd "${PROJECT_DIR}"
source "${PROJECT_DIR}/slurm_helpers.sh"
require_cpu_partition
validate_cpu_python_environment

load_shell_exports python3 alpha_beta_tasks.py \
    --best-k "${BEST_K_FILE}" \
    --config "${CONFIG_FILE}" \
    --task-id "${SLURM_ARRAY_TASK_ID}"
SELECTED_DATASET="${DATASET}"

load_shell_exports \
    python3 read_configs.py --config "${CONFIG_FILE}" --index "${CONFIG_INDEX}"
if [ "${DATASET}" != "${SELECTED_DATASET}" ]; then
    echo "Dataset mismatch between selected-k table and configuration." >&2
    exit 2
fi
CLUSTER_NUMBER="${SELECTED_K}"

DATASET_OUTDIR="${OUTPUT_ROOT}/${DATASET}"
RESULTS_DIR="${DATASET_OUTDIR}/results"
ARTIFACTS_DIR="${DATASET_OUTDIR}/artifacts"
RUN_LOG_DIR="${PROJECT_DIR}/logs/${DATASET}"
RUN_ID="$(date +%Y%m%d_%H%M%S)_job${SLURM_JOB_ID:-local}_task${SLURM_ARRAY_TASK_ID:-0}"
K_LABEL="$(printf 'k%02d' "${CLUSTER_NUMBER}")"
RUN_STEM="${EXPERIMENT_ID}_${K_LABEL}_${MODEL_TAG}_${RUN_ID}"
OUTFILE="${RESULTS_DIR}/${RUN_STEM}_grid.tsv"
MANIFEST="${ARTIFACTS_DIR}/${RUN_STEM}_manifest.json"
mkdir -p "${RESULTS_DIR}" "${ARTIFACTS_DIR}" "${RUN_LOG_DIR}"
exec > >(tee -a "${RUN_LOG_DIR}/${RUN_STEM}.out") \
     2> >(tee -a "${RUN_LOG_DIR}/${RUN_STEM}.err" >&2)

export EXPERIMENT_ID RUN_ID DATASET MODEL MODEL_TAG ARM CLUSTER_NUMBER
export PUBLISHED_BEST_K GROUND_TRUTH_K
export SOURCE_EXPERIMENT_ID SOURCE_RUN_ID LATENT_ARTIFACT ALPHA_BETA_GRID
export OUTFILE MANIFEST SEEDS REPEAT KMEANS_TIMES

python3 - <<'PY'
import json
import os
import time

payload = {
    "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    "experiment_id": os.environ["EXPERIMENT_ID"],
    "run_id": os.environ["RUN_ID"],
    "dataset": os.environ["DATASET"],
    "model": os.environ["MODEL"],
    "arm": os.environ["ARM"],
    "k": int(os.environ["CLUSTER_NUMBER"]),
    "published_best_k": int(os.environ["PUBLISHED_BEST_K"]),
    "ground_truth_k": int(os.environ["GROUND_TRUTH_K"]),
    "alpha_beta_grid": os.environ["ALPHA_BETA_GRID"],
    "seeds": os.environ["SEEDS"],
    "repeat": int(os.environ["REPEAT"]),
    "kmeans_times": int(os.environ["KMEANS_TIMES"]),
    "source_experiment_id": os.environ["SOURCE_EXPERIMENT_ID"],
    "source_run_id": os.environ["SOURCE_RUN_ID"],
    "latent_artifact": os.environ["LATENT_ARTIFACT"],
    "outfile": os.environ["OUTFILE"],
}
with open(os.environ["MANIFEST"], "w", encoding="utf-8") as handle:
    json.dump(payload, handle, indent=2, sort_keys=True)
    handle.write("\n")
PY

LATENT_ARGS=(--arm "${ARM}")
if [ "${ARM}" = "autoencoder" ]; then
    LATENT_ARGS+=(--latents-autoencoder "${LATENT_ARTIFACT}")
else
    LATENT_ARGS+=(--latents-dmkcn "${LATENT_ARTIFACT}")
fi

python3 ablation_block_c_d.py \
    --dataset "${DATASET}" \
    --experiment-id "${EXPERIMENT_ID}" \
    --run-id "${RUN_ID}" \
    --cluster-number "${CLUSTER_NUMBER}" \
    --seeds "${SEEDS}" \
    --repeat "${REPEAT}" \
    --kmeans_times "${KMEANS_TIMES}" \
    --n_neighbour "${N_NEIGHBOUR}" \
    --thread-num "${SLURM_CPUS_PER_TASK}" \
    --red_global "${RED_GLOBAL}" \
    --red_local "${RED_LOCAL}" \
    --alpha-beta-grid "${ALPHA_BETA_GRID}" \
    --grid-d-modes spectral \
    --outfile "${OUTFILE}" \
    "${LATENT_ARGS[@]}"
