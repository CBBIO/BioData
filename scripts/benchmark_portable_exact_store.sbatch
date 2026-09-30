#!/usr/bin/env bash
#SBATCH --job-name=biodata-cuvs-exact
#SBATCH --partition=gpu_partition
#SBATCH --constraint=gpu
#SBATCH --gres=gpu:B200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=256G
#SBATCH --time=24:00:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

set -euo pipefail

: "${FSCRATCH:?FSCRATCH must point to the cluster scratch filesystem.}"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
LAUNCHER_DIRECTORY="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BENCHMARK_SCRIPT="${BENCHMARK_SCRIPT:-$LAUNCHER_DIRECTORY/benchmark_portable_exact_store.py}"
BIODATA_VENV="${BIODATA_VENV:-$FSCRATCH/venvs/biodata}"
BIODATA_INDEX_ROOT="${BIODATA_INDEX_ROOT:-$FSCRATCH/biodata-indexes}"
RESULTS_DIRECTORY="${BIODATA_RESULTS_DIRECTORY:-$FSCRATCH/biodata-benchmarks}"
PYTHON="${PYTHON:-$BIODATA_VENV/bin/python}"

module purge
module load pytorch/2.12.2_cuda132
source "$BIODATA_VENV/bin/activate"

# Isolate the job from packages installed under the login-node home directory.
export PYTHONNOUSERSITE=1

# The benchmark does not generate embeddings, but these prevent accidental model downloads.
export HF_HOME="${HF_HOME:-$FSCRATCH/huggingface}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_HOME/hub}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HUB_CACHE}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

mkdir -p "$RESULTS_DIRECTORY"
cd "$PROJECT_ROOT"

read -r -a BACKENDS <<< "${BIODATA_BACKENDS:-faiss_cpu cuvs_gpu}"
STREAMING_ARGUMENTS=()
if [[ -n "${BIODATA_CUVS_STREAMING_BLOCK_SIZE:-}" ]]; then
  STREAMING_ARGUMENTS=(--cuvs-streaming-block-size "$BIODATA_CUVS_STREAMING_BLOCK_SIZE")
fi

"$PYTHON" "$BENCHMARK_SCRIPT" \
  --index-root "$BIODATA_INDEX_ROOT" \
  --database-label "${BIODATA_DATABASE_LABEL:-biodata-nas}" \
  --embedding-type-id "${BIODATA_EMBEDDING_TYPE_ID:-3}" \
  --layer-index "${BIODATA_LAYER_INDEX:-0}" \
  --metric "${BIODATA_METRIC:-cosine}" \
  --query-count "${BIODATA_QUERY_COUNT:-10}" \
  --query-seed "${BIODATA_QUERY_SEED:-20260919}" \
  --top-k "${BIODATA_TOP_K:-100}" \
  --backends "${BACKENDS[@]}" \
  --device "${BIODATA_CUDA_DEVICE:-cuda:0}" \
  --faiss-threads "${SLURM_CPUS_PER_TASK:-8}" \
  "${STREAMING_ARGUMENTS[@]}" \
  --json-out "$RESULTS_DIRECTORY/portable-exact-store-${SLURM_JOB_ID:-local}.json"
