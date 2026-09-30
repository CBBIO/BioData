# Search Benchmarks

Measure BioData nearest-neighbor backends with reproducible query sets and JSON output.

```bash
poetry run python scripts/benchmark_neighbor_search.py \
  --ids-file notebooks/data/random_protein_ids_10000.txt \
  --embedding-type-id 3 \
  --layer-index 0 \
  --metric cosine \
  --backends pgvector faiss_cpu \
  --json-out /tmp/neighbor-benchmark.json
```

The command searches stored proteins from the ID file. It measures repeated batches for each
requested backend. Configure the database with `config.yaml` or `BIODATA_*` environment variables.

## Portable exact store on Slurm

```bash
module purge
module load pytorch/2.12.2_cuda132
python -m venv "$FSCRATCH/venvs/biodata"
source "$FSCRATCH/venvs/biodata/bin/activate"
python -m pip install --upgrade pip

# Exact retrieval does not need Transformer Engine or embedding-model dependencies.
python -m pip install --no-deps "$HOME/BioData"
python -m pip install \
  'numpy>=2.4.2,<3.0.0' \
  'pyyaml>=6.0.3,<7.0.0' \
  'faiss-cpu>=1.13.2,<2.0.0' \
  'cuvs-cu13>=26.4.0,<27.0.0' \
  'cupy-cuda13x>=14.0.1,<15.0.0'

export BIODATA_INDEX_ROOT=$FSCRATCH/benchmark/biodata-indexes
export BIODATA_VENV=$FSCRATCH/venvs/biodata
sbatch scripts/benchmark_portable_exact_store.sbatch
```

Run bounded-VRAM exact cuVS instead of the resident cuVS benchmark when the full float32 matrix
does not fit on the GPU:

```bash
export BIODATA_BACKENDS=cuvs_streaming
# Omit this variable to choose a conservative size from free VRAM.
export BIODATA_CUVS_STREAMING_BLOCK_SIZE=250000
sbatch scripts/benchmark_portable_exact_store.sbatch
```

`cuvs_streaming` scans the portable exact store on every call. It retains extra candidates at a
canonical distance tie, then applies the same `(distance, protein_id)` ordering as the
other exact backends. It trades this full scan for bounded VRAM.

The install command places the `CBBIO` package in the scratch-backed virtual environment. It
installs only dependencies required by exact local retrieval, so it does not build Transformer
Engine.
`scripts/benchmark_portable_exact_store.sbatch` runs a database-free benchmark from a copied
portable exact store. It measures both the cold state materialization and a warm exact search for
FAISS CPU and cuVS GPU. It selects the same artifact rows for every backend, then reports recall
against FAISS CPU when both succeed.

Copy the complete artifact hierarchy before submitting the job. Preserve the `current` symlink and
omit interrupted generations:

```bash
rsync -a --exclude='.*.tmp' \
  .biodata/indexes/biodata-nas/ \
  'cluster.example:$FSCRATCH/biodata-indexes/biodata-nas/'
```

The shipped Slurm request matches the B200 cluster environment: `gpu_partition`, one B200 GPU,
8 CPUs, 256 GiB of host RAM, and the `pytorch/2.12.2_cuda132` module. It activates
`$FSCRATCH/venvs/biodata` by default, isolates user-site packages, and configures offline Hugging
Face caches under `$FSCRATCH/huggingface`. Override `BIODATA_VENV`, `BIODATA_INDEX_ROOT`, or the
Slurm directives if your cluster differs. Exact cuVS over the current 12 M × 1024 collection needs
a GPU with at least 64 GiB of usable VRAM.

## Corpus scaling

```bash
export POSTGRES_HOST=localhost
export POSTGRES_PORT=5432
export POSTGRES_USER=biodata
export POSTGRES_PASSWORD=secret
export POSTGRES_DB=biodata

poetry run python scripts/benchmark_neighbor_search_corpus_scaling.py \
  --corpus-sizes 10000 100000 1000000 \
  --query-counts 100 1000 5000 \
  --embedding-type-id 3 \
  --layer-index 0 \
  --metric cosine \
  --backends pgvector faiss_cpu cuvs_gpu \
  --device cuda:0 \
  --json-out /tmp/corpus-scaling.json
```

This benchmark samples one candidate corpus per requested size. It reports corpus preparation,
index construction, warm search, batch latency, per-query latency, throughput, and end-to-end time.
It accepts `--dsn` instead of the `POSTGRES_*` variables.

The script creates a session-local temporary table named `benchmark_neighbor_candidates`. It drops
and recreates that temporary table for every corpus size. It does not alter persistent data.

## Approximate search

```bash
poetry run python scripts/benchmark_neighbor_search_corpus_scaling.py \
  --corpus-sizes 1000000 \
  --query-counts 1000 \
  --embedding-type-id 3 \
  --layer-index 0 \
  --metric cosine \
  --backends pgvector faiss_cpu \
  --use-ann \
  --ann-ef-search 200 \
  --ann-candidate-pool 1000
```

`--use-ann` enables the approximate mode supported by each backend. For corpus scaling, pgvector
builds a temporary HNSW index. FAISS builds IVF and cuVS builds CAGRA. The pgvector path reranks
the candidate pool exactly, so `--ann-candidate-pool` trades additional work for recall.

The general neighbor benchmark expects compatible persistent pgvector ANN indexes to exist when
you use `--use-ann` with `--backends pgvector`.

## Results

Each command writes a JSON file. Benchmark output is ignored by Git.

| Field | Description |
|---|---|
| `mean_seconds` | Mean duration for one timed batch |
| `mean_seconds_per_query` | Mean batch duration divided by query count |
| `throughput_qps` | Queries divided by mean batch duration |
| `cold_setup_seconds` | Corpus loading, index construction, and warmup time |
| `end_to_end_seconds` | Setup plus all timed work for a backend and corpus size |

## Exceptions

| Exception | When raised |
|---|---|
| `SystemExit` | Invalid command-line values or incomplete database connection variables |
| `RuntimeError` | Requested backend cannot provide the selected benchmark mode |
