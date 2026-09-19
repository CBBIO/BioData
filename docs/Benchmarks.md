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
