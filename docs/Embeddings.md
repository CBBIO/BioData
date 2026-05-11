# CBBIO Embeddings

`CBBIO.embeddings` provides model-agnostic protein embedding generation. Model-specific generators still expose the low-level `generate(...)` and `generate_batches(...)` methods, but new file and production workflows should use the job API:

```python
from CBBIO.embeddings import (
    FastaBatcher,
    Generator,
    EmbeddingWriter,
    pooler_factory,
    run_embedding_generation,
)

generator = Generator(model_class="protT5", device="cuda:0", dtype="float16")
batcher = FastaBatcher(
    "proteins.fasta",
    batch_size=None,
    max_batch_tokens=32768,
    limit=1000,
    length_sort_window=1000,
    max_sequence_length=4000,
    skipped_path="skipped.tsv",
)
writer = EmbeddingWriter(
    format="npy",
    path="embeddings.npy",
    records_per_shard=10000,
)

result = run_embedding_generation(
    generator,
    batcher,
    writer,
    layer_index=0,
    pooler=pooler_factory("mean"),
    fail_fast=False,
)
```

## Job API

### `Generator(...)`

Factory for model-specific generators:

```python
Generator(model_class="protT5", device="cuda:0", dtype="float16")
Generator(model_class="esm2", name="esm2_t6_8M_UR50D", device="cuda:0")
Generator(model_class="ankh3", name="ElnaggarLab/ankh3-large", prefix="[S2S]")
```

Catalog helpers:

- `available_generator_classes()`
- `available_generator_models(model_class=None)`

### `FastaBatcher(...)`

Streams FASTA records and yields batches:

```python
FastaBatcher(
    path,
    batch_size=None,
    max_batch_tokens=None,
    limit=None,
    length_sort_window=None,
    max_sequence_length=None,
    skipped_path=None,
    id_from="record_id",
)
```

Batching rules:

- `batch_size=None` and `max_batch_tokens=None`: one sequence per batch.
- `batch_size=N`: fixed-size batching.
- `max_batch_tokens=N`: token-budget batching.
- both caps set: both constraints are enforced.
- `length_sort_window=None`: no sorting.
- `max_sequence_length=None`: no length filter.
- `skipped_path=None`: skipped records are counted in memory only.
- `limit` means the first N accepted records after filtering and optional window sorting.

When no `max_sequence_length` is set, the job runner emits an OOM-risk warning. The warning is stronger when `max_batch_tokens` is also unset.

ESM-C batching depends on the installed ESM SDK accepting multiple encoded proteins in one logits call. If your SDK version does not support that path, set `batch_size=1` for `model_class="esmc"`.

### `IterableBatcher(...)`

Use this for in-memory records:

```python
from CBBIO.embeddings import GenerationInput, IterableBatcher

batcher = IterableBatcher(
    [GenerationInput(id="P1", sequence="MTEYKLVVVG")],
    batch_size=None,
    max_batch_tokens=None,
    limit=None,
)
```

### `EmbeddingWriter(...)`

One writer class handles all output formats:

```python
EmbeddingWriter(format="memory")
EmbeddingWriter(format="pkl", path="embeddings.pkl", records_per_shard=10000, payload_format="records")
EmbeddingWriter(format="npy", path="embeddings.npy", records_per_shard=10000)
EmbeddingWriter(format="h5", path="embeddings.h5", compression="gzip", write_batch_size=128)
```

Formats:

- `memory`: stores materialized `EmbeddingRecord` objects on `writer.records`.
- `pkl`: writes numbered pickle shards.
- `npy`: writes numbered `.npy` shards plus `.ids.txt` sidecars.
- `h5`: writes one HDF5 file.

### `run_embedding_generation(...)`

```python
run_embedding_generation(
    generator,
    batcher,
    writer,
    *,
    layer_index=0,
    pooler=None,
    fail_fast=False,
    progress_callback=None,
)
```

The runner requires a non-null batcher and writer. It recursively bisects a batch on CUDA OOM. A protein is recorded as OOM-causing only when it fails as a singleton.

## Poolers

Poolers are run-level configuration:

```python
from CBBIO.embeddings import pooler_factory

pooler_factory("mean")
pooler_factory("none")
```

- `pooler=None`: keep the original model payload shape.
- `pooler_factory("mean")`: mean-pool `n_residues x embedding_dim` into `embedding_dim`.
- `pooler_factory("none")`: identity/no pooling.

The pooler receives the tensor-like residue payload before conversion to Python lists when the model backend exposes tensors.

## Low-Level API

These methods remain supported:

```python
generator.generate(records, layer_index=0, pooler=None, fail_fast=False)

generator.generate_batches(
    records,
    batch_size=512,
    max_batch_tokens=32768,
    layer_index=0,
    pooler=None,
    fail_fast=False,
)

generator.iter_records(records, batch_size=512, layer_index=0)
```

FASTA and persistence helpers also remain:

- `iter_fasta_inputs(path, id_from="record_id")`
- `load_fasta_inputs(path, id_from="record_id")`
- `batch_generation_inputs(records, batch_size=..., max_batch_tokens=None)`
- `load_embedding_records(...)`
- `load_embedding_records_pickle(...)`
- `load_embedding_records_npy(...)`
- `save_embedding_records_pickle(...)`
- `save_embedding_records_npy(...)`
- `save_embedding_records_npy_shards(...)`
- `save_embedding_records_h5(...)`

## Core Types

### `GenerationInput`

- `id: str`
- `sequence: str`
- `description: str | None`
- `metadata: dict[str, Any] | None`

### `EmbeddingRecord`

- `id: str`
- `embedding: Sequence[float] | Sequence[Sequence[float]]`
- `layer_index: int`
- `model_reference: str`
- `shape: tuple[int, ...]`
- `metadata: dict[str, Any] | None`

### `GenerationResult`

- `records: list[EmbeddingRecord]`
- `errors: list[dict[str, Any]]`
- `skipped: list[dict[str, Any]]`
- `model_metadata: ModelMetadata | None`
- `run_metadata: RunMetadata | None`

### `EmbeddingJobResult`

- `record_count`
- `error_count`
- `skipped_count`
- `paths`
- `id_paths`
- `errors`
- `skipped`
- `elapsed_seconds`

## Model Families

- `protT5`: Hugging Face ProtT5; BioData layer convention where `layer_index=0` is the last hidden layer.
- `prostT5`: ProstT5 protein-to-embedding path; BioData layer convention.
- `ankh3`: ANKH3 via `T5Tokenizer` and `T5EncoderModel`.
- `esm2`: ESM2 via `esm.pretrained` or Transformers fallback; native ESM2 layer indexing.
- `esm1b`: ESM-1b via `esm.pretrained` or Transformers fallback; native ESM1b layer indexing.
- `esmc`: ESM-C SDK path. Multi-sequence batches require SDK support for batched logits; use `batch_size=1` if the installed SDK rejects batched inputs.

## Deprecated

These wrappers remain compatible but emit `DeprecationWarning` and will be removed soon.

| Deprecated | Replacement |
| --- | --- |
| `generate_from_fasta(...)` | `FastaBatcher(...)` + `EmbeddingWriter(format="memory")` + `run_embedding_generation(...)` |
| `generate_from_fasta_batches(...)` | `FastaBatcher(...)` + `run_embedding_generation(...)`, or low-level `generator.generate_batches(iter_fasta_inputs(...))` |
| `iter_embedding_records_from_fasta(...)` | `FastaBatcher(...)` + `EmbeddingWriter(format="memory")` + `run_embedding_generation(...)` |
| `generate_fasta_pickle_shards(...)` | `FastaBatcher(...)` + `EmbeddingWriter(format="pkl")` + `run_embedding_generation(...)` |
| `generate_fasta_npy_shards(...)` | `FastaBatcher(...)` + `EmbeddingWriter(format="npy")` + `run_embedding_generation(...)` |
| `generate_fasta_h5(...)` | `FastaBatcher(...)` + `EmbeddingWriter(format="h5")` + `run_embedding_generation(...)` |

`generate_pooled(...)` and `generate_batches_pooled(...)` were experimental and are removed. Use `generate(..., pooler=...)` and `generate_batches(..., pooler=...)`.

## Exceptions

- `EmbeddingGenerationError`: base embedding exception.
- `EmbeddingInputError`: invalid input records, sequences, or file payloads.
- `EmbeddingDependencyError`: missing optional runtime dependency.
- `EmbeddingBackendError`: backend/model inference failure.
