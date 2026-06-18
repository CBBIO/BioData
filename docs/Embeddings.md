# CBBIO Embeddings

`CBBIO.embeddings` provides model-agnostic protein embedding generation. Use the job API for file-backed and production workflows: choose a generator, stream inputs through a batcher, shape the output with a pooler, and persist records with an `EmbeddingWriter`.

```python
from CBBIO import (
    EmbeddingWriter,
    FastaBatcher,
    Generator,
    pooler_factory,
    run_embedding_generation,
)

generator = Generator(model_class="protT5", device="cuda:0", dtype="float16")
batcher = FastaBatcher(
    "proteins.fasta",
    batch_size=None,
    max_batch_tokens=32768,
    length_sort_window=1000,
    max_sequence_length=4000,
    skipped_path="skipped.tsv",
)
writer = EmbeddingWriter(format="h5", path="embeddings.h5", write_batch_size=128)

result = run_embedding_generation(
    generator,
    batcher,
    writer,
    layer_index=[0, 12, 24],
    pooler=None,
    fail_fast=False,
)
```

For large residue-level stores, prefer HDF5 (`format="h5"`). It supports vector and matrix payloads, multiple layers per protein, multiple pool methods per protein/layer, and indexed partial reads.

## Main Workflows

### Generate from FASTA

```python
from CBBIO import EmbeddingWriter, FastaBatcher, Generator, run_embedding_generation

generator = Generator(model_class="esm2", name="esm2_t33_650M_UR50D", device="cuda:0")
batcher = FastaBatcher(
    "proteins.fasta",
    batch_size=None,
    max_batch_tokens=32768,
    max_sequence_length=4000,
)
writer = EmbeddingWriter(format="h5", path="esm2_residue_layers.h5")

run_embedding_generation(
    generator,
    batcher,
    writer,
    layer_index=[0, 16, 33],
    pooler=None,
    fail_fast=False,
)
```

`FastaBatcher` streams records and can batch by record count, padded-token budget, or both. When no `max_sequence_length` is set, the runner emits an OOM-risk warning. The warning is stronger when `max_batch_tokens` is also unset.

### Generate from In-Memory Records

```python
from CBBIO import EmbeddingWriter, GenerationInput, Generator, IterableBatcher, run_embedding_generation

records = [
    GenerationInput(id="P1", sequence="MTEYKLVVVG"),
    GenerationInput(id="P2", sequence="GAGGVGKSAL"),
]

generator = Generator(model_class="protT5", device="cuda:0", dtype="float16")
batcher = IterableBatcher(records, batch_size=16)
writer = EmbeddingWriter(format="memory")

result = run_embedding_generation(generator, batcher, writer, layer_index=0)
memory_records = writer.records
```

### Choose Output Shape

Poolers are run-level output-shaping configuration:

```python
from CBBIO import pooler_factory

pooler_factory("mean")
pooler_factory("none")
pooler_factory("cls")
```

- `pooler=None`: keep the original model payload shape. Residue-level outputs are matrix payloads with shape `(n_residues, embedding_dim)`.
- `pooler_factory("none")`: identity/no pooling. This is equivalent to an unpooled residue matrix when the backend returns residue embeddings.
- `pooler_factory("mean")`: mean-pool a residue matrix into one vector with shape `(embedding_dim,)`.
- `pooler_factory("cls")`: use the model-specific CLS/BOS token path when supported.

Generated pooled records carry `metadata["pooling"]`. H5 stores this as `pool_method`, so a raw residue matrix and pooled vectors can coexist for the same protein and layer. Pooling aliases are normalized: identity/no pooling maps to `none`, and `bos` maps to `cls`.

## Writers and Formats

One writer class handles all persistence targets:

```python
EmbeddingWriter(format="memory")
EmbeddingWriter(format="pkl", path="embeddings.pkl", records_per_shard=10000, payload_format="records")
EmbeddingWriter(format="npy", path="embeddings.npy", records_per_shard=10000)
EmbeddingWriter(format="h5", path="embeddings.h5", compression="gzip", write_batch_size=128)
```

- `memory`: stores materialized `EmbeddingRecord` objects on `writer.records`.
- `pkl`: writes numbered pickle shards.
- `npy`: writes numbered `.npy` shards plus `.ids.txt` sidecars. Use this for same-width vector payloads.
- `h5`: writes one extendable HDF5 file. Use this for large residue-level stores, mixed vector/matrix payloads, or partial reads.

Format helpers remain available:

```python
from CBBIO import (
    load_embedding_records,
    load_embedding_records_h5,
    load_embedding_records_npy,
    load_embedding_records_pickle,
    save_embedding_records_h5,
    save_embedding_records_npy,
    save_embedding_records_npy_shards,
    save_embedding_records_pickle,
    save_embedding_records_pickle_shards,
)
```

`load_embedding_records(path)` dispatches by extension for `.pkl`, `.pickle`, `.npy`, `.npz`, `.h5`, and `.hdf5`.

## H5 IO and Partial Reads

H5 files store record indexes (`id`, `layer_index`, `model_reference`, payload kind, and `pool_method`) separately from the numeric payload. Matrix payloads are stored flat with offsets, so reading a residue segment can load only the requested rows.

### Write Residue-Level Layers

```python
from CBBIO import EmbeddingWriter, FastaBatcher, Generator, run_embedding_generation

generator = Generator(model_class="esm2", name="esm2_t33_650M_UR50D", device="cuda:0")
batcher = FastaBatcher("proteins.fasta", max_batch_tokens=32768, max_sequence_length=4000)
writer = EmbeddingWriter(format="h5", path="esm2_residue_layers.h5", compression="gzip")

run_embedding_generation(
    generator,
    batcher,
    writer,
    layer_index=[0, 16, 33],
    pooler=None,
    fail_fast=False,
)
```

### Read a Full Layer

```python
from CBBIO import load_embedding_records_h5

layer_16 = load_embedding_records_h5(
    "esm2_residue_layers.h5",
    layer_index=16,
    pool_method="none",
)
```

### Read One Protein Across Several Layers

```python
from CBBIO import H5EmbeddingReader

reader = H5EmbeddingReader("esm2_residue_layers.h5")
protein_layers = reader.read_many(
    ids="P12345",
    layer_index=[0, 16, 33],
    pool_method="none",
)
```

### Read One Residue Segment

```python
segment = reader.read(
    "P12345",
    layer_index=16,
    pool_method="none",
    residue_slice=(100, 150),
)

matrix = segment.embedding
```

`residue_slice=(start, end)` follows Python slice semantics and is zero-based, end-exclusive. You can also pass `residue_start=100, residue_end=150`. Residue slicing is only valid for matrix payloads.

### Store and Select Pooled Variants

Generate residue matrices and pooled vectors separately, then append them into one H5 file when you need both lookup styles:

```python
from CBBIO import pooler_factory, save_embedding_records_h5

residue_result = generator.generate(records, layer_index=16, pooler=None)
mean_result = generator.generate(records, layer_index=16, pooler=pooler_factory("mean"))

save_embedding_records_h5("esm2_layers.h5", residue_result.records)
save_embedding_records_h5("esm2_layers.h5", mean_result.records, append=True)
```

Then select the variant explicitly:

```python
reader = H5EmbeddingReader("esm2_layers.h5")

raw_residues = reader.read("P12345", layer_index=16, pool_method="none")
mean_vector = reader.read("P12345", layer_index=16, pool_method="mean")
```

If more than one record matches `reader.read(...)`, it raises `EmbeddingInputError`. Add `layer_index` and/or `pool_method` to make the selection unique.

## Job API Reference

### `Generator(...)`

Factory for model-specific generators:

```python
Generator(model_class="protT5", device="cuda:0", dtype="float16")
Generator(model_class="esm2", name="esm2_t6_8M_UR50D", device="cuda:0")
Generator(model_class="ankh3", name="ElnaggarLab/ankh3-large", prefix="[S2S]")
Generator(model_class="amplify", name="nvidia/AMPLIFY_120M", device="cuda:0")
```

Catalog helpers:

- `available_generator_classes()`
- `available_generator_models(model_class=None)`

### `FastaBatcher(...)`

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
- `limit` means the first N accepted records after filtering and optional window sorting.

ESM-C batching depends on the installed ESM SDK accepting multiple encoded proteins in one logits call. If your SDK version does not support that path, set `batch_size=1` for `model_class="esmc"`.

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

FASTA helpers also remain:

- `iter_fasta_inputs(path, id_from="record_id")`
- `load_fasta_inputs(path, id_from="record_id")`
- `batch_generation_inputs(records, batch_size=..., max_batch_tokens=None)`

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

- `protT5`: Hugging Face ProtT5; native hidden-state layer indexing.
- `prostT5`: ProstT5 protein-to-embedding path; native hidden-state layer indexing.
- `ankh3`: ANKH3 via `T5Tokenizer` and `T5EncoderModel`.
- `amplify`: AMPLIFY 120M and 350M via Transformers `AutoModel`/`AutoTokenizer` with `trust_remote_code=True`; max context length 2048 residues. On CUDA, AMPLIFY defaults to `bfloat16` because its xFormers attention kernels do not support `float32`; pass `dtype="float16"` explicitly if preferred. Short aliases `amplify_120m` and `amplify_350m` select NVIDIA's TransformerEngine-optimized checkpoints and require `transformer_engine.pytorch`, not the bare `transformer-engine` meta package. For CUDA 13 environments, install with `poetry run pip install --no-build-isolation 'transformer-engine[pytorch,core-cu13]==2.16.0'`. Use `amplify_120m_chandar` or `amplify_350m_chandar` for the upstream Chandar Research Lab checkpoints.
- `proteinglm`: ProteinGLM MLM 1B, 3B, and 10B via Transformers `AutoModelForMaskedLM`/`AutoTokenizer` with `trust_remote_code=True`. Short aliases `proteinglm_1b_mlm`, `proteinglm_3b_mlm`, and `proteinglm_10b_mlm` resolve to the Biomap checkpoints. Outputs trim the trailing EOS token and default to the final hidden layer.
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

The old pooled-generation wrappers were experimental and have been removed. Use `generate(..., pooler=...)`, `generate_batches(..., pooler=...)`, or `run_embedding_generation(..., pooler=...)`.

## Exceptions

- `EmbeddingGenerationError`: base embedding exception.
- `EmbeddingInputError`: invalid input records, sequences, or file payloads.
- `EmbeddingDependencyError`: missing optional runtime dependency.
- `EmbeddingBackendError`: backend/model inference failure.
