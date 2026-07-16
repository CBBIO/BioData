# CBBIO Embeddings

`CBBIO.embeddings` provides model-agnostic protein embedding generation. Use the job API for file-backed and production workflows: choose a generator, stream inputs through a batcher, shape the output with a pooler, and persist records with an `EmbeddingWriter`.

---

## Concepts

### EmbeddingRecord

Every generated embedding is returned as an `EmbeddingRecord`:

| Field | Type | Description |
|---|---|---|
| `id` | `str` | Protein identifier (from FASTA header or GenerationInput) |
| `embedding` | `Sequence[float]` or `Sequence[Sequence[float]]` | A 1-D vector (pooled) or 2-D matrix (per-residue) |
| `layer_index` | `int` | Which model layer this came from |
| `model_reference` | `str` | HuggingFace repo or model identifier |
| `shape` | `tuple[int, ...]` | Shape of the embedding array |
| `metadata` | `dict \| None` | Optional extras (e.g. `{"pooling": "mean"}`) |

### Per-residue vs. pooled

By default, models return one embedding vector per amino acid — a matrix of shape `(sequence_length, hidden_dim)`. This is useful for residue-level tasks (PTM site prediction, disorder, binding).

For protein-level tasks, you want a single vector per protein. Use a **pooler** to collapse the matrix:

| Pooler | Description | Use when |
|---|---|---|
| `None` / `"none"` | Keep the raw model output shape | Residue-level tasks, custom pooling |
| `"mean"` | Average across the sequence length axis | Most protein-level tasks |
| `"cls"` | Use the CLS / BOS token position | Models that encode global context in the first token |

Pass a pooler to `run_embedding_generation` or `generator.generate`:

```python
from CBBIO import pooler_factory, run_embedding_generation

result = run_embedding_generation(
    generator, batcher, writer,
    layer_index=33,
    pooler=pooler_factory("mean"),   # → one (D,) vector per protein
)
```

### Layers

`layer_index=0` is the embedding layer (input representations). Higher indices are transformer layers. To extract multiple layers in a single forward pass:

```python
run_embedding_generation(generator, batcher, writer, layer_index=[0, 16, 33])
```

This produces one `EmbeddingRecord` per `(protein, layer)` pair in the output.

---

```python
from CBBIO import (
    EmbeddingWriter,
    FastaBatcher,
    Generator,
    pooler_factory,
    run_embedding_generation,
)

generator = Generator(model_class="prott5", device="cuda:0", dtype="float16")
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

generator = Generator(model_class="prott5", device="cuda:0", dtype="float16")
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
Generator(model_class="prott5", device="cuda:0", dtype="float16")
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

ESM-C loads Biohub checkpoints through Hugging Face Transformers and supports the same fixed-size and token-budget batching controls as the other Hugging Face ESM adapters.

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

### Quick reference

| `model_class` | Default checkpoint | Layers | Poolers | Notes |
|---|---|---|---|---|
| `prott5` | `Rostlab/prot_t5_xl_uniref50` | 24 | `none`, `mean` | Whitespace-separated input; U/Z/O/B → X |
| `prostt5` | `Rostlab/ProstT5` | 24 | `none`, `mean` | Structure-aware ProtT5 variant |
| `ankh3` | `ElnaggarLab/ankh3-large` | varies | `none`, `mean` | T5EncoderModel backbone |
| `amplify` | `nvidia/AMPLIFY_120M` | varies | `none`, `mean`, `cls` | Max 2048 residues; bfloat16 on CUDA |
| `proteinglm` | `biomap-research/proteinglm-1b-mlm` | varies | `none`, `mean` | 1B / 3B / 10B variants |
| `esm2` | `facebook/esm2_t33_650M_UR50D` | 6–48 (by size) | `none`, `mean`, `cls` | ESM2 family: 8M to 15B |
| `esm1b` | `esm1b_t33_650M_UR50S` | 33 | `none`, `mean`, `cls` | Loads via torch.hub; `facebook/esm-1b` is an alias |
| `esmc` | `esmc_600m` | read from model config | `none`, `mean`, `cls` | Biohub ESM-C checkpoints through Hugging Face Transformers |

### ESM-2 model sizes

| Model name | Layers | Parameters |
|---|---|---|
| `facebook/esm2_t6_8M_UR50D` | 6 | 8 M |
| `facebook/esm2_t12_35M_UR50D` | 12 | 35 M |
| `facebook/esm2_t30_150M_UR50D` | 30 | 150 M |
| `facebook/esm2_t33_650M_UR50D` | 33 | 650 M |
| `facebook/esm2_t36_3B_UR50D` | 36 | 3 B |
| `facebook/esm2_t48_15B_UR50D` | 48 | 15 B |

### ProtT5 checkpoints

| Model name | Aliases |
|---|---|
| `Rostlab/prot_t5_xl_uniref50` | `prott5_xl_uniref50`, `prot_t5_xl_uniref50`, `prot-t5-xl-uniref50` |
| `Rostlab/prot_t5_xxl_bfd` | `prott5_xxl_bfd`, `prot_t5_xxl_bfd`, `prot-t5-xxl-bfd` |
| `Rostlab/prot_t5_xl_bfd` | `prott5_xl_bfd`, `prot_t5_xl_bfd`, `prot-t5-xl-bfd` |
| `Rostlab/prot_t5_xxl_uniref50` | `prott5_xxl_uniref50`, `prot_t5_xxl_uniref50`, `prot-t5-xxl-uniref50` |
| `Rostlab/prot_t5_xl_half_uniref50-enc` | `prott5_xl_half_uniref50_enc`, `prot_t5_xl_half_uniref50_enc`, `prot-t5-xl-half-uniref50-enc` |

### Model-specific notes

- **prott5 / prostt5**: Input sequences are whitespace-joined (`M K V L ...`); non-standard amino acids U, Z, O, B are replaced with X automatically.
- **AMPLIFY**: On CUDA, defaults to `bfloat16` because its xFormers attention kernels do not support `float32`. Pass `dtype="float16"` explicitly if preferred. Short aliases `amplify_120m` and `amplify_350m` use NVIDIA TransformerEngine-optimized checkpoints (requires `transformer_engine.pytorch`). For CUDA 13: `pip install --no-build-isolation 'transformer-engine[pytorch,core-cu13]==2.16.0'`. For the upstream Chandar Lab checkpoints use `amplify_120m_chandar` / `amplify_350m_chandar`.
- **ProteinGLM**: Short aliases `proteinglm_1b_mlm`, `proteinglm_3b_mlm`, `proteinglm_10b_mlm` resolve to Biomap checkpoints. Trailing EOS token is trimmed automatically.
- **ESM-1b**: The HuggingFace checkpoint (`facebook/esm-1b`) was trained with post-norm but the HF model code uses pre-norm, which degrades accuracy. The generator loads via `torch.hub` by default to avoid this mismatch.
- **ESM-C**: Short aliases `esmc_300m`, `esmc_600m`, and `esmc_6b` resolve to Biohub checkpoints and load through Hugging Face Transformers. Layer indices remain transformer-layer-only (`0..N-1`); the Hugging Face embedding hidden state is not exposed as an ESM-C layer.

## Error Handling

By default, errors on individual proteins are collected and the run continues. Check `result.errors` and `result.skipped` after the run:

```python
result = run_embedding_generation(generator, batcher, writer)

if result.error_count > 0:
    for err in result.errors:
        print(f"ERROR {err['id']}: {err['error']}")

if result.skipped_count > 0:
    for skip in result.skipped:
        print(f"SKIPPED {skip['id']}: {skip['reason']}")
```

To stop immediately on the first error instead:

```python
result = run_embedding_generation(generator, batcher, writer, fail_fast=True)
```

`FastaBatcher` also tracks sequences that were filtered out before inference (e.g. too long, invalid characters):

```python
batcher = FastaBatcher(
    "proteins.fasta",
    max_sequence_length=4_000,
    skipped_path="skipped.tsv",  # optionally write skip log to file
)
# after run:
print(f"{batcher.skipped_count} sequences were filtered before batching")
```

## Exceptions

- `EmbeddingGenerationError`: base embedding exception.
- `EmbeddingInputError`: invalid input records, sequences, or file payloads.
- `EmbeddingDependencyError`: missing optional runtime dependency (e.g. `torch`, `transformers`).
- `EmbeddingBackendError`: backend/model inference failure or unknown model class.
