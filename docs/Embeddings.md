# CBBIO Embedding Generation Documentation

## General Description
`CBBIO.embeddings` provides model-agnostic building blocks for:
- Defining embedding generation pipelines via adapters.
- Loading sequence inputs from FASTA.
- Loading/saving precomputed embeddings (`.pkl/.pickle/.npy/.npz`).
- Capturing reproducibility metadata (model metadata and run metadata).

Model-specific behavior is implemented in separate modules (for example `CBBIO.embeddings_prott5`).

## Exceptions
- `EmbeddingGenerationError`: base embedding exception.
- `EmbeddingInputError`: invalid input records, sequences, or file payloads.
- `EmbeddingDependencyError`: missing optional runtime dependencies for specific paths.
- `EmbeddingBackendError`: backend/model inference failures.

## Core Data Classes

### `GenerationInput`
Normalized generation input record:
- `id: str`
- `sequence: str`
- `description: str | None`
- `metadata: dict[str, Any] | None`

### `EmbeddingRecord`
Normalized generated/loaded embedding record:
- `id: str`
- `embedding: Sequence[float]` (model-specific modules may store structured numeric payloads)
- `layer_index: int`
- `model_reference: str`
- `shape: tuple[int, ...]`
- `metadata: dict[str, Any] | None`

### `ModelMetadata`
Reproducibility metadata tied to a model setup:
- provider/model identifiers
- optional revisions
- tokenizer identifiers
- device/framework versions
- static model parameters

### `RunMetadata`
Reproducibility metadata tied to one generation run:
- run ID
- UTC timestamp
- sequence count
- requested/resolved layers
- failure count
- run parameters

### `GenerationResult`
Generation return object:
- `records: list[EmbeddingRecord]`
- `errors: list[dict[str, Any]]`
- `skipped: list[dict[str, Any]]`
- `model_metadata: ModelMetadata | None`
- `run_metadata: RunMetadata | None`

## Adapter Interfaces

### `PreprocessorAdapter`
`preprocess(raw_sequence: str) -> str`

### `TokenizerAdapter`
`tokenize(sequence: str) -> Any`

### `ModelAdapter`
`infer(tokens: Any, *, layer_index: int = 0) -> Any`

### `PostprocessorAdapter`
`postprocess(model_output: Any) -> Sequence[float]`

## Orchestrator

### `EmbeddingGenerator`
Model-agnostic orchestrator that runs:
1. Input validation
2. preprocessing
3. tokenization
4. model inference
5. postprocessing
6. output normalization into `EmbeddingRecord`

Main method:
- `generate(records, *, layer_index=0, fail_fast=False) -> GenerationResult`

Layer introspection helpers:
- `available_layers() -> list[int]`
- `num_layers() -> int`
- `family_models() -> list[str]`

Notes:
- These methods depend on model-adapter support.
- For T5-family adapters in this repo (`ProtT5`, `ProstT5`), layer indices follow BioData convention:
  `0 = last hidden layer`.

### `Generator(...)`
Convenience factory for model-specific generators.

Examples:
- `Generator(model_class="protT5", name="Rostlab/prot_t5_xl_uniref50")`
- `Generator(class_="prostT5", name="Rostlab/ProstT5")`
- `Generator(model_class="ankh3", name="ElnaggarLab/ankh3-large", prefix="[S2S]")`
- `Generator(**{"class": "protT5", "name": "Rostlab/prot_t5_xl_uniref50"})`

Factory catalog helpers:
- `available_generator_classes() -> list[str]`
- `available_generator_models(model_class=None) -> dict[str, list[str]] | list[str]`

Factory resolution is metadata-driven:
- each model-specific generator publishes `GENERATOR_CLASS` (canonical name),
  `GENERATOR_ALIASES` (accepted aliases), and `FAMILY_MODELS` (known model names).
- `Generator(...)` and catalog helpers consume those published fields.

## Input Loading Helpers

### `load_fasta_inputs(path, *, id_from="record_id")`
Loads FASTA into `list[GenerationInput]`.
- Uses Biopython `SeqIO`.
- Raises `EmbeddingDependencyError` when Biopython is not installed.
- Validates amino-acid sequences.

### `generate_from_fasta(path, generator, *, layer_index=0, fail_fast=False, id_from="record_id")`
Convenience helper:
- `load_fasta_inputs(...)`
- then `generator.generate(...)`

## Embedding I/O Helpers

### `load_embedding_records(path, *, model_reference="unknown", layer_index=0, ids=None)`
Dispatch loader by extension:
- pickle: `.pkl`, `.pickle`
- NumPy: `.npy`, `.npz`

### `load_embedding_records_pickle(...)`
Supported payloads:
- `{"records": [...]}`
- `{id: embedding_vector}`
- `list[dict | EmbeddingRecord]`

### `load_embedding_records_npy(...)`
Loads `.npy` or single-array `.npz` into normalized records.

### `save_embedding_records_pickle(path, records, *, payload_format="records")`
Saves as:
- `payload_format="records"` -> rich records payload
- `payload_format="mapping"` -> `{id: embedding}`

### `save_embedding_records_npy(path, records)`
Saves numeric matrix to `.npy`.
- requires consistent vector lengths.

## Utility

### `utc_now_iso()`
Returns current UTC timestamp in ISO-8601 format.

## Model-Specific Modules
Current model-specific modules:
- `CBBIO.embeddings_prott5`: ProtT5 adapters/generator with separated model/run metadata population.
  - Layer-indexing convention is aligned to BioData: `layer 0 = last hidden layer`.
  - The reversal from Hugging Face hidden-state indices is handled inside the adapter.
- `CBBIO.embeddings_prostt5`: ProstT5 adapters/generator (protein -> embedding only).
  - Uses `<AA2fold>` prefix in preprocessing.
  - Returns per-residue matrices and does not apply pooling.
  - Layer-indexing convention is aligned to BioData: `layer 0 = last hidden layer`.
- `CBBIO.embeddings_ankh3`: ANKH3 adapters/generator (protein -> embedding only).
  - Uses `T5Tokenizer` + `T5EncoderModel`.
  - Supports configurable prefix via generator parameter:
    - `prefix="[NLU]"` (default)
    - `prefix="[S2S]"` (optional)
  - Returns per-residue matrices and does not apply pooling.
- `CBBIO.embeddings_esmc`: ESM-C adapters/generator (protein -> embedding only, non-transformers path).
  - Uses `esm.models.esmc.ESMC` SDK.
  - Supports model family variants (`esmc_300m`, `esmc_600m`, `esmc_6b` and release tags).
  - Supports optional `use_flash_attention` parameter at generator construction.
  - Returns per-residue matrices and does not apply pooling.
- `CBBIO.embeddings_esm2`: ESM2 adapters/generator (protein -> embedding, `esm` pretrained API path).
  - Uses `esm.pretrained.esm2_*` model loaders and alphabet batch converter.
  - Uses ESM2 native layer indexing (top layer is `num_layers`; layer `0` is embedding layer).
  - Returns per-residue matrices and does not apply pooling.
- `CBBIO.embeddings_esm1b`: ESM-1b adapters/generator (protein -> embedding, `esm` pretrained API path).
  - Uses `esm.pretrained.esm1b_t33_650M_UR50S`.
  - Uses ESM-1b native layer indexing (`0..33` for ESM-1b).
  - Returns per-residue matrices and does not apply pooling.

Contributor guide:
- `docs/ModelSpecificEmbeddingModules.md`
