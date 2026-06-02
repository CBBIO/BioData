# Creating Model-Specific Embedding Modules

## Goal
Keep `CBBIO.embeddings` model-agnostic and place model logic in dedicated files such as:
- `CBBIO/embeddings_prott5.py`
- `CBBIO/embeddings_esm2.py` (future)

This keeps dependencies isolated and makes behavior reproducible and reviewable per model family.

## Required Structure

Each model-specific module should:
1. Reuse base interfaces/types from `CBBIO.embeddings`.
2. Implement model-specific adapters:
   - preprocessor
   - tokenizer adapter
   - model adapter
   - postprocessor
3. Provide one concrete generator class that wires those adapters.
4. Populate `ModelMetadata` at initialization.
5. Populate `RunMetadata` per `generate(...)` execution.
6. Publish generator catalog metadata on the generator class:
   - `GENERATOR_CLASS` (canonical factory name)
   - `GENERATOR_ALIASES` (tuple/list of accepted aliases)
   - `FAMILY_MODELS` (tuple/list of known model names in that family)

## Naming Conventions
- File: `embeddings_<modelname>.py`
- Classes:
  - `<ModelName>Preprocessor`
  - `<ModelName>TokenizerAdapter`
  - `<ModelName>ModelAdapter`
  - `<ModelName>Postprocessor`
  - `<ModelName>EmbeddingGenerator`

Example:
- `ProtT5Preprocessor`
- `ProtT5EmbeddingGenerator`

## Reproducibility Rules

### Model metadata (`ModelMetadata`)
Populate once in generator initialization:
- provider (`huggingface-transformers`, etc.)
- model name/reference
- model/tokenizer revision when available
- device
- framework versions (`torch`, `transformers`, etc.)
- static parameters (representation mode, pooling policy, normalization policy)

### Run metadata (`RunMetadata`)
Populate once per `generate(...)` call:
- unique run ID
- UTC timestamp
- sequence count
- requested/resolved layers
- failure count
- run-specific parameters (batch size, precision, etc. if applicable)

Keep model metadata and run metadata separate. Do not mix mutable run values into `ModelMetadata`.

## Layer Indexing Policy
- Model-specific adapters expose native hidden-state order: layer 0 is the
  earliest returned hidden state and larger indices move deeper through the model.
- Record the chosen policy in `ModelMetadata.parameters`.
- Avoid duplicating layer mappings in notebooks or caller code.

## Dependency Handling
- Import optional heavy dependencies inside code paths that need them.
- Raise `EmbeddingDependencyError` with explicit install guidance.
- Do not require model-specific dependencies for model-agnostic helpers.

## Error Handling
- Use embedding exceptions from `CBBIO.embeddings`:
  - `EmbeddingInputError`
  - `EmbeddingDependencyError`
  - `EmbeddingBackendError`
- Wrap unexpected backend failures into normalized embedding errors.

## Output Contract
- Return normalized `EmbeddingRecord` objects in `GenerationResult.records`.
- Fill:
  - `id`
  - `layer_index`
  - `model_reference`
  - `shape`
  - optional record metadata
- Put recoverable per-record failures in `GenerationResult.errors` when `fail_fast=False`.

## Minimal Implementation Checklist
- Add new module in `CBBIO/`.
- Export new generator and adapters from `CBBIO/__init__.py`.
- Add unit tests for:
  - preprocessing rules
  - dependency error paths
  - layer selection behavior
  - metadata population (`ModelMetadata`, `RunMetadata`)
- Update docs:
  - `docs/Embeddings.md`
  - README module list and links

## Skeleton Template

```python
from CBBIO.embeddings import (
    EmbeddingGenerator,
    PreprocessorAdapter,
    TokenizerAdapter,
    ModelAdapter,
    PostprocessorAdapter,
    ModelMetadata,
    RunMetadata,
)

class MyModelPreprocessor(PreprocessorAdapter):
    ...

class MyModelTokenizerAdapter(TokenizerAdapter):
    ...

class MyModelAdapter(ModelAdapter):
    ...

class MyModelPostprocessor(PostprocessorAdapter):
    ...

class MyModelEmbeddingGenerator(EmbeddingGenerator):
    def __init__(self, ...):
        super().__init__(...)
        self.model_metadata = ModelMetadata(...)

    def generate(self, records, *, layer_index=None, fail_fast=False):
        result = super().generate(records, layer_index=..., fail_fast=fail_fast)
        return type(result)(
            records=result.records,
            errors=result.errors,
            skipped=result.skipped,
            model_metadata=self.model_metadata,
            run_metadata=RunMetadata(...),
        )
```
