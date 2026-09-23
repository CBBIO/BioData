# Embeddings

```python
from CBBIO import GenerationInput, Generator

generator = Generator(model_class="prott5")
query = GenerationInput(id="AMP1_CAEEL", sequence="MKT")
embedding = generator.generate(query)
```

## Generation

::: CBBIO.EmbeddingGenerator
    options:
      members: true

::: CBBIO.Generator

::: CBBIO.run_embedding_generation

::: CBBIO.GenerationInput

::: CBBIO.GenerationResult

::: CBBIO.EmbeddingRecord

## Input and output

::: CBBIO.FastaBatcher

::: CBBIO.IterableBatcher

::: CBBIO.EmbeddingWriter
    options:
      members: true

::: CBBIO.H5EmbeddingReader
    options:
      members: true

## Exceptions

::: CBBIO.EmbeddingGenerationError

::: CBBIO.EmbeddingInputError

::: CBBIO.EmbeddingDependencyError

::: CBBIO.EmbeddingBackendError
