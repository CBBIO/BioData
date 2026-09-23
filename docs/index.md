# BioData API

```python
from CBBIO import connect

client = connect(config_path="config.yaml")
```

BioData provides typed APIs for protein embeddings, probing, ontology analysis, sequence similarity, and nearest-neighbor search.

Use the API reference to inspect signatures, return types, documented errors, and public methods. Read the repository guides when you need a complete workflow or worked example.

## Repository guides

- [Project README](https://github.com/CBBIO/BioData#readme)
- [Examples README](https://github.com/CBBIO/BioData/blob/main/docs/examples/README.md)

## Exceptions

| Exception | When raised |
|---|---|
| `BioDataError` | BioData client operation fails |
| `EmbeddingGenerationError` | Embedding generation fails |
