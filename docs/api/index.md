# API reference

```python
from CBBIO import BioDataClient, Generator, IndexManager

client = BioDataClient(config_path="config.yaml")
generator = Generator(model_class="prott5")
manager = IndexManager(".biodata/indexes")
```

Browse the subsystem pages for the main public interfaces. Use Complete public API when you need an exported symbol that does not fit the primary workflows.

## Exceptions

| Exception | When raised |
|---|---|
| `BioDataError` | BioData client operation fails |
| `EmbeddingGenerationError` | Embedding generation fails |
| `SearchIndexError` | Persistent index operation fails |
