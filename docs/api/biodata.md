# BioData client

```python
from CBBIO import connect

client = connect(config_path="config.yaml")
neighbors = client.find_nearest_neighbors_for_proteins(
    ["AMP1_CAEEL"],
    embedding_type_id=3,
    layer_index=0,
    k=10,
    metric="cosine",
)
```

## Client

::: CBBIO.BioDataClient
    options:
      members: null

## Configuration

::: CBBIO.connect

::: CBBIO.load_config

::: CBBIO.build_dsn

## Types

::: CBBIO.EmbeddingType

::: CBBIO.GOAnnotation

::: CBBIO.Neighbor

## Exceptions

::: CBBIO.BioDataError

::: CBBIO.ConnectionNotOpenError

::: CBBIO.DriverDependencyError

::: CBBIO.NotFoundError
