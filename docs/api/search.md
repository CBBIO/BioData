# Persistent search

```python
from CBBIO import IndexKey, IndexManager

manager = IndexManager(".biodata/indexes")
key = IndexKey(
    database_label="biodata-local",
    embedding_type_id=3,
    layer_index=0,
    dimension=1024,
    metric="cosine",
)
```

## Index manager

::: CBBIO.IndexManager
    options:
      members: null

## Index types

::: CBBIO.IndexKey

::: CBBIO.IndexBuildSpec

::: CBBIO.IndexArtifact

::: CBBIO.IndexManifest

::: CBBIO.IndexInspection

::: CBBIO.IndexCandidate

::: CBBIO.ExactStoreArtifact

::: CBBIO.ExactStoreManifest

::: CBBIO.ExactStoreInspection

## Exceptions

::: CBBIO.SearchIndexError

::: CBBIO.SearchIndexNotFoundError

::: CBBIO.SearchIndexStaleError
