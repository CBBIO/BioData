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

## Build artifacts

```python
from CBBIO import IndexBuildSpec, IndexKey, IndexManager, connect

client = connect(config_path="config.yaml")
manager = IndexManager(".biodata/indexes", search_nprobe=256)
key = IndexKey.from_biodata(
    client,
    database_label="biodata-nas",
    embedding_type_id=3,
    layer_index=0,
    metric="cosine",
)
revision = client.embedding_index_revision(3, 0)

manager.build_ivf_pq(
    key,
    lambda: client.iter_protein_embedding_index_batches(3, 0, batch_size=10_000),
    source_revision=revision,
    spec=IndexBuildSpec(nlist=3475, subquantizers=128, nprobe=256),
)
```

`build_ivf_pq()` reuses a current exact store. If it is absent, it invokes the callback once to
write the shared `float16` matrix and SQLite metadata, then derives IVF-PQ from local files.

| Exact-store state | Default behavior |
|---|---|
| Current | Reuse local files; do not invoke the callback |
| Missing | Materialize local files from the callback |
| Stale or invalid | Raise; pass `overwrite=True` with a callback to replace it |

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
