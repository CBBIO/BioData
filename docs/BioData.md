# CBBIO/BioData.py Documentation

## General Description
`CBBIO.BioData` is a PostgreSQL client layer specialized for the BioData schema and `pgvector`-based embedding workflows. It provides:
- Config loading and override logic (`config.yaml` + `BIODATA_*` environment variables).
- Connection management (`psycopg`, optional `pgvector` registration).
- Generic SQL helpers (`query_one`, `query_all`, `scalar`).
- Domain helpers for proteins, sequences, structures, GO annotations, and embedding search.
- Similarity and nearest-neighbor operations using pgvector distance operators.

The central class is `BioDataClient`, which manages an open DB connection and exposes high-level retrieval/search APIs.

Search implementation note:
- `BioDataClient` is the public entrypoint and facade.
- Neighbor-search routing and backend implementations are delegated to the internal `CBBIO/search/` package.
- This split keeps DB access and domain lookups in `BioData.py` while isolating search-specific logic such as pgvector SQL, backend routing, GPU residency, and diagnostics.

## Exceptions
- `BioDataError`: base module exception.
- `DriverDependencyError`: missing optional runtime dependency (`psycopg`, `pgvector`, `pyyaml`, or `numpy` depending on path).
- `ConnectionNotOpenError`: operation requires an active connection.
- `NotFoundError`: expected DB entity/embedding was not found.

## Configuration and Constants
- `DEFAULT_CONFIG_PATH`: default path to `config.yaml`.
- `REQUIRED_TABLES`: core schema tables checked by `health_check`.
- Default connection/search constants (`DEFAULT_HOST`, `DEFAULT_DSN`, `DEFAULT_SEARCH_METRIC`, etc.) are derived from loaded config.

## Module-Level Functions

### `_default_config_dict()`
Returns hardcoded default config structure with `database`, `client`, and `search` sections.

### `_merge_dicts(base, override)`
Recursively merges two mappings.
- Nested mappings are merged recursively.
- Non-mapping values in `override` replace `base`.

### `_to_config_dict(value)`
Normalizes arbitrary mapping-like inputs into a plain `dict[str, Any]`.
- Returns `{}` for non-mapping values.

### `_parse_bool(value, *, env_name)`
Parses environment-style booleans.
- Truthy: `1,true,yes,on`
- Falsey: `0,false,no,off`
- Raises `BioDataError` on invalid content.

### `_apply_env_overrides(config)`
Applies `BIODATA_*` environment variable overrides to an existing config mapping.
- Supported keys include DB credentials, autocommit/register flags, default metric, and default `k`.

### `load_config(config_path=None, *, strict=False)`
Loads configuration by merging defaults + YAML + env overrides.
- If YAML exists and `pyyaml` is unavailable:
  - raises `DriverDependencyError` only when `strict=True`.
  - otherwise silently falls back to defaults.
- Validates YAML root type (must be mapping).

### `_config_defaults(config)`
Flattens validated config into runtime defaults used by the module/client.
- Computes DSN.
- Validates `default_metric` (`l2`, `cosine`, `inner_product`).
- Raises `BioDataError` on invalid metric.

### `build_dsn(user=..., password=..., host=..., port=..., database=...)`
Builds PostgreSQL DSN string.

### `connect(dsn=None, *, autocommit=None, register_halfvec=None, config_path=None)`
Convenience function to instantiate `BioDataClient`, connect immediately, and return it.

### `_as_optional_str(value)`
Internal converter returning `str(value)` unless `value is None`.

### `_metric_operator(metric)`
Maps metric names to pgvector operators.
- `l2 -> <->`
- `cosine -> <=>`
- `inner_product -> <#>`
- Raises `BioDataError` on unsupported metrics.

### `_embedding_dimension(query_embedding)`
Infers vector dimension for ANN queries.
- Tries `.dimensions()` method, then `.to_list()`, then `len(...)`.
- Raises `BioDataError` if inference fails or dimension `< 1`.

### `_row_to_dict(row, cursor)`
Normalizes row objects to `dict[str, Any]` across different cursor row types.
- Supports dict-like rows, key-indexed rows, or tuple rows via cursor description.

### `_cursor(conn)`
Internal context manager yielding `conn.cursor()` and ensuring cursor close in `finally`.

## Class: `BioDataClient`

### `__init__(dsn=None, *, autocommit=None, register_halfvec=None, config_path=None)`
Initializes client settings from config defaults plus explicit overrides.
- Does not open DB connection.
- Stores runtime defaults for search metric and neighbor count.

### `__enter__()`
Context-manager entry; opens connection and returns `self`.

### `__exit__(exc_type, exc, tb)`
Context-manager exit; closes connection.

### `is_connected` (property)
Returns `True` if a DB connection object exists.

### `connect()`
Opens `psycopg` connection and optionally registers pgvector support.
- Idempotent if already connected.
- Raises `DriverDependencyError` when required packages are missing.

### `close()`
Closes active DB connection if present.

### `transaction()`
Context manager for transaction boundaries.
- On success: commits when `autocommit=False`.
- On exception: rolls back when `autocommit=False`, then re-raises.

### `query_all(sql, params=None)`
Executes SQL and returns all rows as `list[dict[str, Any]]`.
- Requires open connection (`ConnectionNotOpenError` otherwise).

### `query_one(sql, params=None)`
Executes SQL and returns one row as `dict`, or `None` when no result.

### `scalar(sql, params=None)`
Returns first column of first row, or `None` if query has no rows.

### `health_check(*, check_extension=True, check_required_tables=True)`
Returns health metadata dictionary.
- Always includes connection state, current DB name, and server version.
- Optionally checks whether `pgvector` extension is installed.
- Optionally reports missing tables from `REQUIRED_TABLES`.

### `count_sequence_embeddings()`
Returns total row count in `sequence_embeddings`.

### `list_embedding_types()`
Returns all rows from `sequence_embedding_type` as `EmbeddingType` objects.

### `get_protein(protein_id)`
Fetches one protein row by `protein.id`.
- Returns detailed protein fields or `None` if missing.

### `get_protein_by_accession(accession_code)`
Fetches protein metadata by accession code (`accession.code`).
- Includes accession metadata (`is_primary_accession`, `accession_tag`).

### `list_accessions_for_protein(protein_id)`
Lists all accession codes for a protein, ordered with primary first.

### `get_protein_go_annotations(protein_id)`
Returns GO annotations for a protein, joined with GO term category/description.

### `get_protein_sequence(protein_id)`
Returns amino-acid sequence string for protein, or `None` if unavailable.

### `get_protein_species(protein_id)`
Returns organism/species string, or `None`.

### `get_protein_taxonomy_id(protein_id)`
Returns taxonomy ID as string, or `None`.

### `get_protein_species_taxonomy(protein_ids)`
Batch fetch of species + taxonomy.
- Returns mapping: `protein_id -> {"species": str|None, "taxonomy_id": str|None}`.
- Returns `{}` for empty input.

### `get_protein_sequences(protein_ids)`
Batch fetch of protein sequences.
- Returns mapping `protein_id -> sequence`.
- Omits entries with missing IDs/sequences.

### `get_protein_structures(protein_id)`
Returns structures associated with a protein (`structure` table).

### `get_structure_chains(structure_id)`
Returns chain rows for one structure.

### `get_chain_states(chain_id)`
Returns state rows for one chain.

### `get_state_3di_embeddings(state_id)`
Returns `structure_3di` rows for one state.

### `get_protein_context(protein_id, *, include_3di=False)`
High-level context aggregator for one protein.
- Includes protein row, accession list, GO annotations, structures.
- Also builds:
  - `chains_by_structure`
  - `states_by_chain`
  - optionally `structure_3di_by_state` when `include_3di=True`
- Returns `None` if protein does not exist.

### `get_embedding_type_by_name(name)`
Fetches one embedding type by exact `sequence_embedding_type.name`.
- Returns `EmbeddingType` or `None`.

### `distance_to_protein(query_embedding, protein_id, model, layer_index=0, metric=None)`
Computes distance between an in-memory query embedding and a stored protein embedding.
- `model` can be embedding type ID (`int`) or name (`str`).
- Raises `NotFoundError` if target embedding is missing.

### `distance_between_proteins(protein_a_id, protein_b_id, model, layer_index=0, metric=None)`
Computes distance between two proteins at same model/layer.
- Raises `NotFoundError` if either embedding is missing.

### `list_available_layers(embedding_type_id)`
Returns sorted distinct `layer_index` values available for a given embedding type.

### `get_protein_embedding(uniprot_id, embedding_type_id, layer_index=0, *, as_numpy=False)`
Fetches one embedding vector for a protein.
- Returns:
  - raw vector object by default,
  - `numpy.float32` array when `as_numpy=True`,
  - `None` when not found.
- Raises `DriverDependencyError` if `numpy` is required but missing.

### `get_protein_embeddings(protein_ids, embedding_type_id, layer_index=0, *, as_numpy=False)`
Batch version of `get_protein_embedding`.
- Returns `dict[protein_id, embedding]` for found embeddings.
- Returns `{}` for empty input.
- Raises `DriverDependencyError` when `as_numpy=True` without NumPy.

### `find_nearest_neighbors(query_embedding, embedding_type_id, layer_index=0, k=None, *, metric=None, exclude_protein_ids=None, use_ann=False, ann_ef_search=200, ann_candidate_pool=None, backend=None, device=None)`
Nearest-neighbor search with backend routing.
- Supports `backend="auto"|"gpu"|"pgvector"|"faiss_cpu"|"faiss_gpu"|"cuvs_gpu"|"torch_gpu"`.
- `auto` uses static heuristics based on batch size, hardware availability, and resident GPU index state.
- `gpu` prefers accelerated backends and falls back cleanly when unavailable.
- `use_ann=True` is supported by pgvector, FAISS, and cuVS; Torch fallback degrades to exact search.
- Applies metric operator from `_metric_operator`.
- Supports excluding specific protein IDs.
- pgvector ANN mode details:
  - uses inferred embedding dimension for halfvec cast,
  - candidate pool defaults to `max(k*20, 200)` when not set,
  - can set session-level `hnsw.ef_search`.
- Returns list of `Neighbor` dataclass values.
- Diagnostics for the last routed search are exposed via `client.last_search_diagnostics`.
- Raises `BioDataError` if `k < 1`.

Internal implementation:
- public method on `BioDataClient`
- delegated to `CBBIO/search/service.py`
- backend-specific helpers are isolated from the main client module

### `find_nearest_neighbors_for_proteins(protein_ids, embedding_type_id, layer_index=0, k=None, *, metric=None, include_query=False, backend=None, device=None)`
Batch nearest-neighbor search for multiple query proteins.
- Supports the same backend routing options as `find_nearest_neighbors(...)`.
- Uses pgvector SQL for the PostgreSQL path and exact GPU search for FAISS/Torch paths.
- Returns mapping: `query_protein_id -> list[Neighbor]`.
- Query proteins without embeddings are omitted.
- Diagnostics for the last routed search are exposed via `client.last_search_diagnostics`.
- Raises `BioDataError` if `k < 1`.

Internal implementation:
- public method on `BioDataClient`
- delegated to `CBBIO/search/service.py`

### `fetch_go_annotations(protein_ids)`
Batch fetch GO annotations for proteins.
- Returns mapping `protein_id -> list[GOAnnotation]`.
- Returns `{}` for empty input.

### `fetch_protein_go_ids(protein_ids=None)`
Fetches only GO IDs grouped by protein.
- If `protein_ids is None`, scans all annotation rows.
- Returns mapping `protein_id -> set[go_id]`.

### `neighbors_with_go(query_uniprot_id, embedding_type_id, layer_index=0, k=None, *, metric=None, include_query=False, use_ann=False, backend=None, device=None)`
End-to-end helper pipeline:
1. Load query protein embedding.
2. Run nearest-neighbor search.
3. Fetch GO annotations for returned neighbors.
- Returns tuple: `(neighbors, annotations_by_protein)`.
- Raises `NotFoundError` when query embedding does not exist.

### `_resolve_embedding_type_id(model)`
Internal helper resolving `EmbeddingModel` input to numeric embedding type ID.
- Accepts int IDs directly.
- For strings, lookup by embedding type name.
- Raises:
  - `BioDataError` for empty names.
  - `NotFoundError` when name lookup fails.

### `_require_connection()`
Internal guard requiring an active connection.
- Raises `ConnectionNotOpenError` when disconnected.
