# CBBIO.BioData

Python helpers for accessing the BioData PostgreSQL database with `pgvector` (`halfvec`) support.

## Install

```bash
pip install -r requirements.txt
```

## Configuration

Connection and client defaults are loaded from [`config.yaml`](config.yaml).

```yaml
database:
  host: localhost
  port: 5432
  name: BioData
  user: usuario
  password: clave

client:
  autocommit: true
  register_halfvec: true

search:
  default_metric: l2
  default_k: 10
```

Environment variable overrides are also supported:

- `BIODATA_DB_HOST`
- `BIODATA_DB_PORT`
- `BIODATA_DB_NAME`
- `BIODATA_DB_USER`
- `BIODATA_DB_PASSWORD`
- `BIODATA_AUTOCOMMIT`
- `BIODATA_REGISTER_HALFVEC`
- `BIODATA_DEFAULT_METRIC`
- `BIODATA_DEFAULT_K`

## Quick Start

```python
from CBBIO.BioData import BioDataClient

with BioDataClient() as db:
    print(db.health_check())
    print("Total sequence embeddings:", db.count_sequence_embeddings())
```

## Example Script

Run the bundled script:

```bash
python examples/nearest_neighbors.py \
  --query P12345 \
  --embedding-type-name esm2_layer0 \
  --layer 0 \
  --k 10 \
  --metric cosine
```

Additional examples:

```bash
python examples/protein_lookup.py --protein-id P12345 --pretty
python examples/protein_lookup.py --accession Q99999 --pretty
python examples/protein_context.py --protein-id P12345 --include-3di --pretty
python examples/structure_drilldown.py --structure-id AF-P12345-F1 --include-3di --pretty
python examples/go_terms.py --obo /path/to/go-basic.obo --go-id GO:0008150 --pretty
python examples/neighbor_semantic_distance.py --obo /path/to/go-basic.obo --protein-ids-file proteins.txt
```

## Improved Search Features

```python
from CBBIO.BioData import BioDataClient

QUERY_UNIPROT = "P12345"  # replace with a valid protein ID

with BioDataClient() as db:
    embedding_type = db.get_embedding_type_by_name("esm2_layer0")
    if embedding_type is None:
        raise ValueError("Embedding type not found")

    layers = db.list_available_layers(embedding_type.id)
    print("available layers:", layers)

    neighbors, annotations = db.neighbors_with_go(
        query_uniprot_id=QUERY_UNIPROT,
        embedding_type_id=embedding_type.id,
        layer_index=layers[0],
        k=10,
        metric="cosine",  # l2 | cosine | inner_product
    )

    for n in neighbors:
        print(n.protein_id, n.distance)
```

## Public API

- `BioDataClient`
- `build_dsn(...)`
- `connect(...)`
- `load_config(...)`
- `health_check(...)`
- `count_sequence_embeddings()`
- `list_embedding_types()`
- `get_protein(...)`
- `get_protein_by_accession(...)`
- `list_accessions_for_protein(...)`
- `get_protein_go_annotations(...)`
- `get_protein_structures(...)`
- `get_structure_chains(...)`
- `get_chain_states(...)`
- `get_state_3di_embeddings(...)`
- `get_protein_context(...)`
- `get_embedding_type_by_name(...)`
- `list_available_layers(...)`
- `get_protein_embedding(...)`
- `distance_to_protein(...)`
- `distance_between_proteins(...)`
- `find_nearest_neighbors(...)`
- `fetch_go_annotations(...)`
- `neighbors_with_go(...)`

### GO Ontology API (`CBBIO.GO`)

- `GOOntology`
- `load_go(...)`
- `read_annotations_tsv(...)`
- `ancestors(...)`
- `descendants(...)`
- `common_ancestors(...)`
- `minimal_branch_length(...)`
- `prepare_term_counts(...)`
- `information_content(...)`
- `semantic_similarity(...)`

### Shared Types (`CBBIO.types`)

- `DistanceMetric`, `SimilarityMethod`, `EmbeddingModel`, `EmbeddingVector`
- `ProteinID`, `GOID`, `StructureID`
- `EmbeddingType`, `Neighbor`, `GOAnnotation`

## Notes

- `pgvector` registration is done automatically on connect (`halfvec`).
- If `pyyaml` is not available, built-in defaults + env vars are still used.
- Set `as_numpy=True` in `get_protein_embedding(...)` if you want a NumPy array.

## Tests

```bash
python -m pytest -q
```

Tests use mocked DB connections, so they do not require a live PostgreSQL server.

## Type Checking

Strict type checking is configured via [`pyrightconfig.json`](pyrightconfig.json).

```bash
poetry run pyright
```
