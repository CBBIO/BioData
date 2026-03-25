# BioData

Utilities for working with a PostgreSQL BioData database, protein embeddings (`pgvector`), and GO ontology analysis.

## What Is In This Repo

- `CBBIO/BioData.py`: database client (`BioDataClient`) for proteins, embeddings, nearest neighbors, and GO annotations.
- `CBBIO/GO.py`: ontology utilities (`GOOntology`) built on `goatools`.
- `CBBIO/Taxonomy.py`: taxonomy utilities (`TaxonomyOntology`) for NCBI taxdump lineage/LCA/IC analysis.
- `CBBIO/embeddings.py`: model-agnostic embedding generation interfaces, I/O helpers, and metadata types.
- `CBBIO/embeddings_prott5.py`: ProtT5-specific embedding adapters/generator.
- `schema.sql`: database schema.
- `config.yaml`: default DB/client/search configuration.
- `notebooks/`: runnable examples:
  - `protein_lookup.ipynb`
  - `nearest_neighbors.ipynb`
  - `go_terms.ipynb`
  - `taxonomy_clade_examples.ipynb`
  - `sequence_to_neighbors_prott5.ipynb`
  - `distance_vs_semantic_similarity.ipynb`

## Installation

### Private GitHub install (recommended)

Install only the `CBBIO` package and runtime dependencies:

```bash
pip install "biodata @ git+ssh://git@github.com/cbbio/BioData.git@main"
```

Optional extras:

```bash
pip install "biodata[notebooks] @ git+ssh://git@github.com/cbbio/BioData.git@main"
pip install "biodata[dev] @ git+ssh://git@github.com/cbbio/BioData.git@main"
```

### Local repository install

```bash
pip install .
```

With extras:

```bash
pip install ".[notebooks]"
pip install ".[dev]"
```

### Requirements files

- Runtime only: `requirements.txt`
- Dev only: `requirements-dev.txt`
- Notebook only: `requirements-notebooks.txt`

## Database Installation

### 1) Start PostgreSQL + pgvector

```bash
docker run -d --name pgvectorsql \
  -e POSTGRES_USER=usuario \
  -e POSTGRES_PASSWORD=clave \
  -e POSTGRES_DB=BioData \
  -p 5432:5432 \
  pgvector/pgvector:pg16
```

### 2) Create extension

```bash
PGPASSWORD=clave psql -h localhost -U usuario -d BioData -c "CREATE EXTENSION IF NOT EXISTS vector;"
```

### 3) Load schema

```bash
PGPASSWORD=clave psql -h localhost -U usuario -d BioData -f schema.sql
```

### 4) Optional: restore full dataset from backup

If you have a `.backup` dump:

```bash
PGPASSWORD=clave pg_restore -h localhost -U usuario -d BioData /path/to/BioData.backup
```

## Configuration

Defaults are in `config.yaml`:

```yaml
database:
  host: localhost
  port: 5432
  name: BioData
  user: usuario
  password: clave
```

Environment overrides are supported:

- `BIODATA_DB_HOST`
- `BIODATA_DB_PORT`
- `BIODATA_DB_NAME`
- `BIODATA_DB_USER`
- `BIODATA_DB_PASSWORD`
- `BIODATA_AUTOCOMMIT`
- `BIODATA_REGISTER_HALFVEC`
- `BIODATA_DEFAULT_METRIC`
- `BIODATA_DEFAULT_K`

## CBBIO Modules

### `CBBIO.BioData`

Main entrypoint:

- `BioDataClient`

Typical usage:

```python
from CBBIO.BioData import BioDataClient

with BioDataClient() as client:
    print(client.health_check())
    print(client.count_sequence_embeddings())
```

Key capabilities:

- DB helpers: `query_one`, `query_all`, `scalar`, `transaction`
- Protein/context access: `get_protein`, `get_protein_by_accession`, `get_protein_context`
- Sequence/metadata batch fetch: `get_protein_sequences`, `get_protein_species_taxonomy`
- Embeddings and distances: `get_protein_embedding`, `distance_to_protein`, `distance_between_proteins`
- Neighbor search: `find_nearest_neighbors`, `find_nearest_neighbors_for_proteins`, `neighbors_with_go`
- GO annotation retrieval: `fetch_go_annotations`, `fetch_protein_go_ids`

Detailed API reference: `docs/BioData.md`

### `CBBIO.GO`

Main entrypoints:

- `GOOntology`
- `load_go`

Typical usage:

```python
from CBBIO.GO import load_go

go = load_go("go-basic.obo")
```

Key capabilities:

- Term navigation: `term`, `ancestors`, `descendants`, `common_ancestors`
- Relationship helpers: `direct_parents`, `direct_children`, `find_relation`, `best_relation_matches`, `best_relation_map`
- IC and semantic similarity: `prepare_term_counts`, `information_content`, `semantic_similarity`
- Group comparison: `group_similarity` (BMA)
- Category splitting: `split_annotations_by_category` (`mf`, `bp`, `cc`)

Detailed API reference: `docs/GO.md`

### `CBBIO.Taxonomy`

Main entrypoints:

- `TaxonomyOntology`
- `load_taxonomy`

Typical usage:

```python
from CBBIO.Taxonomy import load_taxonomy

tax = load_taxonomy("/path/to/taxdump")
```

Key capabilities:

- Taxon navigation: `taxon`, `lineage`, `ancestors`, `descendants`, `common_ancestors`
- LCA and branch distances: `lowest_common_ancestor`, `minimal_branch_length`
- Taxon IC and similarity: `prepare_taxon_counts`, `information_content`, `lin_similarity` (`observed`, `whole_db`, `subtree`)
- Taxon ID normalization and reusable feature maps: `normalize_taxonomy_id`, `compute_taxon_ic_and_lin_maps`
- Taxonomy annotation reader: `read_taxonomy_annotations_tsv`

Detailed API reference: `docs/Taxonomy.md`

### `CBBIO.embeddings`

Main entrypoints:

- `EmbeddingGenerator`
- `load_fasta_inputs`
- `generate_from_fasta`
- `load_embedding_records` / `save_embedding_records_*`

Key capabilities:

- Model-agnostic adapter interfaces (`PreprocessorAdapter`, `TokenizerAdapter`, `ModelAdapter`, `PostprocessorAdapter`)
- Embedding record I/O (`.pkl/.pickle/.npy/.npz`)
- Reproducibility metadata types (`ModelMetadata`, `RunMetadata`)

Detailed API reference: `docs/Embeddings.md`

Model-specific extension guide: `docs/ModelSpecificEmbeddingModules.md`

## Notebook Examples

All notebooks are under `notebooks/` and include a top setup cell for notebook-only packages when needed.

### `protein_lookup.ipynb`

- Connect to DB
- Fetch protein records and context by protein ID/accession

### `nearest_neighbors.ipynb`

- Resolve embedding type/layer
- Find nearest neighbors and inspect GO annotations

### `go_terms.ipynb`

- Load GO DAG
- Build GO statistics and category-level views

### `taxonomy_clade_examples.ipynb`

- Load NCBI taxonomy (`nodes.dmp` + `names.dmp`)
- Compute LCA ID/rank/clade for a taxon pair
- Check same-clade membership at target ranks (genus/family/order)
- Compute normalized LCA depth, Wu-Palmer taxonomy similarity, taxon IC, and Lin similarity

### `sequence_to_neighbors_prott5.ipynb`

- Generate a ProtT5 embedding from a raw sequence
- Choose pooling strategy on the notebook side (mean/max)
- Query nearest neighbors in BioData and inspect GO annotations

### `distance_vs_semantic_similarity.ipynb`

- Sample random proteins from DB
- Compute query-neighbor distances
- Compute semantic similarity by GO category
- Compute sequence identity
- Build tables and density plots (with linear fit and `R²`)

## Running Tests

```bash
poetry run pytest -q
```

### Integration Tests (Real DB)

Integration tests are in `tests/test_biodata_integration.py` and run against a real PostgreSQL instance only when config is available.

Setup:

```bash
cp config_test.yaml.example config_test.yaml
```

Edit `config_test.yaml` with your test DB credentials.

Run only integration tests:

```bash
poetry run pytest -q tests/test_biodata_integration.py
```

Notes:
- If DB is unavailable or schema is missing, integration tests are skipped with instructions.
- You can override config path with `BIODATA_TEST_CONFIG=/path/to/config_test.yaml`.

### Coverage

```bash
poetry run pytest --cov=CBBIO --cov-report=term-missing -q
```

## Type Checking

```bash
poetry run pyright
```
