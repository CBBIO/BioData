# BioData

Utilities for working with a PostgreSQL BioData database, protein language model embeddings (`pgvector`), residue-level probing tasks, and GO/Taxonomy analysis.

## What Is In This Repo

| Path | Description |
|---|---|
| `CBBIO/BioData.py` | PostgreSQL client (`BioDataClient`) for proteins, embeddings, neighbor search, and GO annotations |
| `CBBIO/embeddings/` | Model-agnostic embedding generation: adapters, batching, pooling, I/O |
| `CBBIO/probing/` | Supervised probing tasks, dataset catalog, residue/protein-level benchmarks |
| `CBBIO/search/` | Search backend routing (pgvector, FAISS, cuVS, Torch GPU) |
| `CBBIO/GO.py` | GO ontology utilities (`GOOntology`) built on `goatools` |
| `CBBIO/Taxonomy.py` | Taxonomy utilities (`TaxonomyOntology`) for NCBI taxdump |
| `CBBIO/similarity.py` | Pairwise sequence alignment via `parasail` |
| `sql/schema.sql` | Database schema |
| `config.yaml.example` | Database and search configuration template |
| `docs/` | Full documentation for all modules |
| `notebooks/data/` | Small notebook input assets tracked with the repo |
| `tests/assets/` | Test configuration templates and fixtures |

---

## Quick Start

```python
from CBBIO import connect, Generator, FastaBatcher, EmbeddingWriter, run_embedding_generation

# Connect to the database
client = connect()
client.health_check()

# Generate embeddings from a FASTA file
generator = Generator(model_class="esm2", device="cuda:0")
batcher   = FastaBatcher("proteins.fasta", max_batch_tokens=32_768)
writer    = EmbeddingWriter(format="h5", path="embeddings.h5")
run_embedding_generation(generator, batcher, writer, layer_index=33)

# Nearest-neighbor search
neighbors = client.find_nearest_neighbors(query_emb, embedding_type_id=1, layer_index=33, k=20)
```

See [docs/GettingStarted.md](docs/GettingStarted.md) for a full tutorial covering all modules.

---

## Installation

### From GitHub

```bash
pip install "biodata @ git+ssh://git@github.com/cbbio/BioData.git@main"
```

Optional extras:

```bash
pip install "biodata[notebooks] @ git+ssh://git@github.com/cbbio/BioData.git@main"
pip install "biodata[dev] @ git+ssh://git@github.com/cbbio/BioData.git@main"
```

### From a local clone

```bash
pip install .           # runtime only
pip install ".[dev]"    # with dev tools
```

---

## Database Setup

### 1. Start PostgreSQL + pgvector

```bash
docker run -d --name pgvectorsql \
  -e POSTGRES_USER=biodata \
  -e POSTGRES_PASSWORD=secret \
  -e POSTGRES_DB=biodata \
  -p 5432:5432 \
  pgvector/pgvector:pg16
```

### 2. Create the extension and schema

```bash
PGPASSWORD=secret psql -h localhost -U biodata -d biodata \
  -c "CREATE EXTENSION IF NOT EXISTS vector;"

PGPASSWORD=secret psql -h localhost -U biodata -d biodata -f sql/schema.sql
```

### 3. Restore a dataset (optional)

```bash
PGPASSWORD=secret pg_restore -h localhost -U biodata -d biodata /path/to/BioData.backup
```

---

## Configuration

Copy the untracked template to `config.yaml` (or pass `config_path=` to `connect()`):

```bash
cp config.yaml.example config.yaml
```

```yaml
database:
  host: localhost
  port: 5432
  user: biodata
  password: ""
  name: biodata

search:
  default_metric: cosine    # l2 | cosine | inner_product
  default_k: 10
```

Environment variable overrides (highest precedence):

| Variable | Overrides |
|---|---|
| `BIODATA_DB_HOST` | `database.host` |
| `BIODATA_DB_PORT` | `database.port` |
| `BIODATA_DB_USER` | `database.user` |
| `BIODATA_DB_PASSWORD` | `database.password` |
| `BIODATA_DB_NAME` | `database.database` |
| `BIODATA_AUTOCOMMIT` | `client.autocommit` |
| `BIODATA_SEARCH_METRIC` | `search.default_metric` |
| `BIODATA_SEARCH_K` | `search.default_k` |

---

## Documentation

Browse the published [API reference on Read the Docs](https://biodata.readthedocs.io/en/latest/).

| Topic | Document |
|---|---|
| Agent quick start | [docs/AgentQuickStart.md](docs/AgentQuickStart.md) |
| Agent workflow recipes | [docs/AgentWorkflows.md](docs/AgentWorkflows.md) |
| Agent public API map | [docs/AgentApiMap.md](docs/AgentApiMap.md) |
| Agent example prompts | [docs/AgentExamplePrompts.md](docs/AgentExamplePrompts.md) |
| Agent code examples | [docs/examples/README.md](docs/examples/README.md) |
| Getting started tutorial | [docs/GettingStarted.md](docs/GettingStarted.md) |
| Search benchmarks | [docs/Benchmarks.md](docs/Benchmarks.md) |
| Embedding generation | [docs/Embeddings.md](docs/Embeddings.md) |
| Probing tasks and benchmarks | [docs/Probing.md](docs/Probing.md) |
| Database client and search | [docs/BioData.md](docs/BioData.md) |
| GO ontology and similarity | [docs/GO.md](docs/GO.md) |
| Taxonomy | [docs/Taxonomy.md](docs/Taxonomy.md) |
| Sequence alignment | [docs/Similarity.md](docs/Similarity.md) |
| Coding style guide | [docs/STYLE.md](docs/STYLE.md) |
| Documentation style guide | [docs/STYLE_DOCS.md](docs/STYLE_DOCS.md) |
| Test style guide | [docs/STYLE_TESTS.md](docs/STYLE_TESTS.md) |
| Versioning and branching guide | [docs/STYLE_VERSIONING.md](docs/STYLE_VERSIONING.md) |

---

## Running Tests

```bash
poetry run pytest -q
```

### Integration tests (real DB required)

```bash
cp tests/assets/config_test.yaml.example config_test.yaml
# edit with your test DB credentials
poetry run pytest -q tests/test_biodata_integration.py
```

Override config path: `BIODATA_TEST_CONFIG=/path/to/config_test.yaml`.

### Coverage

```bash
poetry run pytest --cov=CBBIO --cov-report=term-missing -q
```

### Type checking

```bash
poetry run pyright
```
