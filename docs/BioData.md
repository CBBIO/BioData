# CBBIO BioData

`CBBIO.BioData` is a PostgreSQL client for the BioData schema. It handles connection management, protein and sequence lookups, embedding retrieval, nearest-neighbor search across multiple backends, and GO annotation fetching.

```python
from CBBIO import connect

client = connect()
client.health_check()
```

---

## Configuration

### config.yaml

Place `config.yaml` in your working directory (or pass `config_path=`):

```yaml
database:
  host: localhost
  port: 5432
  user: biodata
  password: ""
  database: biodata

search:
  default_metric: cosine   # l2 | cosine | inner_product
  default_k: 10
```

### Environment variable overrides

Any config value can be overridden with a `BIODATA_*` environment variable. These take the highest precedence.

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

Precedence (highest to lowest): env vars → `config.yaml` → built-in defaults.

### Connecting

```python
from CBBIO import connect

# Reads config.yaml + env vars
client = connect()

# Explicit DSN
client = connect(dsn="postgresql://user:password@host:5432/biodata")

# As a context manager (auto-close)
with connect() as client:
    print(client.health_check())
```

---

## Health Check

```python
info = client.health_check()
# {
#   "connected": True,
#   "database": "biodata",
#   "server_version": "16.2",
#   "pgvector_installed": True,
#   "missing_tables": []
# }
```

---

## Proteins and Sequences

### Single protein

```python
protein = client.get_protein("P12345")
# {"id": "P12345", "accession": "...", "length": 255, ...}

# By accession code
protein = client.get_protein_by_accession("P12345")

# All accession codes for a protein
accessions = client.list_accessions_for_protein("P12345")
```

### Sequences

```python
# Single
seq = client.get_protein_sequence("P12345")   # "MKVL..."

# Batch — returns {protein_id: sequence} for found proteins
seqs = client.get_protein_sequences(["P12345", "Q67890"])
```

### Taxonomy and species

```python
species = client.get_protein_species("P12345")       # "Homo sapiens"
tax_id  = client.get_protein_taxonomy_id("P12345")   # "9606"

# Batch
info = client.get_protein_species_taxonomy(["P12345", "Q67890"])
# {"P12345": {"species": "Homo sapiens", "taxonomy_id": "9606"}, ...}
```

### Full context

```python
context = client.get_protein_context("P12345", include_3di=False)
# Returns None if protein not found; otherwise a dict with:
#   protein, accessions, go_annotations,
#   structures, chains_by_structure, states_by_chain
```

---

## Embeddings

### Listing what is available

```python
# All embedding types in the database
embedding_types = client.list_embedding_types()
for et in embedding_types:
    print(et.id, et.name, et.model_name)

# By name
et = client.get_embedding_type_by_name("esm2_t33_650M_UR50D_layer33")

# Available layers for a given type
layers = client.list_available_layers(embedding_type_id=1)
# [0, 12, 24, 33]
```

### Fetching embeddings

```python
# Single protein, returns vector
emb = client.get_protein_embedding("P12345", embedding_type_id=1, layer_index=33)

# As numpy array
emb_np = client.get_protein_embedding(
    "P12345", embedding_type_id=1, layer_index=33, as_numpy=True
)

# Batch — returns {protein_id: embedding}
embs = client.get_protein_embeddings(
    ["P12345", "Q67890"],
    embedding_type_id=1,
    layer_index=33,
    as_numpy=True,
)
```

### Pairwise distances

```python
# Distance between an in-memory vector and a stored protein
d = client.distance_to_protein(
    query_emb, "P12345", model=1, layer_index=33, metric="cosine"
)

# Distance between two stored proteins
d = client.distance_between_proteins("P12345", "Q67890", model=1, layer_index=33)
```

---

## Nearest-Neighbor Search

### Single query

```python
query_emb = client.get_protein_embedding("P12345", 1, 33, as_numpy=True)

neighbors = client.find_nearest_neighbors(
    query_emb,
    embedding_type_id=1,
    layer_index=33,
    k=20,
    metric="cosine",
)

for nb in neighbors:
    print(nb.protein_id, f"distance={nb.distance:.4f}")
```

### Batch query for stored proteins

```python
results = client.find_nearest_neighbors_for_proteins(
    protein_ids=["P12345", "Q67890"],
    embedding_type_id=1,
    layer_index=33,
    k=10,
)
# → {"P12345": [Neighbor(...), ...], "Q67890": [...]}
```

### Batch query for external embeddings

```python
from CBBIO import GenerationInput, Generator, pooler_factory

generator = Generator(model_class="esm2", device="cuda:0")
generated = generator.generate(
    [
        GenerationInput(id="new_protein_1", sequence="MTEYKLVVVG"),
        GenerationInput(id="new_protein_2", sequence="GAGGVGKSAL"),
    ],
    layer_index=33,
    pooler=pooler_factory("mean"),
)

query_embeddings = {
    record.id: record.embedding
    for record in generated.records
}

results = client.find_nearest_neighbors_for_embeddings(
    query_embeddings,
    embedding_type_id=1,
    layer_index=33,
    k=10,
)
# → {"new_protein_1": [Neighbor(...), ...], "new_protein_2": [...]}
```

Pass embeddings generated with the same model, layer, and pooling method as the stored candidate embeddings.

### Search + GO annotations in one call

The most common workflow — find neighbors and their GO terms together:

```python
neighbors, go_by_protein = client.neighbors_with_go(
    query_uniprot_id="P12345",
    embedding_type_id=1,
    layer_index=33,
    k=20,
    metric="cosine",
)

for nb in neighbors:
    go_terms = go_by_protein.get(nb.protein_id, [])
    print(nb.protein_id, nb.distance, [t.go_id for t in go_terms])
```

---

## Search Backends

The `backend` parameter controls which search engine is used. The default is `"auto"`.

| Backend | Description |
|---|---|
| `"auto"` | Chooses based on batch size and hardware availability |
| `"gpu"` | Prefers GPU; falls back gracefully if no GPU |
| `"pgvector"` | PostgreSQL pgvector extension (always available) |
| `"faiss_cpu"` | FAISS on CPU |
| `"faiss_gpu"` | FAISS on GPU (requires faiss-gpu) |
| `"cuvs_gpu"` | cuVS on NVIDIA GPU (requires cuVS) |
| `"torch_gpu"` | Pure PyTorch GPU search |

```python
# Force pgvector (reliable, no extra deps, slower on large sets)
neighbors = client.find_nearest_neighbors(
    query_emb, 1, 33, k=20, backend="pgvector"
)

# GPU with approximate nearest neighbor (ANN) index
neighbors = client.find_nearest_neighbors(
    query_emb, 1, 33, k=20,
    backend="gpu",
    use_ann=True,
    ann_ef_search=200,
)
```

### ANN search

`use_ann=True` enables approximate nearest neighbor indexing. Supported by pgvector (HNSW), FAISS, and cuVS. Torch falls back to exact search.

```python
neighbors = client.find_nearest_neighbors(
    query_emb, 1, 33, k=50,
    use_ann=True,
    ann_ef_search=200,       # HNSW ef_search parameter (higher = more accurate)
)
```

### Search diagnostics

After every search, diagnostics are available on the client:

```python
neighbors = client.find_nearest_neighbors(query_emb, 1, 33, k=20)
print(client.last_search_diagnostics)
# {"backend": "faiss_gpu", "ann_used": False, "device": "cuda:0", ...}
```

---

## GO Annotations

```python
# Single protein
annotations = client.get_protein_go_annotations("P12345")
# [{"go_id": "GO:0006355", "category": "P", "description": "...", ...}, ...]

# Batch — returns {protein_id: [GOAnnotation, ...]}
annotations = client.fetch_go_annotations(["P12345", "Q67890"])

# Just GO IDs per protein
go_ids = client.fetch_protein_go_ids(["P12345", "Q67890"])
# {"P12345": {"GO:0006355", "GO:0003700"}, "Q67890": {...}}

# All GO IDs in the database (no argument)
all_go_ids = client.fetch_protein_go_ids()
```

---

## Structures

```python
structures = client.get_protein_structures("P12345")
chains     = client.get_structure_chains(structures[0]["id"])
states     = client.get_chain_states(chains[0]["id"])

# 3Di embeddings (for structure-based search)
embeddings_3di = client.get_state_3di_embeddings(states[0]["id"])
```

---

## Exceptions

| Exception | When raised |
|---|---|
| `BioDataError` | Base exception for all BioData errors |
| `DriverDependencyError` | Missing runtime dependency (psycopg, pgvector, numpy, pyyaml) |
| `ConnectionNotOpenError` | Operation requires an open connection |
| `NotFoundError` | Expected protein or embedding does not exist in the database |

```python
from CBBIO import BioDataError, NotFoundError

try:
    emb = client.get_protein_embedding("UNKNOWN", 1, 0)
except NotFoundError:
    print("Protein not in database")
```

---

## Raw SQL Helpers

For queries not covered by the high-level API:

```python
# Returns List[Dict[str, Any]]
rows = client.query_all("SELECT id, accession FROM protein WHERE length > %s", [500])

# Returns one Dict or None
row = client.query_one("SELECT * FROM protein WHERE id = %s", ["P12345"])

# Returns a scalar value or None
count = client.scalar("SELECT COUNT(*) FROM sequence_embeddings")
```
