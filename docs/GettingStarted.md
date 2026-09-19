# Getting Started with CBBIO

This guide walks you through the most common workflows in roughly 30 minutes. Each section links to a deeper reference document once you are ready for it.

---

## Installation

### From GitHub (recommended)

```bash
pip install "biodata @ git+ssh://git@github.com/cbbio/BioData.git@main"
```

Optional extras:

```bash
# Jupyter notebooks
pip install "biodata[notebooks] @ git+ssh://git@github.com/cbbio/BioData.git@main"

# Development tools (tests, linting)
pip install "biodata[dev] @ git+ssh://git@github.com/cbbio/BioData.git@main"
```

### From a local clone

```bash
git clone git@github.com:cbbio/BioData.git
cd BioData
pip install .
```

---

## 1. Generating Embeddings from a FASTA File

The embedding system needs three objects: a **generator** (model wrapper), a **batcher** (reads your FASTA in chunks), and a **writer** (persists results).

```python
from CBBIO import (
    EmbeddingWriter,
    FastaBatcher,
    Generator,
    run_embedding_generation,
)

# Load ESM-2 (650M) on GPU. Omit device= to run on CPU.
generator = Generator(model_class="esm2", device="cuda:0")

# Stream sequences from FASTA, cap padding budget to 32 768 tokens per batch
batcher = FastaBatcher(
    "proteins.fasta",
    batch_size=None,
    max_batch_tokens=32_768,
    max_sequence_length=4_000,
)

# Write per-residue embeddings to an HDF5 file
writer = EmbeddingWriter(format="h5", path="embeddings.h5")

result = run_embedding_generation(generator, batcher, writer)

print(f"Embedded {result.record_count} proteins")
print(f"Skipped {result.skipped_count} (too long or invalid)")
```

To extract a specific layer (or multiple layers) instead of the default:

```python
result = run_embedding_generation(
    generator, batcher, writer,
    layer_index=[0, 16, 33],   # layers to save; None saves all
)
```

To get one mean-pooled vector per protein instead of per-residue matrices:

```python
from CBBIO import pooler_factory

result = run_embedding_generation(
    generator, batcher, writer,
    layer_index=33,
    pooler=pooler_factory("mean"),
)
```

For a complete reference on models, poolers, and output formats see [Embeddings.md](Embeddings.md).

---

## 2. Connecting to the BioData Database

Copy `config.yaml.example` to `config.yaml`, then set your local credentials:

```bash
cp config.yaml.example config.yaml
```

The minimal file:

```yaml
database:
  host: localhost
  port: 5432
  user: biodata
  password: ""
  name: biodata
```

You can also override any value with an environment variable:

```bash
export BIODATA_DB_HOST=db.example.com
export BIODATA_DB_PASSWORD=secret
```

Connect in Python:

```python
from CBBIO import connect

client = connect()            # reads config.yaml + env vars
client.health_check()         # raises if the DB is unreachable
```

Or pass a DSN directly:

```python
from CBBIO import connect

client = connect(dsn="postgresql://biodata:secret@localhost:5432/biodata")
```

---

## 3. Fetching Proteins and Sequences

```python
from CBBIO import connect

client = connect()

# Look up by internal ID
protein = client.get_protein("P12345")
print(protein["accession"], protein["length"])

# Fetch sequences for a list of IDs at once
sequences = client.get_protein_sequences(["P12345", "Q67890", "A11111"])
# → {"P12345": "MKVL...", "Q67890": "MSAS...", ...}
```

---

## 4. Nearest-Neighbor Search

Search by raw embedding vector (e.g. one you just generated):

```python
import numpy as np
from CBBIO import connect

client = connect()

# Get the embedding for a query protein
query_emb = client.get_protein_embedding("P12345", type_id=1, layer=33, as_numpy=True)

# Find the 20 nearest neighbors
neighbors = client.find_nearest_neighbors(
    query_emb,
    type_id=1,
    layer=33,
    k=20,
    metric="cosine",
)

for nb in neighbors:
    print(nb.protein_id, nb.distance)
```

To search and retrieve GO annotations in one call:

```python
neighbors, go_by_protein = client.neighbors_with_go(
    query_id="P12345",
    type_id=1,
    layer=33,
    k=20,
)

for nb in neighbors:
    go_terms = go_by_protein.get(nb.protein_id, [])
    print(nb.protein_id, [t.go_id for t in go_terms])
```

The search backend is selected automatically (pgvector → GPU if available). See [BioData.md](BioData.md) for backend configuration.

---

## 5. Running a Probing Task

Probing trains a small linear or MLP head on top of frozen embeddings to measure what information is encoded at a given layer.

```python
from CBBIO import (
    Generator,
    FastaBatcher,
    EmbeddingWriter,
    LinearProbe,
    run_embedding_generation,
    load_dbptm_benchmark_dataset,
    PredictionSpec,
    ResidueDataset,
    Task,
    run_task_on_layer,
)

# --- Step 1: generate (or load) embeddings ---
generator = Generator(model_class="esm2", device="cuda:0")
batcher   = FastaBatcher("proteins.fasta", batch_size=32)
writer    = EmbeddingWriter(format="memory")

run_embedding_generation(generator, batcher, writer, layer_index=33)

# writer.records is a list of EmbeddingRecord
embeddings = {r.id: r.embedding for r in writer.records}

# --- Step 2: load a dataset ---
# dbPTM CDK phosphorylation benchmark (residue-level binary task)
train_dataset = load_dbptm_benchmark_dataset(
    "/data/probing",
    name="phosphorylation_by_cdk",
    split="train",
    download=True,
)
test_dataset = load_dbptm_benchmark_dataset(
    "/data/probing",
    name="phosphorylation_by_cdk",
    split="test",
)
dataset = ResidueDataset([*train_dataset.examples, *test_dataset.examples])

# --- Step 3: define and run the task ---
task = Task(
    name="cdk_phospho",
    dataset=dataset,
    prediction=PredictionSpec(
        target="phosphorylation_by_cdk",
        objective="binary",
        level="residue",
    ),
    probe=LinearProbe(epochs=100),
)

result = run_task_on_layer(task=task, embeddings=embeddings, layer_index=33)
print(result.metrics)
# → {"accuracy": 0.84, "f1": 0.71, "auroc": 0.89, ...}
```

For the full probing reference, catalog of built-in datasets, and layer sweep examples see [Probing.md](Probing.md).

---

## 6. GO Semantic Similarity

```python
from CBBIO import load_go, read_annotations_tsv

onto = load_go("notebooks/data/go-basic.obo")

# Information-content based similarity
annotations = read_annotations_tsv("annotations.tsv")  # protein_id → [GO IDs]
onto.prepare_ic(annotations)

sim = onto.similarity("GO:0006355", "GO:0006351", method="lin")
print(f"Similarity: {sim:.3f}")

# Ancestry check
print(onto.relation("GO:0006355", "GO:0003700"))
# → "descendant"
```

See [GO.md](GO.md) for all methods.

---

## 7. Taxonomy

```python
from CBBIO import load_taxonomy

tax = load_taxonomy("/path/to/taxdump/")

# Full lineage for human
lineage = tax.lineage("9606", include_self=True)
# → ["1", "131567", "2759", ..., "9606"]

# Name and rank
node = tax.taxon("9606")
print(node["name"], node["rank"])  # Homo sapiens  species
```

See [Taxonomy.md](Taxonomy.md) for all methods.

---

## Where to Go Next

| Topic | Document |
|---|---|
| All embedding models, poolers, output formats | [Embeddings.md](Embeddings.md) |
| Probing datasets, layer sweeps, custom tasks | [Probing.md](Probing.md) |
| BioDataClient, search backends, config | [BioData.md](BioData.md) |
| GO ontology and semantic similarity | [GO.md](GO.md) |
| Taxonomy lineage and information content | [Taxonomy.md](Taxonomy.md) |
| Sequence alignment | [Similarity.md](Similarity.md) |
| Coding style guide | [STYLE.md](STYLE.md) |
| Documentation style guide | [STYLE_DOCS.md](STYLE_DOCS.md) |
| Test style guide | [STYLE_TESTS.md](STYLE_TESTS.md) |
