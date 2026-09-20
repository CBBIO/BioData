# Agent Workflows

```python
from CBBIO import GenerationInput, Generator, IterableBatcher, EmbeddingWriter, run_embedding_generation

records = [
    GenerationInput(id="P1", sequence="MTEYKLVVVG"),
    GenerationInput(id="P2", sequence="GAGGVGKSAL"),
]

generator = Generator(model_class="esm2", device="cuda:0")
batcher = IterableBatcher(records, batch_size=8)
writer = EmbeddingWriter(format="memory")
run_embedding_generation(generator, batcher, writer, layer_index=33)
```

These recipes cover the common changes coding agents make in CBBIO.

## Generate Embeddings

```python
from CBBIO import EmbeddingWriter, FastaBatcher, Generator, pooler_factory, run_embedding_generation

generator = Generator(model_class="esm2", name="facebook/esm2_t33_650M_UR50D", device="cuda:0")
batcher = FastaBatcher("proteins.fasta", max_batch_tokens=32768, max_sequence_length=4000)
writer = EmbeddingWriter(format="h5", path="esm2_mean.h5", compression="gzip")

result = run_embedding_generation(
    generator,
    batcher,
    writer,
    layer_index=33,
    pooler=pooler_factory("mean"),
    fail_fast=False,
)

print(result.record_count)
```

Choose residue-level output with `pooler=None`. Choose protein-level vectors with `pooler_factory("mean")`. Prefer HDF5 for large outputs and mixed payload shapes.

## Read Embeddings

```python
from CBBIO import H5EmbeddingReader, load_embedding_records_h5

records = load_embedding_records_h5("esm2_mean.h5", layer_index=33, pool_method="mean")

reader = H5EmbeddingReader("esm2_residue.h5")
segment = reader.read("P12345", layer_index=33, pool_method="none", residue_start=10, residue_end=40)
```

Use `load_embedding_records()` when dispatch by file extension is enough. Use `H5EmbeddingReader` when you need partial reads from residue matrices.

## Query BioData

```python
from CBBIO import connect

client = connect()
client.health_check()

embedding = client.get_protein_embedding("P12345", embedding_type_id=1, layer_index=33, as_numpy=True)
neighbors = client.find_nearest_neighbors(embedding, embedding_type_id=1, layer_index=33, k=20, metric="cosine")

for neighbor in neighbors:
    print(neighbor.protein_id, neighbor.distance)
```

Read configuration from `config.yaml` or environment variables. Use `neighbors_with_go()` when the caller needs nearest neighbors and GO annotations together.

## Use Persistent Local Search

```python
from CBBIO import IndexBuildSpec, IndexKey, IndexManager, connect

client = connect()
key = IndexKey.from_biodata(
    client,
    database_label="biodata",
    embedding_type_id=3,
    layer_index=0,
    metric="cosine",
)
manager = IndexManager(".biodata/indexes", search_nprobe=128)
revision = client.embedding_index_revision(embedding_type_id=3, layer_index=0)

manager.build_ivf_pq(
    key,
    lambda: client.iter_embedding_index_batches(embedding_type_id=3, layer_index=0),
    source_revision=revision,
    spec=IndexBuildSpec(nlist=3476),
)
client.configure_persistent_index(manager, database_label="biodata")
```

Builds are deliberate maintenance operations, never an implicit side effect of a search. Build
the portable exact store with `build_exact_store()` when exact FAISS CPU or cuVS GPU searches need
to run from local disk. Keep artifacts below `.biodata/indexes/` and commit neither them nor
notebook outputs. See [BioData.md](BioData.md) for the full workflow and error behavior.

## Run a Protein Probe

```python
from CBBIO import LinearProbe, PredictionSpec, Task, load_dataset, run_task_on_layer

dataset = load_dataset("peer:fluorescence", "/data/probing", download=True)
task = Task(
    name="fluorescence_linear",
    dataset=dataset,
    prediction=PredictionSpec(target="log_fluorescence", objective="regression", level="protein"),
    probe=LinearProbe(epochs=100, seed=7),
)

result = run_task_on_layer(task=task, embeddings=embeddings_by_id, layer_index=33)
print(result.metrics)
```

Protein-level probes expect one vector per protein. Generate mean-pooled embeddings before running these tasks.

## Run a Residue Probe

```python
from CBBIO import LinearProbe, PredictionSpec, Task, load_residue_source_dataset, run_task_on_layer

dataset = load_residue_source_dataset("disprot", "/data/probing", download=True)
task = Task(
    name="disorder_linear",
    dataset=dataset,
    prediction=PredictionSpec(target="disorder", objective="binary", level="residue"),
    probe=LinearProbe(epochs=100, seed=7),
)

result = run_task_on_layer(task=task, embeddings=residue_embeddings_by_id, layer_index=33)
print(result.metrics)
```

Residue-level probes expect one matrix per protein with the same sequence length as the labeled example. Keep `pooler=None` during generation.

## Add a Dataset Source

```python
from CBBIO import list_dataset_catalog, search_dataset_catalog

ready_residue_tasks = list_dataset_catalog(level="residue", status="ready")
go_tasks = search_dataset_catalog("go")
print(len(ready_residue_tasks), len(go_tasks))
```

Follow [AddingProbingDataSource.md](AddingProbingDataSource.md). Add provider-specific loading under `CBBIO/probing/sources/`, register metadata in the catalog or collection layer, and add tests that load a minimal fixture without network access.

## Add an Embedding Model

```python
from CBBIO import Generator, available_generator_classes

generator = Generator(model_class="mymodel", device="cpu")
print("mymodel" in available_generator_classes())
```

Follow [ModelSpecificEmbeddingModules.md](ModelSpecificEmbeddingModules.md). Add the generator under `CBBIO/embeddings/models/`, export it from `CBBIO/__init__.py`, register it through the model package, and test dependency errors, preprocessing, layer selection, poolers, and metadata.

Every `EmbeddingGenerator` subclass defines:

| Attribute | Use |
|---|---|
| `GENERATOR_CLASS` | Canonical `Generator(model_class=...)` value |
| `DEFAULT_MODEL_NAME` | Model used when no `name=` is passed |
| `FAMILY_MODELS` | Accepted model names for discovery |
| `SUPPORTED_POOLERS` | Poolers accepted by the generator |

## Use GO and Taxonomy

```python
from CBBIO import load_go, load_taxonomy, read_annotations_tsv

go = load_go("notebooks/data/go-basic.obo")
annotations = read_annotations_tsv("protein_go.tsv")
go.prepare_term_counts(annotations)

taxonomy = load_taxonomy("/data/taxdump")
print(taxonomy.taxon("9606")["name"])
```

Prepare counts before information-content or Lin-style similarity calls. Keep taxonomy normalization helpers imported from `CBBIO.Taxonomy` because they are not top-level exports.

## Use Sequence Alignment

```python
from CBBIO import align_sequences

alignment = align_sequences("MTEYKLVVVG", "MTEYKLVIVG", mode="global")
print(alignment.identity)
```

Use `align_sequences()` instead of adding a new alignment dependency. The implementation wraps `parasail` and normalizes errors through CBBIO exceptions.

## Test Changes

```bash
poetry run pytest -q tests/test_embeddings.py
poetry run pytest -q tests/test_probing.py
poetry run pyright
```

Pick tests by touched subsystem. Use `pytest.importorskip()` for optional dependencies. Do not call `filter_redundant_to_test_mmseqs()` in unit tests because it requires the `mmseqs` binary.

## Exceptions

| Exception | When raised |
|---|---|
| `EmbeddingDependencyError` | Optional model dependency is not installed |
| `EmbeddingGenerationError` | Embedding generation fails at the job level |
| `EmbeddingInputError` | FASTA records, poolers, layers, or payload shapes are invalid |
| `BioDataError` | Database client operation fails |
| `GOCountsNotPreparedError` | GO IC or similarity is called before counts are prepared |
| `TaxonCountsNotPreparedError` | Taxonomy IC or Lin map is called before counts are prepared |
| `SimilarityDependencyError` | Alignment dependency is not installed |
