# Agent Quick Start

```python
from CBBIO import (
    EmbeddingWriter,
    FastaBatcher,
    Generator,
    pooler_factory,
    run_embedding_generation,
)

generator = Generator(model_class="esm2", device="cuda:0")
batcher = FastaBatcher("proteins.fasta", max_batch_tokens=32768, max_sequence_length=4000)
writer = EmbeddingWriter(format="h5", path="embeddings.h5")

run_embedding_generation(
    generator,
    batcher,
    writer,
    layer_index=33,
    pooler=pooler_factory("mean"),
)
```

This guide gives coding agents a fast route through CBBIO without scanning the repository.

## Start Here

```python
from CBBIO import available_generator_classes, available_generator_models

print(available_generator_classes())
print(available_generator_models("esm2"))
```

Use these documents by task:

| Task | Read |
|---|---|
| Generate embeddings | [AgentWorkflows.md](AgentWorkflows.md), [Embeddings.md](Embeddings.md) |
| Train probing tasks | [AgentWorkflows.md](AgentWorkflows.md), [Probing.md](Probing.md) |
| Query PostgreSQL and pgvector | [BioData.md](BioData.md) |
| Use GO or taxonomy utilities | [GO.md](GO.md), [Taxonomy.md](Taxonomy.md) |
| Add an embedding model | [AgentWorkflows.md](AgentWorkflows.md), [ModelSpecificEmbeddingModules.md](ModelSpecificEmbeddingModules.md) |
| Locate public APIs | [AgentApiMap.md](AgentApiMap.md) |
| Change code style safely | [STYLE.md](STYLE.md), [STYLE_TESTS.md](STYLE_TESTS.md), [STYLE_DOCS.md](STYLE_DOCS.md) |

## Import Policy

```python
from CBBIO import Generator, FastaBatcher, EmbeddingWriter, run_embedding_generation
from CBBIO.Taxonomy import compute_taxon_ic_and_lin_maps, normalize_taxonomy_id
```

Use the top-level `CBBIO` namespace in examples, tests, notebooks, and docs. The taxonomy helpers above are the known exception because they are not exported from `CBBIO`.

Do not import from model internals when user code can use `Generator(model_class=...)`.

## Core Objects

```python
from CBBIO import (
    BioDataClient,
    EmbeddingRecord,
    GenerationInput,
    ProteinDataset,
    ResidueDataset,
    Task,
)
```

| Object | Use |
|---|---|
| `Generator` | Loads a protein language model by model family |
| `FastaBatcher` | Streams FASTA records into generation batches |
| `IterableBatcher` | Batches in-memory `GenerationInput` objects |
| `EmbeddingWriter` | Writes generated records to memory, pickle, NumPy, or HDF5 |
| `EmbeddingRecord` | Carries one protein embedding for one layer and pool method |
| `ProteinDataset` | Holds protein-level probing examples |
| `ResidueDataset` | Holds residue-level probing examples |
| `Task` | Bundles dataset, prediction spec, and probe |
| `BioDataClient` | Reads proteins, embeddings, annotations, and neighbors from PostgreSQL |

## Model Families

```python
from CBBIO import Generator

generator = Generator(model_class="prott5", device="cuda:0", dtype="float16")
```

| `model_class` | Default use |
|---|---|
| `"esm2"` | General ESM-2 embeddings |
| `"esm1b"` | ESM-1b via the safe torch-hub loader |
| `"esmc"` | ESM Cambrian models |
| `"prott5"` | ProtT5 protein embeddings |
| `"prostt5"` | ProstT5 sequence or structure-aware embeddings |
| `"ankh3"` | Ankh3 embeddings |
| `"amplify"` | NVIDIA AMPLIFY embeddings |
| `"proteinglm"` | ProteinGLM embeddings |

Use `available_generator_models(model_class)` when you need the accepted model names for a family.

## Editing Map

```python
from CBBIO import connect, load_dataset, load_go, load_taxonomy
```

| Goal | First files to inspect |
|---|---|
| Public exports | `CBBIO/__init__.py` |
| Embedding base contracts | `CBBIO/embeddings/__init__.py` |
| Model generator registration | `CBBIO/embeddings/factory.py` |
| Model-specific behavior | `CBBIO/embeddings/models/` |
| Embedding batching, IO, jobs | `CBBIO/embeddings/batcher.py`, `CBBIO/embeddings/io.py`, `CBBIO/embeddings/jobs.py` |
| Database client | `CBBIO/BioData.py` |
| Search backends | `CBBIO/search/` |
| Probing tasks | `CBBIO/probing/tasks.py`, `CBBIO/probing/runner.py`, `CBBIO/probing/probes.py` |
| Dataset catalog | `CBBIO/probing/catalog.py`, `CBBIO/probing/collections.py`, `CBBIO/probing/sources/` |
| GO utilities | `CBBIO/GO.py` |
| Taxonomy utilities | `CBBIO/Taxonomy.py` |
| Sequence alignment | `CBBIO/similarity.py` |

## Checks

```bash
poetry run pytest -q
poetry run pyright
```

Run the smallest relevant test first while editing. Run the full unit suite before broad changes. Integration tests need a real test database and `config_test.yaml`.

## Exceptions

| Exception | When raised |
|---|---|
| `EmbeddingDependencyError` | Optional model dependency is missing |
| `EmbeddingInputError` | Input records, layers, poolers, or sequences are invalid |
| `EmbeddingBackendError` | Model backend fails during generation |
| `BioDataError` | Database configuration or query operation fails |
| `GOError` | GO ontology operation fails |
| `TaxonomyError` | Taxonomy operation fails |
| `SequenceSimilarityError` | Sequence alignment operation fails |
