# Agent API Map

```python
from CBBIO import (
    Generator,
    connect,
    list_dataset_catalog,
    load_go,
    load_taxonomy,
    align_sequences,
)

print(Generator)
print(connect)
print(len(list_dataset_catalog()))
```

This map lists stable public entry points that coding agents should try before reading internal modules.

## Embeddings

```python
from CBBIO import (
    EmbeddingRecord,
    EmbeddingWriter,
    FastaBatcher,
    GenerationInput,
    Generator,
    IterableBatcher,
    available_generator_classes,
    available_generator_models,
    pooler_factory,
    run_embedding_generation,
)
```

| API | Use |
|---|---|
| `Generator` | Factory for model-specific embedding generators |
| `available_generator_classes()` | Lists accepted `model_class` values |
| `available_generator_models()` | Lists model names by family |
| `GenerationInput` | In-memory generation record |
| `FastaBatcher` | FASTA streaming batcher |
| `IterableBatcher` | In-memory batcher |
| `EmbeddingWriter` | Memory, pickle, NumPy, and HDF5 writer |
| `run_embedding_generation()` | Production generation runner |
| `pooler_factory()` | Creates `"none"`, `"mean"`, and `"cls"` poolers |
| `EmbeddingRecord` | Generated embedding payload and metadata |

## Embedding IO

```python
from CBBIO import (
    H5EmbeddingReader,
    load_embedding_records,
    load_embedding_records_h5,
    save_embedding_records_h5,
)
```

| API | Use |
|---|---|
| `load_embedding_records()` | Loads by extension |
| `load_embedding_records_h5()` | Loads filtered records from HDF5 |
| `load_embedding_records_npy()` | Loads NumPy vector payloads |
| `load_embedding_records_pickle()` | Loads pickle payloads |
| `save_embedding_records_h5()` | Writes records to HDF5 |
| `save_embedding_records_npy()` | Writes same-width vectors to NumPy |
| `save_embedding_records_pickle()` | Writes records to pickle |
| `H5EmbeddingReader` | Reads HDF5 metadata and residue segments |
| `mean_pool_embedding_record()` | Converts a residue matrix record into a vector record |

## BioData Database

```python
from CBBIO import BioDataClient, build_dsn, connect, load_config
```

| API | Use |
|---|---|
| `connect()` | Creates and opens a `BioDataClient` |
| `BioDataClient` | Database client for proteins, embeddings, annotations, and search |
| `build_dsn()` | Builds a PostgreSQL DSN from config values |
| `load_config()` | Loads `config.yaml` plus environment overrides |

Common client methods:

| Method | Use |
|---|---|
| `health_check()` | Verifies the database is reachable |
| `get_protein()` | Fetches one protein record |
| `get_protein_sequences()` | Fetches sequences by protein id |
| `get_protein_embedding()` | Fetches one stored embedding |
| `find_nearest_neighbors()` | Runs vector nearest-neighbor search |
| `find_nearest_neighbors_for_embeddings()` | Runs batch nearest-neighbor search for external embeddings |
| `neighbors_with_go()` | Retrieves neighbors and GO annotations together |

## Probing

```python
from CBBIO import (
    LinearProbe,
    MlpProbe,
    PredictionSpec,
    ProteinDataset,
    ResidueDataset,
    Task,
    TransferProbe,
    load_dataset,
    run_task_on_layer,
)
```

| API | Use |
|---|---|
| `ProteinDataset` | Protein-level examples and splits |
| `ResidueDataset` | Residue-level examples and splits |
| `PredictionSpec` | Target, objective, level, and class metadata |
| `LinearProbe` | Built-in linear Torch probe |
| `MlpProbe` | Built-in MLP Torch probe |
| `TransferProbe` | No-training nearest-neighbor transfer probe |
| `Task` | Dataset plus prediction spec plus probe |
| `run_task_on_layer()` | Trains and evaluates a task on one layer |
| `train_and_evaluate_probe()` | Lower-level protein probe helper |
| `train_and_evaluate_residue_probe()` | Lower-level residue probe helper |

## Dataset Catalog

```python
from CBBIO import (
    get_dataset_collection,
    list_dataset_catalog,
    list_dataset_collections,
    load_dataset,
    search_dataset_catalog,
)
```

| API | Use |
|---|---|
| `list_dataset_collections()` | Lists provider collections |
| `get_dataset_collection()` | Retrieves one provider collection |
| `list_dataset_catalog()` | Lists datasets with optional filters |
| `search_dataset_catalog()` | Searches catalog metadata |
| `load_dataset()` | Loads a qualified dataset id |
| `download_dataset()` | Downloads a qualified dataset id |

Provider-specific loaders remain public for cases where a caller needs provider options:

| Loader | Use |
|---|---|
| `load_peer_dataset()` | PEER benchmark tasks |
| `load_dbptm_benchmark_dataset()` | dbPTM benchmark archives |
| `load_residue_source_dataset()` | Residue annotation sources |
| `load_cafa5_dataset()` | CAFA5 function prediction data |
| `load_ec_dataset()` | Enzyme Commission tasks |
| `load_go_dataset()` | Gene Ontology tasks |

## GO and Taxonomy

```python
from CBBIO import (
    GOOntology,
    TaxonomyOntology,
    load_go,
    load_taxonomy,
    read_annotations_tsv,
    read_taxonomy_annotations_tsv,
)
```

| API | Use |
|---|---|
| `load_go()` | Loads an OBO Gene Ontology file |
| `GOOntology` | GO DAG, ancestors, counts, IC, and semantic similarity |
| `read_annotations_tsv()` | Reads protein-to-GO annotations |
| `load_taxonomy()` | Loads NCBI taxdump files |
| `TaxonomyOntology` | Taxonomy tree, counts, IC, and Lin maps |
| `read_taxonomy_annotations_tsv()` | Reads protein-to-taxonomy annotations |

Taxonomy helpers that are not top-level exports:

```python
from CBBIO.Taxonomy import compute_taxon_ic_and_lin_maps, normalize_taxonomy_id
```

## Similarity

```python
from CBBIO import align_sequences

result = align_sequences("MTEYKLVVVG", "MTEYKLVIVG", mode="global")
print(result.score, result.identity)
```

| API | Use |
|---|---|
| `align_sequences()` | Pairwise sequence alignment |
| `AlignmentResult` | Alignment score, identity, similarity, and aligned strings |
| `AlignmentMode` | Accepted alignment mode values |

## Public Model Classes

```python
from CBBIO import Esm2EmbeddingGenerator, ProtT5EmbeddingGenerator

print(Esm2EmbeddingGenerator.DEFAULT_MODEL_NAME)
print(ProtT5EmbeddingGenerator.SUPPORTED_POOLERS)
```

Use these classes when you are adding or testing a model-specific adapter. Use `Generator` for normal calling code.

| Class | `model_class` |
|---|---|
| `Esm2EmbeddingGenerator` | `"esm2"` |
| `Esm1bEmbeddingGenerator` | `"esm1b"` |
| `EsmcEmbeddingGenerator` | `"esmc"` |
| `ProtT5EmbeddingGenerator` | `"prott5"` |
| `ProstT5EmbeddingGenerator` | `"prostt5"` |
| `Ankh3EmbeddingGenerator` | `"ankh3"` |
| `AmplifyEmbeddingGenerator` | `"amplify"` |
| `ProteinGlmEmbeddingGenerator` | `"proteinglm"` |

## Exceptions

| Exception | When raised |
|---|---|
| `BioDataError` | Base database client error |
| `DriverDependencyError` | PostgreSQL driver dependency is missing |
| `NotFoundError` | Requested database record is absent |
| `EmbeddingGenerationError` | Base embedding generation error |
| `EmbeddingDependencyError` | Optional embedding dependency is missing |
| `EmbeddingInputError` | Embedding input or output shape is invalid |
| `GOError` | Base GO ontology error |
| `TaxonomyError` | Base taxonomy error |
| `SequenceSimilarityError` | Base sequence alignment error |
