# CBBIO Probing

`CBBIO.probing` trains small supervised probes on top of frozen protein language model (PLM) embeddings. It lets you measure what biological information is encoded at each model layer without fine-tuning the PLM itself.

The typical workflow is:
1. Generate (or load) embeddings for the proteins in a dataset.
2. Choose a dataset from the built-in catalog, or supply your own.
3. Define a `Task` (dataset + prediction type + probe architecture).
4. Call `run_task_on_layer()` to train and evaluate.
5. Optionally sweep over layers to find the most informative one.

---

## Concepts

### Levels: protein vs. residue

| Level | Input embedding | Use case |
|---|---|---|
| `"protein"` | One vector per protein `(D,)` | Predicting a property of the whole protein (thermostability, localization, fitness) |
| `"residue"` | One matrix per protein `(L, D)` | Predicting a property of each amino acid (PTM site, disorder, binding residue) |

### ProteinDataset and ResidueDataset

Both are in-memory containers that hold examples, labels, and explicit train/val/test splits.

```python
from CBBIO import ProteinDataset, ResidueDataset, ProteinExample, ResidueExample
```

`ProteinExample` fields:

| Field | Type | Description |
|---|---|---|
| `id` | `str` | Unique protein identifier |
| `sequence` | `str` | Amino acid sequence |
| `labels` | `Mapping[str, Any]` | Named label values (e.g. `{"thermostability": 72.3}`) |
| `split` | `"train" \| "val" \| "test"` | Which split this example belongs to |
| `metadata` | `Mapping[str, Any] \| None` | Optional extra info |

`ResidueExample` fields are the same except `labels` maps each target name to a per-position sequence:

| Field | Type | Description |
|---|---|---|
| `labels` | `Mapping[str, Sequence[Any]]` | Per-residue labels (e.g. `{"ptm_site": [0, 0, 1, 0, ...]}`) |
| `mask` | `Sequence[bool] \| None` | `True` for positions to include; `None` means include all |

### Task and Probes

```python
from CBBIO import LinearProbe, MlpProbe, PredictionSpec, Task
```

**`PredictionSpec`** describes what you are predicting:

| Field | Type | Values |
|---|---|---|
| `target` | `str` | Key into `example.labels` |
| `objective` | `ObjectiveName` | `"regression"`, `"binary"`, `"multiclass"`, `"multilabel"` |
| `level` | `TaskLevel` | `"protein"` or `"residue"` |
| `classes` | `Sequence[str] \| None` | Class names for multiclass (optional, for display) |

`LinearProbe` and `MlpProbe` implement the built-in probe backends. They share these training
parameters:

| Field | Default | Description |
|---|---|---|
| `epochs` | `100` | Training epochs |
| `learning_rate` | `0.01` | Adam learning rate |
| `batch_size` | `None` | Batch size; `None` uses the full training set |
| `weight_decay` | `0.0` | L2 regularization strength |
| `seed` | `7` | Random seed for reproducibility |

**`Task`** bundles everything together:

```python
task = Task(
    name="my_task",
    dataset=dataset,                                # ProteinDataset or ResidueDataset
    prediction=PredictionSpec(
        target="ptm_site",
        objective="binary",
        level="residue",
    ),
    probe=LinearProbe(epochs=200),
)
```

`MlpProbe` adds `hidden_dim`, which defaults to `64`. Use `MlpProbe(hidden_dim=128)` to select the
built-in MLP architecture. For XGBoost, CNNs, and other estimators, see
[Custom Probes](CustomProbes.md).

`ProbeSpec` remains available as a compatibility configuration for existing code.

---

## The Dataset Catalog

```python
from CBBIO import list_dataset_catalog, list_dataset_collections, load_dataset

for collection in list_dataset_collections():
    print(collection.id, len(collection.list_datasets()))

residue_ready = list_dataset_catalog(level="residue", status="ready")
dataset = load_dataset("disprot:all", "/data/probing", split="test", download=True)
```

Each data provider publishes collection metadata and metadata for its datasets. A collection exposes
that provider through one interface for discovery, download, and loading. The catalog combines the
collections into one searchable index. `load_dataset()` resolves a qualified dataset ID and delegates
to the owning collection.

Source-specific functions such as `load_peer_dataset()` and `load_residue_source_dataset()` remain
available as compatibility facades. New code can use `load_dataset()` when it does not need a
provider-specific option.

Key `DatasetMetadata` fields:

| Field | Description |
|---|---|
| `id` | Qualified identifier accepted by `load_dataset()` |
| `collection` | Collection that owns loading for this dataset |
| `status` | `"ready"` = loadable now; `"adapter"` = needs custom import; `"catalog_only"` = metadata only |
| `level` | `"protein"` or `"residue"` |
| `objective` | `"binary"`, `"multiclass"`, `"regression"`, `"multilabel"` |
| `preferred_metric` | Recommended evaluation metric |
| `homepage` | Original dataset homepage |

### Collections

```python
from CBBIO import get_dataset_collection

biolip = get_dataset_collection("biolip")
print(biolip.metadata.homepage)
print([dataset.id for dataset in biolip.list_datasets()])
```

| Collection | Description |
|---|---|
| `peer` | PEER benchmark tasks (fitness, thermostability, localization, fluorescence, …) |
| `dbptm` | dbPTM source data and benchmark archives |
| `musitedeep` | MusiteDeep PTM annotations |
| `disprot` | DisProt disorder annotations |
| `biolip` | BioLiP ligand-class binding annotations |
| `metalpdb` | MetalPDB binding annotations |
| `scannet` | ScanNet binding annotations |
| `netsurfp` | NetSurfP structure annotations |
| `phosphoelm` | PhosphoELM evidence subsets |
| `dtu` | DTU in-house annotation services |

`DatasetCatalogEntry` remains as a compatibility alias for `DatasetMetadata`.

See [Adding a Probing Data Source](AddingProbingDataSource.md) to implement and register another
provider.

---

## Built-in Dataset Loaders

### dbPTM Benchmarks

```python
from CBBIO import list_dbptm_benchmarks, load_dbptm_benchmark_dataset

for benchmark in list_dbptm_benchmarks():
    print(benchmark.name, benchmark.positive_sites, "positive sites")

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
```

These dbPTM benchmarks provide kinase-specific phosphorylation labels with predefined train and test
splits.

Available benchmarks:

| Name | Description |
|---|---|
| `phosphorylation_by_cdk` | CDK phosphorylation sites |
| `phosphorylation_by_mapk` | MAPK phosphorylation sites |
| `phosphorylation_by_pka` | PKA phosphorylation sites |
| `phosphorylation_by_pkc` | PKC phosphorylation sites |
| `phosphorylation_by_ck2` | CK2 phosphorylation sites |

### Residue Sources

```python
from CBBIO import list_residue_sources, download_residue_source, load_residue_source_dataset

for spec in list_residue_sources():
    print(spec.name, spec.category, spec.objective)

download_residue_source("/data/probing", name="disprot")
dataset = load_residue_source_dataset(
    "/data/probing",
    name="disprot",
    split="test",
)
```

Residue sources cover broader PTM and structural annotations. The download operation is idempotent.

### PEER Tasks

```python
from CBBIO import load_peer_dataset, list_peer_tasks

for task_metadata in list_peer_tasks():
    if task_metadata.objective == "regression":
        print(task_metadata.name, task_metadata.preferred_metric)

dataset = load_peer_dataset(
    "/data/probing",
    name="beta_lactamase",
    split="train",
    download=True,
)
```

PEER provides protein-level fitness and function benchmarks as well as residue-level tasks.

---

## End-to-End: Running a Built-in Task

This example runs CDK phosphorylation probing with ESM-2 layer 33.

```python
from CBBIO import (
    Generator,
    FastaBatcher,
    EmbeddingWriter,
    run_embedding_generation,
    load_dbptm_benchmark_dataset,
    LinearProbe,
    PredictionSpec,
    Task,
    run_task_on_layer,
)

# --- Generate embeddings (residue-level, layer 33) ---
generator = Generator(model_class="esm2", device="cuda:0")

# Get all protein sequences from both splits
train_ds = load_dbptm_benchmark_dataset(
    "/data/probing",
    name="phosphorylation_by_cdk",
    split="train",
    download=True,
)
test_ds = load_dbptm_benchmark_dataset(
    "/data/probing",
    name="phosphorylation_by_cdk",
    split="test",
)

# Write sequences to a temporary FASTA (or load from disk if already done)
import tempfile, pathlib
fasta_path = pathlib.Path(tempfile.mktemp(suffix=".fasta"))
with fasta_path.open("w") as f:
    for example in list(train_ds.examples) + list(test_ds.examples):
        f.write(f">{example.id}\n{example.sequence}\n")

batcher = FastaBatcher(str(fasta_path), batch_size=32)
writer  = EmbeddingWriter(format="memory")

run_embedding_generation(generator, batcher, writer, layer_index=33)
embeddings = {r.id: r.embedding for r in writer.records}

# --- Build the combined dataset ---
from CBBIO import ResidueDataset, ResidueExample

examples = list(train_ds.examples) + list(test_ds.examples)
dataset = ResidueDataset(examples)

# --- Define and run the task ---
task = Task(
    name="cdk_phospho_esm2_l33",
    dataset=dataset,
    prediction=PredictionSpec(
        target="phosphorylation_by_cdk",
        objective="binary",
        level="residue",
    ),
    probe=LinearProbe(epochs=100),
)

result = run_task_on_layer(
    task=task,
    embeddings=embeddings,
    model_reference="facebook/esm2_t33_650M_UR50D",
    layer_index=33,
)

print(f"Accuracy: {result.metrics['accuracy']:.3f}")
print(f"AUROC:    {result.metrics['auroc']:.3f}")
print(f"F1:       {result.metrics['f1']:.3f}")
print(f"Elapsed:  {result.elapsed_seconds:.1f}s")
```

---

## Layer Sweep

A layer sweep runs the same task for every layer and shows which layer encodes the most information about your target.

```python
results = []
for layer in range(34):   # ESM-2 650M has 33 transformer layers + layer 0
    result = run_task_on_layer(
        task=task,
        embeddings=embeddings_by_layer[layer],  # pre-computed per layer
        layer_index=layer,
    )
    results.append(result)

# Print AUROC per layer
for r in results:
    print(f"Layer {r.layer_index:2d}  AUROC={r.metrics['auroc']:.3f}")
```

If you can keep all layers in memory, extract them all in one pass:

```python
writer = EmbeddingWriter(format="memory")
run_embedding_generation(generator, batcher, writer, layer_index=list(range(34)))

# writer.records contains one EmbeddingRecord per (protein, layer) pair
from collections import defaultdict
embeddings_by_layer = defaultdict(dict)
for record in writer.records:
    embeddings_by_layer[record.layer_index][record.id] = record.embedding
```

---

## Custom Datasets

### Protein-level

```python
from CBBIO import ProteinDataset, ProteinExample

examples = [
    ProteinExample(
        id="P12345",
        sequence="MKVLILLF...",
        labels={"thermostability": 68.5, "subcell_loc": "cytoplasm"},
        split="train",
    ),
    ProteinExample(
        id="Q67890",
        sequence="MAPLRT...",
        labels={"thermostability": 82.1, "subcell_loc": "nucleus"},
        split="test",
    ),
]

dataset = ProteinDataset(examples)

task = Task(
    name="thermo",
    dataset=dataset,
    prediction=PredictionSpec(target="thermostability", objective="regression", level="protein"),
)
```

### Residue-level

Per-position labels must be the same length as the sequence. Use `mask` to mark positions where the label is defined (e.g. only observed PTM sites vs. background).

```python
from CBBIO import ResidueDataset, ResidueExample

examples = [
    ResidueExample(
        id="P12345",
        sequence="MKVLILL",
        labels={"binding": [0, 0, 1, 0, 1, 0, 0]},
        split="train",
        mask=None,   # use all positions
    ),
]

dataset = ResidueDataset(examples)
```

---

## Reducing Train/Test Redundancy

Use `filter_redundant_to_test_mmseqs` to remove training sequences that are sequence-similar to test sequences. Requires MMseqs2 to be installed.

```python
from CBBIO import filter_redundant_to_test_mmseqs

train_ids = [ex.id for ex in dataset.by_split("train")]
test_ids  = [ex.id for ex in dataset.by_split("test")]

# Sequences dict: {id: amino_acid_sequence}
sequences = {ex.id: ex.sequence for ex in dataset.examples}

report = filter_redundant_to_test_mmseqs(
    train_ids=train_ids,
    test_ids=test_ids,
    sequences=sequences,
    cutoff=0.3,          # 30% sequence identity threshold
    min_coverage=0.3,
)

print(f"Removed {report.removed_count} train sequences")
print(f"Retained {report.retained_count} train sequences")

# Build a filtered dataset
safe_train_ids = set(train_ids) - {hit.removed_id for hit in report.removed}
filtered_examples = [
    ex for ex in dataset.examples
    if ex.split != "train" or ex.id in safe_train_ids
]
clean_dataset = ResidueDataset(filtered_examples)
```

---

## Metrics Reference

### Binary classification

| Metric | Description |
|---|---|
| `accuracy` | Fraction of correct predictions |
| `f1` | F1 score (harmonic mean of precision and recall) |
| `precision` | Positive predictive value |
| `recall` | True positive rate (sensitivity) |
| `auc` | Area under the ROC curve |
| `auprc` | Area under the precision-recall curve (more informative for imbalanced tasks) |
| `mcc` | Matthews Correlation Coefficient |

### Multiclass classification

| Metric | Description |
|---|---|
| `accuracy` | Fraction correct |
| `f1_macro` | Macro-averaged F1 across all classes |
| `f1_weighted` | Weighted-average F1 |

### Regression

| Metric | Description |
|---|---|
| `r2` | Coefficient of determination (R²) |
| `rmse` | Root mean squared error |
| `mae` | Mean absolute error |
| `spearman` | Spearman rank correlation |
| `pearson` | Pearson correlation coefficient |

### Multilabel

Same as binary but computed per-label and averaged.

---

## TaskLayerResult Fields

```python
@dataclass(frozen=True)
class TaskLayerResult:
    task_name: str
    model_reference: str
    layer_index: int
    metrics: dict[str, float]        # metric name → value
    predictions: dict[str, Any]      # protein_id → predicted value
    scores: dict[str, float] | None  # probability scores (classification only)
    train_count: int
    val_count: int
    test_count: int
    elapsed_seconds: float
```

## Exceptions

| Exception | When raised |
|---|---|
| `EmbeddingInputError` | Invalid dataset metadata, collection, split, or input file |
| `EmbeddingDependencyError` | Required probe or dataset dependency is unavailable |
