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
from CBBIO import LinearProbe, MlpProbe, PredictionSpec, Task, TransferProbe
```

**`PredictionSpec`** describes what you are predicting:

| Field | Type | Values |
|---|---|---|
| `target` | `str` | Key into `example.labels` |
| `objective` | `ObjectiveName` | `"regression"`, `"binary"`, `"multiclass"`, `"multilabel"` |
| `level` | `TaskLevel` | `"protein"` or `"residue"` |
| `classes` | `Sequence[str] \| None` | Class names for multiclass (optional, for display) |

`LinearProbe` and `MlpProbe` implement the built-in Torch probe backends. They share these training
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
built-in MLP architecture.

`TransferProbe()` is a no-training transfer head for protein-level tasks. By default, it scores
classes from all training examples with inverse cosine-distance weighted votes and returns the full
score vector. For multilabel tasks, `threshold=None` keeps the full score vector for AP, Fmax, and
GO metrics while using `0.5` for fixed prediction metrics.

Transfer controls:

| Field | Default | Values |
|---|---|---|
| `distance` | `"cosine"` | `"cosine"`, `"euclidean"` |
| `neighbor_selection` | `"all"` | `"all"`, `"knn"`, `"cutoff_distance"` |
| `k` | `10` | Used when `neighbor_selection="knn"` |
| `distance_cutoff` | `None` | Required when `neighbor_selection="cutoff_distance"` |
| `scoring` | `"weighted_voting"` | `"weighted_voting"`, `"voting"` |
| `threshold` | `None` | Binary and multilabel predictions use `0.5`; score-swept metrics use the full scores |
| `search_backend` | `"numpy"` | `"numpy"`, `"auto"`, `"faiss_cpu"`, `"faiss_gpu"`, `"cuvs_gpu"`, `"torch_gpu"` |
| `search_device` | `None` | Accelerator device for non-NumPy backends |
| `search_ann` | `False` | Enables ANN where the selected backend supports it |
| `return_neighbors` | `False` | Adds transferred neighbor ids, distances, weights, and labels to result metadata |

Non-NumPy transfer backends support `neighbor_selection="knn"`. The default NumPy backend remains
available for exact `"all"` and `"cutoff_distance"` transfer.

For XGBoost, CNNs, and other estimators, see
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
| `ec` | Generated Enzyme Commission function-prediction tasks |
| `go` | Generated Gene Ontology function-prediction tasks |
| `cafa` | CAFA5 Kaggle Gene Ontology prediction training and target-subset data |
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

### Gene Ontology Tasks

```python
from CBBIO import load_dataset

dataset = load_dataset(
    "go:go_bp_head",
    "/data/function_datasets/output/go_main_head",
    split=("train", "test"),
)
```

The generated GO collection provides nine protein-level multilabel tasks. Dataset IDs use
`go:<aspect>_<track>`, where `aspect` is `go_bp`, `go_cc`, or `go_mf`, and `track` is `head`
`main`, or `full`.

The `head` and `main` tracks apply class minimum-support filters. The `full` track keeps every
asserted class for proteins that have at least one label in the requested GO aspect. Full splits
use a deterministic hash over `tax_id` and `cluster_id`.

Training targets are the task's asserted GO terms. The loader retains each protein's propagated
terms and the generated `go.obo` path in metadata. When you run a GO task with
`run_task_on_layer()`, CBBIO propagates predicted term scores to their GO ancestors and evaluates
them against those propagated annotations.
Ontology-root terms are excluded, and proteins annotated only with a root term do not contribute
to the GO metric.

The preferred GO metric is `go_fmax`. It follows CAFA protein-centric evaluation: CBBIO sweeps
prediction-score thresholds, averages precision over proteins that have predictions at each
threshold, averages recall over all proteins, and reports the highest F-score. The result also
contains `go_fmax_threshold`, `go_precision_at_fmax`, `go_recall_at_fmax`, and
`go_evaluated_protein_count`.

GO tasks also report fixed-threshold propagated metrics when the probe has a known prediction
threshold. For `TransferProbe(threshold=0.25)`, CBBIO reports `go_propagated_f1`,
`go_propagated_precision`, `go_propagated_recall`, and `go_propagated_threshold` at `0.25`.
These metrics use the same propagated GO score and truth sets as `go_fmax`, but they do not sweep
thresholds.

```python
from CBBIO import go_fixed_threshold_protein_centric_metrics, load_go

ontology = load_go("/data/probing/go.obo")
metrics = go_fixed_threshold_protein_centric_metrics(
    {"P12345": [0.1, 0.8]},
    class_names=["GO:0008151", "GO:0009987"],
    true_labels={"P12345": ["GO:0009987"]},
    ontology=ontology,
    threshold=0.25,
)
print(metrics["go_propagated_f1"])
```

### CAFA5 Kaggle Tasks

```python
from CBBIO import load_dataset

dataset = load_dataset(
    "cafa:cafa5_partial_bp",
    "/data/probing",
    split=("train", "test"),
    download=True,
)
```

The CAFA collection downloads with the Kaggle CLI when `download=True`, then loads
`Train/train_terms.tsv` and `Train/train_sequences.fasta`. Training-only dataset IDs are
`cafa:cafa5_bp`, `cafa:cafa5_cc`, and `cafa:cafa5_mf`; each uses deterministic hash splits over the
labeled training proteins.

Released CAFA5 target-subset IDs use `cafa:cafa5_<subset>_<aspect>`, where `subset` is
`no_knowledge`, `limited`, `partial`, or the corresponding `*_after_t0` publication-date subset.
These datasets use the Kaggle training proteins as `train` and the released target terms as `test`.
The partial-knowledge subsets require `Test (Targets)/known_t0.tsv` or
`Test (Targets)/known_publishedaftert0.tsv`; CBBIO records that path in test-example metadata as
`known_terms_path`. If `Test (Targets)/toi_2025_03.tsv` is present, CBBIO records it as `toi_path`
and restricts GO evaluation to those terms.

Install `kaggle` and configure Kaggle credentials before using `download=True`. The downloader
passes through `KAGGLE_API_TOKEN` when set, or reads `~/.kaggle/access_token` when that file exists.
If `go-basic.obo` or `go.obo` and `IA.txt` are present beside `Train/`, the loader records them in
example metadata so GO tasks can report the challenge metric, `go_weighted_fmax`, with
`go_weighted_precision_at_fmax` and `go_weighted_recall_at_fmax`. The unweighted propagated
`go_fmax` is still reported for comparison. CAFA datasets use the CAFA evaluator threshold step of
`0.001`. Partial-knowledge evaluation excludes propagated known terms from both truth and
predictions before scoring.

### Enzyme Commission Tasks

```python
from CBBIO import load_dataset

dataset = load_dataset(
    "ec:ec_4_main",
    "/data/function_datasets/output/ec_main_head",
    split=("train", "test"),
)
```

The generated EC collection provides multilabel EC-level tasks, single-label variants, and
subclass-holdout variants. Multilabel dataset IDs use `ec:ec_<level>_<track>`, where `level` is
`1` through `4` and `track` is `head`, `main`, or `full`. These tasks prefer `fmax`, a swept
micro-F score over label scores. They also report fixed-threshold `f1`, `macro_f1`,
`weighted_f1`, and `average_precision`.

The `head` and `main` tracks apply class minimum-support filters. The `full` track keeps every
exact EC class for proteins that have at least one EC annotation. Full splits use a deterministic
hash over `tax_id` and `cluster_id`.

Single-label variants use `ec:single_ec_<level>_<track>` and keep proteins with one label at the
requested level.

Subclass-holdout variants use `ec:ec_<level>_subclass_holdout_<track>` for levels `1` through
`3`. They predict a parent EC label while assigning complete child subclasses to one split. No
protein from a held-out child subclass appears in training.

| Dataset level | Predicted label | Held-out subclass |
|---|---|---|
| `ec_1_subclass_holdout_*` | EC level 1 | EC level 2 |
| `ec_2_subclass_holdout_*` | EC level 2 | EC level 3 |
| `ec_3_subclass_holdout_*` | EC level 3 | Exact EC level 4 |

For each parent label, CBBIO keeps at least one child subclass in training. It ranks eligible
child subclasses by frequency and deterministically assigns rare subclasses to test, then
validation, until each reaches about 10% of eligible examples. The task metadata records the
held-out child in `heldout_subclass` and the source split in `original_split`.

### EC Benchmark Tasks

```python
from CBBIO import load_dataset

dataset = load_dataset(
    "clean:ec_4_split30_fold0_halogenase",
    "/data/probing",
)
```

CBBIO also exposes CLEAN and EC-Bench as separate collections because they have their own source
files and evaluation protocols. Both collections derive multilabel targets from exact EC
annotations. Dataset IDs keep the provider first, then the EC level, then the training protocol,
then the optional named test set.

| Collection | Dataset ID pattern |
|---|---|
| CLEAN native split | `clean:ec_<level>_split<threshold>_fold<fold>` |
| CLEAN named test set | `clean:ec_<level>_split<threshold>_fold<fold>_<test_set>` |
| EC-Bench native split | `ecbench:ec_<level>_train_<size>` |
| EC-Bench named test set | `ecbench:ec_<level>_train_<size>_<test_set>` |

CLEAN thresholds are `10`, `30`, `50`, `70`, and `100`. CLEAN folds are `0` through `4`.
EC-Bench train sizes are `30` and `100`. Named test sets are `halogenase`, `price`, and `new`.
The native test split is used when the dataset ID has no test-set suffix.

EC level labels are derived by truncating exact EC annotations:

| Target | Example label |
|---|---|
| `ec_1` | `3` |
| `ec_2` | `3.1` |
| `ec_3` | `3.1.3` |
| `ec_4` | `3.1.3.43` |

Rows with multiple EC annotations keep a deduplicated list at the requested level. Rows with
incomplete EC labels are skipped only when the requested level includes the incomplete segment.

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

## Assigning Dataset Splits

```python
from CBBIO import HashDatasetSplitter, ProteinDataset, ProteinExample

examples = [
    ProteinExample(
        id=f"P{index:05d}",
        sequence="MKTAYIAK",
        labels={"active": index % 2},
        split="train",
    )
    for index in range(20)
]

dataset = ProteinDataset(examples)
splitter = HashDatasetSplitter(salt="activity-split-v1")
split_dataset = splitter.split_dataset(dataset)

print(split_dataset.split_counts())  # {"train": 16, "val": 2, "test": 2}
```

Use splitters when an upstream dataset does not already define train, validation, and test splits.
Splitters sort examples by a salted hash of `example.id`.
The same `salt` and example IDs produce the same split assignments.
Change the `salt` only when you want a new split recipe.

### Loader Overrides

```python
from CBBIO import HashDatasetSplitter, load_dataset

dataset = load_dataset(
    "biolip:all",
    "/data/probing",
    splitter=HashDatasetSplitter(
        salt="biolip-custom-split-v1",
        ratios={"train": 0.7, "val": 0.15, "test": 0.15},
    ),
)
```

Pass `splitter` to `load_dataset()` to override the default generated split recipe.
The override applies only when `split` is `None`.
Datasets with native benchmark splits reject `splitter`.

### Stratified Splits

```python
from CBBIO import ResidueDataset, ResidueExample, StratifiedDatasetSplitter

examples = [
    ResidueExample(
        id=f"HUMAN_{index}",
        sequence="MSTYAS",
        labels={"phosphorylation_site": [0, 1, 0, 0, 0, 0]},
        split="train",
        metadata={"species": "Homo sapiens", "site_type": "S"},
    )
    for index in range(12)
]

dataset = ResidueDataset(examples)
splitter = StratifiedDatasetSplitter(
    metadata_fields=("species", "site_type"),
    salt="phosphorylation-split-v1",
)
split_dataset = splitter.split_dataset(dataset)
```

`StratifiedDatasetSplitter` keeps large metadata groups balanced across splits.
Small groups are pooled by `fallback_metadata_fields`.
The default fallback uses the first field in `metadata_fields`.

### Holdout Splits

```python
from CBBIO import HoldoutDatasetSplitter, ProteinDataset, ProteinExample

dataset = ProteinDataset(
    [
        ProteinExample(
            id="P12345",
            sequence="MKTAYIAK",
            labels={"active": 1},
            split="train",
            metadata={"species": "Homo sapiens"},
        ),
        ProteinExample(
            id="Q67890",
            sequence="MAPLRTLL",
            labels={"active": 0},
            split="train",
            metadata={"species": "Mus musculus"},
        ),
    ]
)

splitter = HoldoutDatasetSplitter(
    metadata_field="species",
    holdout_strategy="random",
    salt="species-holdout-v1",
)
split_dataset = splitter.split_dataset(dataset)
```

`HoldoutDatasetSplitter` selects metadata classes by salted hash and assigns them to `test`.
It adds whole classes until the held-out examples reach the test fraction.
When the test fraction is zero, it uses the validation fraction.
Use `holdout_strategy="rarest_first"` to hold out rare classes first.
Use `holdout_strategy="none"` to disable class holdout.
Pass `holdout_values` only when you need a fixed class list.
It assigns the remaining examples to `train` and `val`.
The default non-held-out ratios are `0.9`, `0.1`, and `0.0` for train, validation, and test.

| Splitter | Use case |
|---|---|
| `HashDatasetSplitter` | Deterministic 80/10/10 splits without stratification |
| `StratifiedDatasetSplitter` | Deterministic splits balanced by metadata fields |
| `HoldoutDatasetSplitter` | Test set defined by a metadata class such as species or family |

---

## Reducing Train/Test Redundancy

Use `filter_redundant_to_test_mmseqs` to remove training sequences that are sequence-similar to test sequences. Requires MMseqs2 to be installed.

```python
from CBBIO import filter_redundant_to_test_mmseqs

filtered_dataset, report = filter_redundant_to_test_mmseqs(
    dataset,
    cutoff=30.0,
    min_coverage=0.8,
)

print(report.removed_count)
```

---

## Metrics Reference

### Binary classification

| Metric | Description |
|---|---|
| `accuracy` | Fraction of correct predictions |
| `f1` | F1 score (harmonic mean of precision and recall) |
| `macro_f1` | Mean of positive-class and negative-class F1 |
| `balanced_accuracy` | Mean of sensitivity and specificity |
| `precision` | Positive predictive value |
| `recall` | True positive rate (sensitivity) |
| `auroc` | Area under the ROC curve |
| `auprc` | Area under the precision-recall curve (more informative for imbalanced tasks) |
| `mcc` | Matthews Correlation Coefficient |

### Multiclass classification

| Metric | Description |
|---|---|
| `accuracy` | Fraction correct |
| `macro_f1` | Macro-averaged F1 across all classes |
| `weighted_f1` | Support-weighted F1 across all classes |

### Regression

| Metric | Description |
|---|---|
| `r2` | Coefficient of determination (R²) |
| `rmse` | Root mean squared error |
| `mae` | Mean absolute error |
| `spearmanr` | Spearman rank correlation |

### Multilabel

| Metric | Description |
|---|---|
| `exact_match` | Fraction of examples whose predicted label set exactly matches the true label set |
| `f1` | Alias for `micro_f1` |
| `micro_f1` | Global F1 over all labels and examples |
| `macro_f1` | Unweighted mean F1 over labels |
| `weighted_f1` | Support-weighted mean F1 over labels |
| `average_precision` | Mean per-label area under the precision-recall curve |
| `fmax` | Best micro-F score after sweeping score thresholds |
| `fmax_threshold` | Score threshold that produced `fmax` |
| `precision_at_fmax` | Micro-precision at the best threshold |
| `recall_at_fmax` | Micro-recall at the best threshold |

GO tasks additionally report propagated metrics. `go_macro_fmax`, `go_macro_precision_at_fmax`,
and `go_macro_recall_at_fmax` use protein-centric macro averaging; `go_fmax`,
`go_precision_at_fmax`, and `go_recall_at_fmax` remain compatibility aliases for those macro
values. `go_micro_fmax`, `go_micro_precision_at_fmax`, and `go_micro_recall_at_fmax` pool
protein-term decisions before computing precision and recall. `go_propagated_f1`,
`go_propagated_precision`, and `go_propagated_recall` use the probe's fixed prediction threshold
when one is available. When IA weights are available, CBBIO also reports weighted macro and micro
Fmax plus the semantic-distance metrics `go_weighted_smin`,
`go_weighted_remaining_uncertainty_at_smin`, and `go_weighted_misinformation_at_smin`.

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
    scores: dict[str, float | list[float]] | None
    metadata: Mapping[str, Any] | None
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
