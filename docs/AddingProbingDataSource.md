# Adding a Probing Data Source

Add a provider through a dataset collection so its metadata, downloads, and loaders remain
consistent with the probing catalog.

```python
from CBBIO import get_dataset_catalog_entry, get_dataset_collection

collection = get_dataset_collection("phosphoelm")
dataset = get_dataset_catalog_entry("phosphoelm:all")

print(collection.metadata.homepage)
print(dataset.target)  # "phosphorylation_site"
```

A provider owns its collection metadata and dataset metadata. The collection exposes discovery,
download, and loading. The catalog indexes the datasets published by every registered collection.

## Provider Contract

```python
from CBBIO import DatasetMetadata, ResidueDataset, get_dataset_collection

collection = get_dataset_collection("phosphoelm")
datasets: list[DatasetMetadata] = collection.list_datasets()
loaded: ResidueDataset = collection.load(
    "/data/probing",
    name="phosphoelm_all",
    split="test",
)
```

Create the provider module under `CBBIO/probing/sources/`. Use a domain name such as
`phosphoelm.py`; do not add parsing code to `catalog.py` or `collections.py`.

The provider module must define:

- One collection metadata constant named `<PROVIDER>_COLLECTION_METADATA`.
- One tuple of dataset metadata named `<PROVIDER>_DATASETS`.
- Public loader functions that return `ProteinDataset` or `ResidueDataset`.
- Download functions when CBBIO can retrieve the source directly.
- An explicit `__all__` containing only public provider names.

Collection metadata requires the following fields:

| Field | Description |
|---|---|
| `id` | Stable lowercase collection identifier |
| `display_name` | Provider name shown to users |
| `description` | One-sentence description of the collection |
| `homepage` | Authoritative provider page, when available |
| `citation` | Publication reference, when available |
| `tags` | Search terms shared by the collection |

Each `DatasetMetadata` must use a qualified ID in the form `<collection>:<dataset>`. Its
`collection` field must equal the collection metadata ID.

| Field | Purpose |
|---|---|
| `id` | Qualified catalog ID, such as `phosphoelm:all` |
| `name` | Name accepted by the collection loader |
| `level` | `"protein"`, `"residue"`, or another supported dataset level |
| `objective` | Probe objective such as `"binary"` or `"regression"` |
| `target` | Label key stored in each example |
| `status` | `"ready"`, `"adapter"`, `"catalog_only"`, or `"blocked"` |
| `loader` | Public loader name, when implemented |
| `download_adapter` | Public download function name, when implemented |
| `import_adapter` | Public raw-data importer name, when implemented |

Use `status="ready"` only when `load_dataset()` can load the dataset. Use `status="adapter"` when
an importer exists but the source still requires user-provided files. Use `status="catalog_only"`
for metadata without an implementation.

All loaders must follow the common collection inputs where they apply:

| Parameter | Meaning |
|---|---|
| `root` | Local data root |
| `name` | Dataset name within the collection |
| `split` | Requested split or splits |
| `target` | Optional label-key override |
| `download` | Download missing source data before loading |
| `max_examples_per_split` | Optional per-split limit |
| `splitter` | Optional generated-split override |

Provider-specific raw-data importers may expose additional keyword-only options. Keep those options
on the source-specific function and preserve the common behavior through the collection.

## Registration

```python
from CBBIO import get_dataset_catalog_entry, list_dataset_collections

assert "phosphoelm" in {collection.id for collection in list_dataset_collections()}
assert get_dataset_catalog_entry("phosphoelm:all").collection == "phosphoelm"
```

Register a standard residue provider in `CBBIO/probing/collections.py`:

1. Import its collection and dataset metadata constants.
2. Add its datasets to `_RESIDUE_DATASET_METADATA`.
3. Add its collection metadata to `_PROVIDER_COLLECTION_METADATA`.

The catalog rebuilds `DATASET_CATALOG` from `DATASET_COLLECTIONS`. Do not copy provider metadata
into `catalog.py`.

The shared residue dispatch facade also requires these changes:

1. Add the source specification and aliases to `CBBIO/probing/sources/_registry.py`.
2. Route download and load operations to the provider functions.
3. Re-export new public functions from `CBBIO/probing/__init__.py` and `CBBIO/__init__.py`.

Add a dedicated `DatasetCollection` subclass when the provider cannot use the standard residue
dispatch behavior. Implement `list_datasets()`, `download()`, and `load()`, then add one instance to
`DATASET_COLLECTIONS`.

## Tests

```python
from CBBIO import get_dataset_catalog_entry, get_dataset_collection


def test_phosphoelm_collection_publishes_its_dataset_metadata() -> None:
    collection = get_dataset_collection("phosphoelm")
    dataset = get_dataset_catalog_entry("phosphoelm:all")

    assert dataset in collection.list_datasets()
    assert dataset.collection == collection.id
```

Add tests to `tests/test_probing.py`. Import only from `CBBIO`, including tests for source-specific
public functions.

Cover these behaviors:

- The collection publishes every provider dataset exactly once.
- Qualified IDs resolve through `get_dataset_catalog_entry()`.
- `load_dataset()` delegates to the collection and returns the expected dataset type.
- Split and target handling preserve labels and masks.
- Invalid rows raise `EmbeddingInputError` with an actionable message.
- Downloads use temporary paths and mocked network calls.

Run the probing tests and type checker:

```bash
poetry run pytest -q tests/test_probing.py
poetry run pyright
```

## Checklist

```python
from CBBIO import list_dataset_catalog

registered_ids = {dataset.id for dataset in list_dataset_catalog(collection="phosphoelm")}
assert "phosphoelm:all" in registered_ids
```

- Read `docs/STYLE.md`, `docs/STYLE_DOCS.md`, and `docs/STYLE_TESTS.md`.
- Keep provider parsing and metadata in its source module.
- Keep shared orchestration in the collection and catalog layers.
- Add complete type annotations and public docstrings.
- Use deterministic splits when the upstream dataset does not define them.
- Preserve source provenance in example metadata.
- Export public names through the top-level `CBBIO` namespace.
- Add or update the provider table in `docs/Probing.md`.
- Run `pytest`, `pyright`, and `git diff --check`.

## Exceptions

| Exception | When raised |
|---|---|
| `EmbeddingInputError` | Invalid metadata, source rows, labels, splits, or dataset names |
| `EmbeddingDependencyError` | Optional dependency required by the provider is unavailable |
