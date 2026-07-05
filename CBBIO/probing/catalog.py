"""Search and route datasets published by registered collections."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

from CBBIO.embeddings import EmbeddingInputError

from .collections import (
    DATASET_COLLECTIONS,
    DatasetCollection,
    DatasetLevel,
    DatasetMetadata,
    DatasetStatus,
    get_dataset_collection,
    list_dataset_collections,
)
from .datasets import ObjectiveName, ProteinDataset, ResidueDataset
from .splitters import DatasetSplitter


def build_dataset_catalog(
    collections: Sequence[DatasetCollection] | None = None,
) -> dict[str, DatasetMetadata]:
    """Build a flattened dataset index from collections."""
    selected = list_dataset_collections() if collections is None else list(collections)
    catalog: dict[str, DatasetMetadata] = {}
    for collection in selected:
        for dataset in collection.list_datasets():
            if dataset.collection != collection.id:
                raise EmbeddingInputError(
                    f"Dataset {dataset.id!r} declares collection {dataset.collection!r}, "
                    f"but was published by {collection.id!r}."
                )
            if dataset.id in catalog:
                raise EmbeddingInputError(f"Duplicate dataset catalog id {dataset.id!r}.")
            catalog[dataset.id] = dataset
    return catalog


def get_dataset_catalog_entry(dataset_id: str) -> DatasetMetadata:
    """Return one registered dataset specification."""
    key = _normalize_catalog_id(dataset_id)
    entry = DATASET_CATALOG.get(key)
    if entry is None:
        supported = ", ".join(sorted(DATASET_CATALOG))
        raise EmbeddingInputError(
            f"Unknown dataset catalog id {dataset_id!r}. Supported values: {supported}."
        )
    return entry


def list_dataset_catalog(
    *,
    collection: str | None = None,
    source: str | None = None,
    category: str | None = None,
    task_class: str | None = None,
    preferred_metric: str | None = None,
    level: DatasetLevel | None = None,
    objective: ObjectiveName | None = None,
    status: DatasetStatus | None = None,
    has_download: bool | None = None,
    has_loader: bool | None = None,
    tag: str | None = None,
) -> list[DatasetMetadata]:
    """Return dataset specifications matching the requested metadata."""
    values = list(DATASET_CATALOG.values())
    if collection is not None:
        normalized = collection.strip().lower()
        values = [value for value in values if value.collection.lower() == normalized]
    if source is not None:
        normalized = source.strip().lower()
        values = [value for value in values if value.source.lower() == normalized]
    if category is not None:
        normalized = category.strip().lower()
        values = [value for value in values if value.category.lower() == normalized]
    if task_class is not None:
        normalized = task_class.strip().lower()
        values = [value for value in values if value.task_class.lower() == normalized]
    if preferred_metric is not None:
        normalized = preferred_metric.strip().lower()
        values = [
            value
            for value in values
            if value.preferred_metric.lower() == normalized
        ]
    if level is not None:
        values = [value for value in values if value.level == level]
    if objective is not None:
        values = [value for value in values if value.objective == objective]
    if status is not None:
        values = [value for value in values if value.status == status]
    if has_download is not None:
        values = [
            value
            for value in values
            if (value.download_url is not None or value.download_adapter is not None)
            == has_download
        ]
    if has_loader is not None:
        values = [
            value
            for value in values
            if (value.loader is not None or value.import_adapter is not None) == has_loader
        ]
    if tag is not None:
        normalized = tag.strip().lower()
        values = [value for value in values if normalized in value.tags]
    return values


def search_dataset_catalog(
    query: str,
    *,
    limit: int | None = 20,
    collection: str | None = None,
    category: str | None = None,
    task_class: str | None = None,
    level: DatasetLevel | None = None,
    objective: ObjectiveName | None = None,
    status: DatasetStatus | None = None,
) -> list[DatasetMetadata]:
    """Search dataset specifications with ranked token matching."""
    tokens = _query_tokens(query)
    if not tokens:
        return []
    candidates = list_dataset_catalog(
        collection=collection,
        category=category,
        task_class=task_class,
        level=level,
        objective=objective,
        status=status,
    )
    scored = [(_catalog_search_score(entry, tokens), entry) for entry in candidates]
    matches = [(score, entry) for score, entry in scored if score > 0]
    matches.sort(key=lambda item: (-item[0], item[1].id))
    entries = [entry for _score, entry in matches]
    return entries if limit is None else entries[:limit]


def download_dataset(
    dataset_id: str,
    root: str | Path,
    *,
    force: bool = False,
) -> list[Path]:
    """Download a catalog dataset through its owning collection."""
    dataset = get_dataset_catalog_entry(dataset_id)
    collection = get_dataset_collection(dataset.collection)
    return collection.download(root, name=dataset.name, force=force)


def load_dataset(
    dataset_id: str,
    root: str | Path,
    *,
    split: str | Sequence[str] | None = None,
    target: str | None = None,
    download: bool = False,
    max_examples_per_split: Mapping[str, int | None] | None = None,
    splitter: DatasetSplitter | None = None,
) -> ProteinDataset | ResidueDataset:
    """Load a catalog dataset through its owning collection."""
    dataset = get_dataset_catalog_entry(dataset_id)
    if dataset.status not in {"ready", "adapter"}:
        raise EmbeddingInputError(
            f"Dataset {dataset.id!r} has status {dataset.status!r} and cannot be loaded."
        )
    collection = get_dataset_collection(dataset.collection)
    return collection.load(
        root,
        name=dataset.name,
        split=split,
        target=target,
        download=download,
        max_examples_per_split=max_examples_per_split,
        splitter=splitter,
    )


def _normalize_catalog_id(dataset_id: str) -> str:
    return dataset_id.strip().lower()


def _query_tokens(query: str) -> tuple[str, ...]:
    return tuple(token for token in query.strip().lower().replace("-", "_").split() if token)


def _catalog_search_score(entry: DatasetMetadata, tokens: tuple[str, ...]) -> int:
    exact_fields = {
        entry.id.lower(),
        entry.name.lower(),
        entry.collection.lower(),
        entry.source.lower(),
        entry.category.lower(),
        entry.task_class.lower(),
        entry.preferred_metric.lower(),
        entry.level.lower(),
        entry.objective.lower(),
        entry.target.lower(),
        *(tag.lower() for tag in entry.tags),
    }
    high = " ".join(
        [entry.id, entry.name, entry.display_name, entry.source, entry.task_class, entry.target]
        + list(entry.tags)
    ).lower()
    medium = " ".join(
        [
            entry.category,
            entry.preferred_metric,
            entry.collection,
            entry.level,
            entry.objective,
            entry.loader or "",
            entry.import_adapter or "",
            entry.download_adapter or "",
        ]
    ).lower()
    low = " ".join(
        [entry.description, entry.notes, entry.homepage or "", entry.dataset_url or ""]
    ).lower()
    score = 0
    for token in tokens:
        if token in exact_fields:
            score += 20
        if token in high:
            score += 8
        if token in medium:
            score += 4
        if token in low:
            score += 2
    return score


DATASET_CATALOG: dict[str, DatasetMetadata] = build_dataset_catalog()


__all__ = [
    "DATASET_CATALOG",
    "DATASET_COLLECTIONS",
    "build_dataset_catalog",
    "download_dataset",
    "get_dataset_catalog_entry",
    "load_dataset",
    "list_dataset_catalog",
    "search_dataset_catalog",
]
