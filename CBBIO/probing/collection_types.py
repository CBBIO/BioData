"""Neutral metadata and interfaces for probing dataset collections."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from CBBIO.embeddings import EmbeddingInputError

from .datasets import ObjectiveName, ProteinDataset, ResidueDataset
from .splitters import DatasetSplitter


DatasetLevel = Literal[
    "protein",
    "residue",
    "residue_pair",
    "protein_pair",
    "protein_ligand",
]
DatasetStatus = Literal["ready", "adapter", "catalog_only", "blocked"]


@dataclass(frozen=True)
class CollectionMetadata:
    """Describe one dataset collection and its authoritative reference."""

    id: str
    display_name: str
    description: str
    homepage: str | None = None
    citation: str | None = None
    tags: tuple[str, ...] = ()


@dataclass(frozen=True)
class DatasetMetadata:
    """Describe one dataset published by a collection."""

    id: str
    name: str
    display_name: str
    collection: str
    source: str
    category: str
    task_class: str
    preferred_metric: str
    description: str
    level: DatasetLevel
    objective: ObjectiveName
    target: str
    status: DatasetStatus
    metrics: tuple[str, ...] = ()
    split_counts: tuple[int, int, int] | None = None
    sample_count: int | None = None
    positive_count: int | None = None
    negative_count: int | None = None
    homepage: str | None = None
    download_url: str | None = None
    dataset_url: str | None = None
    download_adapter: str | None = None
    import_adapter: str | None = None
    loader: str | None = None
    citation: str | None = None
    tags: tuple[str, ...] = ()
    notes: str = ""


class DatasetCollection(ABC):
    """Expose metadata, discovery, download, and loading through one interface."""

    metadata: CollectionMetadata

    @property
    def id(self) -> str:
        """Return the collection identifier."""
        return self.metadata.id

    @property
    def display_name(self) -> str:
        """Return the collection display name."""
        return self.metadata.display_name

    @abstractmethod
    def list_datasets(self) -> list[DatasetMetadata]:
        """Return datasets published by this collection."""

    def get_dataset(self, name: str) -> DatasetMetadata:
        """Return one dataset published by this collection."""
        normalized = str(name).strip().lower().replace("-", "_")
        matches = [
            dataset
            for dataset in self.list_datasets()
            if normalized
            in {
                dataset.id.lower(),
                dataset.name.lower(),
                dataset.id.partition(":")[2].lower(),
            }
        ]
        if len(matches) == 1:
            return matches[0]
        supported = ", ".join(sorted(dataset.name for dataset in self.list_datasets()))
        raise EmbeddingInputError(
            f"Unknown dataset {name!r} in collection {self.id!r}. "
            f"Supported values: {supported}."
        )

    @abstractmethod
    def download(
        self,
        root: str | Path,
        *,
        name: str,
        force: bool = False,
    ) -> list[Path]:
        """Download one dataset and return its local paths."""

    @abstractmethod
    def load(
        self,
        root: str | Path,
        *,
        name: str,
        split: str | Sequence[str] | None = None,
        target: str | None = None,
        download: bool = False,
        max_examples_per_split: Mapping[str, int | None] | None = None,
        splitter: DatasetSplitter | None = None,
    ) -> ProteinDataset | ResidueDataset:
        """Load one dataset into the canonical in-memory representation."""


def residue_dataset_metadata(
    *,
    dataset_id: str,
    name: str,
    display_name: str,
    source: str,
    category: str,
    objective: ObjectiveName,
    target: str,
    status: DatasetStatus,
    homepage: str,
    description: str,
    download_url: str | None = None,
    download_adapter: str | None = None,
    import_adapter: str | None = None,
    loader: str | None = None,
    tags: tuple[str, ...] = (),
    notes: str = "",
) -> DatasetMetadata:
    """Build normalized metadata for one residue-level provider dataset."""
    task_classes = {
        "binding": "binding",
        "disorder": "structure",
        "ptm": "ptms",
        "structure": "structure",
    }
    metrics = {
        "binary": "f1",
        "multiclass": "accuracy" if category == "structure" else "macro_f1",
        "multilabel": "macro_f1",
        "regression": "spearmanr",
    }
    return DatasetMetadata(
        id=dataset_id,
        name=name,
        display_name=display_name,
        collection=dataset_id.partition(":")[0],
        source=source,
        category=category,
        task_class=task_classes.get(category, category),
        preferred_metric=metrics[objective],
        description=description,
        level="residue",
        objective=objective,
        target=target,
        status=status,
        homepage=homepage,
        download_url=download_url,
        download_adapter=download_adapter,
        import_adapter=import_adapter,
        loader=loader,
        tags=tags,
        notes=notes,
    )


__all__ = [
    "CollectionMetadata",
    "DatasetCollection",
    "DatasetLevel",
    "DatasetMetadata",
    "DatasetStatus",
    "residue_dataset_metadata",
]
