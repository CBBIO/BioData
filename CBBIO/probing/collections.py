"""Compatibility exports for collection types and the built-in registry."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import List, Tuple, cast

from CBBIO.embeddings import EmbeddingInputError

from .collection_types import (
    CollectionMetadata,
    DatasetCollection,
    DatasetLevel,
    DatasetMetadata,
    DatasetStatus,
)
from .biolip import BIOLIP_COLLECTION_METADATA, BIOLIP_DATASETS
from .datasets import ObjectiveName, ProteinDataset, ResidueDataset, SplitName
from .dbptm import DBPTM_COLLECTION_METADATA, DBPTM_DATASETS
from .disprot import DISPROT_COLLECTION_METADATA, DISPROT_DATASETS
from .dtu import DTU_COLLECTION_METADATA, DTU_DATASETS
from .musitedeep import MUSITEDEEP_COLLECTION_METADATA, MUSITEDEEP_DATASETS
from .peer import PEER_COLLECTION_METADATA, PEER_DATASETS
from .phosphoelm import PHOSPHOELM_COLLECTION_METADATA, PHOSPHOELM_DATASETS
from .residue_registry import ADAPTER_COLLECTION_METADATA, ADAPTER_DATASETS



class PeerCollection(DatasetCollection):
    """Expose PEER benchmark tasks as one dataset collection."""

    metadata = PEER_COLLECTION_METADATA

    def list_datasets(self) -> List[DatasetMetadata]:
        """Return PEER benchmark dataset specifications."""
        return list(PEER_DATASETS)

    def download(
        self,
        root: str | Path,
        *,
        name: str,
        force: bool = False,
    ) -> List[Path]:
        """Download one native PEER dataset."""
        from .peer import download_peer_dataset

        dataset = self.get_dataset(name)
        return [download_peer_dataset(root, name=dataset.name, force=force)]

    def load(
        self,
        root: str | Path,
        *,
        name: str,
        split: str | Sequence[str] | None = None,
        target: str | None = None,
        download: bool = False,
        max_examples_per_split: Mapping[str, int | None] | None = None,
    ) -> ProteinDataset | ResidueDataset:
        """Load one PEER dataset."""
        from .peer import load_peer_dataset

        if target is not None:
            raise EmbeddingInputError("PEER collection datasets define their target metadata.")
        dataset = self.get_dataset(name)
        return load_peer_dataset(
            root,
            name=dataset.name,
            split=split,
            download=download,
            max_examples_per_split=max_examples_per_split,
        )


class _ResidueDatasetCollection(DatasetCollection):
    """Expose one residue annotation provider as a dataset collection."""

    def __init__(self, metadata: CollectionMetadata) -> None:
        self.metadata = metadata

    def list_datasets(self) -> List[DatasetMetadata]:
        """Return datasets published by this residue annotation provider."""
        return [
            dataset
            for dataset in _RESIDUE_DATASET_METADATA
            if dataset.collection == self.id
        ]

    def download(
        self,
        root: str | Path,
        *,
        name: str,
        force: bool = False,
    ) -> List[Path]:
        """Download one residue annotation dataset."""
        from .residue_sources import download_residue_source

        dataset = self.get_dataset(name)
        return download_residue_source(root, name=dataset.name, force=force)

    def load(
        self,
        root: str | Path,
        *,
        name: str,
        split: str | Sequence[str] | None = None,
        target: str | None = None,
        download: bool = False,
        max_examples_per_split: Mapping[str, int | None] | None = None,
    ) -> ProteinDataset | ResidueDataset:
        """Load one residue annotation dataset."""
        from .residue_sources import load_residue_source_dataset

        if max_examples_per_split is not None:
            raise EmbeddingInputError(
                "max_examples_per_split is only supported by the PEER collection."
            )
        if split is not None and not isinstance(split, str):
            raise EmbeddingInputError("Residue source collections accept one split name.")
        dataset = self.get_dataset(name)
        return load_residue_source_dataset(
            root,
            name=dataset.name,
            target=target,
            split=cast(SplitName | None, split),
            download=download,
        )


class DbptmCollection(DatasetCollection):
    """Expose dbPTM source data and published benchmark archives."""

    metadata = DBPTM_COLLECTION_METADATA

    def list_datasets(self) -> List[DatasetMetadata]:
        """Return dbPTM source and benchmark specifications."""
        return list(DBPTM_DATASETS)

    def download(
        self,
        root: str | Path,
        *,
        name: str,
        force: bool = False,
    ) -> List[Path]:
        """Download one dbPTM source or benchmark dataset."""
        from .residue_sources import download_dbptm_benchmark, download_residue_source

        dataset = self.get_dataset(name)
        if dataset.name == "dbptm":
            return download_residue_source(root, name=dataset.name, force=force)
        return [download_dbptm_benchmark(root, name=dataset.name, force=force)]

    def load(
        self,
        root: str | Path,
        *,
        name: str,
        split: str | Sequence[str] | None = None,
        target: str | None = None,
        download: bool = False,
        max_examples_per_split: Mapping[str, int | None] | None = None,
    ) -> ProteinDataset | ResidueDataset:
        """Load one dbPTM source or benchmark dataset."""
        from .residue_sources import load_dbptm_benchmark_dataset, load_residue_source_dataset

        if max_examples_per_split is not None:
            raise EmbeddingInputError(
                "max_examples_per_split is only supported by the PEER collection."
            )
        if split is not None and not isinstance(split, str):
            raise EmbeddingInputError("The dbPTM collection accepts one split name.")
        dataset = self.get_dataset(name)
        resolved_split = cast(SplitName | None, split)
        if dataset.name == "dbptm":
            return load_residue_source_dataset(
                root,
                name=dataset.name,
                target=target,
                split=resolved_split,
                download=download,
            )
        return load_dbptm_benchmark_dataset(
            root,
            name=dataset.name,
            target=target,
            split=resolved_split or "train",
            download=download,
        )


class DtuCollection(DatasetCollection):
    """Expose catalog-only DTU dataset candidates."""

    metadata = DTU_COLLECTION_METADATA

    def list_datasets(self) -> List[DatasetMetadata]:
        """Return catalog-only DTU dataset specifications."""
        return list(DTU_DATASETS)

    def download(
        self,
        root: str | Path,
        *,
        name: str,
        force: bool = False,
    ) -> List[Path]:
        """Reject downloads because DTU collection adapters are not implemented."""
        _ = root, force
        dataset = self.get_dataset(name)
        raise EmbeddingInputError(
            f"Dataset {dataset.id!r} is catalog-only and has no download adapter."
        )

    def load(
        self,
        root: str | Path,
        *,
        name: str,
        split: str | Sequence[str] | None = None,
        target: str | None = None,
        download: bool = False,
        max_examples_per_split: Mapping[str, int | None] | None = None,
    ) -> ProteinDataset | ResidueDataset:
        """Reject loading because DTU collection adapters are not implemented."""
        _ = root, split, target, download, max_examples_per_split
        dataset = self.get_dataset(name)
        raise EmbeddingInputError(f"Dataset {dataset.id!r} is catalog-only and cannot be loaded.")


def list_dataset_collections() -> List[DatasetCollection]:
    """Return registered dataset collections."""
    return list(DATASET_COLLECTIONS.values())


def get_dataset_collection(collection_id: str) -> DatasetCollection:
    """Return one registered dataset collection."""
    normalized = str(collection_id).strip().lower().replace("-", "_")
    collection = DATASET_COLLECTIONS.get(normalized)
    if collection is None:
        supported = ", ".join(sorted(DATASET_COLLECTIONS))
        raise EmbeddingInputError(
            f"Unknown dataset collection {collection_id!r}. Supported values: {supported}."
        )
    return collection



_RESIDUE_DATASET_METADATA = (
    *MUSITEDEEP_DATASETS,
    *DISPROT_DATASETS,
    *BIOLIP_DATASETS,
    *PHOSPHOELM_DATASETS,
    *ADAPTER_DATASETS,
)

PEER_COLLECTION = PeerCollection()
DBPTM_COLLECTION = DbptmCollection()
DTU_COLLECTION = DtuCollection()

_PROVIDER_COLLECTION_METADATA = (
    MUSITEDEEP_COLLECTION_METADATA,
    DISPROT_COLLECTION_METADATA,
    BIOLIP_COLLECTION_METADATA,
    PHOSPHOELM_COLLECTION_METADATA,
    *ADAPTER_COLLECTION_METADATA,
)
_RESIDUE_COLLECTIONS = {
    metadata.id: _ResidueDatasetCollection(metadata)
    for metadata in _PROVIDER_COLLECTION_METADATA
}

DATASET_COLLECTIONS: dict[str, DatasetCollection] = {
    PEER_COLLECTION.id: PEER_COLLECTION,
    DBPTM_COLLECTION.id: DBPTM_COLLECTION,
    **_RESIDUE_COLLECTIONS,
    DTU_COLLECTION.id: DTU_COLLECTION,
}


__all__ = [
    "DATASET_COLLECTIONS",
    "DBPTM_COLLECTION",
    "DTU_COLLECTION",
    "DatasetCollection",
    "DatasetLevel",
    "DatasetMetadata",
    "DatasetStatus",
    "DbptmCollection",
    "DtuCollection",
    "PEER_COLLECTION",
    "PeerCollection",
    "get_dataset_collection",
    "list_dataset_collections",
]
