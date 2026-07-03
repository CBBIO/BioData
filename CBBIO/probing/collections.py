"""Compatibility exports for collection types and the built-in registry."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import List, cast

from CBBIO.embeddings import EmbeddingInputError

from .collection_types import (
    CollectionMetadata,
    DatasetCollection,
    DatasetLevel,
    DatasetMetadata,
    DatasetStatus,
)
from .datasets import ProteinDataset, ResidueDataset, SplitName
from .splitters import DatasetSplitter
from .sources.biolip import BIOLIP_COLLECTION_METADATA, BIOLIP_DATASETS
from .sources.cafa5 import CAFA5_DATASETS
from .sources.cafa5 import CAFA_COLLECTION_METADATA
from .sources.clean import CLEAN_COLLECTION_METADATA, CLEAN_DATASETS
from .sources.dbptm import DBPTM_COLLECTION_METADATA, DBPTM_DATASETS
from .sources.disprot import DISPROT_COLLECTION_METADATA, DISPROT_DATASETS
from .sources.dtu import DTU_COLLECTION_METADATA, DTU_DATASETS
from .sources.ec import EC_COLLECTION_METADATA, EC_DATASETS
from .sources.ecbench import ECBENCH_COLLECTION_METADATA, ECBENCH_DATASETS
from .sources.go import GO_COLLECTION_METADATA, GO_DATASETS
from .sources.musitedeep import MUSITEDEEP_COLLECTION_METADATA, MUSITEDEEP_DATASETS
from .sources.peer import PEER_COLLECTION_METADATA, PEER_DATASETS
from .sources.phosphoelm import PHOSPHOELM_COLLECTION_METADATA, PHOSPHOELM_DATASETS
from .sources._registry import ADAPTER_COLLECTION_METADATA, ADAPTER_DATASETS


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
        from .sources.peer import download_peer_dataset

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
        splitter: DatasetSplitter | None = None,
    ) -> ProteinDataset | ResidueDataset:
        """Load one PEER dataset."""
        from .sources.peer import load_peer_dataset

        if target is not None:
            raise EmbeddingInputError("PEER collection datasets define their target metadata.")
        if splitter is not None:
            raise EmbeddingInputError(
                "PEER datasets use native benchmark splits and do not accept splitter."
            )
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
        splitter: DatasetSplitter | None = None,
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
            splitter=splitter,
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
        splitter: DatasetSplitter | None = None,
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
                splitter=splitter,
            )
        if splitter is not None:
            raise EmbeddingInputError(
                "dbPTM benchmark datasets define fixed splits and do not accept splitter."
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
        splitter: DatasetSplitter | None = None,
    ) -> ProteinDataset | ResidueDataset:
        """Reject loading because DTU collection adapters are not implemented."""
        _ = root, split, target, download, max_examples_per_split, splitter
        dataset = self.get_dataset(name)
        raise EmbeddingInputError(f"Dataset {dataset.id!r} is catalog-only and cannot be loaded.")


class EcCollection(DatasetCollection):
    """Expose generated enzyme commission prediction datasets."""

    metadata = EC_COLLECTION_METADATA

    def list_datasets(self) -> List[DatasetMetadata]:
        """Return generated EC prediction dataset specifications."""
        return list(EC_DATASETS)

    def download(
        self,
        root: str | Path,
        *,
        name: str,
        force: bool = False,
    ) -> List[Path]:
        """Reject downloads because generated EC datasets are local artifacts."""
        _ = root, force
        dataset = self.get_dataset(name)
        raise EmbeddingInputError(
            f"Dataset {dataset.id!r} does not have a download adapter. "
            "Point load_dataset() at a generated ec_main_head directory."
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
        splitter: DatasetSplitter | None = None,
    ) -> ProteinDataset | ResidueDataset:
        """Load one generated EC prediction dataset."""
        from .sources.ec import load_ec_dataset

        _ = download
        if max_examples_per_split is not None:
            raise EmbeddingInputError("EC datasets do not support max_examples_per_split.")
        if splitter is not None:
            raise EmbeddingInputError("EC datasets define fixed splits and do not accept splitter.")
        dataset = self.get_dataset(name)
        return load_ec_dataset(root, name=dataset.name, split=split, target=target)


class CleanCollection(DatasetCollection):
    """Expose CLEAN enzyme commission benchmark datasets."""

    metadata = CLEAN_COLLECTION_METADATA

    def list_datasets(self) -> List[DatasetMetadata]:
        """Return CLEAN EC benchmark dataset specifications."""
        return list(CLEAN_DATASETS)

    def download(
        self,
        root: str | Path,
        *,
        name: str,
        force: bool = False,
    ) -> List[Path]:
        """Reject downloads because CLEAN datasets are local artifacts."""
        _ = root, force
        dataset = self.get_dataset(name)
        raise EmbeddingInputError(
            f"Dataset {dataset.id!r} does not have a download adapter. "
            "Point load_dataset() at a CLEAN dataset directory."
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
        splitter: DatasetSplitter | None = None,
    ) -> ProteinDataset | ResidueDataset:
        """Load one CLEAN EC benchmark dataset."""
        from .sources.clean import load_clean_dataset

        _ = download
        if max_examples_per_split is not None:
            raise EmbeddingInputError("CLEAN datasets do not support max_examples_per_split.")
        if splitter is not None:
            raise EmbeddingInputError("CLEAN datasets define fixed splits and do not accept splitter.")
        dataset = self.get_dataset(name)
        return load_clean_dataset(root, name=dataset.name, split=split, target=target)


class EcBenchCollection(DatasetCollection):
    """Expose EC-Bench enzyme commission benchmark datasets."""

    metadata = ECBENCH_COLLECTION_METADATA

    def list_datasets(self) -> List[DatasetMetadata]:
        """Return EC-Bench dataset specifications."""
        return list(ECBENCH_DATASETS)

    def download(
        self,
        root: str | Path,
        *,
        name: str,
        force: bool = False,
    ) -> List[Path]:
        """Reject downloads because EC-Bench datasets are local artifacts."""
        _ = root, force
        dataset = self.get_dataset(name)
        raise EmbeddingInputError(
            f"Dataset {dataset.id!r} does not have a download adapter. "
            "Point load_dataset() at an ec-benchmark directory."
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
        splitter: DatasetSplitter | None = None,
    ) -> ProteinDataset | ResidueDataset:
        """Load one EC-Bench dataset."""
        from .sources.ecbench import load_ecbench_dataset

        _ = download
        if max_examples_per_split is not None:
            raise EmbeddingInputError("EC-Bench datasets do not support max_examples_per_split.")
        if splitter is not None:
            raise EmbeddingInputError("EC-Bench datasets define fixed splits and do not accept splitter.")
        dataset = self.get_dataset(name)
        return load_ecbench_dataset(root, name=dataset.name, split=split, target=target)


class GoCollection(DatasetCollection):
    """Expose generated Gene Ontology prediction datasets."""

    metadata = GO_COLLECTION_METADATA

    def list_datasets(self) -> List[DatasetMetadata]:
        """Return generated Gene Ontology dataset specifications."""
        return list(GO_DATASETS)

    def download(
        self,
        root: str | Path,
        *,
        name: str,
        force: bool = False,
    ) -> List[Path]:
        """Reject downloads because generated GO datasets are local artifacts."""
        _ = root, force
        dataset = self.get_dataset(name)
        raise EmbeddingInputError(
            f"Dataset {dataset.id!r} does not have a download adapter. "
            "Point load_dataset() at a generated go_main_head directory."
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
        splitter: DatasetSplitter | None = None,
    ) -> ProteinDataset | ResidueDataset:
        """Load one generated Gene Ontology prediction dataset."""
        from .sources.go import load_go_dataset

        _ = download
        if max_examples_per_split is not None:
            raise EmbeddingInputError("GO datasets do not support max_examples_per_split.")
        if splitter is not None:
            raise EmbeddingInputError("GO datasets define fixed splits and do not accept splitter.")
        dataset = self.get_dataset(name)
        return load_go_dataset(root, name=dataset.name, split=split, target=target)


class CafaCollection(DatasetCollection):
    """Expose local CAFA Kaggle competition training datasets."""

    metadata = CAFA_COLLECTION_METADATA

    def list_datasets(self) -> List[DatasetMetadata]:
        """Return CAFA aspect dataset specifications."""
        return [*CAFA5_DATASETS]

    def download(
        self,
        root: str | Path,
        *,
        name: str,
        force: bool = False,
    ) -> List[Path]:
        """Download CAFA competition files through the Kaggle CLI."""
        from .sources.cafa5 import download_cafa5_dataset

        _ = self.get_dataset(name)
        return download_cafa5_dataset(root, force=force)

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
        """Load one CAFA aspect dataset."""
        from .sources.cafa5 import download_cafa5_dataset, load_cafa5_dataset

        if max_examples_per_split is not None:
            raise EmbeddingInputError("CAFA datasets do not support max_examples_per_split.")
        dataset = self.get_dataset(name)
        if download:
            download_cafa5_dataset(root)
        return load_cafa5_dataset(
            root,
            name=dataset.name,
            split=split,
            target=target,
            splitter=splitter,
        )


def list_dataset_collections() -> List[DatasetCollection]:
    """Return registered dataset collections."""
    return list(DATASET_COLLECTIONS.values())


def get_dataset_collection(collection_id: str) -> DatasetCollection:
    """Return one registered dataset collection."""
    normalized = str(collection_id).strip().lower().replace("-", "_")
    if normalized == "cafa5":
        normalized = "cafa"
    if normalized in {"ec_bench", "ec_benchmark"}:
        normalized = "ecbench"
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
EC_COLLECTION = EcCollection()
CLEAN_COLLECTION = CleanCollection()
ECBENCH_COLLECTION = EcBenchCollection()
GO_COLLECTION = GoCollection()
CAFA_COLLECTION = CafaCollection()

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
    EC_COLLECTION.id: EC_COLLECTION,
    CLEAN_COLLECTION.id: CLEAN_COLLECTION,
    ECBENCH_COLLECTION.id: ECBENCH_COLLECTION,
    GO_COLLECTION.id: GO_COLLECTION,
    CAFA_COLLECTION.id: CAFA_COLLECTION,
    **_RESIDUE_COLLECTIONS,
    DTU_COLLECTION.id: DTU_COLLECTION,
}


__all__ = [
    "DATASET_COLLECTIONS",
    "DBPTM_COLLECTION",
    "CAFA_COLLECTION",
    "CafaCollection",
    "CLEAN_COLLECTION",
    "CleanCollection",
    "DTU_COLLECTION",
    "ECBENCH_COLLECTION",
    "EC_COLLECTION",
    "GO_COLLECTION",
    "DatasetCollection",
    "DatasetLevel",
    "DatasetMetadata",
    "DatasetStatus",
    "DbptmCollection",
    "DtuCollection",
    "EcBenchCollection",
    "EcCollection",
    "GoCollection",
    "PEER_COLLECTION",
    "PeerCollection",
    "get_dataset_collection",
    "list_dataset_collections",
]
