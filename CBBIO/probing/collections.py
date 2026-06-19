"""Common dataset collection interface and built-in collection registry."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import List, Literal, Tuple, cast

from CBBIO.embeddings import EmbeddingInputError

from .datasets import ObjectiveName, ProteinDataset, ResidueDataset, SplitName
from .dtu import DTU_PROBING_CATALOG
from .peer import PEER_CITATION, PEER_TASKS, PeerTaskLevel
from .residue_sources import (
    DBPTM_BENCHMARKS,
    DISPROT_CURRENT_JSON_URL,
    RESIDUE_SOURCE_SPECS,
)


DatasetLevel = PeerTaskLevel
DatasetStatus = Literal["ready", "adapter", "catalog_only", "blocked"]


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
    metrics: Tuple[str, ...] = ()
    split_counts: Tuple[int, int, int] | None = None
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
    tags: Tuple[str, ...] = ()
    notes: str = ""


class DatasetCollection(ABC):
    """Expose dataset discovery, download, and loading through one interface."""

    id: str
    display_name: str

    @abstractmethod
    def list_datasets(self) -> List[DatasetMetadata]:
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
    ) -> List[Path]:
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
    ) -> ProteinDataset | ResidueDataset:
        """Load one dataset into the canonical in-memory representation."""


class PeerCollection(DatasetCollection):
    """Expose PEER benchmark tasks as one dataset collection."""

    id = "peer"
    display_name = "PEER"

    def list_datasets(self) -> List[DatasetMetadata]:
        """Return PEER benchmark dataset specifications."""
        return list(_PEER_DATASET_METADATA)

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

    def __init__(self, collection_id: str, *, display_name: str) -> None:
        self.id = collection_id
        self.display_name = display_name

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

    id = "dbptm"
    display_name = "dbPTM"

    def list_datasets(self) -> List[DatasetMetadata]:
        """Return dbPTM source and benchmark specifications."""
        source = [
            dataset
            for dataset in _RESIDUE_DATASET_METADATA
            if dataset.id == "dbptm:all"
        ]
        return source + list(_DBPTM_BENCHMARK_METADATA)

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

    id = "dtu"
    display_name = "DTU Health Tech"

    def list_datasets(self) -> List[DatasetMetadata]:
        """Return catalog-only DTU dataset specifications."""
        return list(_DTU_DATASET_METADATA)

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


def _peer_dataset_metadata(task: object) -> DatasetMetadata:
    status: DatasetStatus = (
        "ready"
        if getattr(task, "current_loader") is not None
        or getattr(task, "level") in {"protein", "residue"}
        else "blocked"
    )
    task_class = _peer_task_class(getattr(task, "name"), getattr(task, "category"))
    return DatasetMetadata(
        id=f"peer:{getattr(task, 'name')}",
        name=getattr(task, "name"),
        display_name=getattr(task, "display_name"),
        collection="peer",
        source=getattr(task, "source"),
        category=getattr(task, "category"),
        task_class=task_class,
        preferred_metric=getattr(task, "metric"),
        description=_peer_description(task, status),
        level=getattr(task, "level"),
        objective=getattr(task, "objective"),
        target=getattr(task, "target"),
        status=status,
        metrics=(getattr(task, "metric"),),
        split_counts=getattr(task, "split_counts"),
        loader=getattr(task, "current_loader") or "load_peer_dataset",
        citation=PEER_CITATION,
        tags=(
            "peer",
            getattr(task, "short_name").lower(),
            getattr(task, "level"),
            getattr(task, "objective"),
        ),
        notes=f"PEER dataset class: {getattr(task, 'dataset_class')}.",
    )


def _residue_dataset_metadata() -> List[DatasetMetadata]:
    datasets: List[DatasetMetadata] = []
    seen_ids: set[str] = set()
    for source in RESIDUE_SOURCE_SPECS.values():
        dataset_id = _residue_dataset_id(source.name)
        if dataset_id in seen_ids:
            continue
        seen_ids.add(dataset_id)
        download_url = source.download_urls[0] if source.download_urls else None
        if source.name == "disprot":
            download_url = DISPROT_CURRENT_JSON_URL
        has_loader = source.name == "disprot" or source.name.startswith(("biolip", "phosphoelm"))
        status: DatasetStatus = "ready" if has_loader else "adapter"
        collection_id = dataset_id.partition(":")[0]
        datasets.append(
            DatasetMetadata(
                id=dataset_id,
                name=source.name,
                display_name=source.source,
                collection=collection_id,
                source=source.source,
                category=source.category,
                task_class=_task_class(source.category),
                preferred_metric=_preferred_metric(source.objective, source.category),
                description=_residue_description(dataset_id, source, status),
                level="residue",
                objective=source.objective,
                target=source.target,
                status=status,
                homepage=source.homepage,
                download_url=download_url,
                download_adapter=(
                    "download_residue_source"
                    if download_url is not None or source.name == "disprot"
                    else None
                ),
                import_adapter=source.import_adapter,
                loader="load_residue_source_dataset" if has_loader else None,
                tags=("residue", source.category, source.name),
                notes=source.notes,
            )
        )
    return datasets


def _dbptm_dataset_metadata(benchmark: object) -> DatasetMetadata:
    return DatasetMetadata(
        id=f"dbptm:{getattr(benchmark, 'name')}",
        name=getattr(benchmark, "name"),
        display_name=getattr(benchmark, "display_name"),
        collection="dbptm",
        source="dbPTM",
        category="ptm",
        task_class="ptms",
        preferred_metric="f1",
        description=_dbptm_description(benchmark),
        level="residue",
        objective="binary",
        target=getattr(benchmark, "target"),
        status="ready",
        sample_count=getattr(benchmark, "positive_sites") + getattr(benchmark, "negative_sites"),
        positive_count=getattr(benchmark, "positive_sites"),
        negative_count=getattr(benchmark, "negative_sites"),
        homepage="https://biomics.lab.nycu.edu.tw/dbPTM/download.php",
        download_url=getattr(benchmark, "url"),
        download_adapter="download_dbptm_benchmark",
        import_adapter="load_dbptm_benchmark_archive",
        loader="load_dbptm_benchmark_dataset",
        tags=("ptm", "dbptm", "benchmark"),
        notes=(
            f"Benchmark archive {getattr(benchmark, 'archive_name')}; "
            f"{getattr(benchmark, 'protein_count')} proteins."
        ),
    )


def _dtu_dataset_metadata(service: object) -> DatasetMetadata:
    return DatasetMetadata(
        id=f"dtu:{getattr(service, 'name')}",
        name=getattr(service, "name"),
        display_name=getattr(service, "description"),
        collection="dtu",
        source="DTU Health Tech",
        category=getattr(service, "category"),
        task_class=_task_class(getattr(service, "category")),
        preferred_metric=_preferred_metric(
            getattr(service, "objective"),
            getattr(service, "category"),
        ),
        description=_dtu_description(service),
        level=getattr(service, "level"),
        objective=getattr(service, "objective"),
        target=getattr(service, "target"),
        status="catalog_only",
        homepage=getattr(service, "url"),
        dataset_url=getattr(service, "dataset_url"),
        tags=(
            "dtu",
            getattr(service, "category"),
            getattr(service, "level"),
            getattr(service, "objective"),
        ),
        notes=getattr(service, "notes"),
    )


def _residue_dataset_id(name: str) -> str:
    mapping = {
        "dbptm": "dbptm:all",
        "musitedeep": "musitedeep:all",
        "disprot": "disprot:all",
        "biolip": "biolip:all",
        "biolip_all": "biolip:all",
        "biolip_dna": "biolip:dna",
        "biolip_rna": "biolip:rna",
        "biolip_pep": "biolip:pep",
        "biolip_other": "biolip:other",
        "metalpdb": "metalpdb:all",
        "scannet_binding": "scannet:binding",
        "netsurfp": "netsurfp:secondary_structure",
        "phosphoelm_all": "phosphoelm:all",
        "phosphoelm_ltp": "phosphoelm:ltp",
        "phosphoelm_htp": "phosphoelm:htp",
    }
    normalized = str(name).strip().lower()
    return mapping.get(normalized, f"{normalized}:all")


def _peer_task_class(name: str, category: str) -> str:
    if category == "structure_prediction":
        return "structure"
    if category in {
        "protein_protein_interaction_prediction",
        "protein_ligand_interaction_prediction",
    }:
        return "binding"
    if name in {"gb1", "aav", "beta_lactamase"}:
        return "fitness"
    return "function"


def _task_class(category: str) -> str:
    mapping = {
        "binding": "binding",
        "dataset": "structure",
        "disorder": "structure",
        "function_prediction": "function",
        "immunology": "binding",
        "localization_prediction": "function",
        "protein_ligand_interaction_prediction": "binding",
        "protein_protein_interaction_prediction": "binding",
        "ptm": "ptms",
        "sorting": "function",
        "structure": "structure",
        "structure_prediction": "structure",
    }
    return mapping.get(category, category)


def _preferred_metric(objective: ObjectiveName, category: str) -> str:
    if category == "structure" and objective == "multiclass":
        return "accuracy"
    if objective == "binary":
        return "f1"
    if objective in {"multiclass", "multilabel"}:
        return "macro_f1"
    return "spearmanr"


def _split_counts_text(split_counts: Tuple[int, int, int] | None) -> str:
    if split_counts is None:
        return "Split system: not quantified in the catalog entry."
    train, validation, test = split_counts
    return f"Split system: train/validation/test counts are {train}/{validation}/{test}."


def _peer_description(task: object, status: DatasetStatus) -> str:
    task_class = _peer_task_class(getattr(task, "name"), getattr(task, "category"))
    level = getattr(task, "level")
    name = getattr(task, "name")
    current_loader = getattr(task, "current_loader")
    if current_loader == "load_flip_dataset":
        split_text = "Split system: PEER/FLIP named benchmark splits are used."
    elif name == "fold":
        split_text = "Split system: train/valid/test uses the benchmark fold holdout by default."
    elif name == "secondary_structure":
        split_text = "Split system: train/valid/test uses CB513 as the default test split."
    elif level in {"protein", "residue"}:
        split_text = "Split system: native PEER/TorchDrug splits are used when available."
    else:
        split_text = "Split system: cataloged but not currently loadable."
    return (
        f"Source: {getattr(task, 'source')} / PEER. Class: {task_class}. Level: {level}. "
        f"Objective: {getattr(task, 'objective')} for target {getattr(task, 'target')}. "
        f"Preferred metric: {getattr(task, 'metric')}. Status: {status}. {split_text} "
        f"{_split_counts_text(getattr(task, 'split_counts'))}"
    )


def _residue_description(dataset_id: str, source: object, status: DatasetStatus) -> str:
    if dataset_id.startswith("phosphoelm:"):
        split_text = "Split system: deterministic splits stratified by species and residue type."
    elif dataset_id == "disprot:all":
        split_text = "Split system: deterministic splits use disorder content and dataset tags."
    elif dataset_id.startswith("biolip:"):
        split_text = "Split system: deterministic splits are assigned when absent."
    elif status == "adapter":
        split_text = "Split system: adapter-defined until imported."
    else:
        split_text = "Split system: deterministic splits are used when absent."
    task_class = _task_class(getattr(source, "category"))
    preferred_metric = _preferred_metric(
        getattr(source, "objective"),
        getattr(source, "category"),
    )
    return (
        f"Source: {getattr(source, 'source')}. Class: {task_class}. "
        f"Category: {getattr(source, 'category')}. Level: residue. "
        f"Objective: {getattr(source, 'objective')} for target {getattr(source, 'target')}. "
        f"Preferred metric: {preferred_metric}. Status: {status}. {split_text} "
        f"Notes: {getattr(source, 'notes')}"
    )


def _dbptm_description(benchmark: object) -> str:
    return (
        f"Source: dbPTM benchmark archive {getattr(benchmark, 'archive_name')}. "
        "Class: ptms. Category: ptm. Level: residue. Objective: binary. "
        f"Preferred metric: f1. Split system: archive-provided windows; catalog counts are "
        f"{getattr(benchmark, 'positive_sites')} positives, "
        f"{getattr(benchmark, 'negative_sites')} negatives, and "
        f"{getattr(benchmark, 'protein_count')} proteins."
    )


def _dtu_description(service: object) -> str:
    task_class = _task_class(getattr(service, "category"))
    preferred_metric = _preferred_metric(
        getattr(service, "objective"),
        getattr(service, "category"),
    )
    notes = f" Notes: {getattr(service, 'notes')}" if getattr(service, "notes") else ""
    return (
        f"Source: DTU Health Tech service catalog. Class: {task_class}. "
        f"Category: {getattr(service, 'category')}. Level: {getattr(service, 'level')}. "
        f"Objective: {getattr(service, 'objective')} for target {getattr(service, 'target')}. "
        f"Preferred metric: {preferred_metric}. Split system: catalog-only; no local loader. "
        f"{getattr(service, 'description')}{notes}"
    )


_PEER_DATASET_METADATA = tuple(
    _peer_dataset_metadata(task) for task in PEER_TASKS.values()
)
_RESIDUE_DATASET_METADATA = tuple(_residue_dataset_metadata())
_DBPTM_BENCHMARK_METADATA = tuple(
    _dbptm_dataset_metadata(benchmark) for benchmark in DBPTM_BENCHMARKS.values()
)
_DTU_DATASET_METADATA = tuple(
    _dtu_dataset_metadata(service) for service in DTU_PROBING_CATALOG.values()
)

PEER_COLLECTION = PeerCollection()
DBPTM_COLLECTION = DbptmCollection()
DTU_COLLECTION = DtuCollection()

_RESIDUE_COLLECTIONS = {
    collection_id: _ResidueDatasetCollection(collection_id, display_name=display_name)
    for collection_id, display_name in {
        "musitedeep": "MusiteDeep",
        "disprot": "DisProt",
        "biolip": "BioLiP",
        "metalpdb": "MetalPDB",
        "scannet": "ScanNet",
        "netsurfp": "NetSurfP",
        "phosphoelm": "PhosphoELM",
    }.items()
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
