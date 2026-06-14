"""Unified probing dataset catalog."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Literal, Tuple

from CBBIO.embeddings import EmbeddingInputError

from .datasets import ObjectiveName
from .dtu import DTU_PROBING_CATALOG
from .peer import PEER_CITATION, PEER_TASKS, PeerTaskLevel
from .residue_sources import DBPTM_BENCHMARKS, DISPROT_CURRENT_JSON_URL, RESIDUE_SOURCE_SPECS


DatasetCatalogLevel = PeerTaskLevel
DatasetCatalogStatus = Literal["ready", "adapter", "catalog_only", "blocked"]


@dataclass(frozen=True)
class DatasetCatalogEntry:
    """Normalized metadata for a probing dataset or dataset candidate."""

    id: str
    name: str
    display_name: str
    collection: str
    source: str
    category: str
    level: DatasetCatalogLevel
    objective: ObjectiveName
    target: str
    status: DatasetCatalogStatus
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


def build_dataset_catalog() -> Dict[str, DatasetCatalogEntry]:
    """Build the unified dataset catalog from source-specific registries."""

    catalog: Dict[str, DatasetCatalogEntry] = {}
    for entry in _peer_entries() + _residue_source_entries() + _dbptm_benchmark_entries() + _dtu_entries():
        catalog[entry.id] = entry
    return catalog


def get_dataset_catalog_entry(dataset_id: str) -> DatasetCatalogEntry:
    key = _normalize_catalog_id(dataset_id)
    entry = DATASET_CATALOG.get(key)
    if entry is None:
        supported = ", ".join(sorted(DATASET_CATALOG))
        raise EmbeddingInputError(f"Unknown dataset catalog id {dataset_id!r}. Supported values: {supported}.")
    return entry


def list_dataset_catalog(
    *,
    collection: str | None = None,
    source: str | None = None,
    category: str | None = None,
    level: DatasetCatalogLevel | None = None,
    objective: ObjectiveName | None = None,
    status: DatasetCatalogStatus | None = None,
    has_download: bool | None = None,
    has_loader: bool | None = None,
    tag: str | None = None,
) -> List[DatasetCatalogEntry]:
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
    if level is not None:
        values = [value for value in values if value.level == level]
    if objective is not None:
        values = [value for value in values if value.objective == objective]
    if status is not None:
        values = [value for value in values if value.status == status]
    if has_download is not None:
        values = [value for value in values if (value.download_url is not None or value.download_adapter is not None) == has_download]
    if has_loader is not None:
        values = [value for value in values if (value.loader is not None or value.import_adapter is not None) == has_loader]
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
    level: DatasetCatalogLevel | None = None,
    objective: ObjectiveName | None = None,
    status: DatasetCatalogStatus | None = None,
) -> List[DatasetCatalogEntry]:
    """Search catalog entries with simple ranked token matching."""

    tokens = _query_tokens(query)
    if not tokens:
        return []
    candidates = list_dataset_catalog(
        collection=collection,
        category=category,
        level=level,
        objective=objective,
        status=status,
    )
    scored = [
        (_catalog_search_score(entry, tokens), entry)
        for entry in candidates
    ]
    matches = [(score, entry) for score, entry in scored if score > 0]
    matches.sort(key=lambda item: (-item[0], item[1].id))
    entries = [entry for _score, entry in matches]
    return entries if limit is None else entries[:limit]


def _peer_entries() -> List[DatasetCatalogEntry]:
    entries: List[DatasetCatalogEntry] = []
    for task in PEER_TASKS.values():
        status: DatasetCatalogStatus = "ready" if task.current_loader is not None or task.level in {"protein", "residue"} else "blocked"
        entries.append(
            DatasetCatalogEntry(
                id=f"peer:{task.name}",
                name=task.name,
                display_name=task.display_name,
                collection="peer",
                source=task.source,
                category=task.category,
                level=task.level,
                objective=task.objective,
                target=task.target,
                status=status,
                metrics=(task.metric,),
                split_counts=task.split_counts,
                loader=task.current_loader or "load_peer_dataset",
                citation=PEER_CITATION,
                tags=("peer", task.short_name.lower(), task.level, task.objective),
                notes=f"PEER dataset class: {task.dataset_class}.",
            )
        )
    return entries


def _residue_source_entries() -> List[DatasetCatalogEntry]:
    entries: List[DatasetCatalogEntry] = []
    for spec in RESIDUE_SOURCE_SPECS.values():
        download_url = spec.download_urls[0] if spec.download_urls else None
        if spec.name == "disprot":
            download_url = DISPROT_CURRENT_JSON_URL
        status: DatasetCatalogStatus = "ready" if spec.name == "disprot" else "adapter"
        entries.append(
            DatasetCatalogEntry(
                id=f"source:{spec.name}",
                name=spec.name,
                display_name=spec.source,
                collection="residue_sources",
                source=spec.source,
                category=spec.category,
                level="residue",
                objective=spec.objective,
                target=spec.target,
                status=status,
                homepage=spec.homepage,
                download_url=download_url,
                download_adapter="download_residue_source" if download_url is not None or spec.name == "disprot" else None,
                import_adapter=spec.import_adapter,
                loader="load_residue_source_dataset" if spec.name in {"biolip", "disprot"} else None,
                tags=("residue", spec.category, spec.name),
                notes=spec.notes,
            )
        )
    return entries


def _dbptm_benchmark_entries() -> List[DatasetCatalogEntry]:
    return [
        DatasetCatalogEntry(
            id=f"dbptm:{spec.name}",
            name=spec.name,
            display_name=spec.display_name,
            collection="dbptm_benchmark",
            source="dbPTM",
            category="ptm",
            level="residue",
            objective="binary",
            target=spec.target,
            status="ready",
            sample_count=spec.positive_sites + spec.negative_sites,
            positive_count=spec.positive_sites,
            negative_count=spec.negative_sites,
            homepage="https://biomics.lab.nycu.edu.tw/dbPTM/download.php",
            download_url=spec.url,
            download_adapter="download_dbptm_benchmark",
            import_adapter="load_dbptm_benchmark_archive",
            loader="load_dbptm_benchmark_dataset",
            tags=("ptm", "dbptm", "benchmark"),
            notes=f"Benchmark archive {spec.archive_name}; {spec.protein_count} proteins.",
        )
        for spec in DBPTM_BENCHMARKS.values()
    ]


def _dtu_entries() -> List[DatasetCatalogEntry]:
    entries: List[DatasetCatalogEntry] = []
    for service in DTU_PROBING_CATALOG.values():
        entries.append(
            DatasetCatalogEntry(
                id=f"dtu:{service.name}",
                name=service.name,
                display_name=service.description,
                collection="dtu",
                source="DTU Health Tech",
                category=service.category,
                level=service.level,
                objective=service.objective,
                target=service.target,
                status="catalog_only",
                homepage=service.url,
                dataset_url=service.dataset_url,
                tags=("dtu", service.category, service.level, service.objective),
                notes=service.notes,
            )
        )
    return entries


def _normalize_catalog_id(dataset_id: str) -> str:
    text = dataset_id.strip().lower().replace("-", "_")
    if ":" in text:
        return text
    matches = [key for key in DATASET_CATALOG if key.endswith(f":{text}")]
    if len(matches) == 1:
        return matches[0]
    return text


def _query_tokens(query: str) -> Tuple[str, ...]:
    return tuple(token for token in query.strip().lower().replace("-", "_").split() if token)


def _catalog_search_score(entry: DatasetCatalogEntry, tokens: Tuple[str, ...]) -> int:
    exact_fields = {
        entry.id.lower(),
        entry.name.lower(),
        entry.collection.lower(),
        entry.source.lower(),
        entry.category.lower(),
        entry.level.lower(),
        entry.objective.lower(),
        entry.target.lower(),
        *(tag.lower() for tag in entry.tags),
    }
    weighted_text = {
        "high": " ".join(
            [
                entry.id,
                entry.name,
                entry.display_name,
                entry.source,
                entry.target,
                " ".join(entry.tags),
            ]
        ).lower(),
        "medium": " ".join(
            [
                entry.category,
                entry.collection,
                entry.level,
                entry.objective,
                entry.loader or "",
                entry.import_adapter or "",
                entry.download_adapter or "",
            ]
        ).lower(),
        "low": " ".join([entry.notes, entry.homepage or "", entry.dataset_url or "", entry.download_url or ""]).lower(),
    }
    score = 0
    for token in tokens:
        if token in exact_fields:
            score += 20
        if token in weighted_text["high"]:
            score += 8
        if token in weighted_text["medium"]:
            score += 4
        if token in weighted_text["low"]:
            score += 2
    return score


DATASET_CATALOG: Dict[str, DatasetCatalogEntry] = build_dataset_catalog()


__all__ = [
    "DATASET_CATALOG",
    "DatasetCatalogEntry",
    "DatasetCatalogLevel",
    "DatasetCatalogStatus",
    "build_dataset_catalog",
    "get_dataset_catalog_entry",
    "list_dataset_catalog",
    "search_dataset_catalog",
]
