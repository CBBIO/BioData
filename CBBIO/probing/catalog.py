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
    task_class: str
    preferred_metric: str
    description: str
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
    """Return one registered probing dataset catalog entry."""
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
    task_class: str | None = None,
    preferred_metric: str | None = None,
    level: DatasetCatalogLevel | None = None,
    objective: ObjectiveName | None = None,
    status: DatasetCatalogStatus | None = None,
    has_download: bool | None = None,
    has_loader: bool | None = None,
    tag: str | None = None,
) -> List[DatasetCatalogEntry]:
    """Return registered probing dataset catalog entries."""
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
        values = [value for value in values if value.preferred_metric.lower() == normalized]
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
    task_class: str | None = None,
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
        task_class=task_class,
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
                task_class=_peer_task_class(task.name, task.category),
                preferred_metric=task.metric,
                description=_peer_description(task, status),
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
    seen_ids: set[str] = set()
    for spec in RESIDUE_SOURCE_SPECS.values():
        entry_id = _residue_source_catalog_id(spec.name)
        if entry_id in seen_ids:
            continue
        seen_ids.add(entry_id)
        download_url = spec.download_urls[0] if spec.download_urls else None
        if spec.name == "disprot":
            download_url = DISPROT_CURRENT_JSON_URL
        has_native_loader = spec.name == "disprot" or spec.name.startswith(("biolip", "phosphoelm"))
        status: DatasetCatalogStatus = "ready" if has_native_loader else "adapter"
        entries.append(
            DatasetCatalogEntry(
                id=entry_id,
                name=spec.name,
                display_name=spec.source,
                collection="residue_sources",
                source=spec.source,
                category=spec.category,
                task_class=_catalog_task_class(spec.category),
                preferred_metric=_preferred_metric(spec.objective, spec.category),
                description=_residue_source_description(entry_id, spec, status),
                level="residue",
                objective=spec.objective,
                target=spec.target,
                status=status,
                homepage=spec.homepage,
                download_url=download_url,
                download_adapter="download_residue_source" if download_url is not None or spec.name == "disprot" else None,
                import_adapter=spec.import_adapter,
                loader="load_residue_source_dataset" if has_native_loader else None,
                tags=("residue", spec.category, spec.name),
                notes=spec.notes,
            )
        )
    return entries


def _residue_source_catalog_id(name: str) -> str:
    normalized = str(name).strip().lower()
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
    return mapping.get(normalized, f"{normalized}:all")


def _legacy_catalog_id_alias(dataset_id: str) -> str | None:
    text = dataset_id.strip().lower().replace("-", "_")
    aliases = {
        "source:dbptm": "dbptm:all",
        "source:musitedeep": "musitedeep:all",
        "source:disprot": "disprot:all",
        "source:biolip": "biolip:all",
        "source:biolip_all": "biolip:all",
        "source:biolip_dna": "biolip:dna",
        "source:biolip_rna": "biolip:rna",
        "source:biolip_pep": "biolip:pep",
        "source:biolip_other": "biolip:other",
        "source:metalpdb": "metalpdb:all",
        "source:scannet_binding": "scannet:binding",
        "source:netsurfp": "netsurfp:secondary_structure",
        "source:phosphoelm_all": "phosphoelm:all",
        "source:phosphoelm_ltp": "phosphoelm:ltp",
        "source:phosphoelm_htp": "phosphoelm:htp",
        "secondary_structure": "peer:secondary_structure",
        "dbptm": "dbptm:all",
        "musitedeep": "musitedeep:all",
        "disprot": "disprot:all",
        "biolip": "biolip:all",
        "metalpdb": "metalpdb:all",
        "scannet": "scannet:binding",
        "netsurfp": "netsurfp:secondary_structure",
        "phosphoelm": "phosphoelm:all",
        "biolip_all": "biolip:all",
        "biolip_dna": "biolip:dna",
        "biolip_rna": "biolip:rna",
        "biolip_pep": "biolip:pep",
        "biolip_other": "biolip:other",
        "phosphoelm_all": "phosphoelm:all",
        "phosphoelm_ltp": "phosphoelm:ltp",
        "phosphoelm_htp": "phosphoelm:htp",
        "scannet_binding": "scannet:binding",
    }
    return aliases.get(text)


def _dbptm_benchmark_entries() -> List[DatasetCatalogEntry]:
    return [
        DatasetCatalogEntry(
            id=f"dbptm:{spec.name}",
            name=spec.name,
            display_name=spec.display_name,
            collection="dbptm_benchmark",
            source="dbPTM",
            category="ptm",
            task_class="ptms",
            preferred_metric="f1",
            description=_dbptm_benchmark_description(spec),
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
                task_class=_catalog_task_class(service.category),
                preferred_metric=_preferred_metric(service.objective, service.category),
                description=_dtu_description(service),
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
    alias = _legacy_catalog_id_alias(text)
    if alias is not None:
        return alias
    if ":" in text:
        return text
    matches = [key for key in DATASET_CATALOG if key.endswith(f":{text}")]
    if len(matches) == 1:
        return matches[0]
    return text


def _peer_task_class(name: str, category: str) -> str:
    if category == "structure_prediction":
        return "structure"
    if category in {"protein_protein_interaction_prediction", "protein_ligand_interaction_prediction"}:
        return "binding"
    if name in {"gb1", "aav", "beta_lactamase"}:
        return "fitness"
    return "function"


def _catalog_task_class(category: str) -> str:
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
    if objective == "multiclass":
        return "macro_f1"
    if objective == "multilabel":
        return "macro_f1"
    return "spearmanr"


def _split_counts_text(split_counts: Tuple[int, int, int] | None) -> str:
    if split_counts is None:
        return "Split system: not quantified in the catalog entry."
    train, validation, test = split_counts
    return f"Split system: train/validation/test counts are {train}/{validation}/{test}."


def _peer_split_text(task: object) -> str:
    name = getattr(task, "name")
    current_loader = getattr(task, "current_loader")
    level = getattr(task, "level")
    if current_loader == "load_flip_dataset":
        return "Split system: PEER/FLIP named benchmark splits are used through load_flip_dataset."
    if name == "fold":
        return "Split system: native PEER fold classification uses train/valid/test with the benchmark fold holdout by default."
    if name == "secondary_structure":
        return "Split system: native PEER secondary-structure labels use train/valid/test with the CB513 benchmark test split by default."
    if level in {"protein", "residue"}:
        return "Split system: native PEER/TorchDrug train/valid/test splits are used when available."
    return "Split system: PEER benchmark split metadata is cataloged, but this task is not currently loadable."


def _peer_description(task: object, status: DatasetCatalogStatus) -> str:
    task_class = _peer_task_class(getattr(task, "name"), getattr(task, "category"))
    split_text = _peer_split_text(task)
    counts_text = _split_counts_text(getattr(task, "split_counts"))
    return (
        f"Source: {getattr(task, 'source')} / PEER. Class: {task_class}. "
        f"Level: {getattr(task, 'level')}. Objective: {getattr(task, 'objective')} for target "
        f"{getattr(task, 'target')}. Preferred metric: {getattr(task, 'metric')}. "
        f"Status: {status}. {split_text} {counts_text}"
    )


def _residue_source_split_text(entry_id: str, status: DatasetCatalogStatus) -> str:
    if entry_id.startswith("phosphoelm:"):
        return (
            "Split system: deterministic residue dataset splits stratified by species and positive residue type; "
            "unannotated S/T/Y residues are kept as negatives."
        )
    if entry_id == "disprot:all":
        return (
            "Split system: deterministic residue dataset splits are assigned when no source split is provided, "
            "using disorder content and dataset tags for stratification."
        )
    if entry_id.startswith("biolip:"):
        return "Split system: deterministic residue dataset splits are assigned when no source split is provided."
    if status == "adapter":
        return "Split system: adapter-defined until imported; source files must provide or derive residue dataset splits."
    return "Split system: deterministic residue dataset splits are used when no source split is provided."


def _residue_source_description(entry_id: str, spec: object, status: DatasetCatalogStatus) -> str:
    task_class = _catalog_task_class(getattr(spec, "category"))
    preferred = _preferred_metric(getattr(spec, "objective"), getattr(spec, "category"))
    return (
        f"Source: {getattr(spec, 'source')}. Class: {task_class}. Category: {getattr(spec, 'category')}. "
        f"Level: residue. Objective: {getattr(spec, 'objective')} for target {getattr(spec, 'target')}. "
        f"Preferred metric: {preferred}. Status: {status}. "
        f"{_residue_source_split_text(entry_id, status)} Notes: {getattr(spec, 'notes')}"
    )


def _dbptm_benchmark_description(spec: object) -> str:
    return (
        f"Source: dbPTM benchmark archive {getattr(spec, 'archive_name')}. Class: ptms. Category: ptm. "
        f"Level: residue. Objective: binary for target {getattr(spec, 'target')}. Preferred metric: f1. "
        f"Split system: archive-provided positive and negative centered sequence windows are loaded as the requested split; "
        f"catalog counts are {getattr(spec, 'positive_sites')} positives, {getattr(spec, 'negative_sites')} negatives, "
        f"and {getattr(spec, 'protein_count')} proteins."
    )


def _dtu_description(service: object) -> str:
    task_class = _catalog_task_class(getattr(service, "category"))
    preferred = _preferred_metric(getattr(service, "objective"), getattr(service, "category"))
    notes = f" Notes: {getattr(service, 'notes')}" if getattr(service, "notes") else ""
    return (
        f"Source: DTU Health Tech service catalog. Class: {task_class}. Category: {getattr(service, 'category')}. "
        f"Level: {getattr(service, 'level')}. Objective: {getattr(service, 'objective')} for target "
        f"{getattr(service, 'target')}. Preferred metric: {preferred}. "
        f"Split system: catalog-only entry; no local loader or benchmark split is implemented yet. "
        f"{getattr(service, 'description')}{notes}"
    )


def _query_tokens(query: str) -> Tuple[str, ...]:
    return tuple(token for token in query.strip().lower().replace("-", "_").split() if token)


def _catalog_search_score(entry: DatasetCatalogEntry, tokens: Tuple[str, ...]) -> int:
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
    weighted_text = {
        "high": " ".join(
            [
                entry.id,
                entry.name,
                entry.display_name,
                entry.source,
                entry.task_class,
                entry.target,
                " ".join(entry.tags),
            ]
        ).lower(),
        "medium": " ".join(
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
        ).lower(),
        "low": " ".join(
            [entry.description, entry.notes, entry.homepage or "", entry.dataset_url or "", entry.download_url or ""]
        ).lower(),
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
