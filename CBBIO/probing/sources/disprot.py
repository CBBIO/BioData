"""Download and load DisProt residue-level disorder datasets."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import csv
import hashlib
import json
from pathlib import Path
from typing import Any, cast
import urllib.request

from CBBIO.embeddings import EmbeddingInputError

from ..collection_types import CollectionMetadata, DatasetMetadata, residue_dataset_metadata
from ..datasets import ResidueDataset, ResidueExample, SplitName


_DISPROT_API_URL = "https://disprot.org/api/v2/download"
_DISPROT_API_QUERY = "release=current&term_ontology=IDPO&term_ontology=GO"
DISPROT_CURRENT_TSV_URL = f"{_DISPROT_API_URL}?format=tsv&{_DISPROT_API_QUERY}"
DISPROT_CURRENT_JSON_URL = f"{_DISPROT_API_URL}?format=json&{_DISPROT_API_QUERY}"
DISPROT_SPLIT_RATIOS: tuple[tuple[SplitName, float], ...] = (
    ("train", 0.8),
    ("val", 0.1),
    ("test", 0.1),
)

DISPROT_COLLECTION_METADATA = CollectionMetadata(
    id="disprot",
    display_name="DisProt",
    description="Curated intrinsically disordered protein regions.",
    homepage="https://disprot.org/download",
    tags=("disorder", "residue"),
)
DISPROT_DATASETS: tuple[DatasetMetadata, ...] = (
    residue_dataset_metadata(
        dataset_id="disprot:all",
        name="disprot",
        display_name="DisProt",
        source="DisProt",
        category="disorder",
        objective="binary",
        target="disorder",
        status="ready",
        homepage="https://disprot.org/download",
        description=(
            "Source: DisProt. Class: structure. Split system: deterministic splits use "
            "disorder content and dataset tags."
        ),
        download_url=DISPROT_CURRENT_JSON_URL,
        download_adapter="download_residue_source",
        import_adapter="load_disprot_tsv",
        loader="load_residue_source_dataset",
        tags=("residue", "disorder", "disprot"),
    ),
)


def download_disprot_current_tsv(root: str | Path, *, force: bool = False) -> Path:
    """Download the current DisProt TSV export with IDPO and GO terms."""
    output_dir = Path(root).expanduser() / "disprot"
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "disprot_current.tsv"
    if force or not path.exists():
        urllib.request.urlretrieve(DISPROT_CURRENT_TSV_URL, path)
    return path


def download_disprot_current_json(root: str | Path, *, force: bool = False) -> Path:
    """Download the current DisProt JSON export with IDPO and GO terms."""
    output_dir = Path(root).expanduser() / "disprot"
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "disprot_current.json"
    if force or not path.exists():
        urllib.request.urlretrieve(DISPROT_CURRENT_JSON_URL, path)
    return path


def load_disprot_json(
    path: str | Path,
    *,
    target: str = "disorder",
    split: SplitName | None = None,
) -> ResidueDataset:
    """Load DisProt JSON entries by expanding regions to residue labels."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    entries = _disprot_entries(payload)
    if _has_current_region_schema(entries):
        return _load_current_json_entries(entries, target=target, split=split)

    examples: list[ResidueExample] = []
    for index, entry in enumerate(entries):
        entry_map = cast(Mapping[str, Any], entry)
        sequence = _entry_sequence(entry_map)
        labels = [0] * len(sequence)
        for region in _entry_regions(entry_map):
            start, end = _region_bounds(cast(Mapping[str, Any], region))
            _mark_interval(labels, start=start, end=end)
        record_id = str(
            entry_map.get("acc")
            or entry_map.get("disprot_id")
            or entry_map.get("id")
            or index
        )
        examples.append(
            ResidueExample(
                id=record_id,
                sequence=sequence,
                labels={target: labels},
                split=split or "train",
                metadata={"source": "disprot"},
            )
        )
    return ResidueDataset(examples)


def load_disprot_tsv(
    path: str | Path,
    *,
    target: str = "disorder",
    split: SplitName = "train",
    protein_sequences: Mapping[str, str] | None = None,
    region_mode: bool = False,
    term_ids: Sequence[str] = ("IDPO:0000002",),
    term_names: Sequence[str] = ("disorder",),
) -> ResidueDataset:
    """Load the DisProt TSV export into residue-level disorder labels.

    Args:
        path: DisProt TSV export path.
        target: Label name stored in each residue example.
        split: Dataset split assigned to imported examples.
        protein_sequences: Full sequences indexed by UniProt or DisProt identifier.
        region_mode: Create one positive example per annotated region.
        term_ids: IDPO identifiers accepted as disorder annotations.
        term_names: Term names accepted as disorder annotations.

    Returns:
        Residue-level disorder dataset.

    Raises:
        EmbeddingInputError: If rows or required full sequences are unavailable.
    """
    rows = _matching_tsv_rows(path, term_ids=term_ids, term_names=term_names)
    if region_mode:
        return _load_tsv_regions(rows, target=target, split=split)
    if protein_sequences is None:
        raise EmbeddingInputError(
            "DisProt TSV does not include full protein sequences. "
            "Pass protein_sequences or set region_mode=True."
        )
    return _load_tsv_proteins(
        rows,
        target=target,
        split=split,
        protein_sequences=protein_sequences,
    )


def _load_tsv_regions(
    rows: Sequence[Mapping[str, str]],
    *,
    target: str,
    split: SplitName,
) -> ResidueDataset:
    examples: list[ResidueExample] = []
    for row_index, row in enumerate(rows):
        sequence = str(row.get("Region sequence", "")).strip()
        if not sequence:
            raise EmbeddingInputError("DisProt TSV region-mode rows require 'Region sequence'.")
        record_id = str(row.get("Region ID") or row.get("DisProt ID") or row_index)
        examples.append(
            ResidueExample(
                id=record_id,
                sequence=sequence,
                labels={target: [1] * len(sequence)},
                split=split,
                metadata={
                    "source": "disprot",
                    "uniprot_acc": row.get("UniProt ACC", ""),
                    "disprot_id": row.get("DisProt ID", ""),
                    "start": row.get("Start", ""),
                    "end": row.get("End", ""),
                    "term_id": row.get("Term ID", ""),
                    "term_name": row.get("Term name", ""),
                },
            )
        )
    return ResidueDataset(examples)


def _load_tsv_proteins(
    rows: Sequence[Mapping[str, str]],
    *,
    target: str,
    split: SplitName,
    protein_sequences: Mapping[str, str],
) -> ResidueDataset:
    grouped: dict[str, dict[str, Any]] = {}
    for row in rows:
        acc = str(row.get("UniProt ACC", "")).strip()
        disprot_id = str(row.get("DisProt ID", "")).strip()
        record_id = acc or disprot_id
        sequence = protein_sequences.get(acc) or protein_sequences.get(disprot_id)
        if not record_id or sequence is None:
            raise EmbeddingInputError(
                "Missing protein sequence for DisProt TSV row with "
                f"UniProt ACC {acc!r} / DisProt ID {disprot_id!r}."
            )
        group = grouped.setdefault(
            record_id,
            {
                "sequence": sequence,
                "labels": [0] * len(sequence),
                "metadata": {
                    "source": "disprot",
                    "uniprot_acc": acc,
                    "disprot_id": disprot_id,
                },
            },
        )
        if group["sequence"] != sequence:
            raise EmbeddingInputError(f"Conflicting sequences for DisProt record {record_id!r}.")
        start = int(str(row.get("Start", "")).strip())
        end = int(str(row.get("End", start)).strip())
        _mark_interval(cast(list[int], group["labels"]), start=start, end=end)

    return ResidueDataset(
        ResidueExample(
            id=record_id,
            sequence=str(group["sequence"]),
            labels={target: cast(list[int], group["labels"])},
            split=split,
            metadata=cast(dict[str, Any], group["metadata"]),
        )
        for record_id, group in grouped.items()
    )


def _disprot_entries(payload: object) -> Sequence[object]:
    if isinstance(payload, list):
        return cast(Sequence[object], payload)
    if isinstance(payload, Mapping):
        payload_map = cast(Mapping[str, object], payload)
        for key in ("data", "entries", "results"):
            value = payload_map.get(key)
            if isinstance(value, list):
                return cast(Sequence[object], value)
    raise EmbeddingInputError("DisProt JSON must contain a list of entries.")


def _has_current_region_schema(entries: Sequence[object]) -> bool:
    for entry in entries:
        if not isinstance(entry, Mapping):
            continue
        entry_map = cast(Mapping[str, object], entry)
        raw_region = entry_map.get("regions")
        if not isinstance(raw_region, Mapping):
            continue
        region = cast(Mapping[str, object], raw_region)
        if {"term_namespace", "term_name", "term_id"}.intersection(region):
            return True
    return False


def _load_current_json_entries(
    entries: Sequence[object],
    *,
    target: str,
    split: SplitName | None,
) -> ResidueDataset:
    grouped: dict[str, dict[str, Any]] = {}
    for entry in entries:
        if not isinstance(entry, Mapping):
            continue
        entry_map = cast(Mapping[str, Any], entry)
        raw_region = entry_map.get("regions")
        if not isinstance(raw_region, Mapping):
            continue
        region = cast(Mapping[str, Any], raw_region)
        if not _is_disorder_region(region):
            continue

        record_id = str(
            entry_map.get("acc") or entry_map.get("disprot_id") or entry_map.get("id") or ""
        ).strip()
        if not record_id:
            raise EmbeddingInputError(
                "Current DisProt JSON disorder rows require acc, disprot_id, or id."
            )
        _merge_current_region(grouped, record_id=record_id, entry=entry_map, region=region)

    if not grouped:
        raise EmbeddingInputError("No Structural state disorder rows found in DisProt JSON.")
    examples = _current_examples(grouped, target=target, split=split)
    if split is None:
        examples = _assign_stratified_splits(examples)
    return ResidueDataset(examples)


def _merge_current_region(
    grouped: dict[str, dict[str, Any]],
    *,
    record_id: str,
    entry: Mapping[str, Any],
    region: Mapping[str, Any],
) -> None:
    sequence = _entry_sequence(entry)
    disprot_id = str(entry.get("disprot_id") or "").strip()
    dataset_tags = tuple(str(tag) for tag in entry.get("dataset", ()) if str(tag).strip())
    group = grouped.setdefault(
        record_id,
        {
            "sequence": sequence,
            "labels": [0] * len(sequence),
            "metadata": {
                "source": "disprot",
                "uniprot_acc": str(entry.get("acc") or ""),
                "disprot_id": disprot_id,
                "dataset_tags": dataset_tags,
                "disorder_interval_count": 0,
            },
        },
    )
    if group["sequence"] != sequence:
        raise EmbeddingInputError(f"Conflicting sequences for DisProt record {record_id!r}.")
    metadata = cast(dict[str, Any], group["metadata"])
    previous_tags = cast(tuple[str, ...], metadata["dataset_tags"])
    metadata["dataset_tags"] = tuple(sorted(set(previous_tags + dataset_tags)))
    if disprot_id and not metadata.get("disprot_id"):
        metadata["disprot_id"] = disprot_id
    start, end = _region_bounds(region)
    _mark_interval(cast(list[int], group["labels"]), start=start, end=end)
    metadata["disorder_interval_count"] = int(metadata["disorder_interval_count"]) + 1


def _current_examples(
    grouped: Mapping[str, Mapping[str, Any]],
    *,
    target: str,
    split: SplitName | None,
) -> list[ResidueExample]:
    examples: list[ResidueExample] = []
    for record_id, group in grouped.items():
        labels = cast(list[int], group["labels"])
        sequence = str(group["sequence"])
        metadata = dict(cast(dict[str, Any], group["metadata"]))
        metadata["disorder_fraction"] = sum(labels) / len(labels)
        metadata["disorder_content_bin"] = _disorder_content_bin(
            float(metadata["disorder_fraction"])
        )
        metadata["dataset_stratum"] = _dataset_stratum(
            cast(Sequence[str], metadata["dataset_tags"])
        )
        examples.append(
            ResidueExample(
                id=record_id,
                sequence=sequence,
                labels={target: labels},
                split=split or "train",
                metadata=metadata,
            )
        )
    return examples


def _is_disorder_region(region: Mapping[str, Any]) -> bool:
    namespace = str(region.get("term_namespace") or "").strip().lower()
    term_name = str(region.get("term_name") or "").strip().lower()
    return namespace == "structural state" and term_name == "disorder"


def _assign_stratified_splits(examples: Sequence[ResidueExample]) -> list[ResidueExample]:
    primary_groups: dict[tuple[str, str], list[ResidueExample]] = {}
    for example in examples:
        metadata = example.metadata or {}
        key = (
            str(metadata.get("dataset_stratum") or "none"),
            str(metadata.get("disorder_content_bin") or "unknown"),
        )
        primary_groups.setdefault(key, []).append(example)

    assigned: list[ResidueExample] = []
    rare_by_content: dict[str, list[ResidueExample]] = {}
    for key, group in primary_groups.items():
        if len(group) >= 10:
            assigned.extend(_split_group(group))
        else:
            rare_by_content.setdefault(key[1], []).extend(group)
    rare_global: list[ResidueExample] = []
    for group in rare_by_content.values():
        if len(group) >= 10:
            assigned.extend(_split_group(group))
        else:
            rare_global.extend(group)
    if rare_global:
        assigned.extend(_split_group(rare_global))
    return sorted(assigned, key=lambda example: example.id)


def _split_group(group: Sequence[ResidueExample]) -> list[ResidueExample]:
    ordered = sorted(group, key=lambda example: _stable_hash(example.id))
    counts = _split_counts(len(ordered))
    split_names: list[SplitName] = [
        split_name for split_name, count in counts for _ in range(count)
    ]
    return [
        ResidueExample(
            id=example.id,
            sequence=example.sequence,
            labels=example.labels,
            split=split_name,
            mask=example.mask,
            metadata=example.metadata,
        )
        for example, split_name in zip(ordered, split_names)
    ]


def _split_counts(total: int) -> list[tuple[SplitName, int]]:
    if total <= 0:
        return [("train", 0), ("val", 0), ("test", 0)]
    validation = int(round(total * DISPROT_SPLIT_RATIOS[1][1]))
    test = int(round(total * DISPROT_SPLIT_RATIOS[2][1]))
    if total >= 10:
        validation = max(1, validation)
        test = max(1, test)
    elif total >= 2:
        test = max(1, test)
    train = total - validation - test
    while train < 1 and validation > 0:
        validation -= 1
        train += 1
    while train < 1 and test > 0:
        test -= 1
        train += 1
    return [("train", train), ("val", validation), ("test", test)]


def _stable_hash(value: str) -> str:
    return hashlib.sha256(f"disprot-split-v1:{value}".encode("utf-8")).hexdigest()


def _disorder_content_bin(fraction: float) -> str:
    if fraction < 0.1:
        return "0-10%"
    if fraction < 0.3:
        return "10-30%"
    if fraction < 0.6:
        return "30-60%"
    return "60-100%"


def _dataset_stratum(dataset_tags: Sequence[str]) -> str:
    values = sorted({str(tag).strip() for tag in dataset_tags if str(tag).strip()})
    return "+".join(values) if values else "none"


def _entry_sequence(entry: Mapping[str, Any]) -> str:
    sequence_value = entry.get("sequence") or entry.get("protein_sequence") or entry.get("seq")
    if isinstance(sequence_value, Mapping):
        sequence_map = cast(Mapping[str, object], sequence_value)
        sequence_value = (
            sequence_map.get("value")
            or sequence_map.get("sequence")
            or sequence_map.get("content")
        )
    sequence = str(sequence_value or "").strip()
    if not sequence:
        raise EmbeddingInputError("DisProt entry does not contain a protein sequence.")
    return sequence


def _entry_regions(entry: Mapping[str, Any]) -> Sequence[object]:
    raw_regions = (
        entry.get("regions")
        or entry.get("disorder_regions")
        or entry.get("annotations")
        or ()
    )
    if isinstance(raw_regions, Mapping):
        region_map = cast(Mapping[str, object], raw_regions)
        if "start" in region_map or "start_position" in region_map:
            return [region_map]
        return list(region_map.values())
    if isinstance(raw_regions, list):
        return cast(Sequence[object], raw_regions)
    return ()


def _matching_tsv_rows(
    path: str | Path,
    *,
    term_ids: Sequence[str],
    term_names: Sequence[str],
) -> list[dict[str, str]]:
    normalized_ids = {str(value).strip().lower() for value in term_ids if str(value).strip()}
    normalized_names = {str(value).strip().lower() for value in term_names if str(value).strip()}
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        required = {
            "UniProt ACC",
            "DisProt ID",
            "Start",
            "End",
            "Region sequence",
            "Term ID",
            "Term name",
        }
        fieldnames = set(reader.fieldnames or ())
        missing = sorted(required - fieldnames)
        if missing:
            raise EmbeddingInputError(
                f"DisProt TSV is missing required columns: {', '.join(missing)}."
            )
        rows = [
            cast(dict[str, str], row)
            for row in reader
            if str(row.get("Term ID", "")).strip().lower() in normalized_ids
            or str(row.get("Term name", "")).strip().lower() in normalized_names
        ]
    if not rows:
        raise EmbeddingInputError(
            "No matching DisProt TSV rows found for the requested term filters."
        )
    return rows


def _region_bounds(region: Mapping[str, Any]) -> tuple[int, int]:
    start = region.get("start") or region.get("start_position") or region.get("begin")
    end = region.get("end") or region.get("end_position") or region.get("stop") or start
    if start is None or end is None:
        raise EmbeddingInputError("DisProt region is missing start/end coordinates.")
    return int(start), int(end)


def _mark_interval(labels: list[int], *, start: int, end: int) -> None:
    if start < 1 or end < start or end > len(labels):
        raise EmbeddingInputError(
            f"Residue interval {start}-{end} is outside sequence length {len(labels)}."
        )
    for index in range(start - 1, end):
        labels[index] = 1


__all__ = [
    "DISPROT_COLLECTION_METADATA",
    "DISPROT_CURRENT_JSON_URL",
    "DISPROT_CURRENT_TSV_URL",
    "DISPROT_DATASETS",
    "DISPROT_SPLIT_RATIOS",
    "download_disprot_current_json",
    "download_disprot_current_tsv",
    "load_disprot_json",
    "load_disprot_tsv",
]
