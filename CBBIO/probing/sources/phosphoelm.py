"""Load PhosphoELM residue-level phosphorylation datasets."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
import csv
import hashlib
from pathlib import Path
import tarfile
from typing import Any, Literal, cast

from CBBIO.embeddings import EmbeddingInputError

from ..collection_types import CollectionMetadata, DatasetMetadata, residue_dataset_metadata
from ..datasets import ResidueDataset, ResidueExample, SplitName


PhosphoElmSourceFilter = Literal["all", "LTP", "HTP"]

PHOSPHOELM_COLLECTION_METADATA = CollectionMetadata(
    id="phosphoelm",
    display_name="PhosphoELM",
    description="Experimentally validated eukaryotic phosphorylation sites.",
    homepage="http://phospho.elm.eu.org/",
    tags=("ptm", "phosphorylation", "residue"),
)

PHOSPHOELM_DATASETS: tuple[DatasetMetadata, ...] = tuple(
    residue_dataset_metadata(
        dataset_id=f"phosphoelm:{suffix}",
        name=f"phosphoelm_{suffix}",
        display_name=f"PhosphoELM {label}",
        source="Phospho.ELM",
        category="ptm",
        objective="binary",
        target="phosphorylation_site",
        status="ready",
        homepage="http://phospho.elm.eu.org/",
        description=(
            "Source: Phospho.ELM. Class: ptms. Split system: deterministic splits "
            f"stratified by species and positive residue type. Evidence: {label}."
        ),
        import_adapter="load_phosphoelm_dataset",
        loader="load_residue_source_dataset",
        tags=("residue", "ptm", f"phosphoelm_{suffix}"),
    )
    for suffix, label in (("all", "all"), ("ltp", "LTP"), ("htp", "HTP"))
)


def load_phosphoelm_dataset(
    path: str | Path,
    *,
    target: str = "phosphorylation_site",
    split: SplitName | None = None,
    source_filter: PhosphoElmSourceFilter = "all",
    residue_codes: Sequence[str] = ("S", "T", "Y"),
) -> ResidueDataset:
    """Load a PhosphoELM dump as full-protein phosphorylation labels.

    Args:
        path: Plain ``.dump`` file or gzip-compressed dump archive.
        target: Label name stored in each residue example.
        split: Dataset split, or ``None`` to assign deterministic splits.
        source_filter: Evidence subset to retain.
        residue_codes: Residues considered phosphorylation-site candidates.

    Returns:
        Residue-level phosphorylation dataset.

    Raises:
        EmbeddingInputError: If rows, sites, or sequences are invalid.
    """
    resolved_filter = _normalize_source_filter(source_filter)
    candidate_codes = _normalize_residue_codes(residue_codes)
    grouped: dict[str, dict[str, Any]] = {}
    evidence_counts: dict[str, dict[str, int]] = {}
    for row_index, row in enumerate(_iter_rows(path)):
        row_source = str(row.get("source", "")).strip().upper()
        if resolved_filter != "all" and row_source != resolved_filter:
            continue
        _add_row(
            grouped,
            evidence_counts,
            row,
            row_index=row_index,
            row_source=row_source,
            candidate_codes=candidate_codes,
            evidence_filter=resolved_filter,
        )

    if not grouped:
        raise EmbeddingInputError(
            f"No Phospho.ELM rows matched source_filter={resolved_filter!r}."
        )
    examples = _build_examples(
        grouped,
        evidence_counts,
        target=target,
        split=split,
    )
    if split is None:
        examples = _assign_stratified_splits(examples)
    return ResidueDataset(examples)


def _add_row(
    grouped: dict[str, dict[str, Any]],
    evidence_counts: dict[str, dict[str, int]],
    row: Mapping[str, str],
    *,
    row_index: int,
    row_source: str,
    candidate_codes: tuple[str, ...],
    evidence_filter: PhosphoElmSourceFilter,
) -> None:
    acc = str(row.get("acc", "")).strip()
    sequence = str(row.get("sequence", "")).strip()
    code = str(row.get("code", "")).strip().upper()
    if code not in candidate_codes:
        return
    if not acc or not sequence:
        raise EmbeddingInputError(
            f"Phospho.ELM row {row_index} requires non-empty acc and sequence fields."
        )
    try:
        position = int(str(row.get("position", "")).strip())
    except ValueError as exc:
        raise EmbeddingInputError(
            f"Phospho.ELM row {row_index} has invalid position {row.get('position')!r}."
        ) from exc
    if position < 1 or position > len(sequence):
        raise EmbeddingInputError(
            f"Phospho.ELM row {row_index} position {position} is outside "
            f"sequence length {len(sequence)}."
        )
    observed = sequence[position - 1].upper()
    if code != observed:
        raise EmbeddingInputError(
            f"Phospho.ELM row {row_index} code {code!r} does not match "
            f"sequence residue {observed!r}."
        )
    group = grouped.setdefault(
        acc,
        {
            "sequence": sequence,
            "labels": [0] * len(sequence),
            "mask": [residue.upper() in candidate_codes for residue in sequence],
            "metadata": {
                "source": "phosphoelm",
                "evidence_filter": evidence_filter,
                "residue_codes": candidate_codes,
                "species": row.get("species", ""),
            },
        },
    )
    if group["sequence"] != sequence:
        raise EmbeddingInputError(
            f"Conflicting Phospho.ELM sequences for accession {acc!r}."
        )
    cast(list[int], group["labels"])[position - 1] = 1
    counts = evidence_counts.setdefault(acc, {"HTP": 0, "LTP": 0, "other": 0})
    counts[row_source if row_source in {"HTP", "LTP"} else "other"] += 1


def _build_examples(
    grouped: Mapping[str, Mapping[str, Any]],
    evidence_counts: Mapping[str, Mapping[str, int]],
    *,
    target: str,
    split: SplitName | None,
) -> list[ResidueExample]:
    examples: list[ResidueExample] = []
    for acc, group in grouped.items():
        labels = cast(list[int], group["labels"])
        mask = cast(list[bool], group["mask"])
        sequence = str(group["sequence"])
        metadata = dict(cast(dict[str, Any], group["metadata"]))
        positive_counts = _positive_residue_counts(sequence, labels)
        metadata["positive_site_count"] = sum(labels)
        metadata["positive_residue_counts"] = positive_counts
        metadata["positive_residue_stratum"] = _positive_residue_stratum(
            positive_counts
        )
        metadata["candidate_site_count"] = sum(mask)
        metadata["evidence_row_counts"] = evidence_counts[acc]
        examples.append(
            ResidueExample(
                id=acc,
                sequence=sequence,
                labels={target: labels},
                split=split or "train",
                mask=mask,
                metadata=metadata,
            )
        )
    return examples


def _normalize_source_filter(value: str) -> PhosphoElmSourceFilter:
    normalized = str(value).strip().upper()
    if normalized in {"", "ALL"}:
        return "all"
    if normalized in {"LTP", "HTP"}:
        return cast(PhosphoElmSourceFilter, normalized)
    raise EmbeddingInputError("Phospho.ELM source_filter must be one of: all, LTP, HTP.")


def _normalize_residue_codes(values: Sequence[str]) -> tuple[str, ...]:
    codes = tuple(
        dict.fromkeys(str(value).strip().upper() for value in values if str(value).strip())
    )
    if not codes:
        raise EmbeddingInputError(
            "Phospho.ELM residue_codes must contain at least one residue code."
        )
    if any(code not in {"S", "T", "Y"} for code in codes):
        raise EmbeddingInputError(
            "Phospho.ELM residue_codes must be drawn from: S, T, Y."
        )
    return codes


def _iter_rows(path: str | Path) -> Iterable[dict[str, str]]:
    source = Path(path).expanduser()
    if not source.exists():
        raise EmbeddingInputError(f"Phospho.ELM dump does not exist: {source}.")
    if source.name.endswith(".tgz"):
        with tarfile.open(source, "r:gz") as archive:
            members = [
                member
                for member in archive.getmembers()
                if member.isfile() and member.name.endswith(".dump")
            ]
            if not members:
                raise EmbeddingInputError(
                    f"No .dump member found in Phospho.ELM archive {source}."
                )
            handle = archive.extractfile(members[0])
            if handle is None:
                raise EmbeddingInputError(
                    f"Could not read Phospho.ELM archive member {members[0].name}."
                )
            with handle:
                lines = (line.decode("utf-8") for line in handle)
                yield from csv.DictReader(lines, delimiter="\t")
        return
    with source.open("r", encoding="utf-8", newline="") as handle:
        yield from csv.DictReader(handle, delimiter="\t")


def _assign_stratified_splits(
    examples: Sequence[ResidueExample],
) -> list[ResidueExample]:
    primary_groups: dict[tuple[str, str], list[ResidueExample]] = {}
    for example in examples:
        metadata = example.metadata or {}
        key = (
            str(metadata.get("species") or "unknown"),
            str(metadata.get("positive_residue_stratum") or "none"),
        )
        primary_groups.setdefault(key, []).append(example)

    assigned: list[ResidueExample] = []
    rare_by_species: dict[str, list[ResidueExample]] = {}
    for key, group in primary_groups.items():
        if len(group) >= 10:
            assigned.extend(_split_group(group))
        else:
            rare_by_species.setdefault(key[0], []).extend(group)
    rare_global: list[ResidueExample] = []
    for group in rare_by_species.values():
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
    validation = int(round(total * 0.1))
    test = int(round(total * 0.1))
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
    return hashlib.sha256(f"phosphoelm-split-v1:{value}".encode("utf-8")).hexdigest()


def _positive_residue_counts(sequence: str, labels: Sequence[int]) -> dict[str, int]:
    counts = {"S": 0, "T": 0, "Y": 0}
    for residue, label in zip(sequence, labels):
        code = residue.upper()
        if label and code in counts:
            counts[code] += 1
    return counts


def _positive_residue_stratum(counts: Mapping[str, int]) -> str:
    codes = [code for code in ("S", "T", "Y") if int(counts.get(code, 0)) > 0]
    return "+".join(codes) if codes else "none"


__all__ = [
    "PHOSPHOELM_COLLECTION_METADATA",
    "PHOSPHOELM_DATASETS",
    "PhosphoElmSourceFilter",
    "load_phosphoelm_dataset",
]
