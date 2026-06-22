"""Load generic residue labels from interval and per-position tables."""

from __future__ import annotations

from collections.abc import Mapping
import csv
import json
from pathlib import Path
from typing import Any, cast

from CBBIO.embeddings import EmbeddingInputError

from ..datasets import ResidueDataset, ResidueExample, SplitName


def load_interval_residue_tsv(
    path: str | Path,
    *,
    target: str = "target",
    id_field: str = "id",
    sequence_field: str = "sequence",
    start_field: str = "start",
    end_field: str = "end",
    split_field: str | None = "split",
    one_based: bool = True,
    delimiter: str = "\t",
    default_split: SplitName = "train",
) -> ResidueDataset:
    """Load residue labels from inclusive interval or site rows.

    Args:
        path: Delimited input table.
        target: Label name stored in each residue example.
        id_field: Protein identifier column.
        sequence_field: Protein sequence column.
        start_field: Interval start column.
        end_field: Interval end column.
        split_field: Optional dataset split column.
        one_based: Interpret coordinates as one-based when true.
        delimiter: Input field delimiter.
        default_split: Split used when the split column is absent or empty.

    Returns:
        Residue dataset with rows grouped by protein identifier.

    Raises:
        EmbeddingInputError: If rows, coordinates, or sequences are invalid.
    """
    grouped: dict[str, dict[str, Any]] = {}
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter=delimiter)
        if reader.fieldnames is None:
            raise EmbeddingInputError(f"Interval file {path} does not contain a header row.")
        for row in reader:
            record_id = str(row.get(id_field, "")).strip()
            sequence = str(row.get(sequence_field, "")).strip()
            if not record_id or not sequence:
                raise EmbeddingInputError(
                    "Interval rows require non-empty id and sequence fields."
                )
            group = grouped.setdefault(
                record_id,
                {
                    "sequence": sequence,
                    "labels": [0] * len(sequence),
                    "split": _row_split(
                        row,
                        split_field=split_field,
                        default_split=default_split,
                    ),
                },
            )
            if group["sequence"] != sequence:
                raise EmbeddingInputError(
                    f"Conflicting sequences for interval id {record_id!r}."
                )
            start = int(str(row.get(start_field, "")).strip())
            end = int(str(row.get(end_field, start)).strip())
            _mark_interval(
                cast(list[int], group["labels"]),
                start=start,
                end=end,
                one_based=one_based,
            )
    return ResidueDataset(
        ResidueExample(
            id=record_id,
            sequence=str(group["sequence"]),
            labels={target: cast(list[int], group["labels"])},
            split=cast(SplitName, group["split"]),
            metadata={"source": "interval_tsv"},
        )
        for record_id, group in grouped.items()
    )


def load_residue_label_table(
    path: str | Path,
    *,
    target: str = "target",
    id_field: str = "id",
    sequence_field: str = "sequence",
    labels_field: str = "labels",
    split_field: str | None = "split",
    delimiter: str = "\t",
    default_split: SplitName = "train",
) -> ResidueDataset:
    """Load one compact per-residue label sequence per table row."""
    examples: list[ResidueExample] = []
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter=delimiter)
        if reader.fieldnames is None:
            raise EmbeddingInputError(
                f"Residue label table {path} does not contain a header row."
            )
        for row_index, row in enumerate(reader):
            record_id = str(row.get(id_field) or row_index)
            sequence = str(row.get(sequence_field, "")).strip()
            labels = _parse_label_sequence(
                row.get(labels_field, ""),
                expected_length=len(sequence),
            )
            examples.append(
                ResidueExample(
                    id=record_id,
                    sequence=sequence,
                    labels={target: labels},
                    split=_row_split(
                        row,
                        split_field=split_field,
                        default_split=default_split,
                    ),
                    metadata={"source": "residue_label_table", "row_index": row_index},
                )
            )
    return ResidueDataset(examples)


def _row_split(
    row: Mapping[str, str],
    *,
    split_field: str | None,
    default_split: SplitName,
) -> SplitName:
    if split_field is None:
        return default_split
    value = str(row.get(split_field, "")).strip().lower()
    if not value:
        return default_split
    if value in {"train", "training"}:
        return "train"
    if value in {"val", "valid", "validation"}:
        return "val"
    if value == "test":
        return "test"
    raise EmbeddingInputError("Split must be one of: train, val, test.")


def _mark_interval(
    labels: list[int],
    *,
    start: int,
    end: int,
    one_based: bool,
) -> None:
    start_index = start - 1 if one_based else start
    end_index = end - 1 if one_based else end
    if start_index < 0 or end_index >= len(labels) or end_index < start_index:
        raise EmbeddingInputError(
            f"Invalid residue interval start={start} end={end} "
            f"for sequence length {len(labels)}."
        )
    for index in range(start_index, end_index + 1):
        labels[index] = 1


def _parse_label_sequence(value: str | None, *, expected_length: int) -> list[Any]:
    text = "" if value is None else str(value).strip()
    if text.startswith("["):
        parsed = json.loads(text)
        if not isinstance(parsed, list):
            raise EmbeddingInputError("Label JSON must be an array.")
        labels = list(cast(list[object], parsed))
    elif "," in text:
        labels = [_coerce_label(part.strip()) for part in text.split(",")]
    elif " " in text:
        labels = [_coerce_label(part.strip()) for part in text.split() if part.strip()]
    else:
        labels = [_coerce_label(char) for char in text]
    if len(labels) != expected_length:
        raise EmbeddingInputError(
            f"Label length {len(labels)} does not match sequence length {expected_length}."
        )
    return labels


def _coerce_label(value: object) -> Any:
    if isinstance(value, (int, float, bool)):
        return int(value) if isinstance(value, bool) else value
    text = str(value).strip()
    try:
        return int(text)
    except ValueError:
        return text


__all__ = ["load_interval_residue_tsv", "load_residue_label_table"]
