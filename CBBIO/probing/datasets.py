"""Core dataset containers for supervised probing tasks."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
import csv
from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Literal, Tuple, cast

from CBBIO.embeddings import EmbeddingInputError


SplitName = Literal["train", "val", "test"]
TaskLevel = Literal["protein", "residue"]
ObjectiveName = Literal["regression", "binary", "multiclass", "multilabel"]


@dataclass(frozen=True)
class ProteinExample:
    """One protein-level supervised example."""

    id: str
    sequence: str
    labels: Mapping[str, Any]
    split: SplitName
    metadata: Mapping[str, Any] | None = None


class ProteinDataset:
    """In-memory protein-level dataset with labels and explicit splits."""

    def __init__(self, examples: Iterable[ProteinExample]) -> None:
        self.examples = tuple(examples)
        if not self.examples:
            raise EmbeddingInputError("ProteinDataset requires at least one example.")
        seen: set[str] = set()
        for example in self.examples:
            if not example.id.strip():
                raise EmbeddingInputError("ProteinExample.id must be non-empty.")
            if example.id in seen:
                raise EmbeddingInputError(f"Duplicate protein example id: {example.id!r}.")
            seen.add(example.id)
            if not example.sequence.strip():
                raise EmbeddingInputError(f"ProteinExample {example.id!r} has an empty sequence.")
            if example.split not in {"train", "val", "test"}:
                raise EmbeddingInputError(
                    f"ProteinExample {example.id!r} split must be one of: train, val, test."
                )

    def ids(self) -> List[str]:
        return [example.id for example in self.examples]

    def by_split(self, split: SplitName) -> List[ProteinExample]:
        return [example for example in self.examples if example.split == split]

    def target_values(self, target: str) -> Dict[str, Any]:
        resolved = str(target).strip()
        if not resolved:
            raise EmbeddingInputError("Prediction target must be non-empty.")
        missing = [example.id for example in self.examples if resolved not in example.labels]
        if missing:
            sample = ", ".join(missing[:5])
            raise EmbeddingInputError(f"Target {resolved!r} is missing for examples: {sample}.")
        return {example.id: example.labels[resolved] for example in self.examples}

    def split_counts(self) -> Dict[str, int]:
        return {
            "train": len(self.by_split("train")),
            "val": len(self.by_split("val")),
            "test": len(self.by_split("test")),
        }

    def require_training_and_test_splits(self) -> Tuple[List[ProteinExample], List[ProteinExample]]:
        train = self.by_split("train")
        test = self.by_split("test")
        if not train:
            raise EmbeddingInputError("ProteinDataset requires at least one train example.")
        if not test:
            raise EmbeddingInputError("ProteinDataset requires at least one test example.")
        return train, test


@dataclass(frozen=True)
class ResidueExample:
    """One residue-level supervised example."""

    id: str
    sequence: str
    labels: Mapping[str, Sequence[Any]]
    split: SplitName
    mask: Sequence[bool] | None = None
    metadata: Mapping[str, Any] | None = None


class ResidueDataset:
    """In-memory residue-level dataset with per-position labels."""

    def __init__(self, examples: Iterable[ResidueExample]) -> None:
        self.examples = tuple(examples)
        if not self.examples:
            raise EmbeddingInputError("ResidueDataset requires at least one example.")
        seen: set[str] = set()
        for example in self.examples:
            if not example.id.strip():
                raise EmbeddingInputError("ResidueExample.id must be non-empty.")
            if example.id in seen:
                raise EmbeddingInputError(f"Duplicate residue example id: {example.id!r}.")
            seen.add(example.id)
            if not example.sequence.strip():
                raise EmbeddingInputError(f"ResidueExample {example.id!r} has an empty sequence.")
            if example.split not in {"train", "val", "test"}:
                raise EmbeddingInputError(
                    f"ResidueExample {example.id!r} split must be one of: train, val, test."
                )
            sequence_length = len(example.sequence)
            for target, values in example.labels.items():
                if len(values) != sequence_length:
                    raise EmbeddingInputError(
                        f"ResidueExample {example.id!r} target {target!r} has {len(values)} labels "
                        f"for a sequence of length {sequence_length}."
                    )
            if example.mask is not None and len(example.mask) != sequence_length:
                raise EmbeddingInputError(
                    f"ResidueExample {example.id!r} mask has length {len(example.mask)} "
                    f"for a sequence of length {sequence_length}."
                )

    def ids(self) -> List[str]:
        return [example.id for example in self.examples]

    def by_split(self, split: SplitName) -> List[ResidueExample]:
        return [example for example in self.examples if example.split == split]

    def target_values(self, target: str) -> Dict[str, Sequence[Any]]:
        resolved = str(target).strip()
        if not resolved:
            raise EmbeddingInputError("Prediction target must be non-empty.")
        missing = [example.id for example in self.examples if resolved not in example.labels]
        if missing:
            sample = ", ".join(missing[:5])
            raise EmbeddingInputError(f"Target {resolved!r} is missing for residue examples: {sample}.")
        return {example.id: example.labels[resolved] for example in self.examples}

    def split_counts(self) -> Dict[str, int]:
        return {
            "train": len(self.by_split("train")),
            "val": len(self.by_split("val")),
            "test": len(self.by_split("test")),
        }

    def require_training_and_test_splits(self) -> Tuple[List[ResidueExample], List[ResidueExample]]:
        train = self.by_split("train")
        test = self.by_split("test")
        if not train:
            raise EmbeddingInputError("ResidueDataset requires at least one train example.")
        if not test:
            raise EmbeddingInputError("ResidueDataset requires at least one test example.")
        return train, test


def load_residue_label_csv(
    csv_file: str | Path,
    *,
    sequence_field: str = "sequence",
    label_field: str = "labels",
    target: str = "target",
    id_field: str = "id",
    split_field: str = "split",
    mask_field: str | None = None,
) -> ResidueDataset:
    """Load a residue-level dataset from a CSV with one sequence per row.

    Labels may be encoded as JSON arrays, comma/space separated values, or a
    compact string with one character per residue, e.g. ``001010`` or ``HEC``.
    """

    path = Path(csv_file)
    examples: List[ResidueExample] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise EmbeddingInputError(f"Residue label CSV {path} does not contain a header row.")
        for required in (sequence_field, label_field, split_field):
            if required not in reader.fieldnames:
                raise EmbeddingInputError(f"Residue label CSV {path} is missing field {required!r}.")

        for row_index, row in enumerate(reader):
            sequence = str(row.get(sequence_field, "")).strip()
            labels = _parse_residue_label_sequence(row.get(label_field, ""), expected_length=len(sequence))
            mask = None
            if mask_field is not None and str(row.get(mask_field, "")).strip():
                mask_values = _parse_residue_label_sequence(row.get(mask_field, ""), expected_length=len(sequence))
                mask = [bool(_coerce_numeric_label(value)) for value in mask_values]
            examples.append(
                ResidueExample(
                    id=str(row.get(id_field) or row_index),
                    sequence=sequence,
                    labels={target: labels},
                    split=_normalize_split(row.get(split_field, "")),
                    mask=mask,
                    metadata={"source": "residue_csv", "row_index": row_index},
                )
            )
    return ResidueDataset(examples)


def _parse_residue_label_sequence(value: str | None, *, expected_length: int) -> List[Any]:
    text = "" if value is None else str(value).strip()
    if not text:
        raise EmbeddingInputError("Residue label values must be non-empty.")

    parsed: Any
    if text.startswith("["):
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError as exc:
            raise EmbeddingInputError("Could not parse residue labels as JSON.") from exc
        if not isinstance(parsed, list):
            raise EmbeddingInputError("Residue labels JSON must be an array.")
        labels = [_coerce_numeric_label(item) for item in cast(List[object], parsed)]
    elif "," in text:
        labels = [_coerce_numeric_label(item.strip()) for item in text.split(",")]
    elif " " in text:
        labels = [_coerce_numeric_label(item.strip()) for item in text.split() if item.strip()]
    else:
        labels = [_coerce_numeric_label(char) for char in text]

    if len(labels) != expected_length:
        raise EmbeddingInputError(
            f"Residue label length {len(labels)} does not match sequence length {expected_length}."
        )
    return labels


def _coerce_numeric_label(value: object) -> Any:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return value
    text = str(value).strip()
    if text == "":
        return math.nan
    try:
        return int(text)
    except ValueError:
        pass
    try:
        return float(text)
    except ValueError:
        return text


def _normalize_split(value: object) -> SplitName:
    text = str(value).strip().lower()
    if text in {"train", "training"}:
        return "train"
    if text in {"val", "valid", "validation"}:
        return "val"
    if text == "test":
        return "test"
    raise EmbeddingInputError("Split must be one of: train, val, test.")


__all__ = [
    "ObjectiveName",
    "ProteinDataset",
    "ProteinExample",
    "ResidueDataset",
    "ResidueExample",
    "SplitName",
    "TaskLevel",
    "load_residue_label_csv",
]
