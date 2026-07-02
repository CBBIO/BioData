"""Shared helpers for EC benchmark-style protein datasets."""

from __future__ import annotations

from collections.abc import Mapping
import csv
from pathlib import Path
from typing import Any, Literal

from CBBIO.embeddings import EmbeddingInputError

from ..datasets import ProteinExample, SplitName


EcBenchmarkLevel = Literal[1, 2, 3, 4]


def read_ec_benchmark_rows(
    path: Path,
    *,
    split: SplitName,
    target: str,
    level: EcBenchmarkLevel,
    source: str,
    metadata: Mapping[str, Any],
) -> list[ProteinExample]:
    """Read one EC benchmark table into protein examples."""
    if not path.exists():
        raise EmbeddingInputError(f"EC benchmark file not found: {path}.")
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        first_line = handle.readline()
        handle.seek(0)
        delimiter = "\t" if "\t" in first_line else ","
        reader = csv.DictReader(handle, delimiter=delimiter)
        if reader.fieldnames is None:
            raise EmbeddingInputError(f"EC benchmark file {path} is missing a header row.")
        examples: list[ProteinExample] = []
        for line_number, row in enumerate(reader, start=2):
            example = _row_to_example(
                row,
                path=path,
                line_number=line_number,
                split=split,
                target=target,
                level=level,
                source=source,
                metadata=metadata,
            )
            if example is not None:
                examples.append(example)
    return examples


def derive_ec_level_labels(value: str, *, level: EcBenchmarkLevel) -> list[str]:
    """Derive deduplicated EC labels at one hierarchy level."""
    labels: list[str] = []
    seen: set[str] = set()
    for raw_label in value.split(";"):
        label = raw_label.strip()
        if not label:
            continue
        parts = label.split(".")
        if len(parts) < level:
            continue
        selected = parts[:level]
        if any(part in {"", "-"} for part in selected):
            continue
        resolved = ".".join(selected)
        if resolved not in seen:
            labels.append(resolved)
            seen.add(resolved)
    return labels


def _row_to_example(
    row: Mapping[str, str],
    *,
    path: Path,
    line_number: int,
    split: SplitName,
    target: str,
    level: EcBenchmarkLevel,
    source: str,
    metadata: Mapping[str, Any],
) -> ProteinExample | None:
    protein_id = _require_row_value(row, ("id", "Entry", "ID"), path=path, line_number=line_number)
    sequence = _require_row_value(
        row,
        ("seq", "Sequence", "Sequences"),
        path=path,
        line_number=line_number,
    )
    ec_value = _require_row_value(
        row,
        ("ec_number", "EC number", "EC"),
        path=path,
        line_number=line_number,
    )
    labels = derive_ec_level_labels(ec_value, level=level)
    if not labels:
        return None
    return ProteinExample(
        id=protein_id,
        sequence=sequence,
        labels={target: labels},
        split=split,
        metadata={
            **dict(metadata),
            "source": source,
            "raw_ec": ec_value,
            "ec_level": level,
        },
    )


def _require_row_value(
    row: Mapping[str, str],
    keys: tuple[str, ...],
    *,
    path: Path,
    line_number: int,
) -> str:
    for key in keys:
        value = row.get(key)
        if value is not None and value.strip():
            return value.strip()
    supported = ", ".join(keys)
    raise EmbeddingInputError(
        f"EC benchmark row {line_number} in {path} is missing required field: {supported}."
    )


__all__ = [
    "EcBenchmarkLevel",
    "derive_ec_level_labels",
    "read_ec_benchmark_rows",
]
