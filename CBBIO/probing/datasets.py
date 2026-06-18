"""Core dataset containers for supervised probing tasks."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
import csv
from dataclasses import dataclass
import json
import math
from pathlib import Path
import shutil
import subprocess
import tempfile
from typing import Any, Dict, List, Literal, Tuple, cast

from CBBIO.embeddings import EmbeddingDependencyError, EmbeddingInputError


SplitName = Literal["train", "val", "test"]
TaskLevel = Literal["protein", "residue"]
ObjectiveName = Literal["regression", "binary", "multiclass", "multilabel"]


@dataclass(frozen=True)
class MmseqsRedundancyHit:
    """One MMseqs2 hit that caused a train/validation example to be removed."""

    removed_id: str
    reference_id: str
    percent_identity: float
    query_coverage: float
    target_coverage: float
    alignment_length: int
    evalue: float
    bitscore: float


@dataclass(frozen=True)
class MmseqsRedundancyReport:
    """Summary of an MMseqs2 train/validation-to-test redundancy filter run."""

    cutoff: float
    min_coverage: float
    removed_count: int
    retained_count: int
    reference_count: int
    removed: Tuple[MmseqsRedundancyHit, ...]
    hits_path: Path | None = None


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


def filter_redundant_to_test_mmseqs(
    dataset: ProteinDataset | ResidueDataset,
    *,
    cutoff: float = 30.0,
    min_coverage: float = 0.8,
    remove_splits: Sequence[SplitName] = ("train", "val"),
    reference_split: SplitName = "test",
    threads: int | None = None,
) -> Tuple[ProteinDataset | ResidueDataset, MmseqsRedundancyReport]:
    """Remove train/validation examples too similar to the reference split.

    The filter is opt-in and delegates sequence search to the MMseqs2 CLI. Any
    example in ``remove_splits`` is removed when MMseqs2 reports at least one
    hit to ``reference_split`` at or above ``cutoff`` percent identity and
    ``min_coverage`` coverage.
    """

    mmseqs = shutil.which("mmseqs")
    if mmseqs is None:
        raise EmbeddingDependencyError("MMseqs2 executable 'mmseqs' was not found on PATH.")
    cutoff_value = float(cutoff)
    min_coverage_value = float(min_coverage)
    if cutoff_value < 0.0 or cutoff_value > 100.0:
        raise EmbeddingInputError("cutoff must be between 0 and 100 percent.")
    if min_coverage_value < 0.0 or min_coverage_value > 1.0:
        raise EmbeddingInputError("min_coverage must be between 0 and 1.")
    if threads is not None and int(threads) < 1:
        raise EmbeddingInputError("threads must be >= 1 when provided.")

    remove_split_set = {str(split).strip().lower() for split in remove_splits}
    invalid_splits = remove_split_set.difference({"train", "val", "test"})
    if invalid_splits:
        raise EmbeddingInputError("remove_splits must contain only train, val, or test.")
    examples = dataset.examples
    removable = [example for example in examples if example.split in remove_split_set]
    references = [example for example in examples if example.split == reference_split]
    if not references:
        raise EmbeddingInputError(f"Dataset has no examples in reference split {reference_split!r}.")

    if not removable:
        report = MmseqsRedundancyReport(
            cutoff=cutoff_value,
            min_coverage=min_coverage_value,
            removed_count=0,
            retained_count=len(examples),
            reference_count=len(references),
            removed=(),
        )
        return dataset, report

    with tempfile.TemporaryDirectory(prefix="cbbio-mmseqs-") as tmp_name:
        tmp_dir = Path(tmp_name)
        query_fasta = tmp_dir / "remove_candidates.fasta"
        target_fasta = tmp_dir / "reference.fasta"
        hits_path = tmp_dir / "hits.tsv"
        mmseqs_tmp = tmp_dir / "mmseqs_tmp"
        _write_examples_fasta(query_fasta, removable)
        _write_examples_fasta(target_fasta, references)
        command = [
            mmseqs,
            "easy-search",
            str(query_fasta),
            str(target_fasta),
            str(hits_path),
            str(mmseqs_tmp),
            "--min-seq-id",
            f"{cutoff_value / 100.0:.6g}",
            "-c",
            f"{min_coverage_value:.6g}",
            "--cov-mode",
            "0",
            "--format-output",
            "query,target,pident,qcov,tcov,alnlen,qlen,tlen,evalue,bits",
            "-v",
            "1",
        ]
        if threads is not None:
            command.extend(["--threads", str(int(threads))])
        try:
            subprocess.run(command, check=True, capture_output=True, text=True)
        except subprocess.CalledProcessError as exc:
            detail = (exc.stderr or exc.stdout or "").strip()
            suffix = f": {detail}" if detail else "."
            raise EmbeddingDependencyError(f"MMseqs2 easy-search failed{suffix}") from exc

        removed_hits = _read_mmseqs_redundancy_hits(
            hits_path,
            cutoff=cutoff_value,
            min_coverage=min_coverage_value,
        )

    removed_ids = {hit.removed_id for hit in removed_hits}
    kept_examples = [example for example in examples if example.id not in removed_ids]
    if isinstance(dataset, ProteinDataset):
        filtered: ProteinDataset | ResidueDataset = ProteinDataset(cast(Iterable[ProteinExample], kept_examples))
    else:
        filtered = ResidueDataset(cast(Iterable[ResidueExample], kept_examples))
    report = MmseqsRedundancyReport(
        cutoff=cutoff_value,
        min_coverage=min_coverage_value,
        removed_count=len(removed_ids),
        retained_count=len(kept_examples),
        reference_count=len(references),
        removed=tuple(removed_hits),
    )
    return filtered, report


def _write_examples_fasta(path: Path, examples: Sequence[ProteinExample | ResidueExample]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for example in examples:
            handle.write(f">{_fasta_safe_id(example.id)}\n")
            handle.write(f"{_clean_fasta_sequence(example.sequence)}\n")


def _fasta_safe_id(value: str) -> str:
    safe = str(value).strip().replace("\t", "_").replace("\n", "_").replace("\r", "_").replace(" ", "_")
    if not safe:
        raise EmbeddingInputError("Example ids must be non-empty for MMseqs2 filtering.")
    return safe


def _clean_fasta_sequence(value: str) -> str:
    sequence = "".join(str(value).split()).upper()
    if not sequence:
        raise EmbeddingInputError("Example sequences must be non-empty for MMseqs2 filtering.")
    return sequence


def _read_mmseqs_redundancy_hits(
    path: Path,
    *,
    cutoff: float,
    min_coverage: float,
) -> List[MmseqsRedundancyHit]:
    best: Dict[str, MmseqsRedundancyHit] = {}
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            columns = line.split("\t")
            if len(columns) != 10:
                raise EmbeddingInputError(f"MMseqs2 output row {line_number} has {len(columns)} columns; expected 10.")
            hit = MmseqsRedundancyHit(
                removed_id=columns[0],
                reference_id=columns[1],
                percent_identity=float(columns[2]),
                query_coverage=float(columns[3]),
                target_coverage=float(columns[4]),
                alignment_length=int(float(columns[5])),
                evalue=float(columns[8]),
                bitscore=float(columns[9]),
            )
            if hit.percent_identity < cutoff or hit.query_coverage < min_coverage or hit.target_coverage < min_coverage:
                continue
            previous = best.get(hit.removed_id)
            if previous is None or (hit.bitscore, hit.percent_identity) > (previous.bitscore, previous.percent_identity):
                best[hit.removed_id] = hit
    return [best[key] for key in sorted(best)]


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
