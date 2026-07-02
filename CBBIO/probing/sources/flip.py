"""FLIP benchmark dataset loading for probing."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import csv
from dataclasses import dataclass
import math
from pathlib import Path
from typing import Any, Dict, List, Literal, Tuple
import zipfile
from urllib.request import urlretrieve

from CBBIO.embeddings import EmbeddingInputError

from ..datasets import ProteinDataset, ProteinExample, SplitName


FlipDatasetName = Literal["aav", "gb1", "thermostability"]


@dataclass(frozen=True)
class FlipDatasetSpec:
    """Download and split metadata for one FLIP benchmark dataset."""

    name: FlipDatasetName
    url: str
    md5: str
    splits: Tuple[str, ...]
    target_fields: Tuple[str, ...] = ("target",)
    mutation_region: Tuple[int, int] | None = None


FLIP_DATASETS: Dict[str, FlipDatasetSpec] = {
    "aav": FlipDatasetSpec(
        name="aav",
        url="https://github.com/J-SNACKKB/FLIP/raw/d5c35cc716ca93c3c74a0b43eef5b60cbf88521f/splits/aav/splits.zip",
        md5="cabdd41f3386f4949b32ca220db55c58",
        splits=("des_mut", "low_vs_high", "mut_des", "one_vs_many", "sampled", "seven_vs_many", "two_vs_many"),
        mutation_region=(474, 674),
    ),
    "gb1": FlipDatasetSpec(
        name="gb1",
        url="https://github.com/J-SNACKKB/FLIP/raw/d5c35cc716ca93c3c74a0b43eef5b60cbf88521f/splits/gb1/splits.zip",
        md5="14216947834e6db551967c2537332a12",
        splits=("one_vs_rest", "two_vs_rest", "three_vs_rest", "low_vs_high", "sampled"),
    ),
    "thermostability": FlipDatasetSpec(
        name="thermostability",
        url="https://github.com/J-SNACKKB/FLIP/raw/d5c35cc716ca93c3c74a0b43eef5b60cbf88521f/splits/meltome/splits.zip",
        md5="0f8b1e848568f7566713d53594c0ca90",
        splits=("human", "human_cell", "mixed_split"),
    ),
}


def load_flip_csv(
    csv_file: str | Path,
    *,
    sequence_field: str = "sequence",
    target_fields: Sequence[str] | None = ("target",),
    keep_mutation_region: Tuple[int, int] | None = None,
) -> ProteinDataset:
    """Load a local FLIP split CSV into a ``ProteinDataset``.

    FLIP CSVs use ``set=train/test`` and ``validation=True`` to identify the
    validation subset. We translate those into ``train``, ``val``, and ``test``.
    """

    path = Path(csv_file)
    examples: List[ProteinExample] = []
    target_field_set = set(target_fields) if target_fields is not None else None

    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise EmbeddingInputError(f"FLIP CSV {path} does not contain a header row.")
        if sequence_field not in reader.fieldnames:
            raise EmbeddingInputError(f"FLIP CSV {path} is missing sequence field {sequence_field!r}.")

        for row_index, row in enumerate(reader):
            if not _flip_row_has_assigned_split(row):
                continue

            sequence = str(row.get(sequence_field, "")).strip()
            if keep_mutation_region is not None:
                start, end = keep_mutation_region
                sequence = sequence[start:end]

            labels: Dict[str, Any] = {}
            for field, value in row.items():
                if field in {sequence_field, "set", "validation"}:
                    continue
                if target_field_set is not None and field not in target_field_set:
                    continue
                labels[field] = _parse_flip_value(value)

            examples.append(
                ProteinExample(
                    id=str(row.get("id") or row.get("name") or row_index),
                    sequence=sequence,
                    labels=labels,
                    split=_flip_row_split(row),
                    metadata={"source": "flip", "row_index": row_index},
                )
            )

    return ProteinDataset(_ordered_flip_examples(examples))


def load_flip_dataset(
    root: str | Path,
    *,
    name: FlipDatasetName,
    split: str,
    download: bool = False,
    keep_mutation_region: bool = False,
) -> ProteinDataset:
    """Load one named FLIP dataset split, optionally downloading the split zip."""

    spec = FLIP_DATASETS.get(name)
    if spec is None:
        supported = ", ".join(sorted(FLIP_DATASETS))
        raise EmbeddingInputError(f"Unknown FLIP dataset {name!r}. Supported values: {supported}.")
    if split not in spec.splits:
        supported_splits = ", ".join(spec.splits)
        raise EmbeddingInputError(f"Unsupported FLIP split {split!r} for {name!r}. Supported values: {supported_splits}.")

    dataset_root = Path(root).expanduser() / name
    csv_path = dataset_root / "splits" / f"{split}.csv"
    if not csv_path.exists() and download:
        _download_and_extract_flip(spec, dataset_root)
    if not csv_path.exists():
        raise EmbeddingInputError(
            f"FLIP split CSV not found at {csv_path}. Pass download=True or provide an extracted FLIP split directory."
        )

    region = spec.mutation_region if keep_mutation_region and spec.mutation_region is not None else None
    return load_flip_csv(csv_path, target_fields=spec.target_fields, keep_mutation_region=region)


def _download_and_extract_flip(spec: FlipDatasetSpec, dataset_root: Path) -> None:
    dataset_root.mkdir(parents=True, exist_ok=True)
    zip_path = dataset_root / "splits.zip"
    if not zip_path.exists():
        urlretrieve(spec.url, zip_path)
    if not zipfile.is_zipfile(zip_path):
        raise EmbeddingInputError(f"Downloaded FLIP archive is not a zip file: {zip_path}.")
    with zipfile.ZipFile(zip_path) as archive:
        archive.extractall(dataset_root)


def _parse_flip_value(value: str | None) -> Any:
    text = "" if value is None else str(value).strip()
    if text == "":
        return math.nan
    if text in {"True", "False"}:
        return text == "True"
    try:
        return int(text)
    except ValueError:
        pass
    try:
        return float(text)
    except ValueError:
        return text


def _flip_row_has_assigned_split(row: Mapping[str, str]) -> bool:
    raw_set = str(row.get("set", "")).strip().lower()
    validation = str(row.get("validation", "")).strip()
    return raw_set in {"train", "test"} or validation == "True"


def _flip_row_split(row: Mapping[str, str]) -> SplitName:
    if str(row.get("validation", "")).strip() == "True":
        return "val"
    raw_set = str(row.get("set", "")).strip().lower()
    if raw_set == "train":
        return "train"
    if raw_set == "test":
        return "test"
    raise EmbeddingInputError("FLIP rows must use set='train' or set='test', with optional validation=True.")


def _ordered_flip_examples(examples: Sequence[ProteinExample]) -> List[ProteinExample]:
    return (
        [example for example in examples if example.split == "train"]
        + [example for example in examples if example.split == "val"]
        + [example for example in examples if example.split == "test"]
    )


__all__ = [
    "FLIP_DATASETS",
    "FlipDatasetName",
    "FlipDatasetSpec",
    "load_flip_csv",
    "load_flip_dataset",
]
