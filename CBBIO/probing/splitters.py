"""Deterministic splitters for supervised probing datasets."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from dataclasses import replace
import hashlib
from typing import Literal, TypeVar, cast

from CBBIO.embeddings import EmbeddingInputError

from .datasets import ProteinDataset, ProteinExample, ResidueDataset, ResidueExample, SplitName


DatasetExample = ProteinExample | ResidueExample
_DatasetExampleT = TypeVar("_DatasetExampleT", ProteinExample, ResidueExample)
HoldoutStrategy = Literal["none", "random", "rarest_first"]

DEFAULT_SPLIT_RATIOS: Mapping[SplitName, float] = {"train": 0.8, "val": 0.1, "test": 0.1}
DEFAULT_HOLDOUT_REMAINDER_RATIOS: Mapping[SplitName, float] = {
    "train": 0.9,
    "val": 0.1,
    "test": 0.0,
}


class DatasetSplitter(ABC):
    """Assign deterministic train, validation, and test splits to examples."""

    def __init__(
        self,
        *,
        salt: str,
        ratios: Mapping[SplitName, float] = DEFAULT_SPLIT_RATIOS,
    ) -> None:
        self.salt = _normalize_salt(salt)
        self.ratios = _normalize_ratios(ratios)

    @abstractmethod
    def split_examples(self, examples: Sequence[_DatasetExampleT]) -> list[_DatasetExampleT]:
        """Return examples with reassigned split values."""

    def split_dataset(
        self,
        dataset: ProteinDataset | ResidueDataset,
    ) -> ProteinDataset | ResidueDataset:
        """Return a dataset with reassigned split values."""
        if isinstance(dataset, ProteinDataset):
            return ProteinDataset(self.split_examples(dataset.examples))
        return ResidueDataset(self.split_examples(dataset.examples))


class HashDatasetSplitter(DatasetSplitter):
    """Assign splits by sorting examples with a stable salted hash."""

    def split_examples(self, examples: Sequence[_DatasetExampleT]) -> list[_DatasetExampleT]:
        """Return examples split by deterministic salted ID ordering."""
        return _split_ordered_examples(
            sorted(examples, key=lambda example: _stable_hash(self.salt, example.id)),
            ratios=self.ratios,
        )


class StratifiedDatasetSplitter(DatasetSplitter):
    """Assign deterministic splits while preserving metadata strata when possible."""

    def __init__(
        self,
        *,
        metadata_fields: Sequence[str],
        salt: str,
        ratios: Mapping[SplitName, float] = DEFAULT_SPLIT_RATIOS,
        min_group_size: int = 10,
        fallback_metadata_fields: Sequence[str] | None = None,
    ) -> None:
        super().__init__(salt=salt, ratios=ratios)
        self.metadata_fields = _normalize_metadata_fields(metadata_fields)
        self.min_group_size = _normalize_min_group_size(min_group_size)
        self.fallback_metadata_fields = (
            _normalize_metadata_fields(fallback_metadata_fields)
            if fallback_metadata_fields is not None
            else self.metadata_fields[:1]
        )

    def split_examples(self, examples: Sequence[_DatasetExampleT]) -> list[_DatasetExampleT]:
        """Return examples split within metadata strata and rare fallback groups."""
        assigned: list[_DatasetExampleT] = []
        rare_examples: list[_DatasetExampleT] = []
        for group in _metadata_groups(examples, self.metadata_fields).values():
            if len(group) >= self.min_group_size:
                assigned.extend(self._split_group(group))
            else:
                rare_examples.extend(group)

        rare_global: list[_DatasetExampleT] = []
        for group in _metadata_groups(rare_examples, self.fallback_metadata_fields).values():
            if len(group) >= self.min_group_size:
                assigned.extend(self._split_group(group))
            else:
                rare_global.extend(group)
        if rare_global:
            assigned.extend(self._split_group(rare_global))
        return sorted(assigned, key=lambda example: example.id)

    def _split_group(self, group: Sequence[_DatasetExampleT]) -> list[_DatasetExampleT]:
        return _split_ordered_examples(
            sorted(group, key=lambda example: _stable_hash(self.salt, example.id)),
            ratios=self.ratios,
        )


class HoldoutDatasetSplitter(DatasetSplitter):
    """Assign randomly selected metadata classes to test and split the remainder."""

    def __init__(
        self,
        *,
        metadata_field: str,
        holdout_values: Sequence[str] | None = None,
        holdout_strategy: HoldoutStrategy = "random",
        salt: str,
        ratios: Mapping[SplitName, float] = DEFAULT_HOLDOUT_REMAINDER_RATIOS,
    ) -> None:
        super().__init__(salt=salt, ratios=ratios)
        self.metadata_field = _normalize_metadata_field(metadata_field)
        self.holdout_values = _normalize_holdout_values(holdout_values)
        self.holdout_strategy = _normalize_holdout_strategy(holdout_strategy)

    def split_examples(self, examples: Sequence[_DatasetExampleT]) -> list[_DatasetExampleT]:
        """Return examples with metadata holdouts assigned to test."""
        holdout_values = self._resolve_holdout_values(examples)
        held_out: list[_DatasetExampleT] = []
        remaining: list[_DatasetExampleT] = []
        for example in examples:
            value = _metadata_value(example, self.metadata_field)
            if value in holdout_values:
                held_out.append(_replace_split(example, "test"))
            else:
                remaining.append(example)
        remainder_ratios = self._remainder_ratios() if holdout_values else self.ratios
        return _split_ordered_examples(
            sorted(remaining, key=lambda example: _stable_hash(self.salt, example.id)),
            ratios=remainder_ratios,
        ) + sorted(held_out, key=lambda example: example.id)

    def _resolve_holdout_values(self, examples: Sequence[DatasetExample]) -> set[str]:
        if self.holdout_values is not None:
            return self.holdout_values
        if self.holdout_strategy == "none":
            return set()
        class_counts: dict[str, int] = {}
        for example in examples:
            value = _metadata_value(example, self.metadata_field)
            class_counts[value] = class_counts.get(value, 0) + 1
        if not class_counts:
            raise EmbeddingInputError("cannot select holdout classes from an empty dataset.")
        target_count = max(1, int(round(len(examples) * self._holdout_fraction())))
        selected: set[str] = set()
        selected_count = 0
        for value in self._ordered_holdout_values(class_counts):
            selected.add(value)
            selected_count += class_counts[value]
            if selected_count >= target_count:
                return selected
        return selected

    def _holdout_fraction(self) -> float:
        value = 1.0 - self.ratios["train"] - self.ratios["val"]
        if value <= 0.0:
            value = self.ratios["val"]
        if value <= 0.0:
            raise EmbeddingInputError(
                "holdout split requires a positive test or validation ratio."
            )
        return value

    def _remainder_ratios(self) -> Mapping[SplitName, float]:
        total = self.ratios["train"] + self.ratios["val"]
        if total <= 0.0:
            raise EmbeddingInputError("holdout remainder split requires a train or validation ratio.")
        return {
            "train": self.ratios["train"] / total,
            "val": self.ratios["val"] / total,
            "test": 0.0,
        }

    def _ordered_holdout_values(self, class_counts: Mapping[str, int]) -> list[str]:
        if self.holdout_strategy == "random":
            return sorted(
                class_counts,
                key=lambda value: _stable_hash(f"{self.salt}:holdout", value),
            )
        if self.holdout_strategy == "rarest_first":
            return sorted(
                class_counts,
                key=lambda value: (
                    class_counts[value],
                    _stable_hash(f"{self.salt}:holdout", value),
                ),
            )
        return []


def _split_ordered_examples(
    examples: Sequence[_DatasetExampleT],
    *,
    ratios: Mapping[SplitName, float],
) -> list[_DatasetExampleT]:
    split_names: list[SplitName] = [
        split_name
        for split_name, count in _split_counts(len(examples), ratios=ratios)
        for _ in range(count)
    ]
    return [
        _replace_split(example, split_name)
        for example, split_name in zip(examples, split_names)
    ]


def _split_counts(total: int, *, ratios: Mapping[SplitName, float]) -> list[tuple[SplitName, int]]:
    if total <= 0:
        return [("train", 0), ("val", 0), ("test", 0)]
    validation = int(round(total * ratios["val"]))
    test = int(round(total * ratios["test"]))
    if total >= 10:
        if ratios["val"] > 0.0:
            validation = max(1, validation)
        if ratios["test"] > 0.0:
            test = max(1, test)
    elif total >= 2 and ratios["test"] > 0.0:
        test = max(1, test)
    train = total - validation - test
    while train < 1 and validation > 0:
        validation -= 1
        train += 1
    while train < 1 and test > 0:
        test -= 1
        train += 1
    return [("train", train), ("val", validation), ("test", test)]


def _replace_split(example: _DatasetExampleT, split: SplitName) -> _DatasetExampleT:
    return replace(example, split=split)


def _metadata_groups(
    examples: Sequence[_DatasetExampleT],
    fields: Sequence[str],
) -> dict[tuple[str, ...], list[_DatasetExampleT]]:
    groups: dict[tuple[str, ...], list[_DatasetExampleT]] = {}
    for example in examples:
        key = tuple(_metadata_value(example, field) for field in fields)
        groups.setdefault(key, []).append(example)
    return groups


def _metadata_value(example: DatasetExample, field: str) -> str:
    metadata = example.metadata or {}
    fallback = "unknown" if field == "species" else "none"
    return str(metadata.get(field) or fallback)


def _stable_hash(salt: str, value: str) -> str:
    return hashlib.sha256(f"{salt}:{value}".encode("utf-8")).hexdigest()


def _normalize_salt(value: str) -> str:
    salt = str(value).strip()
    if not salt:
        raise EmbeddingInputError("splitter salt must be non-empty.")
    return salt


def _normalize_ratios(ratios: Mapping[SplitName, float]) -> dict[SplitName, float]:
    missing = {"train", "val", "test"}.difference(ratios)
    if missing:
        raise EmbeddingInputError("split ratios must include train, val, and test.")
    split_order: tuple[SplitName, ...] = ("train", "val", "test")
    normalized: dict[SplitName, float] = {split: float(ratios[split]) for split in split_order}
    if any(value < 0.0 for value in normalized.values()):
        raise EmbeddingInputError("split ratios must be non-negative.")
    total = sum(normalized.values())
    if total <= 0.0:
        raise EmbeddingInputError("at least one split ratio must be positive.")
    return {split: value / total for split, value in normalized.items()}


def _normalize_metadata_fields(values: Sequence[str]) -> tuple[str, ...]:
    fields = tuple(_normalize_metadata_field(value) for value in values)
    if not fields:
        raise EmbeddingInputError("metadata_fields must contain at least one field.")
    return fields


def _normalize_metadata_field(value: str) -> str:
    field = str(value).strip()
    if not field:
        raise EmbeddingInputError("metadata field names must be non-empty.")
    return field


def _normalize_min_group_size(value: int) -> int:
    size = int(value)
    if size < 1:
        raise EmbeddingInputError("min_group_size must be >= 1.")
    return size


def _normalize_holdout_values(values: Sequence[str] | None) -> set[str] | None:
    if values is None:
        return None
    normalized = {str(value) for value in values}
    if not normalized:
        raise EmbeddingInputError("holdout_values must contain at least one value.")
    return normalized


def _normalize_holdout_strategy(value: str) -> HoldoutStrategy:
    strategy = str(value).strip().lower()
    if strategy in {"none", "random", "rarest_first"}:
        return cast(HoldoutStrategy, strategy)
    raise EmbeddingInputError("holdout_strategy must be one of: none, random, rarest_first.")


__all__ = [
    "DEFAULT_HOLDOUT_REMAINDER_RATIOS",
    "DEFAULT_SPLIT_RATIOS",
    "DatasetSplitter",
    "HashDatasetSplitter",
    "HoldoutDatasetSplitter",
    "HoldoutStrategy",
    "StratifiedDatasetSplitter",
]
