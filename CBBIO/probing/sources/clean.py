"""Load CLEAN enzyme commission benchmark datasets."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Literal, cast

from CBBIO.embeddings import EmbeddingInputError

from ..collection_types import CollectionMetadata, DatasetMetadata
from ..datasets import ProteinDataset, ProteinExample, SplitName
from ._ec_benchmarks import EcBenchmarkLevel, read_ec_benchmark_rows


CleanSplitThreshold = Literal[10, 30, 50, 70, 100]
CleanFold = Literal[0, 1, 2, 3, 4]
CleanTestSet = Literal["native", "halogenase", "price", "new"]

CLEAN_COLLECTION_METADATA = CollectionMetadata(
    id="clean",
    display_name="CLEAN",
    description="CLEAN enzyme commission prediction benchmark splits.",
    tags=("protein", "function", "enzyme", "ec", "benchmark", "clean"),
)

_CLEAN_LEVELS: tuple[EcBenchmarkLevel, ...] = (1, 2, 3, 4)
_CLEAN_SPLIT_THRESHOLDS: tuple[CleanSplitThreshold, ...] = (10, 30, 50, 70, 100)
_CLEAN_FOLDS: tuple[CleanFold, ...] = (0, 1, 2, 3, 4)
_CLEAN_TEST_SETS: tuple[CleanTestSet, ...] = ("native", "halogenase", "price", "new")
_CLEAN_NATIVE_ROOT_NAME = "CLEAN_all_train_valid_splits"
_CLEAN_ROOT_NAME = "CLEAN"


def load_clean_dataset(
    root: str | Path,
    *,
    name: str,
    split: str | Sequence[str] | None = None,
    target: str | None = None,
) -> ProteinDataset:
    """Load one CLEAN EC benchmark dataset."""
    level, threshold, fold, test_set = _parse_clean_dataset_name(name)
    expected_target = f"ec_{level}"
    resolved_target = target or expected_target
    if resolved_target != expected_target:
        raise EmbeddingInputError(
            f"CLEAN dataset {name!r} only supports target {expected_target!r}."
        )
    root_path = _resolve_clean_root(Path(root).expanduser())
    split_names = _resolve_clean_splits(split)
    examples: list[ProteinExample] = []
    for split_name in split_names:
        path = _clean_split_path(root_path, split_name=split_name, threshold=threshold, fold=fold, test_set=test_set)
        examples.extend(
            read_ec_benchmark_rows(
                path,
                split=split_name,
                target=resolved_target,
                level=level,
                source="CLEAN",
                metadata={
                    "collection": "clean",
                    "split_threshold": threshold,
                    "fold": fold,
                    "test_set": test_set,
                },
            )
        )
    return ProteinDataset(examples)


def _clean_dataset_id(
    *,
    level: EcBenchmarkLevel,
    threshold: CleanSplitThreshold,
    fold: CleanFold,
    test_set: CleanTestSet,
) -> str:
    return f"clean:{_clean_dataset_name(level=level, threshold=threshold, fold=fold, test_set=test_set)}"


def _clean_dataset_name(
    *,
    level: EcBenchmarkLevel,
    threshold: CleanSplitThreshold,
    fold: CleanFold,
    test_set: CleanTestSet,
) -> str:
    base = f"ec_{level}_split{threshold}_fold{fold}"
    return base if test_set == "native" else f"{base}_{test_set}"


def _clean_display_name(
    *,
    level: EcBenchmarkLevel,
    threshold: CleanSplitThreshold,
    fold: CleanFold,
    test_set: CleanTestSet,
) -> str:
    suffix = "" if test_set == "native" else f", {test_set} test"
    return f"CLEAN EC level {level}, split{threshold}, fold {fold}{suffix}"


def _clean_test_description_suffix(test_set: CleanTestSet) -> str:
    return "" if test_set == "native" else f" with {test_set} as the test set"


def _clean_test_notes(
    test_set: CleanTestSet,
    *,
    threshold: CleanSplitThreshold,
    fold: CleanFold,
) -> str:
    if test_set == "native":
        return f"Loads test data from split{threshold}/split{threshold}_test_split_{fold}_curate.csv."
    return f"Loads test data from {test_set}.csv."


def _clean_tags(
    *,
    level: EcBenchmarkLevel,
    threshold: CleanSplitThreshold,
    fold: CleanFold,
    test_set: CleanTestSet,
) -> tuple[str, ...]:
    tags = (
        "protein",
        "function",
        "enzyme",
        "ec",
        "benchmark",
        "clean",
        f"ec_{level}",
        f"split{threshold}",
        f"fold{fold}",
    )
    return tags if test_set == "native" else (*tags, test_set)


def _parse_clean_dataset_name(
    name: str,
) -> tuple[EcBenchmarkLevel, CleanSplitThreshold, CleanFold, CleanTestSet]:
    normalized = str(name).strip().lower().replace("-", "_")
    if normalized.startswith("clean:"):
        normalized = normalized.partition(":")[2]
    parts = normalized.split("_")
    if len(parts) not in {4, 5} or parts[0] != "ec" or not parts[2].startswith("split"):
        raise EmbeddingInputError(
            "CLEAN dataset name must look like 'ec_1_split10_fold0' or "
            "'ec_1_split10_fold0_halogenase'."
        )
    level = _parse_level(parts[1], provider="CLEAN")
    threshold = _parse_threshold(parts[2].removeprefix("split"))
    fold = _parse_fold(parts[3].removeprefix("fold"))
    test_set = "native" if len(parts) == 4 else parts[4]
    if test_set not in _CLEAN_TEST_SETS:
        raise EmbeddingInputError("CLEAN test set must be one of: native, halogenase, price, new.")
    return level, threshold, fold, test_set


def _parse_level(value: str, *, provider: str) -> EcBenchmarkLevel:
    try:
        level = int(value)
    except ValueError as exc:
        raise EmbeddingInputError(f"{provider} EC level must be one of 1, 2, 3, or 4.") from exc
    if level not in _CLEAN_LEVELS:
        raise EmbeddingInputError(f"{provider} EC level must be one of 1, 2, 3, or 4.")
    return level


def _parse_threshold(value: str) -> CleanSplitThreshold:
    try:
        threshold = int(value)
    except ValueError as exc:
        raise EmbeddingInputError("CLEAN split threshold must be one of 10, 30, 50, 70, or 100.") from exc
    if threshold not in _CLEAN_SPLIT_THRESHOLDS:
        raise EmbeddingInputError("CLEAN split threshold must be one of 10, 30, 50, 70, or 100.")
    return threshold


def _parse_fold(value: str) -> CleanFold:
    try:
        fold = int(value)
    except ValueError as exc:
        raise EmbeddingInputError("CLEAN fold must be one of 0, 1, 2, 3, or 4.") from exc
    if fold not in _CLEAN_FOLDS:
        raise EmbeddingInputError("CLEAN fold must be one of 0, 1, 2, 3, or 4.")
    return fold


def _resolve_clean_root(root: Path) -> Path:
    if _has_clean_split_dir(root):
        return root
    nested = root / _CLEAN_NATIVE_ROOT_NAME
    if _has_clean_split_dir(nested):
        return nested
    nested = root / _CLEAN_ROOT_NAME / _CLEAN_NATIVE_ROOT_NAME
    if _has_clean_split_dir(nested):
        return nested
    raise EmbeddingInputError(
        f"No CLEAN dataset layout found under {root}. Expected split10/ or "
        f"{_CLEAN_NATIVE_ROOT_NAME}/split10/."
    )


def _has_clean_split_dir(root: Path) -> bool:
    return any((root / f"split{threshold}").exists() for threshold in _CLEAN_SPLIT_THRESHOLDS)


def _resolve_clean_splits(split: str | Sequence[str] | None) -> tuple[SplitName, ...]:
    if split is None:
        return ("train", "test")
    values = (split,) if isinstance(split, str) else tuple(split)
    resolved: list[SplitName] = []
    for value in values:
        normalized = str(value).strip().lower()
        if normalized == "valid":
            normalized = "val"
        if normalized == "val":
            continue
        if normalized not in {"train", "test"}:
            raise EmbeddingInputError("CLEAN split must be one of: train, test.")
        resolved.append(cast(SplitName, normalized))
    return tuple(resolved)


def _clean_split_path(
    root: Path,
    *,
    split_name: SplitName,
    threshold: CleanSplitThreshold,
    fold: CleanFold,
    test_set: CleanTestSet,
) -> Path:
    if split_name == "train":
        return root / f"split{threshold}" / f"split{threshold}_train_split_{fold}.csv"
    if test_set == "native":
        return root / f"split{threshold}" / f"split{threshold}_test_split_{fold}_curate.csv"
    return root / f"{test_set}.csv"


CLEAN_DATASETS: tuple[DatasetMetadata, ...] = tuple(
    DatasetMetadata(
        id=_clean_dataset_id(level=level, threshold=threshold, fold=fold, test_set=test_set),
        name=_clean_dataset_name(level=level, threshold=threshold, fold=fold, test_set=test_set),
        display_name=_clean_display_name(
            level=level,
            threshold=threshold,
            fold=fold,
            test_set=test_set,
        ),
        collection="clean",
        source="CLEAN",
        category="function_prediction",
        task_class="function",
        preferred_metric="fmax",
        description=(
            "Source: CLEAN EC benchmark. "
            "Class: function_prediction. "
            f"Split system: split{threshold} fold {fold}"
            f"{_clean_test_description_suffix(test_set)}."
        ),
        level="protein",
        objective="multilabel",
        target=f"ec_{level}",
        status="ready",
        metrics=(
            "fmax",
            "f1",
            "macro_f1",
            "weighted_precision",
            "weighted_recall",
            "weighted_f1",
            "average_precision",
        ),
        import_adapter="load_clean_dataset",
        loader="load_clean_dataset",
        tags=_clean_tags(level=level, threshold=threshold, fold=fold, test_set=test_set),
        notes=(
            f"Loads train data from split{threshold}/split{threshold}_train_split_{fold}.csv. "
            f"{_clean_test_notes(test_set, threshold=threshold, fold=fold)} "
            "EC labels are derived from semicolon-delimited exact EC annotations."
        ),
    )
    for threshold in _CLEAN_SPLIT_THRESHOLDS
    for fold in _CLEAN_FOLDS
    for level in _CLEAN_LEVELS
    for test_set in _CLEAN_TEST_SETS
)


__all__ = [
    "CLEAN_COLLECTION_METADATA",
    "CLEAN_DATASETS",
    "CleanFold",
    "CleanSplitThreshold",
    "CleanTestSet",
    "load_clean_dataset",
]
