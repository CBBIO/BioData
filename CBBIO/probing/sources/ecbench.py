"""Load EC-Bench enzyme commission benchmark datasets."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Literal, cast

from CBBIO.embeddings import EmbeddingInputError

from ..collection_types import CollectionMetadata, DatasetMetadata
from ..datasets import ProteinDataset, ProteinExample, SplitName
from ._ec_benchmarks import EcBenchmarkLevel, read_ec_benchmark_rows


EcBenchTrainSize = Literal[30, 100]
EcBenchTestSet = Literal["native", "halogenase", "price", "new"]

ECBENCH_COLLECTION_METADATA = CollectionMetadata(
    id="ecbench",
    display_name="EC-Bench",
    description="EC-Bench enzyme commission prediction benchmark splits.",
    tags=("protein", "function", "enzyme", "ec", "benchmark", "ecbench"),
)

_ECBENCH_LEVELS: tuple[EcBenchmarkLevel, ...] = (1, 2, 3, 4)
_ECBENCH_TRAIN_SIZES: tuple[EcBenchTrainSize, ...] = (30, 100)
_ECBENCH_TEST_SETS: tuple[EcBenchTestSet, ...] = ("native", "halogenase", "price", "new")
_ECBENCH_ROOT_NAME = "ec-benchmark"


def load_ecbench_dataset(
    root: str | Path,
    *,
    name: str,
    split: str | Sequence[str] | None = None,
    target: str | None = None,
) -> ProteinDataset:
    """Load one EC-Bench EC benchmark dataset."""
    level, train_size, test_set = _parse_ecbench_dataset_name(name)
    expected_target = f"ec_{level}"
    resolved_target = target or expected_target
    if resolved_target != expected_target:
        raise EmbeddingInputError(
            f"EC-Bench dataset {name!r} only supports target {expected_target!r}."
        )
    root_path = _resolve_ecbench_root(Path(root).expanduser())
    split_names = _resolve_ecbench_splits(split)
    examples: list[ProteinExample] = []
    for split_name in split_names:
        path = _ecbench_split_path(root_path, split_name=split_name, train_size=train_size, test_set=test_set)
        examples.extend(
            read_ec_benchmark_rows(
                path,
                split=split_name,
                target=resolved_target,
                level=level,
                source="EC-Bench",
                metadata={
                    "collection": "ecbench",
                    "train_size": train_size,
                    "test_set": test_set,
                },
            )
        )
    return ProteinDataset(examples)


def _ecbench_dataset_id(
    *,
    level: EcBenchmarkLevel,
    train_size: EcBenchTrainSize,
    test_set: EcBenchTestSet,
) -> str:
    return f"ecbench:{_ecbench_dataset_name(level=level, train_size=train_size, test_set=test_set)}"


def _ecbench_dataset_name(
    *,
    level: EcBenchmarkLevel,
    train_size: EcBenchTrainSize,
    test_set: EcBenchTestSet,
) -> str:
    base = f"ec_{level}_train_{train_size}"
    return base if test_set == "native" else f"{base}_{test_set}"


def _ecbench_display_name(
    *,
    level: EcBenchmarkLevel,
    train_size: EcBenchTrainSize,
    test_set: EcBenchTestSet,
) -> str:
    suffix = "" if test_set == "native" else f", {test_set} test"
    return f"EC-Bench EC level {level}, train {train_size}{suffix}"


def _ecbench_test_description_suffix(test_set: EcBenchTestSet) -> str:
    return "" if test_set == "native" else f" with {test_set} as the test set"


def _ecbench_test_notes(test_set: EcBenchTestSet) -> str:
    return "Loads test data from test_ec.csv." if test_set == "native" else f"Loads test data from {test_set}.csv."


def _ecbench_tags(
    *,
    level: EcBenchmarkLevel,
    train_size: EcBenchTrainSize,
    test_set: EcBenchTestSet,
) -> tuple[str, ...]:
    tags = (
        "protein",
        "function",
        "enzyme",
        "ec",
        "benchmark",
        "ecbench",
        f"ec_{level}",
        f"train_{train_size}",
    )
    return tags if test_set == "native" else (*tags, test_set)


def _parse_ecbench_dataset_name(
    name: str,
) -> tuple[EcBenchmarkLevel, EcBenchTrainSize, EcBenchTestSet]:
    normalized = str(name).strip().lower().replace("-", "_")
    if normalized.startswith("ecbench:"):
        normalized = normalized.partition(":")[2]
    normalized = _normalize_ecbench_train_alias(normalized)
    parts = normalized.split("_")
    if len(parts) not in {4, 5} or parts[0] != "ec" or parts[2] != "train":
        raise EmbeddingInputError(
            "EC-Bench dataset name must look like 'ec_1_train_30' or "
            "'ec_1_train_30_halogenase'."
        )
    level = _parse_level(parts[1])
    train_size = _parse_train_size(parts[3])
    test_set = "native" if len(parts) == 4 else parts[4]
    if test_set not in _ECBENCH_TEST_SETS:
        raise EmbeddingInputError("EC-Bench test set must be one of: native, halogenase, price, new.")
    return level, train_size, test_set


def _normalize_ecbench_train_alias(value: str) -> str:
    for train_size in _ECBENCH_TRAIN_SIZES:
        value = value.replace(f"train{train_size}", f"train_{train_size}")
    if value.startswith("ecbench_"):
        value = value.removeprefix("ecbench_")
    if value.startswith("ec") and not value.startswith("ec_"):
        value = value.replace("ec", "ec_", 1)
    return value


def _parse_level(value: str) -> EcBenchmarkLevel:
    try:
        level = int(value)
    except ValueError as exc:
        raise EmbeddingInputError("EC-Bench EC level must be one of 1, 2, 3, or 4.") from exc
    if level not in _ECBENCH_LEVELS:
        raise EmbeddingInputError("EC-Bench EC level must be one of 1, 2, 3, or 4.")
    return level


def _parse_train_size(value: str) -> EcBenchTrainSize:
    try:
        train_size = int(value)
    except ValueError as exc:
        raise EmbeddingInputError("EC-Bench train size must be one of 30 or 100.") from exc
    if train_size not in _ECBENCH_TRAIN_SIZES:
        raise EmbeddingInputError("EC-Bench train size must be one of 30 or 100.")
    return train_size


def _resolve_ecbench_root(root: Path) -> Path:
    if (root / "train_30.csv").exists():
        return root
    nested = root / _ECBENCH_ROOT_NAME
    if (nested / "train_30.csv").exists():
        return nested
    raise EmbeddingInputError(
        f"No EC-Bench dataset layout found under {root}. Expected train_30.csv "
        f"or {_ECBENCH_ROOT_NAME}/train_30.csv."
    )


def _resolve_ecbench_splits(split: str | Sequence[str] | None) -> tuple[SplitName, ...]:
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
            raise EmbeddingInputError("EC-Bench split must be one of: train, test.")
        resolved.append(cast(SplitName, normalized))
    return tuple(resolved)


def _ecbench_split_path(
    root: Path,
    *,
    split_name: SplitName,
    train_size: EcBenchTrainSize,
    test_set: EcBenchTestSet,
) -> Path:
    if split_name == "train":
        return root / f"train_{train_size}.csv"
    return root / "test_ec.csv" if test_set == "native" else root / f"{test_set}.csv"


ECBENCH_DATASETS: tuple[DatasetMetadata, ...] = tuple(
    DatasetMetadata(
        id=_ecbench_dataset_id(level=level, train_size=train_size, test_set=test_set),
        name=_ecbench_dataset_name(level=level, train_size=train_size, test_set=test_set),
        display_name=_ecbench_display_name(
            level=level,
            train_size=train_size,
            test_set=test_set,
        ),
        collection="ecbench",
        source="EC-Bench",
        category="function_prediction",
        task_class="function",
        preferred_metric="fmax",
        description=(
            "Source: EC-Bench. "
            "Class: function_prediction. "
            f"Split system: train_{train_size}.csv"
            f"{_ecbench_test_description_suffix(test_set)}."
        ),
        level="protein",
        objective="multilabel",
        target=f"ec_{level}",
        status="ready",
        metrics=("fmax", "f1", "macro_f1", "weighted_f1", "average_precision"),
        import_adapter="load_ecbench_dataset",
        loader="load_ecbench_dataset",
        tags=_ecbench_tags(level=level, train_size=train_size, test_set=test_set),
        notes=(
            f"Loads train data from train_{train_size}.csv. "
            f"{_ecbench_test_notes(test_set)} "
            "EC labels are derived from semicolon-delimited exact EC annotations."
        ),
    )
    for train_size in _ECBENCH_TRAIN_SIZES
    for level in _ECBENCH_LEVELS
    for test_set in _ECBENCH_TEST_SETS
)


__all__ = [
    "ECBENCH_COLLECTION_METADATA",
    "ECBENCH_DATASETS",
    "EcBenchTestSet",
    "EcBenchTrainSize",
    "load_ecbench_dataset",
]
