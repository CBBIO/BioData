"""Load enzyme commission prediction datasets from generated JSONL splits."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import hashlib
import json
from pathlib import Path
from typing import Any, Literal, cast

from CBBIO.embeddings import EmbeddingInputError

from ..collection_types import CollectionMetadata, DatasetMetadata
from ..datasets import ProteinDataset, ProteinExample, SplitName


EcTrack = Literal["head", "main", "full"]
EcLevel = Literal[1, 2, 3, 4]
EcDatasetKind = Literal["multilabel", "single", "subclass_holdout"]

EC_COLLECTION_METADATA = CollectionMetadata(
    id="ec",
    display_name="Enzyme Commission",
    description="Protein-level enzyme commission prediction datasets.",
    tags=("protein", "function", "enzyme", "ec"),
)

_EC_TRACKS: tuple[EcTrack, ...] = ("head", "main", "full")
_EC_LEVELS: tuple[EcLevel, ...] = (1, 2, 3, 4)
_EC_SPLITS: tuple[SplitName, ...] = ("train", "val", "test")
_EC_MANIFEST_NAME = "ec_full_{track}.json"
_EC_DATASET_ROOT_NAME = "ec_main_head"
_EC_TASK_NAME = "ec_full"
_EC3_SUBCLASS_HOLDOUT_SALT = "ec3-subclass-holdout-v1"
_EC_STATS: Mapping[str, Mapping[str, int | float | tuple[int, int, int]]] = {
    "ec_1_head": {
        "split_counts": (28147, 3519, 3519),
        "sample_count": 35185,
        "class_count": 7,
        "support_min": 934,
        "support_median": 2835,
        "support_max": 14911,
        "label_assignments": 35897,
    },
    "ec_2_head": {
        "split_counts": (28147, 3519, 3519),
        "sample_count": 35185,
        "class_count": 45,
        "support_min": 53,
        "support_median": 349,
        "support_max": 7627,
        "label_assignments": 36160,
    },
    "ec_3_head": {
        "split_counts": (28147, 3519, 3519),
        "sample_count": 35185,
        "class_count": 99,
        "support_min": 53,
        "support_median": 182,
        "support_max": 2294,
        "label_assignments": 36305,
    },
    "ec_4_head": {
        "split_counts": (28147, 3519, 3519),
        "sample_count": 35185,
        "class_count": 335,
        "support_min": 50,
        "support_median": 84,
        "support_max": 1732,
        "label_assignments": 36946,
    },
    "ec_1_main": {
        "split_counts": (36323, 4541, 4541),
        "sample_count": 45405,
        "class_count": 7,
        "support_min": 1344,
        "support_median": 4168,
        "support_max": 18885,
        "label_assignments": 46494,
    },
    "ec_2_main": {
        "split_counts": (36323, 4541, 4541),
        "sample_count": 45405,
        "class_count": 57,
        "support_min": 21,
        "support_median": 313,
        "support_max": 8938,
        "label_assignments": 46816,
    },
    "ec_3_main": {
        "split_counts": (36323, 4541, 4541),
        "sample_count": 45405,
        "class_count": 136,
        "support_min": 21,
        "support_median": 150.0,
        "support_max": 2811,
        "label_assignments": 47106,
    },
    "ec_4_main": {
        "split_counts": (36323, 4541, 4541),
        "sample_count": 45405,
        "class_count": 704,
        "support_min": 20,
        "support_median": 47.0,
        "support_max": 1732,
        "label_assignments": 48215,
    },
    "ec_1_full": {
        "split_counts": (48493, 6015, 6112),
        "sample_count": 60620,
        "class_count": 7,
        "support_min": 1695,
        "support_median": 5202,
        "support_max": 23666,
        "label_assignments": 62101,
    },
    "ec_2_full": {
        "split_counts": (48493, 6015, 6112),
        "sample_count": 60620,
        "class_count": 75,
        "support_min": 1,
        "support_median": 269,
        "support_max": 10190,
        "label_assignments": 62646,
    },
    "ec_3_full": {
        "split_counts": (48493, 6015, 6112),
        "sample_count": 60620,
        "class_count": 265,
        "support_min": 1,
        "support_median": 58,
        "support_max": 3056,
        "label_assignments": 63059,
    },
    "ec_4_full": {
        "split_counts": (48493, 6015, 6112),
        "sample_count": 60620,
        "class_count": 5528,
        "support_min": 1,
        "support_median": 2.0,
        "support_max": 1732,
        "label_assignments": 66162,
    },
    "single_ec_1_head": {
        "split_counts": (27551, 3473, 3471),
        "sample_count": 34495,
        "class_count": 7,
        "support_min": 934,
        "support_median": 2670,
        "support_max": 14630,
        "label_assignments": 34495,
    },
    "single_ec_2_head": {
        "split_counts": (27335, 3472, 3469),
        "sample_count": 34276,
        "class_count": 45,
        "support_min": 50,
        "support_median": 348,
        "support_max": 7329,
        "label_assignments": 34276,
    },
    "single_ec_3_head": {
        "split_counts": (27256, 3462, 3461),
        "sample_count": 34179,
        "class_count": 99,
        "support_min": 36,
        "support_median": 156,
        "support_max": 2029,
        "label_assignments": 34179,
    },
    "single_ec_4_head": {
        "split_counts": (26805, 3412, 3406),
        "sample_count": 33623,
        "class_count": 331,
        "support_min": 1,
        "support_median": 78,
        "support_max": 1697,
        "label_assignments": 33623,
    },
    "single_ec_1_main": {
        "split_counts": (35451, 4473, 4478),
        "sample_count": 44402,
        "class_count": 7,
        "support_min": 1344,
        "support_median": 4109,
        "support_max": 18375,
        "label_assignments": 44402,
    },
    "single_ec_2_main": {
        "split_counts": (35194, 4467, 4473),
        "sample_count": 44134,
        "class_count": 57,
        "support_min": 5,
        "support_median": 297,
        "support_max": 8535,
        "label_assignments": 44134,
    },
    "single_ec_3_main": {
        "split_counts": (34973, 4458, 4463),
        "sample_count": 43894,
        "class_count": 136,
        "support_min": 5,
        "support_median": 134.5,
        "support_max": 2425,
        "label_assignments": 43894,
    },
    "single_ec_4_main": {
        "split_counts": (34213, 4368, 4376),
        "sample_count": 42957,
        "class_count": 685,
        "support_min": 1,
        "support_median": 44,
        "support_max": 1674,
        "label_assignments": 42957,
    },
    "single_ec_1_full": {
        "split_counts": (47405, 5879, 5956),
        "sample_count": 59240,
        "class_count": 7,
        "support_min": 1695,
        "support_median": 4652,
        "support_max": 22964,
        "label_assignments": 59240,
    },
    "single_ec_2_full": {
        "split_counts": (47059, 5829, 5920),
        "sample_count": 58808,
        "class_count": 74,
        "support_min": 1,
        "support_median": 279.5,
        "support_max": 9698,
        "label_assignments": 58808,
    },
    "single_ec_3_full": {
        "split_counts": (46812, 5777, 5890),
        "sample_count": 58479,
        "class_count": 264,
        "support_min": 1,
        "support_median": 54.0,
        "support_max": 2887,
        "label_assignments": 58479,
    },
    "single_ec_4_full": {
        "split_counts": (44971, 5548, 5646),
        "sample_count": 56165,
        "class_count": 4871,
        "support_min": 1,
        "support_median": 2,
        "support_max": 1664,
        "label_assignments": 56165,
    },
    "ec_1_subclass_holdout_head": {
        "split_counts": (26968, 3809, 3499),
        "sample_count": 34276,
        "class_count": 7,
        "support_min": 934,
        "support_median": 2668,
        "support_max": 14542,
        "label_assignments": 34276,
    },
    "ec_1_subclass_holdout_main": {
        "split_counts": (34592, 5057, 4485),
        "sample_count": 44134,
        "class_count": 7,
        "support_min": 1344,
        "support_median": 4109,
        "support_max": 18267,
        "label_assignments": 44134,
    },
    "ec_2_subclass_holdout_head": {
        "split_counts": (21406, 2833, 2854),
        "sample_count": 27093,
        "class_count": 23,
        "support_min": 140,
        "support_median": 755,
        "support_max": 7283,
        "label_assignments": 27093,
    },
    "ec_2_subclass_holdout_main": {
        "split_counts": (29773, 3771, 3738),
        "sample_count": 37282,
        "class_count": 30,
        "support_min": 78,
        "support_median": 663.5,
        "support_max": 8388,
        "label_assignments": 37282,
    },
    "ec_3_subclass_holdout_head": {
        "split_counts": (23753, 3015, 2993),
        "sample_count": 29761,
        "class_count": 61,
        "support_min": 27,
        "support_median": 304,
        "support_max": 2017,
        "label_assignments": 29761,
    },
    "ec_3_subclass_holdout_main": {
        "split_counts": (32143, 4040, 4021),
        "sample_count": 40204,
        "class_count": 95,
        "support_min": 43,
        "support_median": 217,
        "support_max": 2349,
        "label_assignments": 40204,
    },
    "ec_1_subclass_holdout_full": {
        "split_counts": (46501, 5995, 6312),
        "sample_count": 58808,
        "class_count": 7,
        "support_min": 1693,
        "support_median": 4642,
        "support_max": 22804,
        "label_assignments": 58808,
    },
    "ec_2_subclass_holdout_full": {
        "split_counts": (42572, 5449, 5339),
        "sample_count": 53360,
        "class_count": 43,
        "support_min": 7,
        "support_median": 650,
        "support_max": 9516,
        "label_assignments": 53360,
    },
    "ec_3_subclass_holdout_full": {
        "split_counts": (44304, 5542, 5543),
        "sample_count": 55389,
        "class_count": 205,
        "support_min": 2,
        "support_median": 87,
        "support_max": 2736,
        "label_assignments": 55389,
    },
}


_EC_MULTILABEL_DATASETS: tuple[DatasetMetadata, ...] = tuple(
    DatasetMetadata(
        id=f"ec:ec_{level}_{track}",
        name=f"ec_{level}_{track}",
        display_name=f"EC level {level} ({track})",
        collection="ec",
        source="UniRef50 EC annotations",
        category="function_prediction",
        task_class="function",
        preferred_metric="fmax",
        description=(
            "Source: UniRef50 EC annotations. "
            "Class: function_prediction. "
            "Split system: fixed train/validation/test splits stratified by species."
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
        split_counts=cast(tuple[int, int, int], _EC_STATS[f"ec_{level}_{track}"]["split_counts"]),
        sample_count=cast(int, _EC_STATS[f"ec_{level}_{track}"]["sample_count"]),
        download_adapter=None,
        import_adapter="load_ec_dataset",
        loader="load_ec_dataset",
        tags=("protein", "function", "enzyme", "ec", f"ec_{level}", track),
        notes=(
            "Loads from ec_main_head/tasks/ec_full/{head,main}/{train,val,test}.jsonl. "
            "Level 4 uses exact EC labels; levels 1-3 use derived metadata labels. "
            f"Classes: {_EC_STATS[f'ec_{level}_{track}']['class_count']}; "
            f"support min/median/max: {_EC_STATS[f'ec_{level}_{track}']['support_min']}/"
            f"{_EC_STATS[f'ec_{level}_{track}']['support_median']}/"
            f"{_EC_STATS[f'ec_{level}_{track}']['support_max']}; "
            f"label assignments: {_EC_STATS[f'ec_{level}_{track}']['label_assignments']}. "
            "The local manifests do not declare an authoritative citation."
        ),
    )
    for track in _EC_TRACKS
    for level in _EC_LEVELS
)

_EC_SINGLE_DATASETS: tuple[DatasetMetadata, ...] = tuple(
    DatasetMetadata(
        id=f"ec:single_ec_{level}_{track}",
        name=f"single_ec_{level}_{track}",
        display_name=f"Single-label EC level {level} ({track})",
        collection="ec",
        source="UniRef50 EC annotations",
        category="function_prediction",
        task_class="function",
        preferred_metric="accuracy",
        description=(
            "Source: UniRef50 EC annotations. "
            "Class: function_prediction. "
            "Split system: fixed train/validation/test splits stratified by species."
        ),
        level="protein",
        objective="multiclass",
        target=f"single_ec_{level}",
        status="ready",
        metrics=("accuracy", "macro_f1", "weighted_f1"),
        split_counts=cast(tuple[int, int, int], _EC_STATS[f"single_ec_{level}_{track}"]["split_counts"]),
        sample_count=cast(int, _EC_STATS[f"single_ec_{level}_{track}"]["sample_count"]),
        download_adapter=None,
        import_adapter="load_ec_dataset",
        loader="load_ec_dataset",
        tags=("protein", "function", "enzyme", "ec", "single_label", f"ec_{level}", track),
        notes=(
            "Loads from ec_main_head/tasks/ec_full/{head,main}/{train,val,test}.jsonl "
            "and keeps only proteins with exactly one label at this EC level. "
            f"Classes: {_EC_STATS[f'single_ec_{level}_{track}']['class_count']}; "
            f"support min/median/max: {_EC_STATS[f'single_ec_{level}_{track}']['support_min']}/"
            f"{_EC_STATS[f'single_ec_{level}_{track}']['support_median']}/"
            f"{_EC_STATS[f'single_ec_{level}_{track}']['support_max']}. "
            "The local manifests do not declare an authoritative citation."
        ),
    )
    for track in _EC_TRACKS
    for level in _EC_LEVELS
)

_EC_SUBCLASS_HOLDOUT_DATASETS: tuple[DatasetMetadata, ...] = tuple(
    DatasetMetadata(
        id=f"ec:ec_{level}_subclass_holdout_{track}",
        name=f"ec_{level}_subclass_holdout_{track}",
        display_name=f"EC level {level} subclass holdout ({track})",
        collection="ec",
        source="UniRef50 EC annotations",
        category="function_prediction",
        task_class="function",
        preferred_metric="accuracy",
        description=(
            "Source: UniRef50 EC annotations. "
            "Class: function_prediction. "
            f"Split system: rare EC level {level + 1} subclasses held out within EC level "
            f"{level} parents until global validation/test example targets are reached."
        ),
        level="protein",
        objective="multiclass",
        target=f"ec_{level}",
        status="ready",
        metrics=("accuracy", "macro_f1", "weighted_f1"),
        split_counts=cast(tuple[int, int, int], _EC_STATS[f"ec_{level}_subclass_holdout_{track}"]["split_counts"]),
        sample_count=cast(int, _EC_STATS[f"ec_{level}_subclass_holdout_{track}"]["sample_count"]),
        download_adapter=None,
        import_adapter="load_ec_dataset",
        loader="load_ec_dataset",
        tags=("protein", "function", "enzyme", "ec", f"ec_{level}", "subclass_holdout", track),
        notes=(
            f"Predicts EC level {level} labels while assigning rare whole EC level {level + 1} "
            f"subclasses to test/validation until each split reaches about 10% of all "
            "eligible examples. "
            f"Classes: {_EC_STATS[f'ec_{level}_subclass_holdout_{track}']['class_count']}; "
            f"support min/median/max: {_EC_STATS[f'ec_{level}_subclass_holdout_{track}']['support_min']}/"
            f"{_EC_STATS[f'ec_{level}_subclass_holdout_{track}']['support_median']}/"
            f"{_EC_STATS[f'ec_{level}_subclass_holdout_{track}']['support_max']}. "
            "Only examples with one parent label and one subclass label are retained."
        ),
    )
    for track in _EC_TRACKS
    for level in (1, 2, 3)
)

EC_DATASETS: tuple[DatasetMetadata, ...] = (
    *_EC_MULTILABEL_DATASETS,
    *_EC_SINGLE_DATASETS,
    *_EC_SUBCLASS_HOLDOUT_DATASETS,
)


def load_ec_dataset(
    root: str | Path,
    *,
    name: str,
    split: str | Sequence[str] | None = None,
    target: str | None = None,
) -> ProteinDataset:
    """Load one generated EC prediction dataset."""
    kind, level, track = _parse_ec_dataset_name(name)
    expected_target = f"single_ec_{level}" if kind == "single" else f"ec_{level}"
    resolved_target = target or expected_target
    if resolved_target != expected_target:
        raise EmbeddingInputError(
            f"EC dataset {name!r} only supports target {expected_target!r}."
        )
    root_path = _resolve_ec_root(Path(root).expanduser())
    split_names = _resolve_splits(split)
    if kind == "subclass_holdout":
        return ProteinDataset(
            _load_ec_subclass_holdout_jsonl(
                root_path,
                level=level,
                track=track,
                target=resolved_target,
                requested_splits=split_names,
            )
        )
    examples: list[ProteinExample] = []
    for split_name in split_names:
        path = root_path / "tasks" / _EC_TASK_NAME / track / f"{split_name}.jsonl"
        examples.extend(
            _load_ec_jsonl(
                path,
                kind=kind,
                level=level,
                target=resolved_target,
                split=split_name,
            )
        )
    return ProteinDataset(examples)


def read_ec_manifest(root: str | Path, *, track: EcTrack) -> Mapping[str, Any]:
    """Read one generated EC dataset manifest."""
    root_path = _resolve_ec_root(Path(root).expanduser())
    path = root_path / "manifests" / _EC_MANIFEST_NAME.format(track=track)
    if not path.exists():
        raise EmbeddingInputError(f"EC manifest file not found: {path}.")
    with path.open("r", encoding="utf-8") as handle:
        loaded = json.load(handle)
    if not isinstance(loaded, dict):
        raise EmbeddingInputError(f"EC manifest file {path} must contain a JSON object.")
    return cast(dict[str, Any], loaded)


def _parse_ec_dataset_name(name: str) -> tuple[EcDatasetKind, EcLevel, EcTrack]:
    normalized = str(name).strip().lower().replace("-", "_")
    if normalized.startswith("ec:"):
        normalized = normalized.partition(":")[2]
    parts = normalized.split("_")
    if len(parts) == 3 and parts[0] == "ec":
        kind: EcDatasetKind = "multilabel"
        level_text = parts[1]
        track = parts[2]
    elif len(parts) == 4 and parts[0] == "single" and parts[1] == "ec":
        kind = "single"
        level_text = parts[2]
        track = parts[3]
    elif len(parts) == 5 and parts[0] == "ec" and parts[2:4] == ["subclass", "holdout"]:
        kind = "subclass_holdout"
        level_text = parts[1]
        track = parts[4]
    else:
        raise EmbeddingInputError(
            "EC dataset name must look like 'ec_1_head', 'ec_4_main', "
            "'single_ec_1_head', 'single_ec_4_main', or "
            "'ec_3_subclass_holdout_head'."
        )
    try:
        level = int(level_text)
    except ValueError as exc:
        raise EmbeddingInputError("EC dataset level must be one of 1, 2, 3, or 4.") from exc
    if level not in _EC_LEVELS:
        raise EmbeddingInputError("EC dataset level must be one of 1, 2, 3, or 4.")
    if kind == "subclass_holdout" and level == 4:
        raise EmbeddingInputError("EC subclass holdout datasets support levels 1, 2, and 3.")
    if track not in _EC_TRACKS:
        raise EmbeddingInputError("EC dataset track must be 'head', 'main', or 'full'.")
    return kind, level, track


def _resolve_ec_root(root: Path) -> Path:
    if (root / "tasks" / _EC_TASK_NAME).exists():
        return root
    nested = root / _EC_DATASET_ROOT_NAME
    if (nested / "tasks" / _EC_TASK_NAME).exists():
        return nested
    raise EmbeddingInputError(
        f"No EC dataset layout found under {root}. Expected tasks/{_EC_TASK_NAME} "
        f"or {_EC_DATASET_ROOT_NAME}/tasks/{_EC_TASK_NAME}."
    )


def _resolve_splits(split: str | Sequence[str] | None) -> tuple[SplitName, ...]:
    if split is None:
        return _EC_SPLITS
    values = (split,) if isinstance(split, str) else tuple(split)
    resolved: list[SplitName] = []
    for value in values:
        normalized = str(value).strip().lower()
        if normalized == "valid":
            normalized = "val"
        if normalized not in _EC_SPLITS:
            raise EmbeddingInputError("EC split must be one of: train, val, test.")
        resolved.append(normalized)
    return tuple(resolved)


def _load_ec_jsonl(
    path: Path,
    *,
    kind: EcDatasetKind,
    level: EcLevel,
    target: str,
    split: SplitName,
) -> list[ProteinExample]:
    if not path.exists():
        raise EmbeddingInputError(f"EC split file not found: {path}.")
    examples: list[ProteinExample] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            text = line.strip()
            if not text:
                continue
            record = _parse_ec_record(text, path=path, line_number=line_number)
            example = _ec_record_to_example(
                record,
                kind=kind,
                level=level,
                target=target,
                split=split,
            )
            if example is not None:
                examples.append(example)
    return examples


def _load_ec_subclass_holdout_jsonl(
    root: Path,
    *,
    level: EcLevel,
    track: EcTrack,
    target: str,
    requested_splits: Sequence[SplitName],
) -> list[ProteinExample]:
    records: list[tuple[Mapping[str, Any], SplitName, str, str]] = []
    parent_to_exact: dict[str, dict[str, int]] = {}
    child_level = level + 1
    for original_split in _EC_SPLITS:
        path = root / "tasks" / _EC_TASK_NAME / track / f"{original_split}.jsonl"
        if not path.exists():
            raise EmbeddingInputError(f"EC split file not found: {path}.")
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                text = line.strip()
                if not text:
                    continue
                record = _parse_ec_record(text, path=path, line_number=line_number)
                parent_labels = _ec_labels(record, level=level)
                child_labels = _ec_labels(record, level=cast(EcLevel, child_level))
                if len(parent_labels) != 1 or len(child_labels) != 1:
                    continue
                parent = parent_labels[0]
                exact = child_labels[0]
                records.append((record, original_split, parent, exact))
                parent_to_exact.setdefault(parent, {})
                parent_to_exact[parent][exact] = parent_to_exact[parent].get(exact, 0) + 1

    exact_to_split = _subclass_holdout_splits(parent_to_exact)

    requested = set(requested_splits)
    examples: list[ProteinExample] = []
    for record, original_split, parent, exact in records:
        split = exact_to_split.get((parent, exact))
        if split is None or split not in requested:
            continue
        record_id = str(record.get("id", "")).strip()
        sequence = str(record.get("sequence", "")).strip()
        metadata = _ec_metadata(record)
        metadata["original_split"] = original_split
        metadata["heldout_subclass"] = exact
        if level == 3:
            metadata["heldout_exact_ec"] = exact
        examples.append(
            ProteinExample(
                id=record_id,
                sequence=sequence,
                labels={target: parent},
                split=split,
                metadata=metadata,
            )
        )
    return examples


def _parse_ec_record(text: str, *, path: Path, line_number: int) -> Mapping[str, Any]:
    try:
        loaded = json.loads(text)
    except json.JSONDecodeError as exc:
        raise EmbeddingInputError(
            f"Invalid JSON in EC split file {path} at line {line_number}."
        ) from exc
    if not isinstance(loaded, dict):
        raise EmbeddingInputError(f"EC split file {path} line {line_number} must be a JSON object.")
    return cast(dict[str, Any], loaded)


def _ec_record_to_example(
    record: Mapping[str, Any],
    *,
    kind: EcDatasetKind,
    level: EcLevel,
    target: str,
    split: SplitName,
) -> ProteinExample | None:
    record_id = str(record.get("id", "")).strip()
    sequence = str(record.get("sequence", "")).strip()
    labels = _ec_labels(record, level=level)
    if kind == "single" and len(labels) != 1:
        return None
    value: list[str] | str = labels[0] if kind == "single" else labels
    metadata = _ec_metadata(record)
    return ProteinExample(
        id=record_id,
        sequence=sequence,
        labels={target: value},
        split=split,
        metadata=metadata,
    )


def _ec_labels(record: Mapping[str, Any], *, level: EcLevel) -> list[str]:
    field = "labels_exact" if level == 4 else f"ec_level{level}"
    source: object = record.get(field)
    if level != 4:
        raw_metadata = record.get("metadata")
        if isinstance(raw_metadata, dict):
            metadata = cast(Mapping[str, Any], raw_metadata)
            source = metadata.get(field)
    if not isinstance(source, list) or not source:
        raise EmbeddingInputError(f"EC record is missing non-empty label field {field!r}.")
    raw_labels = cast(list[object], source)
    labels = [str(label).strip() for label in raw_labels if str(label).strip()]
    if not labels:
        raise EmbeddingInputError(f"EC record label field {field!r} contains no labels.")
    return sorted(set(labels))


def _ec_metadata(record: Mapping[str, Any]) -> dict[str, Any]:
    metadata: dict[str, Any] = {}
    raw_metadata = record.get("metadata")
    if isinstance(raw_metadata, dict):
        metadata.update(cast(dict[str, Any], raw_metadata))
    for field in ("cluster_id", "rep_accession"):
        if field in record:
            metadata[field] = record[field]
    return metadata


def _subclass_holdout_splits(
    parent_to_exact: Mapping[str, Mapping[str, int]],
) -> dict[tuple[str, str], SplitName]:
    assignments: dict[tuple[str, str], SplitName] = {}
    parent_train_children: dict[str, int] = {}
    eligible: list[tuple[str, str, int]] = []
    for parent, exact_counts in parent_to_exact.items():
        if len(exact_counts) < 2:
            continue
        parent_train_children[parent] = len(exact_counts)
        for exact, count in exact_counts.items():
            assignments[(parent, exact)] = "train"
            eligible.append((parent, exact, count))

    total = sum(count for _, _, count in eligible)
    if not eligible or total < 2:
        return assignments

    ordered = sorted(
        eligible,
        key=lambda item: (
            item[2],
            _stable_ec_holdout_hash(f"{item[0]}:{item[1]}"),
        ),
    )
    test_target = max(1, round(total * 0.1))
    validation_target = max(1, round(total * 0.1)) if len(ordered) >= 3 else 0

    cursor = _assign_subclass_holdout_split(
        ordered,
        assignments=assignments,
        parent_train_children=parent_train_children,
        split="test",
        target_examples=test_target,
        start=0,
    )
    _assign_subclass_holdout_split(
        ordered,
        assignments=assignments,
        parent_train_children=parent_train_children,
        split="val",
        target_examples=validation_target,
        start=cursor,
    )

    return assignments


def _assign_subclass_holdout_split(
    ordered: Sequence[tuple[str, str, int]],
    *,
    assignments: dict[tuple[str, str], SplitName],
    parent_train_children: dict[str, int],
    split: SplitName,
    target_examples: int,
    start: int,
) -> int:
    if target_examples < 1:
        return start

    examples = 0
    cursor = start
    while cursor < len(ordered) and examples < target_examples:
        parent, exact, count = ordered[cursor]
        cursor += 1
        if assignments[(parent, exact)] != "train":
            continue
        if parent_train_children[parent] <= 1:
            continue
        assignments[(parent, exact)] = split
        parent_train_children[parent] -= 1
        examples += count
    return cursor


def _stable_ec_holdout_hash(value: str) -> str:
    return hashlib.sha256(f"{_EC3_SUBCLASS_HOLDOUT_SALT}:{value}".encode("utf-8")).hexdigest()


__all__ = [
    "EC_COLLECTION_METADATA",
    "EC_DATASETS",
    "EcDatasetKind",
    "EcLevel",
    "EcTrack",
    "load_ec_dataset",
    "read_ec_manifest",
]
