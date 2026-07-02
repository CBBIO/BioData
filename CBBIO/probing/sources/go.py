"""Load Gene Ontology prediction datasets from generated JSONL splits."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import json
from pathlib import Path
from typing import Any, Literal, cast

from CBBIO.embeddings import EmbeddingInputError

from ..collection_types import CollectionMetadata, DatasetMetadata
from ..datasets import ProteinDataset, ProteinExample, SplitName


GoAspect = Literal["bp", "cc", "mf"]
GoTrack = Literal["head", "main", "full"]

GO_COLLECTION_METADATA = CollectionMetadata(
    id="go",
    display_name="Gene Ontology",
    description="Protein-level Gene Ontology prediction datasets.",
    homepage="https://geneontology.org/",
    tags=("protein", "function", "gene_ontology", "go"),
)

_GO_ASPECTS: tuple[GoAspect, ...] = ("bp", "cc", "mf")
_GO_TRACKS: tuple[GoTrack, ...] = ("head", "main", "full")
_GO_SPLITS: tuple[SplitName, ...] = ("train", "val", "test")
_GO_ASPECT_NAMES: Mapping[GoAspect, str] = {
    "bp": "biological process",
    "cc": "cellular component",
    "mf": "molecular function",
}
_GO_DATASET_ROOT_NAME = "go_main_head"
_GO_STATS: Mapping[str, Mapping[str, int | float | tuple[int, int, int]]] = {
    "go_bp_head": {
        "split_counts": (20948, 2620, 2619),
        "sample_count": 26187,
        "class_count": 249,
        "support_min": 92,
        "support_median": 147,
        "support_max": 1876,
    },
    "go_bp_main": {
        "split_counts": (31310, 3915, 3914),
        "sample_count": 39139,
        "class_count": 1402,
        "support_min": 25,
        "support_median": 50.0,
        "support_max": 1876,
    },
    "go_bp_full": {
        "split_counts": (40782, 5038, 5127),
        "sample_count": 50947,
        "class_count": 15730,
        "support_min": 1,
        "support_median": 4.0,
        "support_max": 1876,
    },
    "go_cc_head": {
        "split_counts": (42708, 5340, 5339),
        "sample_count": 53387,
        "class_count": 161,
        "support_min": 99,
        "support_median": 228,
        "support_max": 13752,
    },
    "go_cc_main": {
        "split_counts": (44261, 5534, 5533),
        "sample_count": 55328,
        "class_count": 434,
        "support_min": 29,
        "support_median": 72.0,
        "support_max": 13752,
    },
    "go_cc_full": {
        "split_counts": (45755, 5734, 5802),
        "sample_count": 57291,
        "class_count": 2693,
        "support_min": 1,
        "support_median": 5,
        "support_max": 13752,
    },
    "go_mf_head": {
        "split_counts": (29819, 3728, 3728),
        "sample_count": 37275,
        "class_count": 100,
        "support_min": 92,
        "support_median": 207.5,
        "support_max": 28624,
    },
    "go_mf_main": {
        "split_counts": (32863, 4109, 4109),
        "sample_count": 41081,
        "class_count": 407,
        "support_min": 27,
        "support_median": 53,
        "support_max": 28624,
    },
    "go_mf_full": {
        "split_counts": (40395, 4994, 5057),
        "sample_count": 50446,
        "class_count": 6876,
        "support_min": 1,
        "support_median": 3.0,
        "support_max": 28624,
    },
}


GO_DATASETS: tuple[DatasetMetadata, ...] = tuple(
    DatasetMetadata(
        id=f"go:go_{aspect}_{track}",
        name=f"go_{aspect}_{track}",
        display_name=f"GO {_GO_ASPECT_NAMES[aspect]} ({track})",
        collection="go",
        source="Gene Ontology annotations",
        category="function_prediction",
        task_class="function",
        preferred_metric="go_fmax",
        description=(
            "Source: Gene Ontology annotations. "
            "Class: function_prediction. "
            "Split system: fixed train/validation/test splits stratified by species."
        ),
        level="protein",
        objective="multilabel",
        target=f"go_{aspect}",
        status="ready",
        metrics=("go_fmax", "go_precision_at_fmax", "go_recall_at_fmax"),
        split_counts=cast(tuple[int, int, int], _GO_STATS[f"go_{aspect}_{track}"]["split_counts"]),
        sample_count=cast(int, _GO_STATS[f"go_{aspect}_{track}"]["sample_count"]),
        homepage="https://geneontology.org/",
        download_adapter=None,
        import_adapter="load_go_dataset",
        loader="load_go_dataset",
        tags=("protein", "function", "gene_ontology", "go", aspect, track),
        notes=(
        f"Loads asserted GO labels from go_main_head/tasks/go_{aspect}/"
        f"{track}/{{train,val,test}}.jsonl. "
            f"Aspect: {_GO_ASPECT_NAMES[aspect]}. "
            f"Classes: {_GO_STATS[f'go_{aspect}_{track}']['class_count']}; "
            f"support min/median/max: {_GO_STATS[f'go_{aspect}_{track}']['support_min']}/"
            f"{_GO_STATS[f'go_{aspect}_{track}']['support_median']}/"
            f"{_GO_STATS[f'go_{aspect}_{track}']['support_max']}."
        ),
    )
    for aspect in _GO_ASPECTS
    for track in _GO_TRACKS
)


def load_go_dataset(
    root: str | Path,
    *,
    name: str,
    split: str | Sequence[str] | None = None,
    target: str | None = None,
) -> ProteinDataset:
    """Load one generated Gene Ontology prediction dataset."""
    aspect, track = _parse_go_dataset_name(name)
    expected_target = f"go_{aspect}"
    resolved_target = target or expected_target
    if resolved_target != expected_target:
        raise EmbeddingInputError(
            f"GO dataset {name!r} only supports target {expected_target!r}."
        )
    root_path = _resolve_go_root(Path(root).expanduser())
    examples: list[ProteinExample] = []
    for split_name in _resolve_splits(split):
        path = root_path / "tasks" / expected_target / track / f"{split_name}.jsonl"
        examples.extend(
            _load_go_jsonl(
                path,
                target=resolved_target,
                split=split_name,
                ontology_path=root_path / "go_ontology" / "go.obo",
            )
        )
    return ProteinDataset(examples)


def read_go_manifest(root: str | Path, *, name: str) -> Mapping[str, Any]:
    """Read the manifest for one generated Gene Ontology prediction dataset."""
    aspect, track = _parse_go_dataset_name(name)
    root_path = _resolve_go_root(Path(root).expanduser())
    path = root_path / "manifests" / f"go_{aspect}_{track}.json"
    if not path.exists():
        raise EmbeddingInputError(f"GO manifest file not found: {path}.")
    with path.open("r", encoding="utf-8") as handle:
        loaded = json.load(handle)
    if not isinstance(loaded, dict):
        raise EmbeddingInputError(f"GO manifest file {path} must contain a JSON object.")
    return cast(dict[str, Any], loaded)


def _parse_go_dataset_name(name: str) -> tuple[GoAspect, GoTrack]:
    normalized = str(name).strip().lower().replace("-", "_")
    if normalized.startswith("go:"):
        normalized = normalized.partition(":")[2]
    parts = normalized.split("_")
    if len(parts) != 3 or parts[0] != "go":
        raise EmbeddingInputError("GO dataset name must look like 'go_bp_head' or 'go_mf_full'.")
    aspect, track = parts[1:]
    if aspect not in _GO_ASPECTS:
        raise EmbeddingInputError("GO dataset aspect must be 'bp', 'cc', or 'mf'.")
    if track not in _GO_TRACKS:
        raise EmbeddingInputError("GO dataset track must be 'head', 'main', or 'full'.")
    return aspect, track


def _resolve_go_root(root: Path) -> Path:
    if (root / "tasks").exists():
        return root
    nested = root / _GO_DATASET_ROOT_NAME
    if (nested / "tasks").exists():
        return nested
    raise EmbeddingInputError(
        f"No GO dataset layout found under {root}. Expected tasks/ or "
        f"{_GO_DATASET_ROOT_NAME}/tasks/."
    )


def _resolve_splits(split: str | Sequence[str] | None) -> tuple[SplitName, ...]:
    if split is None:
        return _GO_SPLITS
    values = (split,) if isinstance(split, str) else tuple(split)
    resolved: list[SplitName] = []
    for value in values:
        normalized = str(value).strip().lower()
        if normalized == "valid":
            normalized = "val"
        if normalized not in _GO_SPLITS:
            raise EmbeddingInputError("GO split must be one of: train, val, test.")
        resolved.append(normalized)
    return tuple(resolved)


def _load_go_jsonl(
    path: Path,
    *,
    target: str,
    split: SplitName,
    ontology_path: Path,
) -> list[ProteinExample]:
    if not path.exists():
        raise EmbeddingInputError(f"GO split file not found: {path}.")
    examples: list[ProteinExample] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            text = line.strip()
            if not text:
                continue
            examples.append(
                _go_record_to_example(
                    _parse_go_record(text, path=path, line_number=line_number),
                    target=target,
                    split=split,
                    ontology_path=ontology_path,
                )
            )
    return examples


def _parse_go_record(text: str, *, path: Path, line_number: int) -> Mapping[str, Any]:
    try:
        loaded = json.loads(text)
    except json.JSONDecodeError as exc:
        raise EmbeddingInputError(
            f"Invalid JSON in GO split file {path} at line {line_number}."
        ) from exc
    if not isinstance(loaded, dict):
        raise EmbeddingInputError(f"GO split file {path} line {line_number} must be a JSON object.")
    return cast(dict[str, Any], loaded)


def _go_record_to_example(
    record: Mapping[str, Any],
    *,
    target: str,
    split: SplitName,
    ontology_path: Path,
) -> ProteinExample:
    record_id = str(record.get("id", "")).strip()
    sequence = str(record.get("sequence", "")).strip()
    raw_labels = record.get("labels_asserted")
    if not isinstance(raw_labels, list) or not raw_labels:
        raise EmbeddingInputError("GO record is missing non-empty label field 'labels_asserted'.")
    raw_go_labels = cast(list[object], raw_labels)
    labels = sorted({str(label).strip() for label in raw_go_labels if str(label).strip()})
    if not labels:
        raise EmbeddingInputError("GO record label field 'labels_asserted' contains no labels.")
    metadata: dict[str, Any] = {}
    raw_metadata = record.get("metadata")
    if isinstance(raw_metadata, dict):
        metadata.update(cast(dict[str, Any], raw_metadata))
    for field in ("cluster_id", "rep_accession", "labels_propagated"):
        if field in record:
            metadata[field] = record[field]
    metadata["go_obo_path"] = str(ontology_path)
    return ProteinExample(
        id=record_id,
        sequence=sequence,
        labels={target: labels},
        split=split,
        metadata=metadata,
    )


__all__ = [
    "GO_COLLECTION_METADATA",
    "GO_DATASETS",
    "GoAspect",
    "GoTrack",
    "load_go_dataset",
    "read_go_manifest",
]
