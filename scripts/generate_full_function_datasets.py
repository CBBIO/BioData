"""Generate unfiltered GO and EC probing datasets from function intermediates."""

from __future__ import annotations

import argparse
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
import hashlib
import json
from pathlib import Path
import shutil
import sqlite3
from statistics import median
from typing import Any, Literal, TextIO, cast

from CBBIO import load_go


SplitName = Literal["train", "val", "test"]

_SPLITS: tuple[SplitName, ...] = ("train", "val", "test")
_GO_ASPECTS: Mapping[str, str] = {
    "bp": "biological_process",
    "cc": "cellular_component",
    "mf": "molecular_function",
}
_GO_TASKS: Mapping[str, str] = {
    "bp": "go_bp",
    "cc": "go_cc",
    "mf": "go_mf",
}
_SPLIT_RATIOS: Mapping[SplitName, float] = {
    "train": 0.8,
    "val": 0.1,
    "test": 0.1,
}


def main() -> None:
    """Run the full dataset generator from command-line arguments."""
    args = _parse_args()
    if args.go_sqlite is not None and args.go_output is not None and args.go_obo is not None:
        generate_full_go_datasets(
            sqlite_path=args.go_sqlite,
            output_root=args.go_output,
            obo_path=args.go_obo,
            seed=args.seed,
        )
    if args.ec_sqlite is not None and args.ec_output is not None:
        generate_full_ec_dataset(
            sqlite_path=args.ec_sqlite,
            output_root=args.ec_output,
            seed=args.seed,
        )


def generate_full_go_datasets(
    *,
    sqlite_path: Path,
    output_root: Path,
    obo_path: Path,
    seed: int = 42,
) -> None:
    """Generate full GO datasets for BP, CC, and MF."""
    output_root.mkdir(parents=True, exist_ok=True)
    ontology_dir = output_root / "go_ontology"
    ontology_dir.mkdir(parents=True, exist_ok=True)
    output_obo = ontology_dir / "go.obo"
    if obo_path.resolve() != output_obo.resolve():
        shutil.copy2(obo_path, output_obo)
    ontology = load_go(str(output_obo))

    with sqlite3.connect(sqlite_path) as connection:
        for aspect, aspect_name in _GO_ASPECTS.items():
            task = _GO_TASKS[aspect]
            records = _go_records(connection, aspect=aspect_name, task=task, ontology=ontology, seed=seed)
            split_counts = _write_jsonl_splits(
                records,
                output_root / "tasks" / task / "full",
            )
            _write_manifest(
                output_root / "manifests" / f"{task}_full.json",
                {
                    "class_support": _support_summary(record["labels_asserted"] for record in records),
                    "input_files": {
                        "go_obo": str(obo_path),
                        "reuse_sqlite": str(sqlite_path),
                    },
                    "minimum_support": None,
                    "seed": seed,
                    "species_stratification": True,
                    "species_stratification_scope": "within_tax_id_hash",
                    "split_counts": split_counts,
                    "task": task,
                    "track": "full",
                },
            )


def generate_full_ec_dataset(
    *,
    sqlite_path: Path,
    output_root: Path,
    seed: int = 42,
) -> None:
    """Generate the full EC dataset with exact labels and derived level metadata."""
    output_root.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(sqlite_path) as connection:
        records = _ec_records(connection, seed=seed)
    split_counts = _write_jsonl_splits(
        records,
        output_root / "tasks" / "ec_full" / "full",
    )
    _write_manifest(
        output_root / "manifests" / "ec_full_full.json",
        {
            "class_support": _support_summary(record["labels_exact"] for record in records),
            "input_files": {
                "reuse_sqlite": str(sqlite_path),
            },
            "minimum_support": None,
            "seed": seed,
            "species_stratification": True,
            "species_stratification_scope": "within_tax_id_hash",
            "split_counts": split_counts,
            "task": "ec_full",
            "track": "full",
        },
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--go-sqlite", type=Path)
    parser.add_argument("--go-output", type=Path)
    parser.add_argument("--go-obo", type=Path)
    parser.add_argument("--ec-sqlite", type=Path)
    parser.add_argument("--ec-output", type=Path)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def _go_records(
    connection: sqlite3.Connection,
    *,
    aspect: str,
    task: str,
    ontology: Any,
    seed: int,
) -> list[dict[str, Any]]:
    labels_by_cluster: dict[str, set[str]] = defaultdict(set)
    for cluster_id, go_id in connection.execute(
        "select cluster_id, go_id from go_labels where aspect = ?",
        (aspect,),
    ):
        labels_by_cluster[str(cluster_id)].add(str(go_id))

    records: list[dict[str, Any]] = []
    for example in _examples_for_clusters(connection, labels_by_cluster.keys()):
        cluster_id = str(example["cluster_id"])
        labels_asserted = sorted(labels_by_cluster[cluster_id])
        labels_propagated = _propagate_go_labels(labels_asserted, ontology)
        split = _split_for_example(seed=seed, tax_id=str(example["tax_id"]), cluster_id=cluster_id)
        records.append(
            {
                "cluster_id": cluster_id,
                "id": f"{task}:full:{cluster_id}",
                "labels_asserted": labels_asserted,
                "labels_propagated": labels_propagated,
                "metadata": {
                    "member_count": _as_int(example["member_count"]),
                    "organism": str(example["organism"]),
                    "sequence_length": _as_int(example["length"]),
                    "task": task,
                    "tax_id": str(example["tax_id"]),
                    "track": "full",
                },
                "rep_accession": str(example["rep_accession"]),
                "sequence": str(example["sequence"]),
                "split": split,
            }
        )
    return sorted(records, key=lambda record: str(record["id"]))


def _ec_records(connection: sqlite3.Connection, *, seed: int) -> list[dict[str, Any]]:
    labels_by_cluster: dict[str, set[str]] = defaultdict(set)
    levels_by_cluster: dict[str, dict[str, set[str]]] = defaultdict(
        lambda: {
            "ec_level1": set(),
            "ec_level2": set(),
            "ec_level3": set(),
        }
    )
    for row in connection.execute(
        "select cluster_id, ec_number, ec_level1, ec_level2, ec_level3 from ec_labels"
    ):
        cluster_id = str(row[0])
        labels_by_cluster[cluster_id].add(str(row[1]))
        levels_by_cluster[cluster_id]["ec_level1"].add(str(row[2]))
        levels_by_cluster[cluster_id]["ec_level2"].add(str(row[3]))
        levels_by_cluster[cluster_id]["ec_level3"].add(str(row[4]))

    records: list[dict[str, Any]] = []
    for example in _examples_for_clusters(connection, labels_by_cluster.keys()):
        cluster_id = str(example["cluster_id"])
        split = _split_for_example(seed=seed, tax_id=str(example["tax_id"]), cluster_id=cluster_id)
        levels = levels_by_cluster[cluster_id]
        records.append(
            {
                "cluster_id": cluster_id,
                "id": f"ec_full:full:{cluster_id}",
                "labels_exact": sorted(labels_by_cluster[cluster_id]),
                "metadata": {
                    "ec_level1": sorted(levels["ec_level1"]),
                    "ec_level2": sorted(levels["ec_level2"]),
                    "ec_level3": sorted(levels["ec_level3"]),
                    "member_count": _as_int(example["member_count"]),
                    "organism": str(example["organism"]),
                    "sequence_length": _as_int(example["length"]),
                    "task": "ec_full",
                    "tax_id": str(example["tax_id"]),
                    "track": "full",
                },
                "rep_accession": str(example["rep_accession"]),
                "sequence": str(example["sequence"]),
                "split": split,
            }
        )
    return sorted(records, key=lambda record: str(record["id"]))


def _examples_for_clusters(
    connection: sqlite3.Connection,
    cluster_ids: Iterable[str],
) -> list[dict[str, object]]:
    identifiers = sorted(set(cluster_ids))
    examples: list[dict[str, object]] = []
    query = (
        "select cluster_id, rep_accession, sequence, length, tax_id, organism, member_count "
        "from examples where cluster_id in ({placeholders})"
    )
    for batch in _batches(identifiers, size=900):
        placeholders = ",".join("?" for _ in batch)
        cursor = connection.execute(query.format(placeholders=placeholders), batch)
        for row in cursor:
            examples.append(
                {
                    "cluster_id": row[0],
                    "rep_accession": row[1],
                    "sequence": row[2],
                    "length": row[3],
                    "tax_id": row[4],
                    "organism": row[5],
                    "member_count": row[6],
                }
            )
    return examples


def _batches(values: Sequence[str], *, size: int) -> Iterable[Sequence[str]]:
    for start in range(0, len(values), size):
        yield values[start : start + size]


def _propagate_go_labels(labels: Sequence[str], ontology: Any) -> list[str]:
    propagated: set[str] = set()
    for label in labels:
        if not ontology.has_term(label):
            propagated.add(label)
            continue
        propagated.update(ontology.ancestors(label, include_self=True))
    return sorted(propagated)


def _as_int(value: object) -> int:
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        return int(value)
    if isinstance(value, float):
        return int(value)
    raise TypeError(f"Expected integer-compatible value, got {type(value).__name__}.")


def _split_for_example(*, seed: int, tax_id: str, cluster_id: str) -> SplitName:
    key = f"{seed}:{tax_id}:{cluster_id}"
    value = int(hashlib.sha256(key.encode("utf-8")).hexdigest()[:12], 16) / float(16**12)
    if value < _SPLIT_RATIOS["train"]:
        return "train"
    if value < _SPLIT_RATIOS["train"] + _SPLIT_RATIOS["val"]:
        return "val"
    return "test"


def _write_jsonl_splits(records: Sequence[Mapping[str, Any]], split_dir: Path) -> dict[str, int]:
    split_dir.mkdir(parents=True, exist_ok=True)
    handles: dict[SplitName, TextIO] = {
        split: (split_dir / f"{split}.jsonl").open("w", encoding="utf-8") for split in _SPLITS
    }
    counts: dict[str, int] = {split: 0 for split in _SPLITS}
    try:
        for record in records:
            split = cast(SplitName, record["split"])
            output_record = dict(record)
            output_record.pop("split")
            handles[split].write(json.dumps(output_record, sort_keys=True) + "\n")
            counts[split] += 1
    finally:
        for handle in handles.values():
            handle.close()
    return counts


def _write_manifest(path: Path, manifest: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _support_summary(label_sets: Iterable[object]) -> dict[str, int | float]:
    support: dict[str, int] = {}
    assignments = 0
    for raw_labels in label_sets:
        labels = cast(Sequence[str], raw_labels)
        assignments += len(labels)
        for label in labels:
            support[label] = support.get(label, 0) + 1
    values = sorted(support.values())
    if not values:
        return {
            "label_assignments": 0,
            "max": 0,
            "median": 0,
            "min": 0,
            "num_classes": 0,
        }
    return {
        "label_assignments": assignments,
        "max": max(values),
        "median": median(values),
        "min": min(values),
        "num_classes": len(values),
    }


if __name__ == "__main__":
    main()
