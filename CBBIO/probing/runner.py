"""Execution helpers for supervised probing tasks."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import time
from typing import Dict, cast

from CBBIO.embeddings import EmbeddingInputError

from .datasets import ProteinDataset, ResidueDataset
from .probes import train_and_evaluate_probe, train_and_evaluate_residue_probe
from .tasks import Task


@dataclass(frozen=True)
class TaskLayerResult:
    task_name: str
    model_reference: str
    layer_index: int
    metrics: Dict[str, float]
    predictions: Dict[str, float | int | str]
    train_count: int
    val_count: int
    test_count: int
    elapsed_seconds: float


def run_task_on_layer(
    *,
    task: Task,
    embeddings: Mapping[str, Sequence[float] | Sequence[Sequence[float]]],
    model_reference: str = "precomputed",
    layer_index: int = 0,
) -> TaskLayerResult:
    """Train and evaluate one task against one model/layer embedding matrix."""

    started = time.perf_counter()
    if task.prediction.level == "protein":
        if not isinstance(task.dataset, ProteinDataset):
            raise EmbeddingInputError("Protein-level tasks require a ProteinDataset.")
        train_examples, test_examples = task.dataset.require_training_and_test_splits()
        labels = task.dataset.target_values(task.prediction.target)
        evaluation = train_and_evaluate_probe(
            train_ids=[example.id for example in train_examples],
            test_ids=[example.id for example in test_examples],
            embeddings=cast(Mapping[str, Sequence[float]], embeddings),
            labels=labels,
            prediction=task.prediction,
            probe=task.probe,
        )
    elif task.prediction.level == "residue":
        if not isinstance(task.dataset, ResidueDataset):
            raise EmbeddingInputError("Residue-level tasks require a ResidueDataset.")
        train_examples, test_examples = task.dataset.require_training_and_test_splits()
        labels = task.dataset.target_values(task.prediction.target)
        masks = {example.id: example.mask for example in task.dataset.examples}
        evaluation = train_and_evaluate_residue_probe(
            train_ids=[example.id for example in train_examples],
            test_ids=[example.id for example in test_examples],
            embeddings=cast(Mapping[str, Sequence[Sequence[float]]], embeddings),
            labels=labels,
            masks=masks,
            prediction=task.prediction,
            probe=task.probe,
        )
    else:
        raise EmbeddingInputError(f"Unsupported task level: {task.prediction.level!r}.")
    split_counts = task.dataset.split_counts()
    return TaskLayerResult(
        task_name=task.name,
        model_reference=model_reference,
        layer_index=int(layer_index),
        metrics=evaluation.metrics,
        predictions=evaluation.predictions,
        train_count=split_counts["train"],
        val_count=split_counts["val"],
        test_count=split_counts["test"],
        elapsed_seconds=time.perf_counter() - started,
    )


__all__ = ["TaskLayerResult", "run_task_on_layer"]
