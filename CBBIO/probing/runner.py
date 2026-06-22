"""Execution helpers for supervised probing tasks."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import time
from typing import Any, Dict, cast

from .backends import (
    ProbeBackend,
    ProbeBackendInput,
    ProbeLabel,
    evaluate_probe_backend_output,
)
from .datasets import ResidueDataset
from .probes import LinearProbe, MlpProbe, ProbeEvaluation
from .tasks import ProbeSpec, Task


@dataclass(frozen=True)
class TaskLayerResult:
    """Result of evaluating one probing task on one embedding layer."""
    task_name: str
    model_reference: str
    layer_index: int
    metrics: Dict[str, float]
    predictions: Dict[str, float | int | str]
    scores: Dict[str, float] | None
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
    feature_mean: Any = None,
    feature_std: Any = None,
    flat_data: Any = None,
) -> TaskLayerResult:
    """Train and evaluate one task against one model/layer embedding matrix."""

    started = time.perf_counter()
    backend = (
        task.probe
        if isinstance(task.probe, ProbeBackend)
        else _backend_from_spec(task.probe)
    )
    evaluation = _run_backend(
        task,
        embeddings=embeddings,
        backend=backend,
        feature_mean=feature_mean,
        feature_std=feature_std,
        flat_data=flat_data,
    )
    split_counts = task.dataset.split_counts()
    return TaskLayerResult(
        task_name=task.name,
        model_reference=model_reference,
        layer_index=int(layer_index),
        metrics=evaluation.metrics,
        predictions=evaluation.predictions,
        scores=evaluation.scores,
        train_count=split_counts["train"],
        val_count=split_counts["val"],
        test_count=split_counts["test"],
        elapsed_seconds=time.perf_counter() - started,
    )


def _run_backend(
    task: Task,
    *,
    embeddings: Mapping[str, Sequence[float] | Sequence[Sequence[float]]],
    backend: ProbeBackend,
    feature_mean: Any,
    feature_std: Any,
    flat_data: Any,
) -> ProbeEvaluation:
    train_examples, test_examples = task.dataset.require_training_and_test_splits()
    labels = task.dataset.target_values(task.prediction.target)
    masks = (
        {example.id: example.mask for example in task.dataset.examples}
        if isinstance(task.dataset, ResidueDataset)
        else {}
    )
    data = ProbeBackendInput(
        level=task.prediction.level,
        prediction=task.prediction,
        train_ids=tuple(example.id for example in train_examples),
        test_ids=tuple(example.id for example in test_examples),
        embeddings=embeddings,
        labels=cast(Mapping[str, ProbeLabel], labels),
        masks=masks,
        feature_mean=feature_mean,
        feature_std=feature_std,
        flat_data=flat_data,
    )
    output = backend.fit_predict(data)
    metrics, predictions, scores = evaluate_probe_backend_output(data, output)
    return ProbeEvaluation(metrics=metrics, predictions=predictions, scores=scores)


def _backend_from_spec(probe: ProbeSpec) -> ProbeBackend:
    if probe.kind == "linear":
        return LinearProbe(
            epochs=probe.epochs,
            learning_rate=probe.learning_rate,
            batch_size=probe.batch_size,
            weight_decay=probe.weight_decay,
            seed=probe.seed,
        )
    return MlpProbe(
        epochs=probe.epochs,
        learning_rate=probe.learning_rate,
        batch_size=probe.batch_size,
        weight_decay=probe.weight_decay,
        seed=probe.seed,
        hidden_dim=probe.hidden_dim,
    )


__all__ = ["TaskLayerResult", "run_task_on_layer"]
