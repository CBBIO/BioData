"""Execution helpers for supervised probing tasks."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import time
from typing import Any, Dict, cast

from CBBIO.embeddings import EmbeddingInputError

from .backends import (
    ProbeBackend,
    ProbeBackendInput,
    ProbeLabel,
    ProbePredictionOutput,
    ProbeScoreOutput,
    evaluate_probe_backend_output,
)
from .datasets import ProteinDataset, ResidueDataset
from .metrics import (
    go_combined_protein_centric_metrics,
    go_fixed_threshold_protein_centric_metrics,
    read_information_accretion_weights,
)
from .probes import LinearProbe, MlpProbe, ProbeEvaluation, TransferProbe
from .tasks import ProbeSpec, Task


@dataclass(frozen=True)
class TaskLayerResult:
    """Result of evaluating one probing task on one embedding layer."""
    task_name: str
    model_reference: str
    layer_index: int
    metrics: Dict[str, float]
    predictions: Dict[str, ProbePredictionOutput]
    scores: Dict[str, ProbeScoreOutput] | None
    metadata: Mapping[str, Any] | None
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
        metadata=evaluation.metadata,
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
    if _is_go_task(task):
        metrics.update(
            _evaluate_go_metrics(
                task,
                scores=scores,
                fixed_threshold=_go_fixed_threshold(backend),
            )
        )
    return ProbeEvaluation(
        metrics=metrics,
        predictions=predictions,
        scores=scores,
        metadata=output.metadata,
    )


def _is_go_task(task: Task) -> bool:
    return (
        isinstance(task.dataset, ProteinDataset)
        and task.prediction.objective == "multilabel"
        and task.prediction.target in {"go_bp", "go_cc", "go_mf"}
    )


def _evaluate_go_metrics(
    task: Task,
    *,
    scores: Mapping[str, ProbeScoreOutput] | None,
    fixed_threshold: float | None,
) -> dict[str, float]:
    if scores is None:
        raise EmbeddingInputError("GO evaluation requires multilabel prediction scores.")
    test_examples = task.dataset.by_split("test")
    true_labels: dict[str, Sequence[str]] = {}
    ontology_paths: set[str] = set()
    ia_paths: set[str] = set()
    sources: set[str] = set()
    for example in test_examples:
        metadata = example.metadata or {}
        raw_labels = metadata.get("labels_propagated")
        ontology_path = metadata.get("go_obo_path")
        ia_path = metadata.get("ia_path")
        source = metadata.get("source")
        if not isinstance(raw_labels, Sequence) or isinstance(raw_labels, str | bytes):
            raise EmbeddingInputError(
                f"GO example {example.id!r} is missing propagated labels in metadata."
            )
        if not isinstance(ontology_path, str) or not ontology_path.strip():
            raise EmbeddingInputError(f"GO example {example.id!r} is missing its ontology path.")
        propagated_labels = cast(Sequence[object], raw_labels)
        true_labels[example.id] = [str(label) for label in propagated_labels]
        ontology_paths.add(ontology_path)
        if isinstance(ia_path, str) and ia_path.strip():
            ia_paths.add(ia_path)
        if isinstance(source, str) and source.strip():
            sources.add(source)
    if len(ontology_paths) != 1:
        raise EmbeddingInputError("GO evaluation requires one shared ontology path.")
    if len(ia_paths) > 1:
        raise EmbeddingInputError("GO weighted evaluation requires one shared IA weight path.")
    if sources.intersection({"cafa5", "cafa6"}) and not ia_paths:
        raise EmbeddingInputError(
            "CAFA weighted GO evaluation requires an IA.txt information-accretion weight file."
        )
    score_rows: dict[str, Sequence[float]] = {}
    for example_id, score in scores.items():
        if not isinstance(score, Sequence) or isinstance(score, str | bytes):
            raise EmbeddingInputError(f"GO prediction scores for {example_id!r} must be a sequence.")
        score_rows[example_id] = [float(value) for value in score]
    scored_examples = [*task.dataset.by_split("train"), *test_examples]
    class_names = sorted(
        {
            str(label)
            for example in scored_examples
            for label in example.labels[task.prediction.target]
        }
    )
    from CBBIO.GO import GOError, load_go

    try:
        ontology = load_go(next(iter(ontology_paths)))
    except GOError as exc:
        raise EmbeddingInputError("GO evaluation could not load the generated GO ontology.") from exc
    try:
        term_weights = (
            read_information_accretion_weights(next(iter(ia_paths)))
            if ia_paths
            else None
        )
        metrics = go_combined_protein_centric_metrics(
            score_rows,
            class_names=class_names,
            true_labels=true_labels,
            ontology=ontology,
            term_weights=term_weights,
            threshold_count=_go_threshold_count(sources),
        )
        if fixed_threshold is not None:
            metrics.update(
                go_fixed_threshold_protein_centric_metrics(
                    score_rows,
                    class_names=class_names,
                    true_labels=true_labels,
                    ontology=ontology,
                    threshold=fixed_threshold,
                )
            )
        return metrics
    except ValueError as exc:
        raise EmbeddingInputError(f"GO evaluation failed: {exc}") from exc


def _go_fixed_threshold(backend: ProbeBackend) -> float | None:
    if isinstance(backend, TransferProbe):
        return backend.threshold
    if isinstance(backend, LinearProbe | MlpProbe):
        return 0.5
    return None


def _go_threshold_count(sources: set[str]) -> int:
    if sources.intersection({"cafa5", "cafa6"}):
        return 51
    return 101


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
