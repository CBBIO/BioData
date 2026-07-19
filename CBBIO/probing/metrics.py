"""Metric helpers for probe evaluation."""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
import math
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import numpy as _np
import numpy.typing as _npt

if TYPE_CHECKING:
    from CBBIO.GO import GOOntology


@dataclass(frozen=True)
class _PreparedGoMetricInputs:
    truth: Mapping[str, set[str]]
    scores: Mapping[str, Mapping[str, float]]


@dataclass(frozen=True)
class GoMetricContext:
    """Reusable GO metric preparation shared across score matrices."""

    class_names: tuple[str, ...]
    class_ancestor_map: Mapping[str, Sequence[str]]
    score_terms: tuple[str, ...]
    class_edge_indices: Any
    ancestor_edge_indices: Any
    truth: Mapping[str, set[str]]
    excluded_terms: Mapping[str, set[str]]
    interest_terms: set[str] | None
    term_weights: Mapping[str, float] | None
    thresholds: tuple[float, ...]


@dataclass(frozen=True)
class _PreparedGoScoreTerm:
    term: str
    score: float
    weight: float
    is_true: bool


@dataclass(frozen=True)
class _PreparedGoThresholdRow:
    truth: set[str]
    terms: tuple[_PreparedGoScoreTerm, ...]
    truth_weight: float


@dataclass(frozen=True)
class _FlatGoThresholdRows:
    protein_count: int
    protein_indices: Any
    scores: Any
    true_values: Any
    weights: Any
    truth_counts: Any
    truth_weights: Any


@dataclass(frozen=True)
class _SemanticDistanceMetrics:
    threshold: float
    remaining_uncertainty: float
    misinformation: float


@dataclass(frozen=True)
class _ProteinCentricThresholdMetrics:
    f_score: float
    precision: float
    recall: float


@dataclass(frozen=True)
class _GoThresholdStatistics:
    protein_count: int
    macro_precision: Any
    macro_recall: Any
    micro_precision: Any
    micro_recall: Any
    weighted_protein_count: int
    weighted_macro_precision: Any
    weighted_macro_recall: Any
    weighted_micro_precision: Any
    weighted_micro_recall: Any
    remaining_uncertainty: Any
    misinformation: Any


def regression_metrics(y_true: Sequence[float], y_pred: Sequence[float]) -> dict[str, float]:
    """Compute regression metrics for true and predicted values."""
    _require_same_non_empty_length(y_true, y_pred)
    true_values = _np.asarray(y_true, dtype=_np.float64)
    pred_values = _np.asarray(y_pred, dtype=_np.float64)
    errors = pred_values - true_values
    squared_errors = errors * errors
    mae = float(cast(float, _np.mean(_np.abs(errors))))
    rmse = math.sqrt(float(cast(float, _np.mean(squared_errors))))
    centered_true = true_values - _np.mean(true_values)
    ss_tot = float(cast(float, _np.dot(centered_true, centered_true)))
    ss_res = float(cast(float, _np.sum(squared_errors)))
    r2 = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
    return {
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
        "spearmanr": _spearmanr_arrays(true_values, pred_values),
    }


def spearmanr(y_true: Sequence[float], y_pred: Sequence[float]) -> float:
    """Spearman rank correlation with average ranks for ties."""

    _require_same_non_empty_length(y_true, y_pred)
    true_values = _np.asarray(y_true, dtype=_np.float64)
    pred_values = _np.asarray(y_pred, dtype=_np.float64)
    return _spearmanr_arrays(true_values, pred_values)


def binary_metrics(y_true: Sequence[int], y_pred: Sequence[int], y_score: Sequence[float]) -> dict[str, float]:
    """Compute binary classification metrics for labels, predictions, and scores."""
    _require_same_non_empty_length(y_true, y_pred)
    _require_same_non_empty_length(y_true, y_score)
    true_values = _np.asarray(y_true, dtype=_np.int8)
    pred_values = _np.asarray(y_pred, dtype=_np.int8)
    score_values = _np.asarray(y_score, dtype=_np.float64)
    true_positive = true_values == 1
    predicted_positive = pred_values == 1
    tp = int(_np.count_nonzero(true_positive & predicted_positive))
    tn = int(_np.count_nonzero(~true_positive & ~predicted_positive))
    fp = int(_np.count_nonzero(~true_positive & predicted_positive))
    fn = int(_np.count_nonzero(true_positive & ~predicted_positive))
    accuracy = float(tp + tn) / float(len(y_true))
    precision = float(tp) / float(tp + fp) if tp + fp else 0.0
    recall = float(tp) / float(tp + fn) if tp + fn else 0.0
    specificity = float(tn) / float(tn + fp) if tn + fp else 0.0
    balanced_accuracy = (recall + specificity) / 2.0
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
    negative_precision = float(tn) / float(tn + fn) if tn + fn else 0.0
    negative_recall = specificity
    negative_f1 = (
        2.0 * negative_precision * negative_recall / (negative_precision + negative_recall)
        if negative_precision + negative_recall
        else 0.0
    )
    macro_f1 = (negative_f1 + f1) / 2.0
    mcc_denominator = math.sqrt(float(tp + fp) * float(tp + fn) * float(tn + fp) * float(tn + fn))
    mcc = float(tp * tn - fp * fn) / mcc_denominator if mcc_denominator else 0.0
    auroc, auprc = _binary_ranking_metrics(true_values, score_values)
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "macro_f1": macro_f1,
        "balanced_accuracy": balanced_accuracy,
        "mcc": mcc,
        "auroc": auroc,
        "auprc": auprc,
    }


def multiclass_metrics(y_true: Sequence[int], y_pred: Sequence[int], *, class_count: int) -> dict[str, float]:
    """Compute multiclass classification metrics for labels and predictions."""
    _require_same_non_empty_length(y_true, y_pred)
    count = int(class_count)
    if count < 1:
        raise ValueError("Multiclass metrics require class_count >= 1.")
    true_values = _np.asarray(y_true, dtype=_np.int64)
    pred_values = _np.asarray(y_pred, dtype=_np.int64)
    if bool(_np.any((true_values < 0) | (true_values >= count))):
        raise ValueError("Multiclass true labels must be valid class indices.")
    if bool(_np.any((pred_values < 0) | (pred_values >= count))):
        raise ValueError("Multiclass predictions must be valid class indices.")

    supports = _np.bincount(true_values, minlength=count).astype(_np.float64, copy=False)
    predicted_counts = _np.bincount(pred_values, minlength=count).astype(_np.float64, copy=False)
    matched = true_values == pred_values
    true_positives = _np.bincount(true_values[matched], minlength=count).astype(
        _np.float64,
        copy=False,
    )
    precision = _np.divide(
        true_positives,
        predicted_counts,
        out=_np.zeros(count, dtype=_np.float64),
        where=predicted_counts != 0.0,
    )
    recall = _np.divide(
        true_positives,
        supports,
        out=_np.zeros(count, dtype=_np.float64),
        where=supports != 0.0,
    )
    f1_values = _np.divide(
        2.0 * precision * recall,
        precision + recall,
        out=_np.zeros(count, dtype=_np.float64),
        where=(precision + recall) != 0.0,
    )
    return {
        "accuracy": float(cast(float, _np.mean(matched))),
        "macro_f1": float(cast(float, _np.mean(f1_values))),
        "weighted_f1": float(cast(float, _np.sum(f1_values * supports))) / float(len(y_true)),
    }


def multilabel_metrics(
    y_true: Sequence[Sequence[int]],
    y_pred: Sequence[Sequence[int]],
    y_score: Sequence[Sequence[float]],
    *,
    threshold_count: int = 101,
) -> dict[str, float]:
    """Compute multilabel classification metrics for indicator matrices."""
    _require_same_non_empty_length(y_true, y_pred)
    _require_same_non_empty_length(y_true, y_score)
    class_count = len(y_true[0])
    if class_count < 1:
        raise ValueError("Multilabel metric inputs must contain at least one class.")
    for true_row, pred_row, score_row in zip(y_true, y_pred, y_score):
        if len(true_row) != class_count or len(pred_row) != class_count or len(score_row) != class_count:
            raise ValueError("Multilabel metric rows must have the same class count.")

    true_matrix = _np.asarray(y_true, dtype=_np.int8)
    pred_matrix = _np.asarray(y_pred, dtype=_np.int8)
    score_matrix = _np.asarray(y_score, dtype=_np.float64)
    exact_match = float(cast(float, _np.mean(_np.all(true_matrix == pred_matrix, axis=1))))
    true_positive_by_class = _np.sum((true_matrix == 1) & (pred_matrix == 1), axis=0)
    false_positive_by_class = _np.sum((true_matrix == 0) & (pred_matrix == 1), axis=0)
    false_negative_by_class = _np.sum((true_matrix == 1) & (pred_matrix == 0), axis=0)
    precision_by_class = _np.divide(
        true_positive_by_class,
        true_positive_by_class + false_positive_by_class,
        out=_np.zeros(class_count, dtype=_np.float64),
        where=(true_positive_by_class + false_positive_by_class) != 0,
    )
    recall_by_class = _np.divide(
        true_positive_by_class,
        true_positive_by_class + false_negative_by_class,
        out=_np.zeros(class_count, dtype=_np.float64),
        where=(true_positive_by_class + false_negative_by_class) != 0,
    )
    macro_f1_values = _np.divide(
        2.0 * precision_by_class * recall_by_class,
        precision_by_class + recall_by_class,
        out=_np.zeros(class_count, dtype=_np.float64),
        where=(precision_by_class + recall_by_class) != 0.0,
    )
    supports = _np.sum(true_matrix, axis=0)
    average_precision = sum(
        _binary_auprc_array(true_matrix[:, class_index], score_matrix[:, class_index])
        for class_index in range(class_count)
    ) / float(class_count)
    micro_tp = int(_np.sum(true_positive_by_class))
    micro_fp = int(_np.sum(false_positive_by_class))
    micro_fn = int(_np.sum(false_negative_by_class))
    micro_precision = float(micro_tp) / float(micro_tp + micro_fp) if micro_tp + micro_fp else 0.0
    micro_recall = float(micro_tp) / float(micro_tp + micro_fn) if micro_tp + micro_fn else 0.0
    micro_f1 = (
        2.0 * micro_precision * micro_recall / (micro_precision + micro_recall)
        if micro_precision + micro_recall
        else 0.0
    )
    support_total = int(_np.sum(supports))
    weighted_precision = (
        float(cast(float, _np.sum(precision_by_class * supports))) / float(support_total)
        if support_total
        else 0.0
    )
    weighted_recall = (
        float(cast(float, _np.sum(recall_by_class * supports))) / float(support_total)
        if support_total
        else 0.0
    )
    weighted_f1 = (
        float(cast(float, _np.sum(macro_f1_values * supports))) / float(support_total)
        if support_total
        else 0.0
    )
    ranking_metrics = _multilabel_fmax(
        y_true=y_true,
        y_score=y_score,
        thresholds=_score_thresholds(threshold_count, metric_name="multilabel"),
    )
    return {
        "exact_match": exact_match,
        "f1": micro_f1,
        "micro_f1": micro_f1,
        "macro_f1": float(cast(float, _np.mean(macro_f1_values))),
        "weighted_precision": weighted_precision,
        "weighted_recall": weighted_recall,
        "weighted_f1": weighted_f1,
        "average_precision": average_precision,
        **ranking_metrics,
    }


def _multilabel_fmax(
    *,
    y_true: Sequence[Sequence[int]],
    y_score: Sequence[Sequence[float]],
    thresholds: Sequence[float],
) -> dict[str, float]:
    true_values = _np.ravel(_np.asarray(y_true, dtype=_np.int8))
    score_values = _np.ravel(_np.asarray(y_score, dtype=_np.float64))
    unique_thresholds = _np.asarray(sorted(set(thresholds)), dtype=_np.float64)
    predicted_histogram = _np.zeros(unique_thresholds.size, dtype=_np.int64)
    true_positive_histogram = _np.zeros(unique_thresholds.size, dtype=_np.int64)
    chunk_size = 1_000_000
    for start in range(0, score_values.size, chunk_size):
        end = min(start + chunk_size, score_values.size)
        score_chunk = score_values[start:end]
        threshold_indices = _np.searchsorted(
            unique_thresholds,
            score_chunk,
            side="right",
        ) - 1
        included = threshold_indices >= 0
        included_indices = threshold_indices[included]
        predicted_histogram += _np.bincount(
            included_indices,
            minlength=unique_thresholds.size,
        )
        true_positive_histogram += _np.bincount(
            included_indices,
            weights=true_values[start:end][included],
            minlength=unique_thresholds.size,
        ).astype(_np.int64, copy=False)
    predicted_positive = _np.cumsum(predicted_histogram[::-1])[::-1]
    true_positive = _np.cumsum(true_positive_histogram[::-1])[::-1]
    actual_positive = int(_np.sum(true_values))
    precision = _np.divide(
        true_positive,
        predicted_positive,
        out=_np.zeros(unique_thresholds.size, dtype=_np.float64),
        where=predicted_positive != 0,
    )
    recall = true_positive.astype(_np.float64) / float(actual_positive) if actual_positive else _np.zeros(
        unique_thresholds.size,
        dtype=_np.float64,
    )
    f_scores = _np.divide(
        2.0 * precision * recall,
        precision + recall,
        out=_np.zeros(unique_thresholds.size, dtype=_np.float64),
        where=(precision + recall) != 0.0,
    )
    threshold_metrics = {
        float(threshold): (float(f_score), float(precision_value), float(recall_value))
        for threshold, f_score, precision_value, recall_value in zip(
            unique_thresholds,
            f_scores,
            precision,
            recall,
        )
    }
    best = {
        "fmax": -1.0,
        "fmax_threshold": 0.0,
        "precision_at_fmax": 0.0,
        "recall_at_fmax": 0.0,
    }
    for threshold in thresholds:
        f_score, precision, recall = threshold_metrics[threshold]
        if f_score > best["fmax"]:
            best = {
                "fmax": f_score,
                "fmax_threshold": threshold,
                "precision_at_fmax": precision,
                "recall_at_fmax": recall,
            }
    return best


def go_protein_centric_metrics(
    predicted_scores: Mapping[str, Sequence[float]],
    *,
    class_names: Sequence[str],
    true_labels: Mapping[str, Sequence[str]],
    ontology: GOOntology,
    threshold_count: int = 101,
) -> dict[str, float]:
    """Compute CAFA-style protein-centric F-max using propagated GO terms."""
    context = prepare_go_metric_context(
        class_names=class_names,
        true_labels=true_labels,
        ontology=ontology,
        threshold_count=threshold_count,
    )
    return evaluate_go_metric_context(predicted_scores, context=context)


def go_combined_protein_centric_metrics(
    predicted_scores: Mapping[str, Sequence[float]],
    *,
    class_names: Sequence[str],
    true_labels: Mapping[str, Sequence[str]],
    ontology: GOOntology,
    term_weights: Mapping[str, float] | None = None,
    known_labels: Mapping[str, Sequence[str]] | None = None,
    terms_of_interest: Sequence[str] | None = None,
    threshold_count: int = 101,
) -> dict[str, float]:
    """Compute unweighted and optional IA-weighted GO F-max with shared propagation."""
    context = prepare_go_metric_context(
        class_names=class_names,
        true_labels=true_labels,
        ontology=ontology,
        term_weights=term_weights,
        known_labels=known_labels,
        terms_of_interest=terms_of_interest,
        threshold_count=threshold_count,
    )
    return evaluate_go_metric_context(predicted_scores, context=context)


def evaluate_go_metric_context(
    predicted_scores: Mapping[str, Sequence[float]],
    *,
    context: GoMetricContext,
    fixed_threshold: float | None = None,
) -> dict[str, float]:
    """Compute GO metrics from a reusable prepared GO metric context."""
    if fixed_threshold is not None and not 0.0 <= float(fixed_threshold) <= 1.0:
        raise ValueError("GO fixed-threshold metrics require threshold between 0 and 1.")
    protein_ids = tuple(
        protein_id
        for protein_id, labels in context.truth.items()
        if labels
    )
    if not protein_ids:
        raise ValueError("GO truth labels contain no non-root terms.")
    thresholds = tuple(
        sorted(
            set(context.thresholds).union(
                () if fixed_threshold is None else (float(fixed_threshold),)
            )
        )
    )
    statistics = _go_threshold_statistics(
        predicted_scores,
        context=context,
        protein_ids=protein_ids,
        thresholds=thresholds,
    )
    context_indices = _np.searchsorted(
        _np.asarray(thresholds, dtype=_np.float64),
        _np.asarray(context.thresholds, dtype=_np.float64),
    )
    metrics = _go_unweighted_metrics_from_statistics(
        statistics,
        indices=context_indices,
        thresholds=context.thresholds,
    )
    if fixed_threshold is not None:
        fixed_index = int(_np.searchsorted(thresholds, float(fixed_threshold)))
        metrics.update(
            _go_fixed_threshold_metrics_from_statistics(
                statistics,
                index=fixed_index,
                threshold=float(fixed_threshold),
            )
        )
    if context.term_weights is None:
        return metrics
    if not context.term_weights:
        raise ValueError("GO weighted metrics require information-accretion weights.")
    if statistics.weighted_protein_count < 1:
        raise ValueError("GO truth labels contain no positive-weight non-root terms.")
    metrics.update(
        _go_weighted_metrics_from_statistics(
            statistics,
            indices=context_indices,
            thresholds=context.thresholds,
        )
    )
    return metrics


def go_fixed_threshold_protein_centric_metrics(
    predicted_scores: Mapping[str, Sequence[float]],
    *,
    class_names: Sequence[str],
    true_labels: Mapping[str, Sequence[str]],
    ontology: GOOntology,
    threshold: float,
    known_labels: Mapping[str, Sequence[str]] | None = None,
    terms_of_interest: Sequence[str] | None = None,
) -> dict[str, float]:
    """Compute propagated protein-centric GO F1 at one score threshold."""
    if not 0.0 <= float(threshold) <= 1.0:
        raise ValueError("GO fixed-threshold metrics require threshold between 0 and 1.")
    context = prepare_go_metric_context(
        class_names=class_names,
        true_labels=true_labels,
        ontology=ontology,
        known_labels=known_labels,
        terms_of_interest=terms_of_interest,
        threshold_count=2,
    )
    metrics = evaluate_go_metric_context(
        predicted_scores,
        context=context,
        fixed_threshold=float(threshold),
    )
    return {
        name: value
        for name, value in metrics.items()
        if name.startswith("go_propagated_")
    }


def go_weighted_protein_centric_metrics(
    predicted_scores: Mapping[str, Sequence[float]],
    *,
    class_names: Sequence[str],
    true_labels: Mapping[str, Sequence[str]],
    ontology: GOOntology,
    term_weights: Mapping[str, float],
    threshold_count: int = 101,
) -> dict[str, float]:
    """Compute IA-weighted CAFA protein-centric F-max."""
    if not term_weights:
        raise ValueError("GO weighted metrics require information-accretion weights.")
    context = prepare_go_metric_context(
        class_names=class_names,
        true_labels=true_labels,
        ontology=ontology,
        term_weights=term_weights,
        threshold_count=threshold_count,
    )
    metrics = evaluate_go_metric_context(predicted_scores, context=context)
    return {
        name: value
        for name, value in metrics.items()
        if name.startswith("go_weighted_")
    }


def _go_weighted_fmax(  # pyright: ignore[reportUnusedFunction]
    *,
    propagated_truth: Mapping[str, set[str]],
    propagated_scores: Mapping[str, Mapping[str, float]],
    term_weights: Mapping[str, float],
    thresholds: Sequence[float],
) -> dict[str, float]:
    return _go_weighted_metrics(
        propagated_truth=propagated_truth,
        propagated_scores=propagated_scores,
        term_weights=term_weights,
        thresholds=thresholds,
        include_semantic_distance=False,
    )


def _go_weighted_metrics(
    *,
    propagated_truth: Mapping[str, set[str]],
    propagated_scores: Mapping[str, Mapping[str, float]],
    term_weights: Mapping[str, float],
    thresholds: Sequence[float],
    include_semantic_distance: bool,
) -> dict[str, float]:
    rows = _prepare_go_threshold_rows(
        propagated_truth=propagated_truth,
        propagated_scores=propagated_scores,
        term_weights=term_weights,
    )
    best = {
        "go_weighted_fmax": -1.0,
        "go_weighted_macro_fmax": -1.0,
        "go_weighted_fmax_threshold": 0.0,
        "go_weighted_macro_fmax_threshold": 0.0,
        "go_weighted_precision_at_fmax": 0.0,
        "go_weighted_recall_at_fmax": 0.0,
        "go_weighted_macro_precision_at_fmax": 0.0,
        "go_weighted_macro_recall_at_fmax": 0.0,
        "go_weighted_evaluated_protein_count": float(len(propagated_truth)),
    }
    micro_best = {
        "go_weighted_micro_fmax": -1.0,
        "go_weighted_micro_fmax_threshold": 0.0,
        "go_weighted_micro_precision_at_fmax": 0.0,
        "go_weighted_micro_recall_at_fmax": 0.0,
    }
    distance_best = _SemanticDistanceMetrics(
        threshold=0.0,
        remaining_uncertainty=math.inf,
        misinformation=math.inf,
    )
    best_distance = math.inf
    for threshold, threshold_metrics, micro_metrics, distance_metrics in _iter_weighted_go_threshold_metrics(
        rows=rows,
        thresholds=thresholds,
        include_semantic_distance=include_semantic_distance,
    ):
        if threshold_metrics.f_score > best["go_weighted_fmax"]:
            best = {
                "go_weighted_fmax": threshold_metrics.f_score,
                "go_weighted_macro_fmax": threshold_metrics.f_score,
                "go_weighted_fmax_threshold": threshold,
                "go_weighted_macro_fmax_threshold": threshold,
                "go_weighted_precision_at_fmax": threshold_metrics.precision,
                "go_weighted_recall_at_fmax": threshold_metrics.recall,
                "go_weighted_macro_precision_at_fmax": threshold_metrics.precision,
                "go_weighted_macro_recall_at_fmax": threshold_metrics.recall,
                "go_weighted_evaluated_protein_count": float(len(propagated_truth)),
            }
        if micro_metrics.f_score > micro_best["go_weighted_micro_fmax"]:
            micro_best = {
                "go_weighted_micro_fmax": micro_metrics.f_score,
                "go_weighted_micro_fmax_threshold": threshold,
                "go_weighted_micro_precision_at_fmax": micro_metrics.precision,
                "go_weighted_micro_recall_at_fmax": micro_metrics.recall,
            }
        if include_semantic_distance:
            distance = math.sqrt(
                distance_metrics.remaining_uncertainty**2 + distance_metrics.misinformation**2
            )
            if distance < best_distance:
                distance_best = distance_metrics
                best_distance = distance
    best.update(micro_best)
    if include_semantic_distance:
        best.update(
            {
                "go_weighted_smin": best_distance,
                "go_weighted_smin_threshold": distance_best.threshold,
                "go_weighted_remaining_uncertainty_at_smin": distance_best.remaining_uncertainty,
                "go_weighted_misinformation_at_smin": distance_best.misinformation,
            }
        )
    return best


def _go_weighted_metrics_from_matrices(  # pyright: ignore[reportUnusedFunction]
    *,
    score_matrix: Any,
    truth_matrix: Any,
    inclusion_matrix: Any,
    term_weights: Any,
    truth_weights: Any,
    thresholds: Sequence[float],
    include_semantic_distance: bool,
) -> dict[str, float]:
    best = {
        "go_weighted_fmax": -1.0,
        "go_weighted_macro_fmax": -1.0,
        "go_weighted_fmax_threshold": 0.0,
        "go_weighted_macro_fmax_threshold": 0.0,
        "go_weighted_precision_at_fmax": 0.0,
        "go_weighted_recall_at_fmax": 0.0,
        "go_weighted_macro_precision_at_fmax": 0.0,
        "go_weighted_macro_recall_at_fmax": 0.0,
        "go_weighted_evaluated_protein_count": float(score_matrix.shape[0]),
    }
    micro_best = {
        "go_weighted_micro_fmax": -1.0,
        "go_weighted_micro_fmax_threshold": 0.0,
        "go_weighted_micro_precision_at_fmax": 0.0,
        "go_weighted_micro_recall_at_fmax": 0.0,
    }
    distance_best = _SemanticDistanceMetrics(
        threshold=0.0,
        remaining_uncertainty=math.inf,
        misinformation=math.inf,
    )
    best_distance = math.inf
    truth_weight_total = float(_np.sum(truth_weights))
    truth_weight_matrix = truth_matrix * term_weights
    for threshold in thresholds:
        threshold_metrics, micro_metrics, distance_metrics = _weighted_threshold_metrics_from_matrices(
            score_matrix=score_matrix,
            truth_weight_matrix=truth_weight_matrix,
            inclusion_matrix=inclusion_matrix,
            term_weights=term_weights,
            truth_weights=truth_weights,
            truth_weight_total=truth_weight_total,
            threshold=threshold,
            include_semantic_distance=include_semantic_distance,
        )
        if threshold_metrics.f_score > best["go_weighted_fmax"]:
            best = {
                "go_weighted_fmax": threshold_metrics.f_score,
                "go_weighted_macro_fmax": threshold_metrics.f_score,
                "go_weighted_fmax_threshold": threshold,
                "go_weighted_macro_fmax_threshold": threshold,
                "go_weighted_precision_at_fmax": threshold_metrics.precision,
                "go_weighted_recall_at_fmax": threshold_metrics.recall,
                "go_weighted_macro_precision_at_fmax": threshold_metrics.precision,
                "go_weighted_macro_recall_at_fmax": threshold_metrics.recall,
                "go_weighted_evaluated_protein_count": float(score_matrix.shape[0]),
            }
        if micro_metrics.f_score > micro_best["go_weighted_micro_fmax"]:
            micro_best = {
                "go_weighted_micro_fmax": micro_metrics.f_score,
                "go_weighted_micro_fmax_threshold": threshold,
                "go_weighted_micro_precision_at_fmax": micro_metrics.precision,
                "go_weighted_micro_recall_at_fmax": micro_metrics.recall,
            }
        if include_semantic_distance:
            distance = math.sqrt(
                distance_metrics.remaining_uncertainty**2 + distance_metrics.misinformation**2
            )
            if distance < best_distance:
                distance_best = distance_metrics
                best_distance = distance
    best.update(micro_best)
    if include_semantic_distance:
        best.update(
            {
                "go_weighted_smin": best_distance,
                "go_weighted_smin_threshold": distance_best.threshold,
                "go_weighted_remaining_uncertainty_at_smin": distance_best.remaining_uncertainty,
                "go_weighted_misinformation_at_smin": distance_best.misinformation,
            }
        )
    return best


def _weighted_threshold_metrics_from_matrices(
    *,
    score_matrix: Any,
    truth_weight_matrix: Any,
    inclusion_matrix: Any,
    term_weights: Any,
    truth_weights: Any,
    truth_weight_total: float,
    threshold: float,
    include_semantic_distance: bool,
) -> tuple[_ProteinCentricThresholdMetrics, _ProteinCentricThresholdMetrics, _SemanticDistanceMetrics]:
    predicted = (score_matrix >= threshold) & inclusion_matrix
    predicted_weight = cast(Any, _np.sum(predicted * term_weights, axis=1, dtype=_np.float64))
    true_positive_weight = cast(Any, _np.sum(predicted * truth_weight_matrix, axis=1, dtype=_np.float64))
    has_prediction = predicted_weight > 0.0
    precision = (
        cast(float, _np.mean(true_positive_weight[has_prediction] / predicted_weight[has_prediction]))
        if bool(_np.any(has_prediction))
        else 0.0
    )
    recall = float(cast(float, _np.mean(true_positive_weight / truth_weights)))
    micro_true_positive_weight = float(_np.sum(true_positive_weight))
    micro_predicted_weight = float(_np.sum(predicted_weight))
    micro_precision = (
        micro_true_positive_weight / micro_predicted_weight
        if micro_predicted_weight
        else 0.0
    )
    micro_recall = (
        micro_true_positive_weight / truth_weight_total
        if truth_weight_total
        else 0.0
    )
    remaining_uncertainty = 0.0
    misinformation = 0.0
    if include_semantic_distance:
        remaining_uncertainty = float(_np.sum(truth_weights - true_positive_weight))
        misinformation = float(_np.sum(predicted_weight - true_positive_weight))
    return (
        _ProteinCentricThresholdMetrics(
            f_score=_f_score(precision, recall),
            precision=precision,
            recall=recall,
        ),
        _ProteinCentricThresholdMetrics(
            f_score=_f_score(micro_precision, micro_recall),
            precision=micro_precision,
            recall=micro_recall,
        ),
        _SemanticDistanceMetrics(
            threshold=threshold,
            remaining_uncertainty=remaining_uncertainty / float(score_matrix.shape[0]),
            misinformation=misinformation / float(score_matrix.shape[0]),
        ),
    )


def _prepare_go_threshold_rows(
    *,
    propagated_truth: Mapping[str, set[str]],
    propagated_scores: Mapping[str, Mapping[str, float]],
    term_weights: Mapping[str, float] | None = None,
) -> tuple[_PreparedGoThresholdRow, ...]:
    rows: list[_PreparedGoThresholdRow] = []
    for protein_id, truth in propagated_truth.items():
        terms = tuple(
            sorted(
                (
                    _PreparedGoScoreTerm(
                        term=term,
                        score=float(score),
                        weight=(
                            float(term_weights.get(term, 0.0))
                            if term_weights is not None
                            else 1.0
                        ),
                        is_true=term in truth,
                    )
                    for term, score in propagated_scores[protein_id].items()
                ),
                key=lambda item: item.score,
                reverse=True,
            )
        )
        truth_weight = (
            _term_weight_sum(truth, term_weights=term_weights)
            if term_weights is not None
            else float(len(truth))
        )
        rows.append(
            _PreparedGoThresholdRow(
                truth=set(truth),
                terms=terms,
                truth_weight=truth_weight,
            )
        )
    return tuple(rows)


def _flatten_go_threshold_rows(rows: Sequence[_PreparedGoThresholdRow]) -> _FlatGoThresholdRows:
    protein_indices: list[int] = []
    scores: list[float] = []
    true_values: list[float] = []
    weights: list[float] = []
    truth_counts: list[float] = []
    truth_weights: list[float] = []
    for protein_index, row in enumerate(rows):
        truth_counts.append(float(len(row.truth)))
        truth_weights.append(row.truth_weight)
        for term in row.terms:
            protein_indices.append(protein_index)
            scores.append(term.score)
            true_values.append(1.0 if term.is_true else 0.0)
            weights.append(term.weight)
    score_array = _np.asarray(scores, dtype=_np.float64)
    if score_array.size:
        order = _np.argsort(-score_array, kind="stable")
        protein_index_array = _np.asarray(protein_indices, dtype=_np.int64)[order]
        score_array = score_array[order]
        true_value_array = _np.asarray(true_values, dtype=_np.float64)[order]
        weight_array = _np.asarray(weights, dtype=_np.float64)[order]
    else:
        protein_index_array = _np.empty(0, dtype=_np.int64)
        true_value_array = _np.empty(0, dtype=_np.float64)
        weight_array = _np.empty(0, dtype=_np.float64)
    return _FlatGoThresholdRows(
        protein_count=len(rows),
        protein_indices=protein_index_array,
        scores=score_array,
        true_values=true_value_array,
        weights=weight_array,
        truth_counts=_np.asarray(truth_counts, dtype=_np.float64),
        truth_weights=_np.asarray(truth_weights, dtype=_np.float64),
    )


def _iter_unweighted_go_threshold_metrics(  # pyright: ignore[reportUnusedFunction]
    *,
    rows: Sequence[_PreparedGoThresholdRow],
    thresholds: Sequence[float],
) -> Iterator[tuple[float, _ProteinCentricThresholdMetrics, _ProteinCentricThresholdMetrics]]:
    flat = _flatten_go_threshold_rows(rows)
    predicted = _np.zeros(flat.protein_count, dtype=_np.float64)
    true_positive = _np.zeros(flat.protein_count, dtype=_np.float64)
    metrics_by_threshold: dict[float, tuple[_ProteinCentricThresholdMetrics, _ProteinCentricThresholdMetrics]] = {}
    cursor = 0
    for threshold in sorted(set(thresholds), reverse=True):
        end = cursor
        while end < flat.scores.size and flat.scores[end] >= threshold:
            end += 1
        if end > cursor:
            protein_indices = flat.protein_indices[cursor:end]
            _np.add.at(predicted, protein_indices, 1.0)
            _np.add.at(true_positive, protein_indices, flat.true_values[cursor:end])
            cursor = end
        has_prediction = predicted > 0.0
        precision = (
            float(_np.mean(true_positive[has_prediction] / predicted[has_prediction]))
            if bool(_np.any(has_prediction))
            else 0.0
        )
        recall = float(cast(float, _np.mean(true_positive / flat.truth_counts)))
        micro_true_positive = float(_np.sum(true_positive))
        micro_predicted = float(_np.sum(predicted))
        truth_total = float(_np.sum(flat.truth_counts))
        micro_precision = micro_true_positive / micro_predicted if micro_predicted else 0.0
        micro_recall = micro_true_positive / truth_total if truth_total else 0.0
        metrics_by_threshold[threshold] = (
            _ProteinCentricThresholdMetrics(
                f_score=_f_score(precision, recall),
                precision=precision,
                recall=recall,
            ),
            _ProteinCentricThresholdMetrics(
                f_score=_f_score(micro_precision, micro_recall),
                precision=micro_precision,
                recall=micro_recall,
            ),
        )
    for threshold in thresholds:
        macro_metrics, micro_metrics = metrics_by_threshold[threshold]
        yield threshold, macro_metrics, micro_metrics


def _iter_weighted_go_threshold_metrics(
    *,
    rows: Sequence[_PreparedGoThresholdRow],
    thresholds: Sequence[float],
    include_semantic_distance: bool,
) -> Iterator[
    tuple[
        float,
        _ProteinCentricThresholdMetrics,
        _ProteinCentricThresholdMetrics,
        _SemanticDistanceMetrics,
    ]
]:
    flat = _flatten_go_threshold_rows(rows)
    predicted_weight = _np.zeros(flat.protein_count, dtype=_np.float64)
    true_positive_weight = _np.zeros(flat.protein_count, dtype=_np.float64)
    metrics_by_threshold: dict[
        float,
        tuple[_ProteinCentricThresholdMetrics, _ProteinCentricThresholdMetrics, _SemanticDistanceMetrics],
    ] = {}
    cursor = 0
    for threshold in sorted(set(thresholds), reverse=True):
        end = cursor
        while end < flat.scores.size and flat.scores[end] >= threshold:
            end += 1
        if end > cursor:
            protein_indices = flat.protein_indices[cursor:end]
            weights = flat.weights[cursor:end]
            _np.add.at(predicted_weight, protein_indices, weights)
            _np.add.at(true_positive_weight, protein_indices, weights * flat.true_values[cursor:end])
            cursor = end
        has_prediction = predicted_weight > 0.0
        precision = (
            float(_np.mean(true_positive_weight[has_prediction] / predicted_weight[has_prediction]))
            if bool(_np.any(has_prediction))
            else 0.0
        )
        recall = float(cast(float, _np.mean(true_positive_weight / flat.truth_weights)))
        micro_true_positive_weight = float(_np.sum(true_positive_weight))
        micro_predicted_weight = float(_np.sum(predicted_weight))
        truth_weight_total = float(_np.sum(flat.truth_weights))
        micro_precision = (
            micro_true_positive_weight / micro_predicted_weight
            if micro_predicted_weight
            else 0.0
        )
        micro_recall = (
            micro_true_positive_weight / truth_weight_total
            if truth_weight_total
            else 0.0
        )
        remaining_uncertainty = 0.0
        misinformation = 0.0
        if include_semantic_distance:
            remaining_uncertainty = float(_np.sum(flat.truth_weights - true_positive_weight))
            misinformation = float(_np.sum(predicted_weight - true_positive_weight))
        metrics_by_threshold[threshold] = (
            _ProteinCentricThresholdMetrics(
                f_score=_f_score(precision, recall),
                precision=precision,
                recall=recall,
            ),
            _ProteinCentricThresholdMetrics(
                f_score=_f_score(micro_precision, micro_recall),
                precision=micro_precision,
                recall=micro_recall,
            ),
            _SemanticDistanceMetrics(
                threshold=threshold,
                remaining_uncertainty=remaining_uncertainty / float(len(rows)),
                misinformation=misinformation / float(len(rows)),
            ),
        )
    for threshold in thresholds:
        macro_metrics, micro_metrics, distance_metrics = metrics_by_threshold[threshold]
        yield threshold, macro_metrics, micro_metrics, distance_metrics


def read_information_accretion_weights(path: str | Path) -> dict[str, float]:
    """Read CAFA information-accretion weights from a local text file."""
    weight_path = Path(path).expanduser()
    if not weight_path.exists():
        raise ValueError(f"Information-accretion weight file not found: {weight_path}.")
    weights: dict[str, float] = {}
    with weight_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            text = line.strip()
            if not text or text.startswith("#"):
                continue
            parts = text.replace(",", "\t").split()
            if len(parts) < 2:
                raise ValueError(
                    f"Information-accretion weight file {weight_path} line {line_number} "
                    "must contain a GO term and numeric weight."
                )
            term = parts[0].strip()
            if not term:
                raise ValueError(
                    f"Information-accretion weight file {weight_path} line {line_number} "
                    "has an empty GO term."
                )
            try:
                weights[term] = float(parts[1])
            except ValueError as exc:
                raise ValueError(
                    f"Information-accretion weight file {weight_path} line {line_number} "
                    "has a non-numeric weight."
                ) from exc
    if not weights:
        raise ValueError(f"Information-accretion weight file {weight_path} contains no weights.")
    return weights


def cafa6_weighted_fmax_mean(metrics_by_aspect: Mapping[str, Mapping[str, float]]) -> dict[str, float]:
    """Average CAFA6 IA-weighted F-max values across MF, BP, and CC."""
    values: list[float] = []
    for aspect in ("mf", "bp", "cc"):
        metrics = _metrics_for_aspect(metrics_by_aspect, aspect)
        if "go_weighted_fmax" not in metrics:
            raise ValueError(f"CAFA6 {aspect} metrics are missing 'go_weighted_fmax'.")
        values.append(float(metrics["go_weighted_fmax"]))
    return {
        "cafa6_weighted_fmax_mean": sum(values) / float(len(values)),
        "cafa6_mf_weighted_fmax": values[0],
        "cafa6_bp_weighted_fmax": values[1],
        "cafa6_cc_weighted_fmax": values[2],
    }


def _prepare_go_metric_inputs(  # pyright: ignore[reportUnusedFunction]
    predicted_scores: Mapping[str, Sequence[float]],
    *,
    class_names: Sequence[str],
    true_labels: Mapping[str, Sequence[str]],
    ontology: GOOntology,
) -> _PreparedGoMetricInputs:
    context = prepare_go_metric_context(
        class_names=class_names,
        true_labels=true_labels,
        ontology=ontology,
        threshold_count=2,
    )
    return _prepare_go_score_inputs_from_context(predicted_scores, context=context)


def prepare_go_metric_context(
    *,
    class_names: Sequence[str],
    true_labels: Mapping[str, Sequence[str]],
    ontology: GOOntology,
    term_weights: Mapping[str, float] | None = None,
    known_labels: Mapping[str, Sequence[str]] | None = None,
    terms_of_interest: Sequence[str] | None = None,
    threshold_count: int = 101,
) -> GoMetricContext:
    """Prepare reusable ontology propagation inputs for GO metric scoring."""

    normalized_class_names = tuple(str(term).strip() for term in class_names)
    if not normalized_class_names:
        raise ValueError("GO metrics require at least one predicted GO class.")
    if not true_labels:
        raise ValueError("GO metric inputs must contain at least one truth label row.")
    non_root_cache: dict[str, bool] = {}
    truth_ancestor_map = _go_non_root_ancestor_map(
        _unique_go_terms_from_label_rows(true_labels.values()),
        ontology=ontology,
        non_root_cache=non_root_cache,
    )
    class_ancestor_map = _go_non_root_ancestor_map(
        normalized_class_names,
        ontology=ontology,
        non_root_cache=non_root_cache,
    )
    score_terms, class_edge_indices, ancestor_edge_indices = _go_score_propagation_edges(
        normalized_class_names,
        ancestor_map=class_ancestor_map,
    )
    propagated_truth = {
        protein_id: _propagate_go_terms_from_map(labels, ancestor_map=truth_ancestor_map)
        for protein_id, labels in true_labels.items()
    }
    known_terms: dict[str, set[str]] = {}
    if known_labels is not None:
        known_ancestor_map = _go_non_root_ancestor_map(
            _unique_go_terms_from_label_rows(known_labels.values()),
            ontology=ontology,
            non_root_cache=non_root_cache,
        )
        known_terms = {
            protein_id: _propagate_go_terms_from_map(labels, ancestor_map=known_ancestor_map)
            for protein_id, labels in known_labels.items()
        }
    interest_terms = (
        {
            normalized
            for term in terms_of_interest
            if (normalized := str(term).strip())
            and _is_non_root_go_term_cached(normalized, ontology=ontology, cache=non_root_cache)
        }
        if terms_of_interest is not None
        else None
    )
    truth: dict[str, set[str]] = {}
    for protein_id, labels in propagated_truth.items():
        filtered_truth = labels.difference(known_terms.get(protein_id, set()))
        if interest_terms is not None:
            filtered_truth = filtered_truth.intersection(interest_terms)
        truth[protein_id] = filtered_truth
    return GoMetricContext(
        class_names=normalized_class_names,
        class_ancestor_map=class_ancestor_map,
        score_terms=score_terms,
        class_edge_indices=class_edge_indices,
        ancestor_edge_indices=ancestor_edge_indices,
        truth=truth,
        excluded_terms=known_terms,
        interest_terms=interest_terms,
        term_weights=term_weights,
        thresholds=tuple(_score_thresholds(threshold_count, metric_name="GO")),
    )


def _go_threshold_statistics(
    predicted_scores: Mapping[str, Sequence[float]],
    *,
    context: GoMetricContext,
    protein_ids: Sequence[str],
    thresholds: Sequence[float],
) -> _GoThresholdStatistics:
    if not predicted_scores:
        raise ValueError("GO metric inputs must contain at least one prediction.")
    if set(predicted_scores) != set(context.truth):
        raise ValueError("GO prediction and truth identifiers must match.")
    threshold_values = _np.asarray(thresholds, dtype=_np.float64)
    threshold_count = int(threshold_values.size)
    class_count = len(context.class_names)
    term_count = len(context.score_terms)
    term_indices = {term: index for index, term in enumerate(context.score_terms)}
    all_term_indices = _np.arange(term_count, dtype=_np.int64)
    if context.interest_terms is None:
        base_inclusion = _np.ones(term_count, dtype=bool)
    else:
        base_inclusion = _np.asarray(
            [term in context.interest_terms for term in context.score_terms],
            dtype=bool,
        )
    term_weights = (
        _np.asarray(
            [float(context.term_weights.get(term, 0.0)) for term in context.score_terms],
            dtype=_np.float64,
        )
        if context.term_weights is not None
        else None
    )

    macro_precision_sum = _np.zeros(threshold_count, dtype=_np.float64)
    macro_prediction_count = _np.zeros(threshold_count, dtype=_np.int64)
    macro_recall_sum = _np.zeros(threshold_count, dtype=_np.float64)
    micro_predicted = _np.zeros(threshold_count, dtype=_np.float64)
    micro_true_positive = _np.zeros(threshold_count, dtype=_np.float64)
    truth_total = 0.0

    weighted_macro_precision_sum = _np.zeros(threshold_count, dtype=_np.float64)
    weighted_macro_prediction_count = _np.zeros(threshold_count, dtype=_np.int64)
    weighted_macro_recall_sum = _np.zeros(threshold_count, dtype=_np.float64)
    weighted_micro_predicted = _np.zeros(threshold_count, dtype=_np.float64)
    weighted_micro_true_positive = _np.zeros(threshold_count, dtype=_np.float64)
    remaining_uncertainty = _np.zeros(threshold_count, dtype=_np.float64)
    misinformation = _np.zeros(threshold_count, dtype=_np.float64)
    truth_weight_total = 0.0
    weighted_protein_count = 0

    for protein_id in protein_ids:
        raw_scores = _np.asarray(predicted_scores[protein_id], dtype=_np.float64)
        if raw_scores.shape != (class_count,):
            raise ValueError("GO score rows must match the class count.")
        propagated_scores = _np.zeros(term_count, dtype=_np.float64)
        if term_count:
            _np.maximum.at(
                propagated_scores,
                context.ancestor_edge_indices,
                raw_scores[context.class_edge_indices],
            )

        inclusion = base_inclusion
        excluded = context.excluded_terms.get(protein_id)
        if excluded:
            inclusion = base_inclusion.copy()
            for term in excluded:
                term_index = term_indices.get(term)
                if term_index is not None:
                    inclusion[term_index] = False
        included_indices = all_term_indices[inclusion]
        activation_indices = _np.searchsorted(
            threshold_values,
            propagated_scores[included_indices],
            side="right",
        ) - 1
        activated = activation_indices >= 0
        activation_indices = activation_indices[activated]
        active_term_indices = included_indices[activated]

        labels = context.truth[protein_id]
        truth_count = float(len(labels))
        truth_total += truth_count
        true_mask = _np.zeros(term_count, dtype=bool)
        for term in labels:
            term_index = term_indices.get(term)
            if term_index is not None:
                true_mask[term_index] = True
        predicted_counts = _reverse_cumulative_histogram(
            activation_indices,
            size=threshold_count,
        )
        true_positive_counts = _reverse_cumulative_histogram(
            activation_indices,
            size=threshold_count,
            weights=true_mask[active_term_indices],
        )
        has_prediction = predicted_counts > 0.0
        macro_precision_sum[has_prediction] += (
            true_positive_counts[has_prediction] / predicted_counts[has_prediction]
        )
        macro_prediction_count += has_prediction
        macro_recall_sum += true_positive_counts / truth_count
        micro_predicted += predicted_counts
        micro_true_positive += true_positive_counts

        if term_weights is None:
            continue
        truth_weight = _term_weight_sum(labels, term_weights=context.term_weights or {})
        if truth_weight <= 0.0:
            continue
        weighted_protein_count += 1
        truth_weight_total += truth_weight
        predicted_weights = _reverse_cumulative_histogram(
            activation_indices,
            size=threshold_count,
            weights=term_weights[active_term_indices],
        )
        true_positive_weights = _reverse_cumulative_histogram(
            activation_indices,
            size=threshold_count,
            weights=term_weights[active_term_indices] * true_mask[active_term_indices],
        )
        has_weighted_prediction = predicted_weights > 0.0
        weighted_macro_precision_sum[has_weighted_prediction] += (
            true_positive_weights[has_weighted_prediction]
            / predicted_weights[has_weighted_prediction]
        )
        weighted_macro_prediction_count += has_weighted_prediction
        weighted_macro_recall_sum += true_positive_weights / truth_weight
        weighted_micro_predicted += predicted_weights
        weighted_micro_true_positive += true_positive_weights
        remaining_uncertainty += truth_weight - true_positive_weights
        misinformation += predicted_weights - true_positive_weights

    protein_count = len(protein_ids)
    macro_precision = _np.divide(
        macro_precision_sum,
        macro_prediction_count,
        out=_np.zeros(threshold_count, dtype=_np.float64),
        where=macro_prediction_count != 0,
    )
    micro_precision = _np.divide(
        micro_true_positive,
        micro_predicted,
        out=_np.zeros(threshold_count, dtype=_np.float64),
        where=micro_predicted != 0.0,
    )
    weighted_macro_precision = _np.divide(
        weighted_macro_precision_sum,
        weighted_macro_prediction_count,
        out=_np.zeros(threshold_count, dtype=_np.float64),
        where=weighted_macro_prediction_count != 0,
    )
    weighted_micro_precision = _np.divide(
        weighted_micro_true_positive,
        weighted_micro_predicted,
        out=_np.zeros(threshold_count, dtype=_np.float64),
        where=weighted_micro_predicted != 0.0,
    )
    return _GoThresholdStatistics(
        protein_count=protein_count,
        macro_precision=macro_precision,
        macro_recall=macro_recall_sum / float(protein_count),
        micro_precision=micro_precision,
        micro_recall=micro_true_positive / truth_total,
        weighted_protein_count=weighted_protein_count,
        weighted_macro_precision=weighted_macro_precision,
        weighted_macro_recall=(
            weighted_macro_recall_sum / float(weighted_protein_count)
            if weighted_protein_count
            else weighted_macro_recall_sum
        ),
        weighted_micro_precision=weighted_micro_precision,
        weighted_micro_recall=(
            weighted_micro_true_positive / truth_weight_total
            if truth_weight_total
            else weighted_micro_true_positive
        ),
        remaining_uncertainty=(
            remaining_uncertainty / float(weighted_protein_count)
            if weighted_protein_count
            else remaining_uncertainty
        ),
        misinformation=(
            misinformation / float(weighted_protein_count)
            if weighted_protein_count
            else misinformation
        ),
    )


def _reverse_cumulative_histogram(
    indices: Any,
    *,
    size: int,
    weights: Any = None,
) -> Any:
    histogram = _np.bincount(indices, weights=weights, minlength=size).astype(
        _np.float64,
        copy=False,
    )
    return _np.cumsum(histogram[::-1])[::-1]


def _go_unweighted_metrics_from_statistics(
    statistics: _GoThresholdStatistics,
    *,
    indices: Any,
    thresholds: Sequence[float],
) -> dict[str, float]:
    precision = statistics.macro_precision[indices]
    recall = statistics.macro_recall[indices]
    f_scores = _f_score_array(precision, recall)
    best_index = int(_np.argmax(f_scores))
    micro_precision = statistics.micro_precision[indices]
    micro_recall = statistics.micro_recall[indices]
    micro_f_scores = _f_score_array(micro_precision, micro_recall)
    micro_best_index = int(_np.argmax(micro_f_scores))
    threshold = float(thresholds[best_index])
    micro_threshold = float(thresholds[micro_best_index])
    return {
        "go_fmax": float(f_scores[best_index]),
        "go_macro_fmax": float(f_scores[best_index]),
        "go_fmax_threshold": threshold,
        "go_macro_fmax_threshold": threshold,
        "go_precision_at_fmax": float(precision[best_index]),
        "go_recall_at_fmax": float(recall[best_index]),
        "go_macro_precision_at_fmax": float(precision[best_index]),
        "go_macro_recall_at_fmax": float(recall[best_index]),
        "go_evaluated_protein_count": float(statistics.protein_count),
        "go_micro_fmax": float(micro_f_scores[micro_best_index]),
        "go_micro_fmax_threshold": micro_threshold,
        "go_micro_precision_at_fmax": float(micro_precision[micro_best_index]),
        "go_micro_recall_at_fmax": float(micro_recall[micro_best_index]),
    }


def _go_fixed_threshold_metrics_from_statistics(
    statistics: _GoThresholdStatistics,
    *,
    index: int,
    threshold: float,
) -> dict[str, float]:
    precision = float(statistics.macro_precision[index])
    recall = float(statistics.macro_recall[index])
    return {
        "go_propagated_f1": _f_score(precision, recall),
        "go_propagated_precision": precision,
        "go_propagated_recall": recall,
        "go_propagated_threshold": threshold,
        "go_propagated_evaluated_protein_count": float(statistics.protein_count),
    }


def _go_weighted_metrics_from_statistics(
    statistics: _GoThresholdStatistics,
    *,
    indices: Any,
    thresholds: Sequence[float],
) -> dict[str, float]:
    precision = statistics.weighted_macro_precision[indices]
    recall = statistics.weighted_macro_recall[indices]
    f_scores = _f_score_array(precision, recall)
    best_index = int(_np.argmax(f_scores))
    micro_precision = statistics.weighted_micro_precision[indices]
    micro_recall = statistics.weighted_micro_recall[indices]
    micro_f_scores = _f_score_array(micro_precision, micro_recall)
    micro_best_index = int(_np.argmax(micro_f_scores))
    remaining_uncertainty = statistics.remaining_uncertainty[indices]
    misinformation = statistics.misinformation[indices]
    semantic_distance = _np.sqrt(remaining_uncertainty**2 + misinformation**2)
    distance_best_index = int(_np.argmin(semantic_distance))
    threshold = float(thresholds[best_index])
    return {
        "go_weighted_fmax": float(f_scores[best_index]),
        "go_weighted_macro_fmax": float(f_scores[best_index]),
        "go_weighted_fmax_threshold": threshold,
        "go_weighted_macro_fmax_threshold": threshold,
        "go_weighted_precision_at_fmax": float(precision[best_index]),
        "go_weighted_recall_at_fmax": float(recall[best_index]),
        "go_weighted_macro_precision_at_fmax": float(precision[best_index]),
        "go_weighted_macro_recall_at_fmax": float(recall[best_index]),
        "go_weighted_evaluated_protein_count": float(statistics.weighted_protein_count),
        "go_weighted_micro_fmax": float(micro_f_scores[micro_best_index]),
        "go_weighted_micro_fmax_threshold": float(thresholds[micro_best_index]),
        "go_weighted_micro_precision_at_fmax": float(micro_precision[micro_best_index]),
        "go_weighted_micro_recall_at_fmax": float(micro_recall[micro_best_index]),
        "go_weighted_smin": float(semantic_distance[distance_best_index]),
        "go_weighted_smin_threshold": float(thresholds[distance_best_index]),
        "go_weighted_remaining_uncertainty_at_smin": float(
            remaining_uncertainty[distance_best_index]
        ),
        "go_weighted_misinformation_at_smin": float(misinformation[distance_best_index]),
    }


def _f_score_array(precision: Any, recall: Any) -> Any:
    return _np.divide(
        2.0 * precision * recall,
        precision + recall,
        out=_np.zeros_like(precision, dtype=_np.float64),
        where=(precision + recall) != 0.0,
    )


def _filter_go_score_rows(
    score_rows: Mapping[str, Mapping[str, float]],
    *,
    context: GoMetricContext,
) -> dict[str, Mapping[str, float]]:
    scores: dict[str, Mapping[str, float]] = {}
    for protein_id, score_row in score_rows.items():
        excluded = context.excluded_terms.get(protein_id, set())
        filtered_scores = {
            term: score
            for term, score in score_row.items()
            if term not in excluded
        }
        if context.interest_terms is not None:
            filtered_scores = {
                term: score
                for term, score in filtered_scores.items()
                if term in context.interest_terms
            }
        scores[protein_id] = filtered_scores
    return scores


def _prepare_go_score_inputs_from_context(
    predicted_scores: Mapping[str, Sequence[float]],
    *,
    context: GoMetricContext,
) -> _PreparedGoMetricInputs:
    if not predicted_scores:
        raise ValueError("GO metric inputs must contain at least one prediction.")
    if set(predicted_scores) != set(context.truth):
        raise ValueError("GO prediction and truth identifiers must match.")
    propagated_scores = _propagate_go_score_rows_from_context(predicted_scores, context=context)
    return _PreparedGoMetricInputs(
        truth=context.truth,
        scores=_filter_go_score_rows(propagated_scores, context=context),
    )


def _go_score_truth_matrices(  # pyright: ignore[reportUnusedFunction]
    predicted_scores: Mapping[str, Sequence[float]],
    *,
    context: GoMetricContext,
    protein_ids: Sequence[str],
) -> tuple[Any, Any, Any, Any]:
    if not predicted_scores:
        raise ValueError("GO metric inputs must contain at least one prediction.")
    if set(predicted_scores) != set(context.truth):
        raise ValueError("GO prediction and truth identifiers must match.")
    class_count = len(context.class_names)
    protein_count = len(protein_ids)
    term_count = len(context.score_terms)
    score_matrix = _np.zeros((protein_count, term_count), dtype=_np.float64)
    truth_matrix = _np.zeros((protein_count, term_count), dtype=bool)
    inclusion_matrix = _go_default_inclusion_matrix(
        protein_count=protein_count,
        context=context,
    )
    term_indices = {term: index for index, term in enumerate(context.score_terms)}
    truth_counts: list[float] = []
    for protein_index, protein_id in enumerate(protein_ids):
        raw_scores = _np.asarray([float(value) for value in predicted_scores[protein_id]], dtype=_np.float64)
        if raw_scores.shape != (class_count,):
            raise ValueError("GO score rows must match the class count.")
        if term_count:
            _np.maximum.at(
                score_matrix[protein_index],
                context.ancestor_edge_indices,
                raw_scores[context.class_edge_indices],
            )
        labels = context.truth[protein_id]
        truth_counts.append(float(len(labels)))
        for term in labels:
            term_index = term_indices.get(term)
            if term_index is not None:
                truth_matrix[protein_index, term_index] = True
        excluded = context.excluded_terms.get(protein_id)
        if excluded:
            for term in excluded:
                term_index = term_indices.get(term)
                if term_index is not None:
                    inclusion_matrix[protein_index, term_index] = False
    return (
        score_matrix,
        truth_matrix,
        inclusion_matrix,
        _np.asarray(truth_counts, dtype=_np.float64),
    )


def _go_default_inclusion_matrix(
    *,
    protein_count: int,
    context: GoMetricContext,
) -> Any:
    if context.interest_terms is None:
        return _np.ones((protein_count, len(context.score_terms)), dtype=bool)
    interest_mask = _np.asarray(
        [term in context.interest_terms for term in context.score_terms],
        dtype=bool,
    )
    return _np.broadcast_to(interest_mask, (protein_count, interest_mask.size)).copy()


def _go_score_propagation_edges(
    class_names: Sequence[str],
    *,
    ancestor_map: Mapping[str, Sequence[str]],
) -> tuple[tuple[str, ...], Any, Any]:
    term_indices: dict[str, int] = {}
    score_terms: list[str] = []
    class_edge_indices: list[int] = []
    ancestor_edge_indices: list[int] = []
    for class_index, class_name in enumerate(class_names):
        for ancestor in ancestor_map.get(class_name, ()):
            ancestor_index = term_indices.get(ancestor)
            if ancestor_index is None:
                ancestor_index = len(score_terms)
                term_indices[ancestor] = ancestor_index
                score_terms.append(ancestor)
            class_edge_indices.append(class_index)
            ancestor_edge_indices.append(ancestor_index)
    return (
        tuple(score_terms),
        _np.asarray(class_edge_indices, dtype=_np.int64),
        _np.asarray(ancestor_edge_indices, dtype=_np.int64),
    )


def _propagate_go_score_rows_from_context(
    predicted_scores: Mapping[str, Sequence[float]],
    *,
    context: GoMetricContext,
) -> dict[str, Mapping[str, float]]:
    propagated_scores: dict[str, Mapping[str, float]] = {}
    class_count = len(context.class_names)
    term_count = len(context.score_terms)
    for protein_id, scores in predicted_scores.items():
        score_row = _np.asarray([float(value) for value in scores], dtype=float)
        if score_row.shape != (class_count,):
            raise ValueError("GO score rows must match the class count.")
        propagated_row = _np.zeros(term_count, dtype=float)
        if term_count:
            _np.maximum.at(
                propagated_row,
                context.ancestor_edge_indices,
                score_row[context.class_edge_indices],
            )
        propagated_scores[protein_id] = {
            term: float(propagated_row[index])
            for index, term in enumerate(context.score_terms)
        }
    return propagated_scores


def _select_score_rows(  # pyright: ignore[reportUnusedFunction]
    propagated_scores: Mapping[str, Mapping[str, float]],
    propagated_truth: Mapping[str, set[str]],
) -> dict[str, Mapping[str, float]]:
    return {
        protein_id: propagated_scores[protein_id]
        for protein_id in propagated_truth
    }


def _score_thresholds(threshold_count: int, *, metric_name: str) -> list[float]:
    count = int(threshold_count)
    if count < 2:
        raise ValueError(f"{metric_name} threshold_count must be at least 2.")
    return [index / float(count - 1) for index in range(count)]


def _f_score(precision: float, recall: float) -> float:
    return 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0


def _unique_go_terms_from_label_rows(label_rows: Iterable[Sequence[str]]) -> list[str]:
    terms: dict[str, None] = {}
    for labels in label_rows:
        for term in labels:
            normalized = str(term).strip()
            if normalized:
                terms[normalized] = None
    return list(terms)


def _go_non_root_ancestor_map(
    terms: Sequence[str],
    *,
    ontology: GOOntology,
    non_root_cache: dict[str, bool] | None = None,
) -> dict[str, tuple[str, ...]]:
    cache = non_root_cache if non_root_cache is not None else {}
    ancestor_map: dict[str, tuple[str, ...]] = {}
    for term in terms:
        normalized = str(term).strip()
        if not normalized or normalized in ancestor_map:
            continue
        ancestor_map[normalized] = tuple(
            ancestor
            for ancestor in ontology.ancestors(normalized, include_self=True)
            if _is_non_root_go_term_cached(ancestor, ontology=ontology, cache=cache)
        )
    return ancestor_map


def _propagate_go_terms_from_map(
    terms: Sequence[str],
    *,
    ancestor_map: Mapping[str, Sequence[str]],
) -> set[str]:
    propagated: set[str] = set()
    for term in terms:
        normalized = str(term).strip()
        if normalized:
            propagated.update(ancestor_map.get(normalized, ()))
    return propagated


def _is_non_root_go_term_cached(
    term: str,
    *,
    ontology: GOOntology,
    cache: dict[str, bool],
) -> bool:
    if term not in cache:
        cache[term] = _is_non_root_go_term(term, ontology)
    return cache[term]


def _is_non_root_go_term(term: str, ontology: GOOntology) -> bool:
    return bool(ontology.direct_parents(term))


def _term_weight_sum(terms: Sequence[str] | set[str], *, term_weights: Mapping[str, float]) -> float:
    return sum(float(term_weights.get(term, 0.0)) for term in terms)


def _metrics_for_aspect(metrics_by_aspect: Mapping[str, Mapping[str, float]], aspect: str) -> Mapping[str, float]:
    for key in (
        aspect,
        f"go_{aspect}",
        f"cafa5_{aspect}",
        f"cafa6_{aspect}",
        f"cafa:cafa5_{aspect}",
        f"cafa:cafa6_{aspect}",
        f"cafa5:cafa5_{aspect}",
        f"cafa6:cafa6_{aspect}",
    ):
        metrics = metrics_by_aspect.get(key)
        if metrics is not None:
            return metrics
    raise ValueError(f"CAFA6 metrics are missing aspect {aspect!r}.")


def _binary_ranking_metrics(y_true: Any, y_score: Any) -> tuple[float, float]:
    true_values = _np.asarray(y_true, dtype=_np.int8)
    score_values = _np.asarray(y_score, dtype=_np.float64)
    positives = int(_np.count_nonzero(true_values == 1))
    negatives = int(true_values.size) - positives
    order = _np.argsort(score_values, kind="stable")
    sorted_scores = score_values[order]
    sorted_true = true_values[order]

    auroc = 0.0
    if positives > 0 and negatives > 0:
        group_starts = _np.flatnonzero(
            _np.concatenate(
                (_np.asarray([True]), sorted_scores[:-1] != sorted_scores[1:])
            )
        )
        group_ends = _np.concatenate(
            (group_starts[1:], _np.asarray([true_values.size]))
        )
        average_ranks = (group_starts + 1 + group_ends) / 2.0
        positive_counts = _np.add.reduceat(
            (sorted_true == 1).astype(_np.int64, copy=False),
            group_starts,
        )
        rank_sum = float(_np.sum(average_ranks * positive_counts))
        auroc = (rank_sum - positives * (positives + 1) / 2.0) / float(
            positives * negatives
        )

    if positives == 0:
        return auroc, 0.0
    descending_scores = sorted_scores[::-1]
    descending_true = sorted_true[::-1]
    cumulative_true = _np.cumsum(descending_true, dtype=_np.int64)
    group_ends = _np.flatnonzero(
        _np.concatenate(
            (
                descending_scores[:-1] != descending_scores[1:],
                _np.asarray([True]),
            )
        )
    )
    true_at_group_end = cumulative_true[group_ends].astype(_np.float64)
    true_by_group = _np.diff(
        _np.concatenate((_np.asarray([0.0]), true_at_group_end))
    )
    precision = true_at_group_end / (group_ends + 1)
    auprc = float(_np.sum((true_by_group / float(positives)) * precision))
    return auroc, auprc


def _binary_auprc_array(y_true: Any, y_score: Any) -> float:
    true_values = _np.asarray(y_true, dtype=_np.int8)
    positives = int(_np.sum(true_values))
    if positives == 0:
        return 0.0
    score_values = _np.asarray(y_score, dtype=_np.float64)
    order = _np.argsort(-score_values, kind="stable")
    sorted_scores = score_values[order]
    sorted_true = true_values[order]
    cumulative_true = _np.cumsum(sorted_true, dtype=_np.int64)
    group_ends = _np.flatnonzero(
        _np.concatenate((sorted_scores[:-1] != sorted_scores[1:], _np.asarray([True])))
    )
    true_at_group_end = cumulative_true[group_ends].astype(_np.float64)
    true_by_group = _np.diff(_np.concatenate((_np.asarray([0.0]), true_at_group_end)))
    precision = true_at_group_end / (group_ends + 1)
    return float(_np.sum((true_by_group / float(positives)) * precision))


def _spearmanr_arrays(
    true_values: _npt.NDArray[_np.float64],
    pred_values: _npt.NDArray[_np.float64],
) -> float:
    true_ranks = _average_ranks_array(true_values)
    pred_ranks = _average_ranks_array(pred_values)
    centered_true = true_ranks - _np.mean(true_ranks)
    centered_pred = pred_ranks - _np.mean(pred_ranks)
    numerator = float(cast(float, _np.dot(centered_true, centered_pred)))
    true_sum_squares = float(cast(float, _np.dot(centered_true, centered_true)))
    pred_sum_squares = float(cast(float, _np.dot(centered_pred, centered_pred)))
    denominator = math.sqrt(true_sum_squares * pred_sum_squares)
    return numerator / denominator if denominator > 0.0 else 0.0


def _average_ranks_array(
    values: _npt.NDArray[_np.float64],
) -> _npt.NDArray[_np.float64]:
    order = _np.argsort(values, kind="stable")
    sorted_values = values[order]
    group_starts = _np.flatnonzero(
        _np.concatenate(
            (_np.asarray([True]), sorted_values[:-1] != sorted_values[1:])
        )
    )
    group_ends = _np.concatenate(
        (group_starts[1:], _np.asarray([values.size]))
    )
    average_ranks = (group_starts + 1 + group_ends) / 2.0
    ranks = _np.empty(values.size, dtype=_np.float64)
    ranks[order] = _np.repeat(average_ranks, group_ends - group_starts)
    return ranks


def _require_same_non_empty_length(a: Sequence[object], b: Sequence[object]) -> None:
    if len(a) < 1:
        raise ValueError("Metric inputs must be non-empty.")
    if len(a) != len(b):
        raise ValueError("Metric inputs must have the same length.")


__all__ = [
    "GoMetricContext",
    "binary_metrics",
    "cafa6_weighted_fmax_mean",
    "evaluate_go_metric_context",
    "go_combined_protein_centric_metrics",
    "go_fixed_threshold_protein_centric_metrics",
    "go_protein_centric_metrics",
    "go_weighted_protein_centric_metrics",
    "multiclass_metrics",
    "multilabel_metrics",
    "prepare_go_metric_context",
    "read_information_accretion_weights",
    "regression_metrics",
    "spearmanr",
]
