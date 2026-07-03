"""Metric helpers for probe evaluation."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import math
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from CBBIO.GO import GOOntology


@dataclass(frozen=True)
class _PreparedGoMetricInputs:
    truth: Mapping[str, set[str]]
    scores: Mapping[str, Mapping[str, float]]


@dataclass(frozen=True)
class _ProteinCentricThresholdMetrics:
    f_score: float
    precision: float
    recall: float


def regression_metrics(y_true: Sequence[float], y_pred: Sequence[float]) -> dict[str, float]:
    """Compute regression metrics for true and predicted values."""
    _require_same_non_empty_length(y_true, y_pred)
    count = float(len(y_true))
    errors = [float(pred) - float(true) for true, pred in zip(y_true, y_pred)]
    mae = sum(abs(error) for error in errors) / count
    rmse = math.sqrt(sum(error * error for error in errors) / count)
    mean_true = sum(float(value) for value in y_true) / count
    ss_tot = sum((float(value) - mean_true) ** 2 for value in y_true)
    ss_res = sum(error * error for error in errors)
    r2 = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
    return {"mae": mae, "rmse": rmse, "r2": r2, "spearmanr": spearmanr(y_true, y_pred)}


def spearmanr(y_true: Sequence[float], y_pred: Sequence[float]) -> float:
    """Spearman rank correlation with average ranks for ties."""

    _require_same_non_empty_length(y_true, y_pred)
    true_ranks = _average_ranks([float(value) for value in y_true])
    pred_ranks = _average_ranks([float(value) for value in y_pred])
    return _pearsonr(true_ranks, pred_ranks)


def binary_metrics(y_true: Sequence[int], y_pred: Sequence[int], y_score: Sequence[float]) -> dict[str, float]:
    """Compute binary classification metrics for labels, predictions, and scores."""
    _require_same_non_empty_length(y_true, y_pred)
    _require_same_non_empty_length(y_true, y_score)
    tp = sum(1 for true, pred in zip(y_true, y_pred) if int(true) == 1 and int(pred) == 1)
    tn = sum(1 for true, pred in zip(y_true, y_pred) if int(true) == 0 and int(pred) == 0)
    fp = sum(1 for true, pred in zip(y_true, y_pred) if int(true) == 0 and int(pred) == 1)
    fn = sum(1 for true, pred in zip(y_true, y_pred) if int(true) == 1 and int(pred) == 0)
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
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "macro_f1": macro_f1,
        "balanced_accuracy": balanced_accuracy,
        "mcc": mcc,
        "auroc": _binary_auroc(y_true, y_score),
        "auprc": _binary_auprc(y_true, y_score),
    }


def multiclass_metrics(y_true: Sequence[int], y_pred: Sequence[int], *, class_count: int) -> dict[str, float]:
    """Compute multiclass classification metrics for labels and predictions."""
    _require_same_non_empty_length(y_true, y_pred)
    # Build confusion matrix in O(N) instead of the previous O(class_count × N) triple scan.
    conf = [[0] * class_count for _ in range(class_count)]
    for t, p in zip(y_true, y_pred):
        conf[int(t)][int(p)] += 1
    correct = sum(conf[c][c] for c in range(class_count))
    accuracy = float(correct) / float(len(y_true))
    f1_values: list[float] = []
    supports: list[int] = []
    for c in range(class_count):
        tp = conf[c][c]
        fp = sum(conf[r][c] for r in range(class_count)) - tp
        fn = sum(conf[c][r] for r in range(class_count)) - tp
        precision = float(tp) / float(tp + fp) if tp + fp else 0.0
        recall = float(tp) / float(tp + fn) if tp + fn else 0.0
        f1_values.append(2.0 * precision * recall / (precision + recall) if precision + recall else 0.0)
        supports.append(sum(conf[c]))
    macro_f1 = sum(f1_values) / float(len(f1_values)) if f1_values else 0.0
    support_total = sum(supports)
    weighted_f1 = (
        sum(f1 * support for f1, support in zip(f1_values, supports)) / float(support_total)
        if support_total
        else 0.0
    )
    return {"accuracy": accuracy, "macro_f1": macro_f1, "weighted_f1": weighted_f1}


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

    exact_match = sum(
        1 for true_row, pred_row in zip(y_true, y_pred)
        if [int(value) for value in true_row] == [int(value) for value in pred_row]
    ) / float(len(y_true))
    macro_f1_values: list[float] = []
    supports: list[int] = []
    average_precision_values: list[float] = []
    micro_tp = 0
    micro_fp = 0
    micro_fn = 0
    for class_index in range(class_count):
        true_values = [int(row[class_index]) for row in y_true]
        pred_values = [int(row[class_index]) for row in y_pred]
        score_values = [float(row[class_index]) for row in y_score]
        tp = sum(1 for true, pred in zip(true_values, pred_values) if true == 1 and pred == 1)
        fp = sum(1 for true, pred in zip(true_values, pred_values) if true == 0 and pred == 1)
        fn = sum(1 for true, pred in zip(true_values, pred_values) if true == 1 and pred == 0)
        precision = float(tp) / float(tp + fp) if tp + fp else 0.0
        recall = float(tp) / float(tp + fn) if tp + fn else 0.0
        macro_f1_values.append(2.0 * precision * recall / (precision + recall) if precision + recall else 0.0)
        supports.append(sum(true_values))
        average_precision_values.append(_binary_auprc(true_values, score_values))
        micro_tp += tp
        micro_fp += fp
        micro_fn += fn
    micro_precision = float(micro_tp) / float(micro_tp + micro_fp) if micro_tp + micro_fp else 0.0
    micro_recall = float(micro_tp) / float(micro_tp + micro_fn) if micro_tp + micro_fn else 0.0
    micro_f1 = (
        2.0 * micro_precision * micro_recall / (micro_precision + micro_recall)
        if micro_precision + micro_recall
        else 0.0
    )
    support_total = sum(supports)
    weighted_f1 = (
        sum(f1 * support for f1, support in zip(macro_f1_values, supports)) / float(support_total)
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
        "macro_f1": sum(macro_f1_values) / float(class_count),
        "weighted_f1": weighted_f1,
        "average_precision": sum(average_precision_values) / float(class_count),
        **ranking_metrics,
    }


def _multilabel_fmax(
    *,
    y_true: Sequence[Sequence[int]],
    y_score: Sequence[Sequence[float]],
    thresholds: Sequence[float],
) -> dict[str, float]:
    best = {
        "fmax": -1.0,
        "fmax_threshold": 0.0,
        "precision_at_fmax": 0.0,
        "recall_at_fmax": 0.0,
    }
    for threshold in thresholds:
        true_positive = 0
        false_positive = 0
        false_negative = 0
        for true_row, score_row in zip(y_true, y_score):
            for true_value, score in zip(true_row, score_row):
                predicted = float(score) >= threshold
                actual = int(true_value) == 1
                if predicted and actual:
                    true_positive += 1
                elif predicted:
                    false_positive += 1
                elif actual:
                    false_negative += 1
        precision = (
            float(true_positive) / float(true_positive + false_positive)
            if true_positive + false_positive
            else 0.0
        )
        recall = (
            float(true_positive) / float(true_positive + false_negative)
            if true_positive + false_negative
            else 0.0
        )
        f_score = _f_score(precision, recall)
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
    prepared = _prepare_go_metric_inputs(
        predicted_scores,
        class_names=class_names,
        true_labels=true_labels,
        ontology=ontology,
    )
    propagated_truth = {
        protein_id: labels
        for protein_id, labels in prepared.truth.items()
        if labels
    }
    if not propagated_truth:
        raise ValueError("GO truth labels contain no non-root terms.")
    propagated_scores = _select_score_rows(prepared.scores, propagated_truth)
    thresholds = _score_thresholds(threshold_count, metric_name="GO")
    return _go_unweighted_fmax(
        propagated_truth=propagated_truth,
        propagated_scores=propagated_scores,
        thresholds=thresholds,
    )


def go_combined_protein_centric_metrics(
    predicted_scores: Mapping[str, Sequence[float]],
    *,
    class_names: Sequence[str],
    true_labels: Mapping[str, Sequence[str]],
    ontology: GOOntology,
    term_weights: Mapping[str, float] | None = None,
    threshold_count: int = 101,
) -> dict[str, float]:
    """Compute unweighted and optional IA-weighted GO F-max with shared propagation."""
    prepared = _prepare_go_metric_inputs(
        predicted_scores,
        class_names=class_names,
        true_labels=true_labels,
        ontology=ontology,
    )
    propagated_truth = {
        protein_id: labels
        for protein_id, labels in prepared.truth.items()
        if labels
    }
    if not propagated_truth:
        raise ValueError("GO truth labels contain no non-root terms.")
    thresholds = _score_thresholds(threshold_count, metric_name="GO")
    metrics = _go_unweighted_fmax(
        propagated_truth=propagated_truth,
        propagated_scores=_select_score_rows(prepared.scores, propagated_truth),
        thresholds=thresholds,
    )
    if term_weights is None:
        return metrics
    if not term_weights:
        raise ValueError("GO weighted metrics require information-accretion weights.")
    weighted_truth = {
        protein_id: labels
        for protein_id, labels in propagated_truth.items()
        if _term_weight_sum(labels, term_weights=term_weights) > 0.0
    }
    if not weighted_truth:
        raise ValueError("GO truth labels contain no positive-weight non-root terms.")
    metrics.update(
        _go_weighted_fmax(
            propagated_truth=weighted_truth,
            propagated_scores=_select_score_rows(prepared.scores, weighted_truth),
            term_weights=term_weights,
            thresholds=thresholds,
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
) -> dict[str, float]:
    """Compute propagated protein-centric GO F1 at one score threshold."""
    if not 0.0 <= float(threshold) <= 1.0:
        raise ValueError("GO fixed-threshold metrics require threshold between 0 and 1.")
    prepared = _prepare_go_metric_inputs(
        predicted_scores,
        class_names=class_names,
        true_labels=true_labels,
        ontology=ontology,
    )
    propagated_truth = {
        protein_id: labels
        for protein_id, labels in prepared.truth.items()
        if labels
    }
    if not propagated_truth:
        raise ValueError("GO truth labels contain no non-root terms.")
    propagated_scores = _select_score_rows(prepared.scores, propagated_truth)
    return _go_unweighted_fixed_threshold(
        propagated_truth=propagated_truth,
        propagated_scores=propagated_scores,
        threshold=float(threshold),
    )


def _go_unweighted_fmax(
    *,
    propagated_truth: Mapping[str, set[str]],
    propagated_scores: Mapping[str, Mapping[str, float]],
    thresholds: Sequence[float],
) -> dict[str, float]:
    best = {
        "go_fmax": -1.0,
        "go_fmax_threshold": 0.0,
        "go_precision_at_fmax": 0.0,
        "go_recall_at_fmax": 0.0,
        "go_evaluated_protein_count": float(len(propagated_truth)),
    }
    for threshold in thresholds:
        threshold_metrics = _go_unweighted_at_threshold(
            propagated_truth=propagated_truth,
            propagated_scores=propagated_scores,
            threshold=threshold,
        )
        if threshold_metrics.f_score > best["go_fmax"]:
            best = {
                "go_fmax": threshold_metrics.f_score,
                "go_fmax_threshold": threshold,
                "go_precision_at_fmax": threshold_metrics.precision,
                "go_recall_at_fmax": threshold_metrics.recall,
                "go_evaluated_protein_count": float(len(propagated_truth)),
            }
    return best


def _go_unweighted_fixed_threshold(
    *,
    propagated_truth: Mapping[str, set[str]],
    propagated_scores: Mapping[str, Mapping[str, float]],
    threshold: float,
) -> dict[str, float]:
    threshold_metrics = _go_unweighted_at_threshold(
        propagated_truth=propagated_truth,
        propagated_scores=propagated_scores,
        threshold=threshold,
    )
    return {
        "go_propagated_f1": threshold_metrics.f_score,
        "go_propagated_precision": threshold_metrics.precision,
        "go_propagated_recall": threshold_metrics.recall,
        "go_propagated_threshold": threshold,
        "go_propagated_evaluated_protein_count": float(len(propagated_truth)),
    }


def _go_unweighted_at_threshold(
    *,
    propagated_truth: Mapping[str, set[str]],
    propagated_scores: Mapping[str, Mapping[str, float]],
    threshold: float,
) -> _ProteinCentricThresholdMetrics:
    precision_values: list[float] = []
    recall_values: list[float] = []
    for protein_id, truth in propagated_truth.items():
        predicted = _terms_at_threshold(propagated_scores[protein_id], threshold=threshold)
        intersection = predicted.intersection(truth)
        if predicted:
            precision_values.append(len(intersection) / float(len(predicted)))
        recall_values.append(len(intersection) / float(len(truth)))
    precision = sum(precision_values) / float(len(precision_values)) if precision_values else 0.0
    recall = sum(recall_values) / float(len(recall_values))
    return _ProteinCentricThresholdMetrics(
        f_score=_f_score(precision, recall),
        precision=precision,
        recall=recall,
    )


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
    prepared = _prepare_go_metric_inputs(
        predicted_scores,
        class_names=class_names,
        true_labels=true_labels,
        ontology=ontology,
    )
    propagated_truth = {
        protein_id: labels
        for protein_id, labels in prepared.truth.items()
        if _term_weight_sum(labels, term_weights=term_weights) > 0.0
    }
    if not propagated_truth:
        raise ValueError("GO truth labels contain no positive-weight non-root terms.")
    return _go_weighted_fmax(
        propagated_truth=propagated_truth,
        propagated_scores=_select_score_rows(prepared.scores, propagated_truth),
        term_weights=term_weights,
        thresholds=_score_thresholds(threshold_count, metric_name="GO"),
    )


def _go_weighted_fmax(
    *,
    propagated_truth: Mapping[str, set[str]],
    propagated_scores: Mapping[str, Mapping[str, float]],
    term_weights: Mapping[str, float],
    thresholds: Sequence[float],
) -> dict[str, float]:
    best = {
        "go_weighted_fmax": -1.0,
        "go_weighted_fmax_threshold": 0.0,
        "go_weighted_precision_at_fmax": 0.0,
        "go_weighted_recall_at_fmax": 0.0,
        "go_weighted_evaluated_protein_count": float(len(propagated_truth)),
    }
    for threshold in thresholds:
        threshold_metrics = _go_weighted_at_threshold(
            propagated_truth=propagated_truth,
            propagated_scores=propagated_scores,
            term_weights=term_weights,
            threshold=threshold,
        )
        if threshold_metrics.f_score > best["go_weighted_fmax"]:
            best = {
                "go_weighted_fmax": threshold_metrics.f_score,
                "go_weighted_fmax_threshold": threshold,
                "go_weighted_precision_at_fmax": threshold_metrics.precision,
                "go_weighted_recall_at_fmax": threshold_metrics.recall,
                "go_weighted_evaluated_protein_count": float(len(propagated_truth)),
            }
    return best


def _go_weighted_at_threshold(
    *,
    propagated_truth: Mapping[str, set[str]],
    propagated_scores: Mapping[str, Mapping[str, float]],
    term_weights: Mapping[str, float],
    threshold: float,
) -> _ProteinCentricThresholdMetrics:
    precision_values: list[float] = []
    recall_values: list[float] = []
    for protein_id, truth in propagated_truth.items():
        predicted = _terms_at_threshold(propagated_scores[protein_id], threshold=threshold)
        intersection_weight = _term_weight_sum(
            predicted.intersection(truth),
            term_weights=term_weights,
        )
        predicted_weight = _term_weight_sum(predicted, term_weights=term_weights)
        truth_weight = _term_weight_sum(truth, term_weights=term_weights)
        if predicted_weight > 0.0:
            precision_values.append(intersection_weight / predicted_weight)
        recall_values.append(intersection_weight / truth_weight if truth_weight > 0.0 else 0.0)
    precision = sum(precision_values) / float(len(precision_values)) if precision_values else 0.0
    recall = sum(recall_values) / float(len(recall_values))
    return _ProteinCentricThresholdMetrics(
        f_score=_f_score(precision, recall),
        precision=precision,
        recall=recall,
    )


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


def _propagate_go_terms(terms: Sequence[str], ontology: GOOntology) -> set[str]:
    propagated: set[str] = set()
    for term in terms:
        normalized = str(term).strip()
        if not normalized:
            continue
        propagated.update(ontology.ancestors(normalized, include_self=True))
    propagated = {term for term in propagated if _is_non_root_go_term(term, ontology)}
    return propagated


def _prepare_go_metric_inputs(
    predicted_scores: Mapping[str, Sequence[float]],
    *,
    class_names: Sequence[str],
    true_labels: Mapping[str, Sequence[str]],
    ontology: GOOntology,
) -> _PreparedGoMetricInputs:
    if not predicted_scores:
        raise ValueError("GO metric inputs must contain at least one prediction.")
    if not class_names:
        raise ValueError("GO metrics require at least one predicted GO class.")
    if set(predicted_scores) != set(true_labels):
        raise ValueError("GO prediction and truth identifiers must match.")
    propagated_truth = {
        protein_id: _propagate_go_terms(labels, ontology)
        for protein_id, labels in true_labels.items()
    }
    propagated_scores = {
        protein_id: _propagate_go_scores(scores, class_names=class_names, ontology=ontology)
        for protein_id, scores in predicted_scores.items()
    }
    return _PreparedGoMetricInputs(truth=propagated_truth, scores=propagated_scores)


def _select_score_rows(
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


def _terms_at_threshold(scores: Mapping[str, float], *, threshold: float) -> set[str]:
    return {
        term
        for term, score in scores.items()
        if score >= threshold
    }


def _f_score(precision: float, recall: float) -> float:
    return 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0


def _propagate_go_scores(
    scores: Sequence[float],
    *,
    class_names: Sequence[str],
    ontology: GOOntology,
) -> dict[str, float]:
    if len(scores) != len(class_names):
        raise ValueError("GO score rows must match the class count.")
    propagated: dict[str, float] = {}
    for term, value in zip(class_names, scores):
        score = float(value)
        for ancestor in ontology.ancestors(str(term), include_self=True):
            if not _is_non_root_go_term(ancestor, ontology):
                continue
            propagated[ancestor] = max(propagated.get(ancestor, 0.0), score)
    return propagated


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


def _binary_auroc(y_true: Sequence[int], y_score: Sequence[float]) -> float:
    positives = sum(1 for value in y_true if int(value) == 1)
    negatives = len(y_true) - positives
    if positives == 0 or negatives == 0:
        return 0.0

    pairs = sorted(
        ((float(score), int(label)) for label, score in zip(y_true, y_score)),
        key=lambda item: item[0],
    )
    rank_sum = 0.0
    index = 0
    while index < len(pairs):
        end = index + 1
        while end < len(pairs) and pairs[end][0] == pairs[index][0]:
            end += 1
        average_rank = (index + 1 + end) / 2.0
        rank_sum += average_rank * sum(1 for _score, label in pairs[index:end] if label == 1)
        index = end
    return (rank_sum - positives * (positives + 1) / 2.0) / float(positives * negatives)


def _binary_auprc(y_true: Sequence[int], y_score: Sequence[float]) -> float:
    positives = sum(1 for value in y_true if int(value) == 1)
    if positives == 0:
        return 0.0

    pairs = sorted(
        ((float(score), int(label)) for label, score in zip(y_true, y_score)),
        key=lambda item: item[0],
        reverse=True,
    )
    area = 0.0
    tp = 0
    fp = 0
    previous_recall = 0.0
    index = 0
    while index < len(pairs):
        end = index + 1
        while end < len(pairs) and pairs[end][0] == pairs[index][0]:
            end += 1
        for _score, label in pairs[index:end]:
            if label == 1:
                tp += 1
            else:
                fp += 1
        recall = float(tp) / float(positives)
        precision = float(tp) / float(tp + fp) if tp + fp else 1.0
        area += (recall - previous_recall) * precision
        previous_recall = recall
        index = end
    return area


def _average_ranks(values: Sequence[float]) -> list[float]:
    ordered = sorted((float(value), index) for index, value in enumerate(values))
    ranks = [0.0] * len(ordered)
    cursor = 0
    while cursor < len(ordered):
        end = cursor + 1
        while end < len(ordered) and ordered[end][0] == ordered[cursor][0]:
            end += 1
        average_rank = (cursor + 1 + end) / 2.0
        for _value, original_index in ordered[cursor:end]:
            ranks[original_index] = average_rank
        cursor = end
    return ranks


def _pearsonr(a: Sequence[float], b: Sequence[float]) -> float:
    _require_same_non_empty_length(a, b)
    count = float(len(a))
    mean_a = sum(float(value) for value in a) / count
    mean_b = sum(float(value) for value in b) / count
    centered_a = [float(value) - mean_a for value in a]
    centered_b = [float(value) - mean_b for value in b]
    numerator = sum(left * right for left, right in zip(centered_a, centered_b))
    denom_a = sum(value * value for value in centered_a)
    denom_b = sum(value * value for value in centered_b)
    denominator = math.sqrt(denom_a * denom_b)
    return numerator / denominator if denominator > 0.0 else 0.0


def _require_same_non_empty_length(a: Sequence[object], b: Sequence[object]) -> None:
    if not a:
        raise ValueError("Metric inputs must be non-empty.")
    if len(a) != len(b):
        raise ValueError("Metric inputs must have the same length.")


__all__ = [
    "binary_metrics",
    "cafa6_weighted_fmax_mean",
    "go_combined_protein_centric_metrics",
    "go_fixed_threshold_protein_centric_metrics",
    "go_protein_centric_metrics",
    "go_weighted_protein_centric_metrics",
    "multiclass_metrics",
    "multilabel_metrics",
    "read_information_accretion_weights",
    "regression_metrics",
    "spearmanr",
]
