"""Metric helpers for probe evaluation."""

from __future__ import annotations

import math
from typing import Dict, List, Sequence


def regression_metrics(y_true: Sequence[float], y_pred: Sequence[float]) -> Dict[str, float]:
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


def binary_metrics(y_true: Sequence[int], y_pred: Sequence[int], y_score: Sequence[float]) -> Dict[str, float]:
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


def multiclass_metrics(y_true: Sequence[int], y_pred: Sequence[int], *, class_count: int) -> Dict[str, float]:
    """Compute multiclass classification metrics for labels and predictions."""
    _require_same_non_empty_length(y_true, y_pred)
    # Build confusion matrix in O(N) instead of the previous O(class_count × N) triple scan.
    conf = [[0] * class_count for _ in range(class_count)]
    for t, p in zip(y_true, y_pred):
        conf[int(t)][int(p)] += 1
    correct = sum(conf[c][c] for c in range(class_count))
    accuracy = float(correct) / float(len(y_true))
    f1_values: List[float] = []
    for c in range(class_count):
        tp = conf[c][c]
        fp = sum(conf[r][c] for r in range(class_count)) - tp
        fn = sum(conf[c][r] for r in range(class_count)) - tp
        precision = float(tp) / float(tp + fp) if tp + fp else 0.0
        recall = float(tp) / float(tp + fn) if tp + fn else 0.0
        f1_values.append(2.0 * precision * recall / (precision + recall) if precision + recall else 0.0)
    macro_f1 = sum(f1_values) / float(len(f1_values)) if f1_values else 0.0
    return {"accuracy": accuracy, "macro_f1": macro_f1}


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


def _average_ranks(values: Sequence[float]) -> List[float]:
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


__all__ = ["binary_metrics", "multiclass_metrics", "regression_metrics", "spearmanr"]
