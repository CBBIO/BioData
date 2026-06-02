"""Metric helpers for probe evaluation."""

from __future__ import annotations

import math
from typing import Dict, List, Sequence


def regression_metrics(y_true: Sequence[float], y_pred: Sequence[float]) -> Dict[str, float]:
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
    _require_same_non_empty_length(y_true, y_pred)
    _require_same_non_empty_length(y_true, y_score)
    tp = sum(1 for true, pred in zip(y_true, y_pred) if int(true) == 1 and int(pred) == 1)
    tn = sum(1 for true, pred in zip(y_true, y_pred) if int(true) == 0 and int(pred) == 0)
    fp = sum(1 for true, pred in zip(y_true, y_pred) if int(true) == 0 and int(pred) == 1)
    fn = sum(1 for true, pred in zip(y_true, y_pred) if int(true) == 1 and int(pred) == 0)
    accuracy = float(tp + tn) / float(len(y_true))
    precision = float(tp) / float(tp + fp) if tp + fp else 0.0
    recall = float(tp) / float(tp + fn) if tp + fn else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1, "auroc": _binary_auroc(y_true, y_score)}


def multiclass_metrics(y_true: Sequence[int], y_pred: Sequence[int], *, class_count: int) -> Dict[str, float]:
    _require_same_non_empty_length(y_true, y_pred)
    accuracy = sum(1 for true, pred in zip(y_true, y_pred) if int(true) == int(pred)) / float(len(y_true))
    f1_values: List[float] = []
    for class_index in range(class_count):
        tp = sum(1 for true, pred in zip(y_true, y_pred) if int(true) == class_index and int(pred) == class_index)
        fp = sum(1 for true, pred in zip(y_true, y_pred) if int(true) != class_index and int(pred) == class_index)
        fn = sum(1 for true, pred in zip(y_true, y_pred) if int(true) == class_index and int(pred) != class_index)
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
