"""Extension interfaces for custom supervised probe backends."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, TypeAlias, cast

import numpy as _np
import numpy.typing as _npt

from CBBIO.embeddings import EmbeddingInputError

from .datasets import TaskLevel
from .metrics import binary_metrics, multiclass_metrics, multilabel_metrics, regression_metrics

if TYPE_CHECKING:
    from .tasks import PredictionSpec


ProbeFeature: TypeAlias = Sequence[float] | Sequence[Sequence[float]]
ProbeLabel: TypeAlias = object | Sequence[object]
ProbePrediction: TypeAlias = float | int | str
ProbePredictionOutput: TypeAlias = ProbePrediction | Sequence[ProbePrediction]
ProbeScoreRow: TypeAlias = Sequence[float] | _npt.NDArray[_np.float64]
ProbeScoreOutput: TypeAlias = float | ProbeScoreRow


@dataclass(frozen=True)
class ProbeBackendInput:
    """Canonical train/test data passed to a custom probe backend."""

    level: TaskLevel
    prediction: PredictionSpec
    train_ids: tuple[str, ...]
    test_ids: tuple[str, ...]
    embeddings: Mapping[str, ProbeFeature]
    labels: Mapping[str, ProbeLabel]
    masks: Mapping[str, Sequence[bool] | None]
    feature_mean: Any = None
    feature_std: Any = None
    flat_data: Any = None


@dataclass(frozen=True)
class ProbeBackendOutput:
    """Predictions and optional scores produced by a custom backend.

    Multilabel backends may return ``score_matrix`` instead of ``scores``. Its
    rows must follow ``ProbeBackendInput.test_ids`` and its columns must follow
    ``PredictionSpec.classes`` or the inferred sorted class order.
    """

    predictions: Mapping[str, ProbePredictionOutput]
    scores: Mapping[str, ProbeScoreOutput] | None = None
    metadata: Mapping[str, Any] | None = None
    score_matrix: Any = None


class ProbeBackend(ABC):
    """Fit a custom probe and predict the canonical test split."""

    @abstractmethod
    def fit_predict(self, data: ProbeBackendInput) -> ProbeBackendOutput:
        """Fit on ``data.train_ids`` and return predictions for ``data.test_ids``."""
        raise NotImplementedError


def evaluate_probe_backend_output(
    data: ProbeBackendInput,
    output: ProbeBackendOutput,
) -> tuple[dict[str, float], dict[str, ProbePredictionOutput], dict[str, ProbeScoreOutput] | None]:
    """Validate custom backend output and compute canonical probe metrics."""

    if output.score_matrix is not None and not (
        data.level == "protein" and data.prediction.objective == "multilabel"
    ):
        raise EmbeddingInputError(
            "Custom probe score_matrix is supported only for protein-level "
            "multilabel tasks."
        )
    if data.level == "protein":
        return _evaluate_protein_output(data, output)
    if data.prediction.objective == "multilabel":
        raise EmbeddingInputError("Custom residue probe backends do not support multilabel evaluation.")
    if data.level == "residue":
        return cast(
            tuple[dict[str, float], dict[str, ProbePredictionOutput], dict[str, ProbeScoreOutput] | None],
            _evaluate_residue_output(data, output),
        )
    raise EmbeddingInputError(f"Unsupported custom probe level: {data.level!r}.")


def _evaluate_protein_output(
    data: ProbeBackendInput,
    output: ProbeBackendOutput,
) -> tuple[dict[str, float], dict[str, ProbePredictionOutput], dict[str, ProbeScoreOutput] | None]:
    _require_output_ids(data.test_ids, output.predictions, name="predictions")
    if data.prediction.objective == "multilabel":
        return _evaluate_protein_multilabel_output(data, output)
    predictions = {
        item: _require_scalar(output.predictions[item], item=item, name="prediction")
        for item in data.test_ids
    }
    predictions = _normalize_predictions(data, predictions)
    scores = _protein_scores(data, output)
    truth = {
        item: _require_scalar(data.labels[item], item=item, name="label")
        for item in data.test_ids
    }
    metrics = _evaluate_predictions(
        data,
        truth=truth,
        predictions=predictions,
        scores=scores,
    )
    return (
        metrics,
        cast(dict[str, ProbePredictionOutput], predictions),
        cast(dict[str, ProbeScoreOutput] | None, scores),
    )


def _evaluate_protein_multilabel_output(
    data: ProbeBackendInput,
    output: ProbeBackendOutput,
) -> tuple[dict[str, float], dict[str, ProbePredictionOutput], dict[str, ProbeScoreOutput] | None]:
    classes = _multilabel_names(data)
    class_index = {name: index for index, name in enumerate(classes)}
    predictions: dict[str, ProbePredictionOutput] = {}
    true_matrix = _np.zeros((len(data.test_ids), len(classes)), dtype=_np.int8)
    pred_matrix = _np.zeros_like(true_matrix)
    score_matrix = _multilabel_score_matrix(
        data,
        output,
        class_count=len(classes),
    )
    scores: dict[str, ProbeScoreOutput] = {}
    for row_index, item in enumerate(data.test_ids):
        true_labels = _multilabel_values(data.labels[item], item=item, name="labels")
        predicted_labels = _multilabel_values(output.predictions[item], item=item, name="predictions")
        true_indices = _multilabel_indices(true_labels, item=item, indices=class_index, name="label")
        pred_indices = _multilabel_indices(
            predicted_labels,
            item=item,
            indices=class_index,
            name="prediction",
        )
        true_matrix[row_index, true_indices] = 1
        pred_matrix[row_index, pred_indices] = 1
        predictions[item] = list(predicted_labels)
        scores[item] = score_matrix[row_index]
    return (
        multilabel_metrics(
            cast(Sequence[Sequence[int]], true_matrix),
            cast(Sequence[Sequence[int]], pred_matrix),
            cast(Sequence[Sequence[float]], score_matrix),
        ),
        predictions,
        scores,
    )


def _multilabel_score_matrix(
    data: ProbeBackendInput,
    output: ProbeBackendOutput,
    *,
    class_count: int,
) -> _npt.NDArray[_np.float64]:
    expected_shape = (len(data.test_ids), class_count)
    if output.score_matrix is not None:
        if output.scores is not None:
            raise EmbeddingInputError(
                "Multilabel custom probe backends must return either scores or "
                "score_matrix, not both."
            )
        return _require_float_matrix(output.score_matrix, expected_shape=expected_shape)
    if output.scores is None:
        raise EmbeddingInputError(
            "Multilabel custom probe backends must return scores or score_matrix."
        )
    _require_output_ids(data.test_ids, output.scores, name="scores")
    score_matrix = _np.empty(expected_shape, dtype=_np.float64)
    for row_index, item in enumerate(data.test_ids):
        score_values = _require_float_sequence(output.scores[item], item=item)
        if len(score_values) != class_count:
            raise EmbeddingInputError(
                f"Custom probe score length {len(score_values)} does not match "
                f"class count {class_count} for {item!r}."
            )
        score_matrix[row_index] = score_values
    return score_matrix


def _require_float_matrix(
    value: object,
    *,
    expected_shape: tuple[int, int],
) -> _npt.NDArray[_np.float64]:
    try:
        matrix = _np.asarray(value, dtype=_np.float64)
    except (TypeError, ValueError) as exc:
        raise EmbeddingInputError(
            "Custom probe score_matrix must contain numeric values."
        ) from exc
    if matrix.shape != expected_shape:
        raise EmbeddingInputError(
            f"Custom probe score_matrix shape {matrix.shape} does not match "
            f"expected shape {expected_shape}."
        )
    if not bool(_np.isfinite(matrix).all()):
        raise EmbeddingInputError(
            "Custom probe score_matrix must contain only finite numeric values."
        )
    return matrix


def _evaluate_residue_output(
    data: ProbeBackendInput,
    output: ProbeBackendOutput,
) -> tuple[dict[str, float], dict[str, ProbePrediction], dict[str, float] | None]:
    _require_output_ids(data.test_ids, output.predictions, name="predictions")
    if data.prediction.objective == "binary":
        if output.scores is None:
            raise EmbeddingInputError(
                "Binary custom probe backends must return scores for every test residue."
            )
        _require_output_ids(data.test_ids, output.scores, name="scores")

    truth: dict[str, ProbePrediction] = {}
    predictions: dict[str, ProbePrediction] = {}
    scores: dict[str, float] | None = (
        {} if data.prediction.objective == "binary" else None
    )
    for item in data.test_ids:
        target_values = _require_sequence(data.labels[item], item=item, name="labels")
        predicted_values = _require_sequence(
            output.predictions[item],
            item=item,
            name="predictions",
        )
        if len(predicted_values) != len(target_values):
            raise EmbeddingInputError(
                f"Custom probe prediction length {len(predicted_values)} does not match "
                f"label length {len(target_values)} for {item!r}."
            )
        score_values: Sequence[float] | None = None
        if output.scores is not None:
            score_values = _require_float_sequence(output.scores[item], item=item)
            if len(score_values) != len(target_values):
                raise EmbeddingInputError(
                    f"Custom probe score length {len(score_values)} does not match "
                    f"label length {len(target_values)} for {item!r}."
                )
        mask = data.masks.get(item)
        if mask is not None and len(mask) != len(target_values):
            raise EmbeddingInputError(
                f"Residue mask length {len(mask)} does not match label length "
                f"{len(target_values)} for {item!r}."
            )
        for index, target in enumerate(target_values):
            if mask is not None and not bool(mask[index]):
                continue
            residue_id = f"{item}:{index + 1}"
            truth[residue_id] = _require_scalar(target, item=residue_id, name="label")
            predictions[residue_id] = _require_scalar(
                predicted_values[index],
                item=residue_id,
                name="prediction",
            )
            if scores is not None and score_values is not None:
                scores[residue_id] = float(score_values[index])
    if not predictions:
        raise EmbeddingInputError(
            "Custom residue probe output does not contain any unmasked test residues."
        )
    predictions = _normalize_predictions(data, predictions)
    metrics = _evaluate_predictions(
        data,
        truth=truth,
        predictions=predictions,
        scores=scores,
    )
    return metrics, predictions, scores


def _evaluate_predictions(
    data: ProbeBackendInput,
    *,
    truth: Mapping[str, ProbePrediction],
    predictions: Mapping[str, ProbePrediction],
    scores: Mapping[str, float] | None,
) -> dict[str, float]:
    ordered_ids = list(truth)
    objective = data.prediction.objective
    if objective == "regression":
        return regression_metrics(
            [float(truth[item]) for item in ordered_ids],
            [float(predictions[item]) for item in ordered_ids],
        )
    if objective == "binary":
        if scores is None:
            raise EmbeddingInputError(
                "Binary custom probe backends must return scores for every test example."
            )
        return binary_metrics(
            [_binary_value(truth[item], item=item) for item in ordered_ids],
            [_binary_value(predictions[item], item=item) for item in ordered_ids],
            [float(scores[item]) for item in ordered_ids],
        )
    if objective == "multiclass":
        classes = _multiclass_names(data)
        class_indices = {name: index for index, name in enumerate(classes)}
        return multiclass_metrics(
            [
                _class_index(
                    truth[item],
                    item=item,
                    indices=class_indices,
                    allow_index=False,
                )
                for item in ordered_ids
            ],
            [
                _class_index(
                    predictions[item],
                    item=item,
                    indices=class_indices,
                    allow_index=True,
                )
                for item in ordered_ids
            ],
            class_count=len(classes),
        )
    raise EmbeddingInputError(f"Unsupported custom probe objective: {objective!r}.")


def _protein_scores(
    data: ProbeBackendInput,
    output: ProbeBackendOutput,
) -> dict[str, float] | None:
    if data.prediction.objective != "binary":
        return None
    if output.scores is None:
        raise EmbeddingInputError(
            "Binary custom probe backends must return scores for every test example."
        )
    _require_output_ids(data.test_ids, output.scores, name="scores")
    return {
        item: float(_require_scalar(output.scores[item], item=item, name="score"))
        for item in data.test_ids
    }


def _multiclass_names(data: ProbeBackendInput) -> list[str]:
    if data.prediction.classes is not None:
        classes = [str(value) for value in data.prediction.classes]
    else:
        values: set[str] = set()
        for item in (*data.train_ids, *data.test_ids):
            label = data.labels[item]
            if data.level == "protein":
                values.add(str(_require_scalar(label, item=item, name="label")))
            else:
                labels = _require_sequence(label, item=item, name="labels")
                mask = data.masks.get(item)
                for index, value in enumerate(labels):
                    if mask is None or bool(mask[index]):
                        values.add(str(_require_scalar(value, item=item, name="label")))
        classes = sorted(values)
    if len(classes) < 2:
        raise EmbeddingInputError("Multiclass custom probes require at least two classes.")
    return classes


def _multilabel_names(data: ProbeBackendInput) -> list[str]:
    if data.prediction.classes is not None:
        classes = [str(value) for value in data.prediction.classes]
    else:
        values: set[str] = set()
        for item in (*data.train_ids, *data.test_ids):
            values.update(_multilabel_values(data.labels[item], item=item, name="labels"))
        classes = sorted(values)
    if not classes:
        raise EmbeddingInputError("Multilabel custom probes require at least one class.")
    return classes


def _multilabel_values(value: object, *, item: str, name: str) -> list[str]:
    values = _require_sequence(value, item=item, name=name)
    return sorted({str(label).strip() for label in values if str(label).strip()})


def _multilabel_indices(
    values: Sequence[str],
    *,
    item: str,
    indices: Mapping[str, int],
    name: str,
) -> list[int]:
    resolved: list[int] = []
    for value in values:
        if value not in indices:
            raise EmbeddingInputError(f"Unknown multilabel {name} {value!r} for example {item!r}.")
        resolved.append(indices[value])
    return resolved


def _class_index(
    value: ProbePrediction,
    *,
    item: str,
    indices: Mapping[str, int],
    allow_index: bool,
) -> int:
    if allow_index and isinstance(value, int) and value in indices.values():
        return value
    key = str(value)
    if key not in indices:
        supported = ", ".join(indices)
        raise EmbeddingInputError(
            f"Unknown custom probe class {value!r} for {item!r}. "
            f"Supported classes: {supported}."
        )
    return indices[key]


def _normalize_predictions(
    data: ProbeBackendInput,
    predictions: Mapping[str, ProbePrediction],
) -> dict[str, ProbePrediction]:
    objective = data.prediction.objective
    if objective == "regression":
        return {item: float(value) for item, value in predictions.items()}
    if objective == "binary":
        return {
            item: _binary_value(value, item=item)
            for item, value in predictions.items()
        }
    if objective == "multiclass":
        classes = _multiclass_names(data)
        indices = {name: index for index, name in enumerate(classes)}
        normalized: dict[str, ProbePrediction] = {}
        for item, value in predictions.items():
            index = _class_index(
                value,
                item=item,
                indices=indices,
                allow_index=True,
            )
            normalized[item] = classes[index]
        return normalized
    raise EmbeddingInputError(f"Unsupported custom probe objective: {objective!r}.")


def _binary_value(value: ProbePrediction, *, item: str) -> int:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int) and value in {0, 1}:
        return value
    if isinstance(value, float) and value in {0.0, 1.0}:
        return int(value)
    text = str(value).strip().lower()
    if text in {"0", "false", "no", "negative"}:
        return 0
    if text in {"1", "true", "yes", "positive"}:
        return 1
    raise EmbeddingInputError(
        f"Binary custom probe value for {item!r} must be 0/1 or boolean-like."
    )


def _require_output_ids(
    ids: Sequence[str],
    values: Mapping[str, object],
    *,
    name: str,
) -> None:
    missing = [item for item in ids if item not in values]
    if missing:
        sample = ", ".join(missing[:5])
        raise EmbeddingInputError(f"Custom probe {name} are missing test examples: {sample}.")


def _require_scalar(value: object, *, item: str, name: str) -> ProbePrediction:
    if isinstance(value, bool | int | float | str):
        return value
    raise EmbeddingInputError(f"Custom probe {name} for {item!r} must be a scalar value.")


def _require_sequence(value: object, *, item: str, name: str) -> Sequence[object]:
    if isinstance(value, Sequence) and not isinstance(value, str | bytes):
        return cast(Sequence[object], value)
    raise EmbeddingInputError(f"Custom probe {name} for {item!r} must be a sequence.")


def _require_float_sequence(value: object, *, item: str) -> Sequence[float]:
    values = _require_sequence(value, item=item, name="scores")
    try:
        return [float(cast(float | int | str, score)) for score in values]
    except (TypeError, ValueError) as exc:
        raise EmbeddingInputError(
            f"Custom probe scores for {item!r} must contain numeric values."
        ) from exc


__all__ = [
    "ProbeBackend",
    "ProbeBackendInput",
    "ProbeBackendOutput",
    "ProbeFeature",
    "ProbeLabel",
    "ProbePrediction",
    "ProbePredictionOutput",
    "ProbeScoreOutput",
    "ProbeScoreRow",
    "evaluate_probe_backend_output",
]
