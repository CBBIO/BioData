"""PyTorch probe training for supervised embedding evaluation."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple, cast

from CBBIO.embeddings import EmbeddingDependencyError, EmbeddingInputError

from .datasets import ObjectiveName
from .metrics import binary_metrics, multiclass_metrics, regression_metrics
from .tasks import PredictionSpec, ProbeSpec


@dataclass(frozen=True)
class ProbeEvaluation:
    metrics: Dict[str, float]
    predictions: Dict[str, float | int | str]
    scores: Dict[str, float] | None = None


def train_and_evaluate_probe(
    *,
    train_ids: Sequence[str],
    test_ids: Sequence[str],
    embeddings: Mapping[str, Sequence[float]],
    labels: Mapping[str, object],
    prediction: PredictionSpec,
    probe: ProbeSpec,
) -> ProbeEvaluation:
    """Train a small PyTorch probe and evaluate it on the test split."""

    torch = _import_torch()
    if prediction.objective == "multilabel":
        raise EmbeddingInputError("Multilabel probes are not supported in this first probing slice.")

    ordered_ids = list(train_ids) + list(test_ids)
    input_dim = _validate_embeddings(ordered_ids, embeddings)
    encoded_labels, output_dim, class_names = _encode_labels(
        ordered_ids,
        labels,
        objective=prediction.objective,
        classes=prediction.classes,
    )

    torch.manual_seed(int(probe.seed))
    model = _build_probe_model(torch, probe=probe, input_dim=input_dim, output_dim=output_dim)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(probe.learning_rate),
        weight_decay=float(probe.weight_decay),
    )
    loss_fn = _loss_fn(torch, prediction.objective)

    train_x = _tensor_for_ids(torch, train_ids, embeddings)
    feature_mean, feature_std = _feature_standardization_stats(torch, train_x)
    train_x = _standardize_features(train_x, mean=feature_mean, std=feature_std)
    train_y = _label_tensor(torch, [encoded_labels[item] for item in train_ids], prediction.objective)
    batch_size = int(probe.batch_size or len(train_ids))

    model.train()
    for _epoch in range(int(probe.epochs)):
        for start in range(0, len(train_ids), batch_size):
            end = start + batch_size
            batch_x = train_x[start:end]
            batch_y = train_y[start:end]
            optimizer.zero_grad()
            logits = model(batch_x)
            loss = _compute_loss(logits, batch_y, loss_fn, prediction.objective)
            loss.backward()
            optimizer.step()

    model.eval()
    with torch.no_grad():
        test_x = _tensor_for_ids(torch, test_ids, embeddings)
        test_x = _standardize_features(test_x, mean=feature_mean, std=feature_std)
        logits = model(test_x)

    return _evaluate_outputs(
        torch,
        test_ids=test_ids,
        logits=logits,
        labels={item: encoded_labels[item] for item in test_ids},
        objective=prediction.objective,
        class_names=class_names,
    )


def train_and_evaluate_residue_probe(
    *,
    train_ids: Sequence[str],
    test_ids: Sequence[str],
    embeddings: Mapping[str, Sequence[Sequence[float]]],
    labels: Mapping[str, Sequence[object]],
    masks: Mapping[str, Sequence[bool] | None],
    prediction: PredictionSpec,
    probe: ProbeSpec,
) -> ProbeEvaluation:
    """Train and evaluate a residue-level probe by flattening valid residues."""

    train_flat = _flatten_residue_examples(train_ids, embeddings=embeddings, labels=labels, masks=masks)
    test_flat = _flatten_residue_examples(test_ids, embeddings=embeddings, labels=labels, masks=masks)
    if not train_flat[0]:
        raise EmbeddingInputError("Residue-level probe requires at least one valid train residue.")
    if not test_flat[0]:
        raise EmbeddingInputError("Residue-level probe requires at least one valid test residue.")
    flat_embeddings = {**train_flat[1], **test_flat[1]}
    flat_labels = {**train_flat[2], **test_flat[2]}
    return train_and_evaluate_probe(
        train_ids=train_flat[0],
        test_ids=test_flat[0],
        embeddings=flat_embeddings,
        labels=flat_labels,
        prediction=prediction,
        probe=probe,
    )


def _import_torch() -> Any:
    try:
        import torch  # type: ignore

        return torch
    except Exception as exc:
        raise EmbeddingDependencyError(
            "PyTorch is required for probing. Install torch or run probes in an environment that provides it."
        ) from exc


def _validate_embeddings(ids: Sequence[str], embeddings: Mapping[str, Sequence[float]]) -> int:
    missing = [item for item in ids if item not in embeddings]
    if missing:
        sample = ", ".join(missing[:5])
        raise EmbeddingInputError(f"Missing embeddings for examples: {sample}.")
    input_dim = len(embeddings[ids[0]])
    if input_dim < 1:
        raise EmbeddingInputError("Embeddings must be non-empty vectors.")
    for item in ids:
        if len(embeddings[item]) != input_dim:
            raise EmbeddingInputError("All embeddings for a probe task must have the same dimension.")
    return input_dim


def _flatten_residue_examples(
    ids: Sequence[str],
    *,
    embeddings: Mapping[str, Sequence[Sequence[float]]],
    labels: Mapping[str, Sequence[object]],
    masks: Mapping[str, Sequence[bool] | None],
) -> Tuple[List[str], Dict[str, Sequence[float]], Dict[str, object]]:
    flat_ids: List[str] = []
    flat_embeddings: Dict[str, Sequence[float]] = {}
    flat_labels: Dict[str, object] = {}
    missing_embeddings = [item for item in ids if item not in embeddings]
    if missing_embeddings:
        sample = ", ".join(missing_embeddings[:5])
        raise EmbeddingInputError(f"Missing residue embeddings for examples: {sample}.")
    missing_labels = [item for item in ids if item not in labels]
    if missing_labels:
        sample = ", ".join(missing_labels[:5])
        raise EmbeddingInputError(f"Missing residue labels for examples: {sample}.")
    for item in ids:
        matrix = embeddings[item]
        target_values = labels[item]
        mask = masks.get(item)
        if len(matrix) != len(target_values):
            raise EmbeddingInputError(
                f"Residue embedding length {len(matrix)} does not match label length {len(target_values)} for {item!r}."
            )
        if mask is not None and len(mask) != len(target_values):
            raise EmbeddingInputError(
                f"Residue mask length {len(mask)} does not match label length {len(target_values)} for {item!r}."
            )
        for index, (vector, label) in enumerate(zip(matrix, target_values)):
            if mask is not None and not bool(mask[index]):
                continue
            if not vector:
                raise EmbeddingInputError(f"Residue embedding for {item!r} position {index + 1} is empty.")
            flat_id = f"{item}:{index + 1}"
            flat_ids.append(flat_id)
            flat_embeddings[flat_id] = vector
            flat_labels[flat_id] = label
    return flat_ids, flat_embeddings, flat_labels


def _build_probe_model(torch: Any, *, probe: ProbeSpec, input_dim: int, output_dim: int) -> Any:
    if probe.kind == "linear":
        return torch.nn.Linear(input_dim, output_dim)
    if probe.kind == "mlp":
        return torch.nn.Sequential(
            torch.nn.Linear(input_dim, int(probe.hidden_dim)),
            torch.nn.ReLU(),
            torch.nn.Linear(int(probe.hidden_dim), output_dim),
        )
    raise EmbeddingInputError(f"Unsupported probe kind: {probe.kind!r}.")


def _loss_fn(torch: Any, objective: ObjectiveName) -> Any:
    if objective == "regression":
        return torch.nn.MSELoss()
    if objective == "binary":
        return torch.nn.BCEWithLogitsLoss()
    if objective == "multiclass":
        return torch.nn.CrossEntropyLoss()
    raise EmbeddingInputError(f"Unsupported probe objective: {objective!r}.")


def _compute_loss(logits: Any, target: Any, loss_fn: Any, objective: ObjectiveName) -> Any:
    if objective in {"regression", "binary"}:
        return loss_fn(logits.reshape(-1), target)
    return loss_fn(logits, target)


def _tensor_for_ids(torch: Any, ids: Sequence[str], embeddings: Mapping[str, Sequence[float]]) -> Any:
    return torch.tensor([[float(value) for value in embeddings[item]] for item in ids], dtype=torch.float32)


def _label_tensor(torch: Any, values: Sequence[float | int], objective: ObjectiveName) -> Any:
    if objective in {"regression", "binary"}:
        return torch.tensor([float(value) for value in values], dtype=torch.float32)
    return torch.tensor([int(value) for value in values], dtype=torch.long)


def _feature_standardization_stats(torch: Any, values: Any) -> Tuple[Any, Any]:
    mean = values.mean(dim=0, keepdim=True)
    std = values.std(dim=0, unbiased=False, keepdim=True)
    return mean, torch.clamp(std, min=1e-6)


def _standardize_features(values: Any, *, mean: Any, std: Any) -> Any:
    return (values - mean) / std


def _encode_labels(
    ids: Sequence[str],
    labels: Mapping[str, object],
    *,
    objective: ObjectiveName,
    classes: Sequence[str] | None,
) -> Tuple[Dict[str, float | int], int, List[str] | None]:
    if objective == "regression":
        return ({item: float(cast(float | int | str, labels[item])) for item in ids}, 1, None)
    if objective == "binary":
        return ({item: _as_binary(labels[item], item) for item in ids}, 1, None)
    if objective == "multiclass":
        class_names = list(classes) if classes is not None else _infer_classes([labels[item] for item in ids])
        class_index = {name: index for index, name in enumerate(class_names)}
        encoded: Dict[str, float | int] = {}
        for item in ids:
            value = labels[item]
            key = str(value)
            if key not in class_index:
                raise EmbeddingInputError(f"Unknown class label {value!r} for example {item!r}.")
            encoded[item] = class_index[key]
        return encoded, len(class_names), class_names
    raise EmbeddingInputError(f"Unsupported probe objective: {objective!r}.")


def _as_binary(value: object, example_id: str) -> int:
    if isinstance(value, bool):
        return 1 if value else 0
    if isinstance(value, int) and value in {0, 1}:
        return int(value)
    if isinstance(value, float) and value in {0.0, 1.0}:
        return int(value)
    text = str(value).strip().lower()
    if text in {"0", "false", "no", "negative"}:
        return 0
    if text in {"1", "true", "yes", "positive"}:
        return 1
    raise EmbeddingInputError(f"Binary target for example {example_id!r} must be 0/1 or boolean-like.")


def _infer_classes(values: Sequence[object]) -> List[str]:
    classes = sorted({str(value) for value in values})
    if len(classes) < 2:
        raise EmbeddingInputError("Multiclass objective requires at least two classes.")
    return classes


def _evaluate_outputs(
    torch: Any,
    *,
    test_ids: Sequence[str],
    logits: Any,
    labels: Mapping[str, float | int],
    objective: ObjectiveName,
    class_names: Sequence[str] | None,
) -> ProbeEvaluation:
    if objective == "regression":
        values = cast(List[float], logits.reshape(-1).detach().cpu().tolist())
        true_values = [float(labels[item]) for item in test_ids]
        return ProbeEvaluation(
            metrics=regression_metrics(true_values, values),
            predictions={item: float(value) for item, value in zip(test_ids, values)},
        )
    if objective == "binary":
        scores = cast(List[float], torch.sigmoid(logits.reshape(-1)).detach().cpu().tolist())
        pred_labels = [1 if score >= 0.5 else 0 for score in scores]
        true_labels = [int(labels[item]) for item in test_ids]
        return ProbeEvaluation(
            metrics=binary_metrics(true_labels, pred_labels, scores),
            predictions={item: int(value) for item, value in zip(test_ids, pred_labels)},
            scores={item: float(value) for item, value in zip(test_ids, scores)},
        )
    if objective == "multiclass":
        pred_indices = cast(List[int], torch.argmax(logits, dim=1).detach().cpu().tolist())
        true_indices = [int(labels[item]) for item in test_ids]
        names = list(class_names or [])
        predictions: Dict[str, float | int | str] = {}
        for item, index in zip(test_ids, pred_indices):
            predictions[item] = names[index] if names else int(index)
        return ProbeEvaluation(
            metrics=multiclass_metrics(true_indices, pred_indices, class_count=max(1, len(names))),
            predictions=predictions,
        )
    raise EmbeddingInputError(f"Unsupported probe objective: {objective!r}.")


__all__ = ["ProbeEvaluation", "train_and_evaluate_probe", "train_and_evaluate_residue_probe"]
