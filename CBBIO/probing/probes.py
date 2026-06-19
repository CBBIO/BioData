"""PyTorch probe training for supervised embedding evaluation."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import numpy as _np
from typing import Any, Dict, List, Tuple, cast

from CBBIO.embeddings import EmbeddingDependencyError, EmbeddingInputError

from .datasets import ObjectiveName
from .metrics import binary_metrics, multiclass_metrics, regression_metrics
from .tasks import PredictionSpec, ProbeSpec


DEFAULT_RESIDUE_BATCH_SIZE = 8192


@dataclass(frozen=True)
class ProbeEvaluation:
    """Evaluation metrics, predictions, and optional scores from a probe."""
    metrics: Dict[str, float]
    predictions: Dict[str, float | int | str]
    scores: Dict[str, float] | None = None


@dataclass(frozen=True)
class ResidueDataFlat:
    """Pre-flattened residue embeddings for a single layer.

    Build once per layer with ``compute_residue_flat_data`` and pass into
    ``train_and_evaluate_residue_probe`` as ``flat_data`` to replace per-epoch
    per-protein tensor assembly with direct zero-copy slicing.

    Reuse across seeds: embeddings and masks are seed-independent.
    Evict between layers to bound peak RAM to one layer at a time.
    """

    x_train: Any          # numpy float32 (N_train_residues, D)
    train_label_ids: List[str]   # flat IDs "protein_id:position"
    x_test: Any           # numpy float32 (N_test_residues, D)
    test_label_ids: List[str]


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

    device = _probe_device(torch)
    torch.manual_seed(int(probe.seed))
    model = _build_probe_model(torch, probe=probe, input_dim=input_dim, output_dim=output_dim)
    model = model.to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(probe.learning_rate),
        weight_decay=float(probe.weight_decay),
    )
    loss_fn = _loss_fn(torch, prediction.objective)

    train_x = _tensor_for_ids(torch, train_ids, embeddings)
    feature_mean, feature_std = _feature_standardization_stats(torch, train_x)
    feature_mean = feature_mean.to(device)
    feature_std = feature_std.to(device)
    train_x = _standardize_features(train_x.to(device), mean=feature_mean, std=feature_std)
    train_y = _label_tensor(torch, [encoded_labels[item] for item in train_ids], prediction.objective).to(device)
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
    with torch.inference_mode():
        test_x = _tensor_for_ids(torch, test_ids, embeddings).to(device)
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


def compute_residue_feature_stats(
    *,
    train_ids: Sequence[str],
    embeddings: Mapping[str, Sequence[Sequence[float]]],
    masks: Mapping[str, Sequence[bool] | None],
) -> Tuple[Any, Any]:
    """Return (feature_mean, feature_std) CPU tensors shaped (1, D) for residue probing.

    Pass these into ``train_and_evaluate_residue_probe`` as ``feature_mean`` /
    ``feature_std`` to skip the internal stats pass when probing multiple seeds
    against the same layer: the stats depend only on the embeddings and masks,
    not the random seed, so they are identical across seeds.
    """
    torch = _import_torch()
    return _residue_feature_standardization_stats(torch, train_ids, embeddings=embeddings, masks=masks)


def compute_residue_flat_data(
    *,
    train_ids: Sequence[str],
    test_ids: Sequence[str],
    embeddings: Mapping[str, Sequence[Sequence[float]]],
    masks: Mapping[str, Sequence[bool] | None],
) -> ResidueDataFlat:
    """Pre-flatten residue embeddings into contiguous numpy arrays.

    Call once per layer and pass the result into ``train_and_evaluate_residue_probe``
    as ``flat_data``.  This replaces per-epoch per-protein tensor assembly
    (O(N_epochs × N_proteins) allocations) with a single ``np.concatenate`` and
    direct zero-copy slicing during the training loop.

    Reuse the returned object across seeds for the same layer — embeddings and
    masks are seed-independent.  Evict between layers to keep peak extra RAM at
    one layer at a time.
    """
    x_train, train_label_ids = _flatten_residue_embeddings(train_ids, embeddings=embeddings, masks=masks)
    x_test, test_label_ids = _flatten_residue_embeddings(test_ids, embeddings=embeddings, masks=masks)
    return ResidueDataFlat(
        x_train=x_train,
        train_label_ids=train_label_ids,
        x_test=x_test,
        test_label_ids=test_label_ids,
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
    feature_mean: Any = None,
    feature_std: Any = None,
    flat_data: ResidueDataFlat | None = None,
) -> ProbeEvaluation:
    """Train and evaluate a residue-level probe without per-residue dict expansion.

    ``feature_mean`` / ``feature_std``: optional pre-computed CPU tensors
    (see ``compute_residue_feature_stats``).  Skip the internal stats pass when
    calling for multiple seeds on the same layer.

    ``flat_data``: optional pre-flattened contiguous numpy arrays
    (see ``compute_residue_flat_data``).  Replaces per-epoch per-protein tensor
    assembly with direct zero-copy slicing.  Eliminates the ``pending_tensors``
    accumulation in ``_iter_residue_batches`` that causes RAM growth with large
    probe batch sizes.  Reuse across seeds for the same layer; evict between layers.
    """

    torch = _import_torch()
    if prediction.objective == "multilabel":
        raise EmbeddingInputError("Multilabel probes are not supported in this first probing slice.")

    ordered_ids = list(train_ids) + list(test_ids)
    _validate_residue_examples(ordered_ids, embeddings=embeddings, labels=labels, masks=masks)
    ordered_labels = _flatten_residue_labels(ordered_ids, labels=labels, masks=masks)
    encoded_labels, output_dim, class_names = _encode_labels(
        ordered_labels[0],
        ordered_labels[1],
        objective=prediction.objective,
        classes=prediction.classes,
    )

    if flat_data is not None:
        train_count = int(flat_data.x_train.shape[0])
        test_count = int(flat_data.x_test.shape[0])
        input_dim = int(flat_data.x_train.shape[1])
    else:
        train_count = _valid_residue_count(train_ids, embeddings=embeddings, masks=masks)
        test_count = _valid_residue_count(test_ids, embeddings=embeddings, masks=masks)
        input_dim = _residue_input_dim(train_ids, embeddings=embeddings, masks=masks)

    if train_count < 1:
        raise EmbeddingInputError("Residue-level probe requires at least one valid train residue.")
    if test_count < 1:
        raise EmbeddingInputError("Residue-level probe requires at least one valid test residue.")

    device = _probe_device(torch)
    torch.manual_seed(int(probe.seed))
    model = _build_probe_model(torch, probe=probe, input_dim=input_dim, output_dim=output_dim)
    model = model.to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(probe.learning_rate),
        weight_decay=float(probe.weight_decay),
    )
    loss_fn = _loss_fn(torch, prediction.objective)

    if feature_mean is None or feature_std is None:
        feature_mean, feature_std = _residue_feature_standardization_stats(
            torch, train_ids, embeddings=embeddings, masks=masks,
        )
    feature_mean = feature_mean.to(device)
    feature_std = feature_std.to(device)
    batch_size = int(probe.batch_size or min(train_count, DEFAULT_RESIDUE_BATCH_SIZE))
    x_pin: Any = _try_alloc_pinned(torch, device, rows=batch_size, cols=input_dim)

    model.train()
    if flat_data is not None:
        # Pre-flattened path: x_train is a contiguous numpy array; torch.as_tensor shares
        # its memory (zero copy).  Training loop does direct slicing — no pending_tensors
        # accumulation, no torch.cat, no per-protein Python overhead.
        x_train_t = torch.as_tensor(flat_data.x_train, dtype=torch.float32)
        y_train_t = _label_tensor(
            torch,
            [encoded_labels[lid] for lid in flat_data.train_label_ids],
            prediction.objective,
        )
        for _epoch in range(int(probe.epochs)):
            for start in range(0, train_count, batch_size):
                end = min(train_count, start + batch_size)
                batch_x_dev = _send_to_device(x_train_t[start:end], device, x_pin)
                batch_y = y_train_t[start:end].to(device)
                batch_x_dev = _standardize_features(batch_x_dev, mean=feature_mean, std=feature_std)
                optimizer.zero_grad()
                logits = model(batch_x_dev)
                loss = _compute_loss(logits, batch_y, loss_fn, prediction.objective)
                loss.backward()
                optimizer.step()
    else:
        for _epoch in range(int(probe.epochs)):
            for batch_x, batch_label_ids in _iter_residue_batches(
                torch, train_ids, embeddings=embeddings, masks=masks, batch_size=batch_size,
            ):
                batch_x_dev = _send_to_device(batch_x, device, x_pin)
                batch_y = _label_tensor(torch, [encoded_labels[item] for item in batch_label_ids], prediction.objective).to(device)
                batch_x_dev = _standardize_features(batch_x_dev, mean=feature_mean, std=feature_std)
                optimizer.zero_grad()
                logits = model(batch_x_dev)
                loss = _compute_loss(logits, batch_y, loss_fn, prediction.objective)
                loss.backward()
                optimizer.step()

    model.eval()
    test_logits: List[Any] = []
    test_label_ids_used: List[str] = []
    with torch.inference_mode():
        if flat_data is not None:
            x_test_t = torch.as_tensor(flat_data.x_test, dtype=torch.float32)
            n_test = int(flat_data.x_test.shape[0])
            for start in range(0, n_test, batch_size):
                end = min(n_test, start + batch_size)
                batch_x_dev = _send_to_device(x_test_t[start:end], device, x_pin)
                batch_x_dev = _standardize_features(batch_x_dev, mean=feature_mean, std=feature_std)
                test_logits.append(model(batch_x_dev).cpu())
            test_label_ids_used = list(flat_data.test_label_ids)
        else:
            for batch_x, batch_label_ids in _iter_residue_batches(
                torch, test_ids, embeddings=embeddings, masks=masks, batch_size=batch_size,
            ):
                batch_x_dev = _send_to_device(batch_x, device, x_pin)
                batch_x_dev = _standardize_features(batch_x_dev, mean=feature_mean, std=feature_std)
                test_logits.append(model(batch_x_dev).cpu())
                test_label_ids_used.extend(batch_label_ids)

    return _evaluate_outputs(
        torch,
        test_ids=test_label_ids_used,
        logits=torch.cat(test_logits, dim=0),
        labels={item: encoded_labels[item] for item in test_label_ids_used},
        objective=prediction.objective,
        class_names=class_names,
    )


def _import_torch() -> Any:
    try:
        import torch  # type: ignore

        return torch
    except Exception as exc:
        raise EmbeddingDependencyError(
            "PyTorch is required for probing. Install torch or run probes in an environment that provides it."
        ) from exc


def _probe_device(torch: Any) -> Any:
    cuda_available = getattr(torch, "cuda", None)
    if cuda_available is not None and callable(getattr(cuda_available, "is_available", None)):
        if cuda_available.is_available():
            return torch.device("cuda")
    return torch.device("cpu")


def _try_alloc_pinned(torch: Any, device: Any, *, rows: int, cols: int) -> Any:
    """Return a pinned float32 buffer of shape (rows, cols), or None on failure/CPU."""
    if getattr(device, "type", None) != "cuda":
        return None
    try:
        return torch.empty(rows, cols, dtype=torch.float32).pin_memory()
    except Exception:
        return None


def _send_to_device(batch_cpu: Any, device: Any, x_pin: Any) -> Any:
    """Copy batch into pinned buffer and start an async H2D transfer, or fall back to blocking."""
    if x_pin is None:
        return batch_cpu.to(device)
    bsz = int(batch_cpu.shape[0])
    x_pin[:bsz].copy_(batch_cpu)
    return x_pin[:bsz].to(device, non_blocking=True)


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


def _validate_residue_examples(
    ids: Sequence[str],
    *,
    embeddings: Mapping[str, Sequence[Sequence[float]]],
    labels: Mapping[str, Sequence[object]],
    masks: Mapping[str, Sequence[bool] | None],
) -> None:
    missing_embeddings = [item for item in ids if item not in embeddings]
    if missing_embeddings:
        sample = ", ".join(missing_embeddings[:5])
        raise EmbeddingInputError(f"Missing residue embeddings for examples: {sample}.")
    missing_labels = [item for item in ids if item not in labels]
    if missing_labels:
        sample = ", ".join(missing_labels[:5])
        raise EmbeddingInputError(f"Missing residue labels for examples: {sample}.")
    input_dim: int | None = None
    for item in ids:
        matrix = embeddings[item]
        target_values = labels[item]
        mask = masks.get(item)
        rows, cols = _matrix_shape(matrix)
        if rows != len(target_values):
            raise EmbeddingInputError(
                f"Residue embedding length {rows} does not match label length {len(target_values)} for {item!r}."
            )
        if mask is not None and len(mask) != len(target_values):
            raise EmbeddingInputError(
                f"Residue mask length {len(mask)} does not match label length {len(target_values)} for {item!r}."
            )
        if input_dim is None:
            input_dim = cols
        elif cols != input_dim:
            raise EmbeddingInputError("All residue embeddings for a probe task must have the same dimension.")


def _flatten_residue_labels(
    ids: Sequence[str],
    *,
    labels: Mapping[str, Sequence[object]],
    masks: Mapping[str, Sequence[bool] | None],
) -> Tuple[List[str], Dict[str, object]]:
    flat_ids: List[str] = []
    flat_labels: Dict[str, object] = {}
    for item in ids:
        target_values = labels[item]
        mask = masks.get(item)
        for index, label in enumerate(target_values):
            if mask is not None and not bool(mask[index]):
                continue
            flat_id = f"{item}:{index + 1}"
            flat_ids.append(flat_id)
            flat_labels[flat_id] = label
    return flat_ids, flat_labels


def _valid_residue_count(
    ids: Sequence[str],
    *,
    embeddings: Mapping[str, Sequence[Sequence[float]]],
    masks: Mapping[str, Sequence[bool] | None],
) -> int:
    count = 0
    for item in ids:
        rows, _cols = _matrix_shape(embeddings[item])
        count += len(_valid_residue_indices(rows, masks.get(item)))
    return count


def _residue_input_dim(
    ids: Sequence[str],
    *,
    embeddings: Mapping[str, Sequence[Sequence[float]]],
    masks: Mapping[str, Sequence[bool] | None],
) -> int:
    for item in ids:
        rows, cols = _matrix_shape(embeddings[item])
        if _valid_residue_indices(rows, masks.get(item)):
            return cols
    raise EmbeddingInputError("Residue-level probe requires at least one valid train residue.")


def _residue_feature_standardization_stats(
    torch: Any,
    ids: Sequence[str],
    *,
    embeddings: Mapping[str, Sequence[Sequence[float]]],
    masks: Mapping[str, Sequence[bool] | None],
) -> Tuple[Any, Any]:
    feature_sum = None
    feature_sumsq = None
    count = 0
    for matrix, _label_ids in _iter_residue_batches(
        torch,
        ids,
        embeddings=embeddings,
        masks=masks,
        batch_size=DEFAULT_RESIDUE_BATCH_SIZE,
    ):
        if feature_sum is None:
            feature_sum = matrix.sum(dim=0, keepdim=True)
            feature_sumsq = (matrix * matrix).sum(dim=0, keepdim=True)
        else:
            feature_sum += matrix.sum(dim=0, keepdim=True)
            feature_sumsq += (matrix * matrix).sum(dim=0, keepdim=True)
        count += int(matrix.shape[0])
    if feature_sum is None or feature_sumsq is None or count < 1:
        raise EmbeddingInputError("Residue-level probe requires at least one valid train residue.")
    mean = feature_sum / float(count)
    variance = torch.clamp(feature_sumsq / float(count) - mean * mean, min=0.0)
    return mean, torch.clamp(torch.sqrt(variance), min=1e-6)


def _flatten_residue_embeddings(
    ids: Sequence[str],
    *,
    embeddings: Mapping[str, Sequence[Sequence[float]]],
    masks: Mapping[str, Sequence[bool] | None],
) -> Tuple[Any, List[str]]:
    """Concatenate per-protein embedding matrices into one contiguous float32 array."""
    arrays: List[Any] = []
    label_ids: List[str] = []
    for item in ids:
        matrix = embeddings[item]
        to_array = getattr(matrix, "__array__", None)
        arr = _np.asarray(cast(Any, to_array)(), dtype=_np.float32) if callable(to_array) else _np.array(matrix, dtype=_np.float32)
        if arr.dtype != _np.float32:
            arr = arr.astype(_np.float32)
        mask = masks.get(item)
        keep = _valid_residue_indices(int(arr.shape[0]), mask)
        if not keep:
            continue
        arrays.append(arr if mask is None else arr[keep])
        label_ids.extend(f"{item}:{i + 1}" for i in keep)
    if not arrays:
        return _np.empty((0, 0), dtype=_np.float32), []
    return _np.concatenate(arrays, axis=0), label_ids


def _iter_residue_batches(
    torch: Any,
    ids: Sequence[str],
    *,
    embeddings: Mapping[str, Sequence[Sequence[float]]],
    masks: Mapping[str, Sequence[bool] | None],
    batch_size: int,
) -> Any:
    resolved_batch_size = max(1, int(batch_size))
    pending_tensors: List[Any] = []
    pending_label_ids: List[str] = []
    pending_count = 0
    for item in ids:
        tensor = _residue_matrix_tensor(torch, item, embeddings[item])
        keep_indices = _valid_residue_indices(int(tensor.shape[0]), masks.get(item))
        if not keep_indices:
            continue
        index_tensor = torch.tensor(keep_indices, dtype=torch.long)
        tensor = tensor.index_select(0, index_tensor)
        label_ids = [f"{item}:{index + 1}" for index in keep_indices]
        start = 0
        while start < len(label_ids):
            available = resolved_batch_size - pending_count
            end = min(len(label_ids), start + available)
            pending_tensors.append(tensor[start:end])
            pending_label_ids.extend(label_ids[start:end])
            pending_count += end - start
            start = end
            if pending_count >= resolved_batch_size:
                yield torch.cat(pending_tensors, dim=0), pending_label_ids
                pending_tensors = []
                pending_label_ids = []
                pending_count = 0
    if pending_count:
        yield torch.cat(pending_tensors, dim=0), pending_label_ids


def _residue_matrix_tensor(torch: Any, item: str, matrix: Sequence[Sequence[float]]) -> Any:
    to_array = getattr(matrix, "__array__", None)
    if callable(to_array):
        matrix = cast(Sequence[Sequence[float]], to_array())
    tensor = torch.as_tensor(matrix, dtype=torch.float32)
    if tensor.ndim != 2 or int(tensor.shape[0]) < 1 or int(tensor.shape[1]) < 1:
        raise EmbeddingInputError(f"Residue embedding matrix for {item!r} must be non-empty and rank 2.")
    return tensor


def _valid_residue_indices(length: int, mask: Sequence[bool] | None) -> List[int]:
    if mask is None:
        return list(range(length))
    return [index for index in range(length) if bool(mask[index])]


def _matrix_shape(matrix: Sequence[Sequence[float]]) -> Tuple[int, int]:
    shape = getattr(matrix, "shape", None)
    if shape is not None:
        if len(shape) != 2:
            raise EmbeddingInputError("Residue embeddings must be rank-2 matrices.")
        rows = int(shape[0])
        cols = int(shape[1])
        if rows < 1 or cols < 1:
            raise EmbeddingInputError("Residue embedding matrix is empty.")
        return rows, cols
    rows = len(matrix)
    if rows < 1:
        raise EmbeddingInputError("Residue embedding matrix is empty.")
    cols = len(matrix[0])
    if cols < 1:
        raise EmbeddingInputError("Residue embedding vectors must be non-empty.")
    for row in matrix:
        if len(row) != cols:
            raise EmbeddingInputError("Residue embedding matrix rows must have the same dimension.")
    return rows, cols


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


__all__ = [
    "ProbeEvaluation",
    "ResidueDataFlat",
    "compute_residue_feature_stats",
    "compute_residue_flat_data",
    "train_and_evaluate_probe",
    "train_and_evaluate_residue_probe",
]
