"""PyTorch probe training for supervised embedding evaluation."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
import numpy as _np
import numpy.typing as _npt
from typing import Any, Dict, List, Literal, Tuple, TypeAlias, cast

from CBBIO.BioData import DriverDependencyError
from CBBIO.embeddings import EmbeddingDependencyError, EmbeddingInputError
from CBBIO.search.engines import build_search_state, search_state
from CBBIO.search.utils import (
    import_faiss,
    preferred_cuvs_device,
    preferred_faiss_device,
    preferred_torch_device,
)
from CBBIO.search.types import ResolvedSearchBackend
from CBBIO.types import DistanceMetric

from .backends import (
    ProbeBackend,
    ProbeFeature,
    ProbeBackendInput,
    ProbeBackendOutput,
    ProbePrediction,
    ProbePredictionOutput,
    ProbeScoreOutput,
)
from .datasets import ObjectiveName
from .metrics import binary_metrics, multiclass_metrics, multilabel_metrics, regression_metrics
from .tasks import PredictionSpec, ProbeKind, ProbeSpec


DEFAULT_RESIDUE_BATCH_SIZE = 8192
_CPU_MANUAL_MULTICLASS_MIN_CLASSES = 128
TransferDistance: TypeAlias = Literal["cosine", "euclidean"]
TransferNeighborSelection: TypeAlias = Literal["knn", "all", "cutoff_distance"]
TransferScoring: TypeAlias = Literal[
    "voting",
    "weighted_voting",
    "nearest_similarity",
]
TransferSearchBackend: TypeAlias = Literal[
    "numpy",
    "auto",
    "faiss_cpu",
    "faiss_gpu",
    "cuvs_gpu",
    "torch_gpu",
]


@dataclass(frozen=True)
class ProbeEvaluation:
    """Evaluation metrics, predictions, and optional scores from a probe."""
    metrics: Dict[str, float]
    predictions: Dict[str, ProbePredictionOutput]
    scores: Dict[str, ProbeScoreOutput] | None = None
    metadata: Mapping[str, Any] | None = None
    score_matrix: Any = None


@dataclass(frozen=True)
class _TorchProbeBackend(ProbeBackend, ABC):
    """Shared training configuration for built-in PyTorch probes."""

    epochs: int = 100
    learning_rate: float = 0.01
    batch_size: int | None = None
    weight_decay: float = 0.0
    seed: int = 7

    def __post_init__(self) -> None:
        self._probe_spec()

    @property
    @abstractmethod
    def kind(self) -> ProbeKind:
        """Return the built-in probe architecture name."""
        raise NotImplementedError

    def fit_predict(self, data: ProbeBackendInput) -> ProbeBackendOutput:
        """Train the built-in probe and return canonical test outputs."""
        evaluation = self.fit_evaluate(data)
        if data.level == "residue":
            return _expand_residue_evaluation(data, evaluation)
        return ProbeBackendOutput(
            predictions=evaluation.predictions,
            scores=evaluation.scores,
            score_matrix=evaluation.score_matrix,
        )

    def fit_evaluate(self, data: ProbeBackendInput) -> ProbeEvaluation:
        """Train the built-in probe and return its computed evaluation."""
        probe = self._probe_spec()
        if data.level == "protein":
            return train_and_evaluate_probe(
                train_ids=data.train_ids,
                test_ids=data.test_ids,
                embeddings=cast(Mapping[str, Sequence[float]], data.embeddings),
                labels=cast(Mapping[str, object], data.labels),
                prediction=data.prediction,
                probe=probe,
                flat_data=cast(ProteinDataFlat | None, data.flat_data),
            )
        if data.level == "residue":
            return train_and_evaluate_residue_probe(
                train_ids=data.train_ids,
                test_ids=data.test_ids,
                embeddings=cast(
                    Mapping[str, Sequence[Sequence[float]]],
                    data.embeddings,
                ),
                labels=cast(Mapping[str, Sequence[object]], data.labels),
                masks=data.masks,
                prediction=data.prediction,
                probe=probe,
                feature_mean=data.feature_mean,
                feature_std=data.feature_std,
                flat_data=cast(ResidueDataFlat | None, data.flat_data),
            )
        raise EmbeddingInputError(f"Unsupported built-in probe level: {data.level!r}.")

    def _probe_spec(self) -> ProbeSpec:
        return ProbeSpec(
            kind=self.kind,
            epochs=self.epochs,
            learning_rate=self.learning_rate,
            batch_size=self.batch_size,
            weight_decay=self.weight_decay,
            hidden_dim=self._resolved_hidden_dim(),
            seed=self.seed,
        )

    def _resolved_hidden_dim(self) -> int:
        return 64


@dataclass(frozen=True)
class LinearProbe(_TorchProbeBackend):
    """Train a linear probe through the common backend interface."""

    @property
    def kind(self) -> ProbeKind:
        """Return the linear architecture name."""
        return "linear"


@dataclass(frozen=True)
class MlpProbe(_TorchProbeBackend):
    """Train an MLP probe through the common backend interface."""

    hidden_dim: int = 64

    @property
    def kind(self) -> ProbeKind:
        """Return the MLP architecture name."""
        return "mlp"

    def _resolved_hidden_dim(self) -> int:
        return self.hidden_dim


@dataclass(frozen=True)
class TransferProbe(ProbeBackend):
    """Transfer labels from nearest training embeddings."""

    k: int = 10
    threshold: float | None = None
    distance: TransferDistance = "cosine"
    neighbor_selection: TransferNeighborSelection = "all"
    distance_cutoff: float | None = None
    scoring: TransferScoring = "weighted_voting"
    search_backend: TransferSearchBackend = "numpy"
    search_device: str | None = None
    search_ann: bool = False
    return_neighbors: bool = False

    def __post_init__(self) -> None:
        if self.k < 1:
            raise EmbeddingInputError("TransferProbe.k must be >= 1.")
        if self.threshold is not None and not 0.0 <= self.threshold <= 1.0:
            raise EmbeddingInputError("TransferProbe.threshold must be between 0 and 1.")
        if self.distance not in {"cosine", "euclidean"}:
            raise EmbeddingInputError("TransferProbe.distance must be one of: cosine, euclidean.")
        if self.neighbor_selection not in {"knn", "all", "cutoff_distance"}:
            raise EmbeddingInputError(
                "TransferProbe.neighbor_selection must be one of: knn, all, cutoff_distance."
            )
        if self.neighbor_selection == "cutoff_distance" and self.distance_cutoff is None:
            raise EmbeddingInputError(
                "TransferProbe.distance_cutoff is required when neighbor_selection='cutoff_distance'."
            )
        if self.distance_cutoff is not None and self.distance_cutoff < 0.0:
            raise EmbeddingInputError("TransferProbe.distance_cutoff must be >= 0 when provided.")
        if self.scoring not in {
            "voting",
            "weighted_voting",
            "nearest_similarity",
        }:
            raise EmbeddingInputError(
                "TransferProbe.scoring must be one of: voting, weighted_voting, "
                "nearest_similarity."
            )
        if self.search_backend not in {"numpy", "auto", "faiss_cpu", "faiss_gpu", "cuvs_gpu", "torch_gpu"}:
            raise EmbeddingInputError(
                "TransferProbe.search_backend must be one of: numpy, auto, faiss_cpu, faiss_gpu, cuvs_gpu, torch_gpu."
            )
        if self.search_backend != "numpy" and self.neighbor_selection != "knn":
            raise EmbeddingInputError(
                "TransferProbe accelerated search backends support neighbor_selection='knn' only."
            )

    def fit_predict(self, data: ProbeBackendInput) -> ProbeBackendOutput:
        """Predict by transferred votes from selected training embeddings."""
        if data.level != "protein":
            raise EmbeddingInputError("TransferProbe supports protein-level tasks only.")
        if data.prediction.objective == "regression":
            return self._predict_regression(data)
        if data.prediction.objective == "binary":
            return self._predict_binary(data)
        if data.prediction.objective == "multiclass":
            return self._predict_multiclass(data)
        if data.prediction.objective == "multilabel":
            return self._predict_multilabel(data)
        raise EmbeddingInputError(f"Unsupported TransferProbe objective: {data.prediction.objective!r}.")

    def _predict_regression(self, data: ProbeBackendInput) -> ProbeBackendOutput:
        predictions: Dict[str, ProbePredictionOutput] = {}
        weighted_neighbors = self._weighted_neighbors(data)
        for item, neighbor_weights in weighted_neighbors.items():
            total = sum(weight for _neighbor_id, _distance, weight in neighbor_weights)
            value = sum(
                float(cast(float | int | str, data.labels[neighbor_id])) * weight
                for neighbor_id, _distance, weight in neighbor_weights
            ) / total
            predictions[item] = float(value)
        return ProbeBackendOutput(
            predictions=predictions,
            metadata=self._neighbor_metadata(data, weighted_neighbors) if self.return_neighbors else None,
        )

    def _predict_binary(self, data: ProbeBackendInput) -> ProbeBackendOutput:
        predictions: Dict[str, ProbePredictionOutput] = {}
        scores: Dict[str, ProbeScoreOutput] = {}
        weighted_neighbors = self._weighted_neighbors(data)
        for item, neighbor_weights in weighted_neighbors.items():
            if self.scoring == "nearest_similarity":
                score = max(
                    (
                        weight
                        for neighbor_id, _distance, weight in neighbor_weights
                        if _as_binary(data.labels[neighbor_id], neighbor_id) == 1
                    ),
                    default=0.0,
                )
            else:
                total = sum(weight for _neighbor_id, _distance, weight in neighbor_weights)
                score = sum(
                    _as_binary(data.labels[neighbor_id], neighbor_id) * weight
                    for neighbor_id, _distance, weight in neighbor_weights
                ) / total
            threshold = 0.5 if self.threshold is None else self.threshold
            scores[item] = float(score)
            predictions[item] = 1 if score >= threshold else 0
        return ProbeBackendOutput(
            predictions=predictions,
            scores=scores,
            metadata=self._neighbor_metadata(data, weighted_neighbors) if self.return_neighbors else None,
        )

    def _predict_multiclass(self, data: ProbeBackendInput) -> ProbeBackendOutput:
        classes = _transfer_classes(data)
        predictions: Dict[str, ProbePredictionOutput] = {}
        weighted_neighbors = self._weighted_neighbors(data)
        for item, neighbor_weights in weighted_neighbors.items():
            votes = dict.fromkeys(classes, 0.0)
            for neighbor_id, _distance, weight in neighbor_weights:
                label = str(data.labels[neighbor_id])
                if label not in votes:
                    raise EmbeddingInputError(
                        f"Unknown multiclass label {label!r} for training example {neighbor_id!r}."
                    )
                votes[label] += weight
            predictions[item] = max(classes, key=lambda label: (votes[label], -classes.index(label)))
        return ProbeBackendOutput(
            predictions=predictions,
            metadata=self._neighbor_metadata(data, weighted_neighbors) if self.return_neighbors else None,
        )

    def _predict_multilabel(self, data: ProbeBackendInput) -> ProbeBackendOutput:
        classes = _transfer_multilabel_classes(data)
        class_indices = {label: index for index, label in enumerate(classes)}
        training_label_indices: dict[str, _np.ndarray[Any, _np.dtype[_np.int64]]] = {}
        predictions: Dict[str, ProbePredictionOutput] = {}
        score_matrix = _np.zeros(
            (len(data.test_ids), len(classes)),
            dtype=_np.float64,
        )
        weighted_neighbors = self._weighted_neighbors(data)
        for row_index, item in enumerate(data.test_ids):
            neighbor_weights = weighted_neighbors[item]
            row = score_matrix[row_index]
            for neighbor_id, _distance, weight in neighbor_weights:
                label_indices = training_label_indices.get(neighbor_id)
                if label_indices is None:
                    neighbor_labels = _as_multilabel(data.labels[neighbor_id], neighbor_id)
                    unknown_labels = [
                        label for label in neighbor_labels if label not in class_indices
                    ]
                    if unknown_labels:
                        raise EmbeddingInputError(
                            f"Unknown multilabel class {unknown_labels[0]!r} for "
                            f"training example {neighbor_id!r}."
                        )
                    label_indices = _np.fromiter(
                        (class_indices[label] for label in neighbor_labels),
                        dtype=_np.int64,
                        count=len(neighbor_labels),
                    )
                    training_label_indices[neighbor_id] = label_indices
                if self.scoring == "nearest_similarity":
                    row[label_indices] = _np.maximum(row[label_indices], weight)
                else:
                    row[label_indices] += weight
            if self.scoring != "nearest_similarity":
                total = sum(weight for _neighbor_id, _distance, weight in neighbor_weights)
                row /= total
            threshold = 0.5 if self.threshold is None else self.threshold
            predictions[item] = [
                classes[int(index)]
                for index in _np.flatnonzero(row >= threshold)
            ]
        return ProbeBackendOutput(
            predictions=predictions,
            score_matrix=score_matrix,
            metadata=self._neighbor_metadata(data, weighted_neighbors) if self.return_neighbors else None,
        )

    def _weighted_neighbors(self, data: ProbeBackendInput) -> Dict[str, list[tuple[str, float, float]]]:
        train_matrix = _embedding_matrix(data.train_ids, data.embeddings)
        test_matrix = _embedding_matrix(data.test_ids, data.embeddings)
        _validate_distance_matrix(train_matrix, "train", distance=self.distance)
        _validate_distance_matrix(test_matrix, "test", distance=self.distance)
        if self.search_backend != "numpy":
            return self._weighted_neighbors_from_search(data, train_matrix=train_matrix, test_matrix=test_matrix)
        distances_by_test = _distance_matrix(train_matrix, test_matrix, distance=self.distance)
        weighted: Dict[str, list[tuple[str, float, float]]] = {}
        for row_index, item in enumerate(data.test_ids):
            distances = distances_by_test[row_index]
            order = self._selected_neighbor_indices(distances, item=item)
            selected_distances = distances[order]
            weights = self._neighbor_weights(selected_distances)
            weighted_neighbors = [
                (data.train_ids[int(index)], float(distance), float(weight))
                for index, distance, weight in zip(order, selected_distances, weights)
                if float(weight) > 0.0
            ]
            if not weighted_neighbors:
                raise EmbeddingInputError(f"TransferProbe found no positive transfer weights for example {item!r}.")
            weighted[item] = weighted_neighbors
        return weighted

    def _weighted_neighbors_from_search(
        self,
        data: ProbeBackendInput,
        *,
        train_matrix: Any,
        test_matrix: Any,
    ) -> Dict[str, list[tuple[str, float, float]]]:
        metric: DistanceMetric = "cosine" if self.distance == "cosine" else "l2"
        backend, device = _resolve_transfer_search_backend(self.search_backend, self.search_device)
        try:
            state = build_search_state(
                backend=backend,
                item_ids=data.train_ids,
                vectors=train_matrix,
                metric=metric,
                device=device,
                ann_requested=self.search_ann,
            )
            neighbors_by_test = search_state(
                state,
                query_ids=data.test_ids,
                query_vectors=test_matrix,
                k=self.k,
            )
        except DriverDependencyError as exc:
            raise EmbeddingDependencyError(str(exc)) from exc
        weighted: Dict[str, list[tuple[str, float, float]]] = {}
        for item in data.test_ids:
            neighbors = neighbors_by_test[str(item)]
            distances = _np.asarray([neighbor.distance for neighbor in neighbors], dtype=_np.float32)
            weights = self._neighbor_weights(distances)
            weighted_neighbors = [
                (neighbor.protein_id, float(neighbor.distance), float(weight))
                for neighbor, weight in zip(neighbors, weights)
                if float(weight) > 0.0
            ]
            if not weighted_neighbors:
                raise EmbeddingInputError(f"TransferProbe found no positive transfer weights for example {item!r}.")
            weighted[str(item)] = weighted_neighbors
        return weighted

    def _neighbor_metadata(
        self,
        data: ProbeBackendInput,
        weighted_neighbors: Mapping[str, Sequence[tuple[str, float, float]]],
    ) -> dict[str, object]:
        return {
            "transfer_neighbors": {
                item: [
                    {
                        "id": neighbor_id,
                        "distance": float(distance),
                        "weight": float(weight),
                        "label": data.labels[neighbor_id],
                    }
                    for neighbor_id, distance, weight in weighted_neighbors[item]
                ]
                for item in data.test_ids
            }
        }

    def _selected_neighbor_indices(self, distances: Any, *, item: str) -> Any:
        order = _np.argsort(distances, kind="stable")
        if self.neighbor_selection == "knn":
            return order[: min(int(self.k), int(len(order)))]
        if self.neighbor_selection == "all":
            return order
        cutoff = float(cast(float, self.distance_cutoff))
        selected = order[distances[order] <= cutoff]
        if int(len(selected)) < 1:
            raise EmbeddingInputError(
                f"TransferProbe found no neighbors within distance_cutoff={cutoff} for example {item!r}."
            )
        return selected

    def _neighbor_weights(self, distances: Any) -> list[float]:
        if self.scoring == "voting":
            return [1.0] * int(len(distances))
        if self.scoring in {"weighted_voting", "nearest_similarity"}:
            return [
                _bounded_similarity(float(distance), distance=self.distance)
                for distance in distances
            ]
        raise EmbeddingInputError(f"Unsupported TransferProbe.scoring value {self.scoring!r}.")


@dataclass(frozen=True)
class ProteinDataFlat:
    """Pre-standardized protein embeddings for one task split and layer.

    Build once per layer with ``compute_protein_flat_data`` and reuse across
    probe seeds that have the same train and test identifiers.
    """

    x_train: _npt.NDArray[_np.float32]
    train_ids: tuple[str, ...]
    x_test: _npt.NDArray[_np.float32]
    test_ids: tuple[str, ...]


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
    flat_data: ProteinDataFlat | None = None,
) -> ProbeEvaluation:
    """Train a small PyTorch probe and evaluate it on the test split."""

    torch = _import_torch()
    ordered_ids = list(train_ids) + list(test_ids)
    if flat_data is None:
        input_dim = _validate_embeddings(ordered_ids, embeddings)
    else:
        input_dim = _validate_protein_flat_data(
            flat_data,
            train_ids=train_ids,
            test_ids=test_ids,
        )
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

    feature_mean: Any = None
    feature_std: Any = None
    if flat_data is None:
        train_x = _tensor_for_ids(torch, train_ids, embeddings)
        feature_mean, feature_std = _feature_standardization_stats(torch, train_x)
        feature_mean = feature_mean.to(device)
        feature_std = feature_std.to(device)
        train_x = _standardize_features(
            train_x.to(device),
            mean=feature_mean,
            std=feature_std,
        )
    else:
        train_x = torch.from_numpy(flat_data.x_train).to(device)
    train_y = _label_tensor(torch, [encoded_labels[item] for item in train_ids], prediction.objective).to(device)
    batch_size = int(probe.batch_size or len(train_ids))
    use_manual_multiclass_gradient = (
        probe.kind == "linear"
        and prediction.objective == "multiclass"
        and output_dim >= _CPU_MANUAL_MULTICLASS_MIN_CLASSES
        and getattr(device, "type", None) == "cpu"
    )

    model.train()
    for _epoch in range(int(probe.epochs)):
        for start in range(0, len(train_ids), batch_size):
            end = start + batch_size
            batch_x = train_x[start:end]
            batch_y = train_y[start:end]
            if use_manual_multiclass_gradient:
                _train_linear_multiclass_batch(
                    torch,
                    model=model,
                    optimizer=optimizer,
                    batch_x=batch_x,
                    batch_y=batch_y,
                )
            else:
                optimizer.zero_grad()
                logits = model(batch_x)
                loss = _compute_loss(logits, batch_y, loss_fn, prediction.objective)
                loss.backward()
                optimizer.step()

    model.eval()
    with torch.inference_mode():
        if flat_data is None:
            test_x = _tensor_for_ids(torch, test_ids, embeddings).to(device)
            test_x = _standardize_features(
                test_x,
                mean=feature_mean,
                std=feature_std,
            )
        else:
            test_x = torch.from_numpy(flat_data.x_test).to(device)
        logits = model(test_x)

    return _evaluate_outputs(
        torch,
        test_ids=test_ids,
        logits=logits,
        labels={item: encoded_labels[item] for item in test_ids},
        objective=prediction.objective,
        class_names=class_names,
    )


def compute_protein_flat_data(
    *,
    train_ids: Sequence[str],
    test_ids: Sequence[str],
    embeddings: Mapping[str, Sequence[float]],
) -> ProteinDataFlat:
    """Prepare standardized protein embeddings for reuse across probe seeds."""

    torch = _import_torch()
    if len(train_ids) < 1:
        raise EmbeddingInputError(
            "Prepared protein data requires at least one train identifier."
        )
    if len(test_ids) < 1:
        raise EmbeddingInputError(
            "Prepared protein data requires at least one test identifier."
        )
    ordered_ids = list(train_ids) + list(test_ids)
    _validate_embeddings(ordered_ids, embeddings)
    train_x = _tensor_for_ids(torch, train_ids, embeddings)
    feature_mean, feature_std = _feature_standardization_stats(torch, train_x)
    train_x = _standardize_features(
        train_x,
        mean=feature_mean,
        std=feature_std,
    )
    test_x = _standardize_features(
        _tensor_for_ids(torch, test_ids, embeddings),
        mean=feature_mean,
        std=feature_std,
    )
    return ProteinDataFlat(
        x_train=train_x.numpy(),
        train_ids=tuple(train_ids),
        x_test=test_x.numpy(),
        test_ids=tuple(test_ids),
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
    train_label_ids = (
        flat_data.train_label_ids
        if flat_data is not None
        else ordered_labels[0][:train_count]
    )
    y_train_t = _label_tensor(
        torch,
        [encoded_labels[label_id] for label_id in train_label_ids],
        prediction.objective,
    )

    model.train()
    if flat_data is not None:
        # Pre-flattened path: x_train is a contiguous numpy array; torch.as_tensor shares
        # its memory (zero copy).  Training loop does direct slicing — no pending_tensors
        # accumulation, no torch.cat, no per-protein Python overhead.
        x_train_t = torch.as_tensor(flat_data.x_train, dtype=torch.float32)
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
            label_start = 0
            for batch_x, _batch_label_ids in _iter_residue_batches(
                torch,
                train_ids,
                embeddings=embeddings,
                masks=masks,
                batch_size=batch_size,
                include_label_ids=False,
            ):
                label_end = label_start + int(batch_x.shape[0])
                batch_x_dev = _send_to_device(batch_x, device, x_pin)
                batch_y = y_train_t[label_start:label_end].to(device)
                batch_x_dev = _standardize_features(batch_x_dev, mean=feature_mean, std=feature_std)
                optimizer.zero_grad()
                logits = model(batch_x_dev)
                loss = _compute_loss(logits, batch_y, loss_fn, prediction.objective)
                loss.backward()
                optimizer.step()
                label_start = label_end

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


def _embedding_matrix(ids: Sequence[str], embeddings: Mapping[str, ProbeFeature]) -> Any:
    missing = [item for item in ids if item not in embeddings]
    if missing:
        sample = ", ".join(missing[:5])
        raise EmbeddingInputError(f"Missing embeddings for examples: {sample}.")
    try:
        matrix = _np.asarray(
            [[float(value) for value in cast(Sequence[float], embeddings[item])] for item in ids],
            dtype=_np.float32,
        )
    except (TypeError, ValueError) as exc:
        raise EmbeddingInputError("TransferProbe embeddings must be numeric vectors.") from exc
    if matrix.ndim != 2 or matrix.shape[1] < 1:
        raise EmbeddingInputError("Embeddings must be non-empty vectors.")
    return matrix


def _validate_distance_matrix(
    matrix: Any,
    split_name: str,
    *,
    distance: TransferDistance,
) -> None:
    if not _np.isfinite(matrix).all():
        raise EmbeddingInputError(f"TransferProbe {split_name} embeddings must contain finite values.")
    if distance == "cosine":
        norms = _np.linalg.norm(matrix, axis=1)
        if bool((norms <= 0.0).any()):
            raise EmbeddingInputError(
                f"TransferProbe {split_name} embeddings must not contain zero vectors for cosine distance."
            )


def _distance_matrix(train_matrix: Any, test_matrix: Any, *, distance: TransferDistance) -> Any:
    if distance == "cosine":
        train_unit = _normalize_rows(train_matrix)
        test_unit = _normalize_rows(test_matrix)
        similarities = test_unit @ train_unit.T
        return 1.0 - similarities
    differences = test_matrix[:, None, :] - train_matrix[None, :, :]
    return _np.linalg.norm(differences, axis=2)


def _bounded_similarity(distance_value: float, *, distance: TransferDistance) -> float:
    if distance == "cosine":
        return max(0.0, min(1.0, 1.0 - float(distance_value) / 2.0))
    return 1.0 / (1.0 + max(0.0, float(distance_value)))


def _normalize_rows(matrix: Any) -> Any:
    norms = _np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / norms


def _resolve_transfer_search_backend(
    backend: TransferSearchBackend,
    device: str | None,
) -> tuple[ResolvedSearchBackend, str]:
    if backend == "auto":
        faiss_device = preferred_faiss_device(device)
        if faiss_device is not None:
            return "faiss_gpu", faiss_device
        cuvs_device = preferred_cuvs_device(device)
        if cuvs_device is not None:
            return "cuvs_gpu", cuvs_device
        torch_device = preferred_torch_device(device)
        if torch_device is not None:
            return "torch_gpu", torch_device
        if import_faiss(allow_missing=True) is not None:
            return "faiss_cpu", "cpu"
        raise EmbeddingDependencyError(
            "TransferProbe search_backend='auto' could not find faiss, cuVS, or torch acceleration."
        )
    if backend == "faiss_cpu":
        return "faiss_cpu", "cpu"
    if backend == "faiss_gpu":
        faiss_device = preferred_faiss_device(device)
        if faiss_device is None:
            raise EmbeddingDependencyError("TransferProbe search_backend='faiss_gpu' requires FAISS GPU.")
        return "faiss_gpu", faiss_device
    if backend == "cuvs_gpu":
        cuvs_device = preferred_cuvs_device(device)
        if cuvs_device is None:
            raise EmbeddingDependencyError("TransferProbe search_backend='cuvs_gpu' requires cuVS and a CUDA device.")
        return "cuvs_gpu", cuvs_device
    if backend == "torch_gpu":
        torch_device = preferred_torch_device(device)
        if torch_device is None:
            raise EmbeddingDependencyError("TransferProbe search_backend='torch_gpu' requires torch and an accelerator.")
        return "torch_gpu", torch_device
    raise EmbeddingInputError(f"Unsupported TransferProbe.search_backend: {backend!r}.")


def _transfer_classes(data: ProbeBackendInput) -> list[str]:
    if data.prediction.classes is not None:
        classes = [str(value) for value in data.prediction.classes]
    else:
        classes = sorted({str(data.labels[item]) for item in (*data.train_ids, *data.test_ids)})
    if len(classes) < 2:
        raise EmbeddingInputError("Multiclass TransferProbe requires at least two classes.")
    return classes


def _transfer_multilabel_classes(data: ProbeBackendInput) -> list[str]:
    if data.prediction.classes is not None:
        classes = [str(value) for value in data.prediction.classes]
    else:
        classes = sorted(
            {
                label
                for item in (*data.train_ids, *data.test_ids)
                for label in _as_multilabel(data.labels[item], item)
            }
        )
    if not classes:
        raise EmbeddingInputError("Multilabel TransferProbe requires at least one class.")
    return classes


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


def _validate_protein_flat_data(
    flat_data: ProteinDataFlat,
    *,
    train_ids: Sequence[str],
    test_ids: Sequence[str],
) -> int:
    if flat_data.train_ids != tuple(train_ids):
        raise EmbeddingInputError(
            "Prepared protein train identifiers do not match the task train split."
        )
    if flat_data.test_ids != tuple(test_ids):
        raise EmbeddingInputError(
            "Prepared protein test identifiers do not match the task test split."
        )
    if flat_data.x_train.ndim != 2 or flat_data.x_test.ndim != 2:
        raise EmbeddingInputError("Prepared protein embeddings must be rank-2 matrices.")
    if flat_data.x_train.dtype != _np.float32 or flat_data.x_test.dtype != _np.float32:
        raise EmbeddingInputError(
            "Prepared protein embeddings must use float32 matrices."
        )
    if int(flat_data.x_train.shape[0]) != len(train_ids):
        raise EmbeddingInputError(
            "Prepared protein train embeddings do not match the train split length."
        )
    if int(flat_data.x_test.shape[0]) != len(test_ids):
        raise EmbeddingInputError(
            "Prepared protein test embeddings do not match the test split length."
        )
    input_dim = int(flat_data.x_train.shape[1])
    if input_dim < 1 or int(flat_data.x_test.shape[1]) != input_dim:
        raise EmbeddingInputError(
            "Prepared protein train and test embeddings must have the same non-empty dimension."
        )
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
        mask = masks.get(item)
        count += rows if mask is None else sum(bool(value) for value in mask)
    return count


def _residue_input_dim(
    ids: Sequence[str],
    *,
    embeddings: Mapping[str, Sequence[Sequence[float]]],
    masks: Mapping[str, Sequence[bool] | None],
) -> int:
    for item in ids:
        _rows, cols = _matrix_shape(embeddings[item])
        mask = masks.get(item)
        if mask is None or any(bool(value) for value in mask):
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
        include_label_ids=False,
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
    include_label_ids: bool = True,
) -> Any:
    resolved_batch_size = max(1, int(batch_size))
    pending_tensors: List[Any] = []
    pending_label_ids: List[str] = []
    pending_count = 0
    for item in ids:
        tensor = _residue_matrix_tensor(torch, item, embeddings[item])
        row_count = int(tensor.shape[0])
        mask = masks.get(item)
        if mask is None or all(bool(value) for value in mask):
            selected_count = row_count
            label_ids = (
                [f"{item}:{index + 1}" for index in range(row_count)]
                if include_label_ids
                else None
            )
        else:
            keep_indices = _valid_residue_indices(row_count, mask)
            if not keep_indices:
                continue
            index_tensor = torch.tensor(keep_indices, dtype=torch.long)
            tensor = tensor.index_select(0, index_tensor)
            selected_count = len(keep_indices)
            label_ids = (
                [f"{item}:{index + 1}" for index in keep_indices]
                if include_label_ids
                else None
            )
        start = 0
        while start < selected_count:
            available = resolved_batch_size - pending_count
            end = min(selected_count, start + available)
            pending_tensors.append(tensor[start:end])
            if label_ids is not None:
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
    if objective == "multilabel":
        return torch.nn.BCEWithLogitsLoss()
    if objective == "multiclass":
        return torch.nn.CrossEntropyLoss()
    raise EmbeddingInputError(f"Unsupported probe objective: {objective!r}.")


def _compute_loss(logits: Any, target: Any, loss_fn: Any, objective: ObjectiveName) -> Any:
    if objective in {"regression", "binary"}:
        return loss_fn(logits.reshape(-1), target)
    return loss_fn(logits, target)


def _train_linear_multiclass_batch(
    torch: Any,
    *,
    model: Any,
    optimizer: Any,
    batch_x: Any,
    batch_y: Any,
) -> None:
    # High-class-count CPU probes spend most of their time constructing the
    # autograd graph. This is the exact gradient of mean cross-entropy for a
    # linear layer and leaves parameter updates to the configured optimizer.
    with torch.no_grad():
        logits = model(batch_x)
        output_gradient = torch.softmax(logits, dim=1)
        row_indices = torch.arange(int(batch_y.shape[0]), device=batch_y.device)
        output_gradient[row_indices, batch_y] -= 1.0
        output_gradient /= int(batch_y.shape[0])
        model.weight.grad = output_gradient.transpose(0, 1).matmul(batch_x)
        if model.bias is not None:
            model.bias.grad = output_gradient.sum(dim=0)
        optimizer.step()


def _tensor_for_ids(
    torch: Any,
    ids: Sequence[str],
    embeddings: Mapping[str, Sequence[float]],
) -> Any:
    values = _np.asarray([embeddings[item] for item in ids], dtype=_np.float32)
    return torch.from_numpy(values)


def _label_tensor(torch: Any, values: Sequence[float | int | Sequence[float]], objective: ObjectiveName) -> Any:
    if objective in {"regression", "binary"}:
        scalar_values = cast(Sequence[float | int], values)
        return torch.tensor([float(value) for value in scalar_values], dtype=torch.float32)
    if objective == "multilabel":
        return torch.tensor(
            [[float(entry) for entry in cast(Sequence[float], value)] for value in values],
            dtype=torch.float32,
        )
    scalar_values = cast(Sequence[float | int], values)
    return torch.tensor([int(value) for value in scalar_values], dtype=torch.long)


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
) -> Tuple[Dict[str, float | int | List[float]], int, List[str] | None]:
    if objective == "regression":
        return ({item: float(cast(float | int | str, labels[item])) for item in ids}, 1, None)
    if objective == "binary":
        return ({item: _as_binary(labels[item], item) for item in ids}, 1, None)
    if objective == "multiclass":
        class_names = list(classes) if classes is not None else _infer_classes([labels[item] for item in ids])
        class_index = {name: index for index, name in enumerate(class_names)}
        encoded: Dict[str, float | int | List[float]] = {}
        for item in ids:
            value = labels[item]
            key = str(value)
            if key not in class_index:
                raise EmbeddingInputError(f"Unknown class label {value!r} for example {item!r}.")
            encoded[item] = class_index[key]
        return encoded, len(class_names), class_names
    if objective == "multilabel":
        label_sets = {item: _as_multilabel(labels[item], item) for item in ids}
        class_names = list(classes) if classes is not None else _infer_multilabel_classes(label_sets.values())
        class_index = {name: index for index, name in enumerate(class_names)}
        encoded: Dict[str, float | int | List[float]] = {}
        for item, item_labels in label_sets.items():
            vector = [0.0] * len(class_names)
            for label in item_labels:
                if label not in class_index:
                    raise EmbeddingInputError(f"Unknown multilabel class {label!r} for example {item!r}.")
                vector[class_index[label]] = 1.0
            encoded[item] = vector
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


def _as_multilabel(value: object, example_id: str) -> List[str]:
    if isinstance(value, str | bytes) or not isinstance(value, Sequence):
        raise EmbeddingInputError(
            f"Multilabel target for example {example_id!r} must be a sequence of labels."
        )
    raw_labels = cast(Sequence[object], value)
    return sorted({str(label).strip() for label in raw_labels if str(label).strip()})


def _infer_multilabel_classes(values: Iterable[Sequence[str]]) -> List[str]:
    classes = sorted({label for labels in values for label in labels})
    if not classes:
        raise EmbeddingInputError("Multilabel objective requires at least one class.")
    return classes


def _evaluate_outputs(
    torch: Any,
    *,
    test_ids: Sequence[str],
    logits: Any,
    labels: Mapping[str, float | int | Sequence[float]],
    objective: ObjectiveName,
    class_names: Sequence[str] | None,
) -> ProbeEvaluation:
    if objective == "regression":
        values = cast(List[float], logits.reshape(-1).detach().cpu().tolist())
        true_values = [float(cast(float | int, labels[item])) for item in test_ids]
        return ProbeEvaluation(
            metrics=regression_metrics(true_values, values),
            predictions={item: float(value) for item, value in zip(test_ids, values)},
        )
    if objective == "binary":
        binary_scores = cast(List[float], torch.sigmoid(logits.reshape(-1)).detach().cpu().tolist())
        pred_labels = [1 if score >= 0.5 else 0 for score in binary_scores]
        true_labels = [int(cast(float | int, labels[item])) for item in test_ids]
        return ProbeEvaluation(
            metrics=binary_metrics(true_labels, pred_labels, binary_scores),
            predictions={item: int(value) for item, value in zip(test_ids, pred_labels)},
            scores={item: float(value) for item, value in zip(test_ids, binary_scores)},
        )
    if objective == "multiclass":
        pred_indices = cast(List[int], torch.argmax(logits, dim=1).detach().cpu().tolist())
        true_indices = [int(cast(float | int, labels[item])) for item in test_ids]
        names = list(class_names or [])
        predictions: Dict[str, ProbePredictionOutput] = {}
        for item, index in zip(test_ids, pred_indices):
            predictions[item] = names[index] if names else int(index)
        return ProbeEvaluation(
            metrics=multiclass_metrics(true_indices, pred_indices, class_count=max(1, len(names))),
            predictions=predictions,
        )
    if objective == "multilabel":
        names = list(class_names or [])
        if not names:
            raise EmbeddingInputError("Multilabel evaluation requires class names.")
        score_matrix = torch.sigmoid(logits).detach().cpu().numpy()
        pred_matrix = (score_matrix >= 0.5).astype(_np.int8, copy=False)
        true_matrix = _np.asarray(
            [cast(Sequence[float], labels[item]) for item in test_ids],
            dtype=_np.int8,
        )
        predictions = {
            item: [names[index] for index, value in enumerate(row) if value == 1]
            for item, row in zip(test_ids, pred_matrix)
        }
        return ProbeEvaluation(
            metrics=multilabel_metrics(
                cast(Sequence[Sequence[int]], true_matrix),
                cast(Sequence[Sequence[int]], pred_matrix),
                cast(Sequence[Sequence[float]], score_matrix),
            ),
            predictions=predictions,
            score_matrix=score_matrix,
        )
    raise EmbeddingInputError(f"Unsupported probe objective: {objective!r}.")


def _expand_residue_evaluation(
    data: ProbeBackendInput,
    evaluation: ProbeEvaluation,
) -> ProbeBackendOutput:
    predictions: dict[str, list[ProbePrediction]] = {}
    scores: dict[str, list[float]] | None = (
        {} if evaluation.scores is not None else None
    )
    for item in data.test_ids:
        label_values = data.labels[item]
        if not isinstance(label_values, Sequence) or isinstance(label_values, str | bytes):
            raise EmbeddingInputError(
                f"Residue labels for {item!r} must be a sequence."
            )
        label_sequence = cast(Sequence[object], label_values)
        item_predictions: list[ProbePrediction] = [0] * len(label_sequence)
        item_scores = [0.0] * len(label_sequence) if scores is not None else None
        mask = data.masks.get(item)
        for index in range(len(label_sequence)):
            if mask is not None and not bool(mask[index]):
                continue
            residue_id = f"{item}:{index + 1}"
            if residue_id not in evaluation.predictions:
                raise EmbeddingInputError(
                    f"Built-in probe did not return a prediction for {residue_id!r}."
                )
            item_predictions[index] = cast(ProbePrediction, evaluation.predictions[residue_id])
            if item_scores is not None and evaluation.scores is not None:
                item_scores[index] = float(cast(float, evaluation.scores[residue_id]))
        predictions[item] = item_predictions
        if scores is not None and item_scores is not None:
            scores[item] = item_scores
    return ProbeBackendOutput(predictions=predictions, scores=scores)


__all__ = [
    "LinearProbe",
    "MlpProbe",
    "ProteinDataFlat",
    "ProbeEvaluation",
    "ResidueDataFlat",
    "TransferDistance",
    "TransferNeighborSelection",
    "TransferProbe",
    "TransferScoring",
    "TransferSearchBackend",
    "compute_protein_flat_data",
    "compute_residue_feature_stats",
    "compute_residue_flat_data",
    "train_and_evaluate_probe",
    "train_and_evaluate_residue_probe",
]
