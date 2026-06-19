"""Pooling helpers for embedding tensors and records."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Literal, Protocol, Sequence, TypeAlias, cast

from .. import (
    EmbeddingGenerationError,
    EmbeddingInputError,
    EmbeddingRecord,
    as_float_matrix,
    as_float_vector,
)

PoolerName: TypeAlias = Literal["none", "mean", "cls", "bos"]


class EmbeddingPooler(Protocol):
    """Protocol for transforming residue embeddings into output payloads."""
    @property
    def name(self) -> str:
        """Return the canonical pooler name."""

        ...
    def __call__(self, residue_tensor: object) -> object: ...


PoolerInput: TypeAlias = EmbeddingPooler | PoolerName | str | None


@dataclass(frozen=True)
class IdentityPooler:
    """Pooler that preserves residue-level embeddings."""
    name: str = "none"

    def __call__(self, residue_tensor: object) -> object:
        return residue_tensor


@dataclass(frozen=True)
class MeanPooler:
    """Pooler that averages residue-level embeddings."""
    name: str = "mean"

    def __call__(self, residue_tensor: object) -> object:
        pooled = _try_tensor_mean_pool_raw(residue_tensor)
        if pooled is not None:
            return pooled
        return mean_pool_matrix(as_float_matrix(residue_tensor))


@dataclass(frozen=True)
class ClsPooler:
    """Pooler placeholder for CLS/BOS-style outputs."""
    name: str = "cls"

    def __call__(self, residue_tensor: object) -> object:
        return residue_tensor


def pooler_factory(name: PoolerName | str | None) -> EmbeddingPooler | None:
    """Create a pooler instance from a pooler name."""
    if name is None:
        return None
    normalized = str(name).strip().lower()
    if normalized in {"", "none", "identity"}:
        return IdentityPooler()
    if normalized == "mean":
        return MeanPooler()
    if normalized in {"cls", "bos"}:
        return ClsPooler()
    raise EmbeddingInputError(f"Unsupported embedding pooler: {name!r}.")


def resolve_pooler(pooler: PoolerInput) -> EmbeddingPooler | None:
    """Resolve a pooler name or instance to a pooler object."""
    if pooler is None:
        return None
    if isinstance(pooler, str):
        resolved = pooler_factory(pooler)
        if isinstance(resolved, IdentityPooler):
            return None
        return resolved
    return pooler


def materialize_embedding_payload(value: object) -> tuple[Sequence[float] | Sequence[Sequence[float]], tuple[int, ...]]:
    """Convert tensor-like embedding output into a serializable payload."""
    import numpy as _np

    shape = getattr(value, "shape", None)
    if shape is not None:
        try:
            ndim = len(shape)
            if ndim in {1, 2}:
                # Fast path for tensor-like objects: avoid .tolist() which boxes every
                # element as a Python float (~28 bytes each).  For residue matrices
                # (L, D) this saves O(L*D) allocations — the dominant cost at large
                # batch sizes.  Returns a numpy float32 array which satisfies the
                # Sequence[float] | Sequence[Sequence[float]] contract and is accepted
                # directly by np.array() in the notebook with zero extra copy.
                detach = getattr(value, "detach", None)
                if callable(detach):
                    tensor = cast(Any, detach())
                    arr = _np.asarray(tensor.cpu().float().numpy(), dtype=_np.float32)
                    return cast(Sequence[float] | Sequence[Sequence[float]], arr), tuple(int(d) for d in arr.shape)
                # numpy arrays: ensure float32, return as-is
                to_numpy = getattr(value, "__array__", None)
                if callable(to_numpy):
                    arr = _np.asarray(value, dtype=_np.float32)
                    return cast(Sequence[float] | Sequence[Sequence[float]], arr), tuple(int(d) for d in arr.shape)
        except (TypeError, RuntimeError, AttributeError):
            pass
        try:
            if len(shape) == 1:
                vector = as_float_vector(value)
                return vector, (len(vector),)
            if len(shape) == 2:
                matrix = as_float_matrix(value)
                return matrix, _matrix_shape(matrix)
        except TypeError:
            pass

    tolist = getattr(value, "tolist", None)
    if callable(tolist):
        value = tolist()
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise EmbeddingInputError("Embedding payload must be a vector or row-major matrix.")
    values = cast(Sequence[object], value)
    if not values:
        raise EmbeddingInputError("Embedding payload is empty.")
    first = values[0]
    if isinstance(first, Sequence) and not isinstance(first, (str, bytes, bytearray)):
        matrix = as_float_matrix(values)
        return matrix, _matrix_shape(matrix)
    vector = as_float_vector(values)
    return vector, (len(vector),)


def as_mean_pooled_vector(value: object) -> List[float]:
    """Mean-pool a tensor-like or row-major embedding into one vector."""

    vector = _try_tensor_mean_pool(value)
    if vector is not None:
        return vector

    matrix = as_float_matrix(value)
    return mean_pool_matrix(matrix)


def mean_pool_embedding_record(record: EmbeddingRecord) -> EmbeddingRecord:
    """Return a vector-valued record by mean-pooling row-major embeddings."""

    emb = record.embedding
    # Peek at shape before materialising to Python — avoids O(L*D) list() conversion.
    # Default to () (empty tuple) so plain Python lists (no .shape) land in the else branch.
    ndim = len(getattr(emb, "shape", ()))
    if ndim == 2:
        vector = as_mean_pooled_vector(emb)
    elif ndim == 1:
        vector = as_float_vector(emb)
    else:
        # Plain Python list — check first element to decide
        values = list(emb)
        if not values:
            raise EmbeddingInputError(f"Record {record.id!r} has an empty embedding.")
        first = values[0]
        if isinstance(first, Sequence) and not isinstance(first, (str, bytes, bytearray)):
            vector = as_mean_pooled_vector(values)
        else:
            vector = as_float_vector(values)

    return EmbeddingRecord(
        id=record.id,
        embedding=vector,
        layer_index=int(record.layer_index),
        model_reference=record.model_reference,
        shape=(len(vector),),
        metadata=record.metadata,
    )


def mean_pool_matrix(matrix: Sequence[Sequence[float]]) -> List[float]:
    """Mean-pool a row-major embedding matrix into one vector."""
    # Fast path: numpy arrays avoid the O(L*D) Python element loop
    shape = getattr(matrix, "shape", None)
    if shape is not None:
        try:
            import numpy as _np
            arr = _np.asarray(matrix, dtype=_np.float32)
            if arr.ndim != 2 or arr.shape[0] == 0:
                raise EmbeddingInputError("Cannot mean-pool an empty embedding matrix.")
            return arr.mean(axis=0).tolist()
        except EmbeddingInputError:
            raise
        except Exception:
            pass

    rows = [as_float_vector(row) for row in matrix]
    if not rows or not rows[0]:
        raise EmbeddingInputError("Cannot mean-pool an empty embedding matrix.")

    dims = len(rows[0])
    if any(len(row) != dims for row in rows):
        raise EmbeddingInputError("Cannot mean-pool matrix with inconsistent row lengths.")

    pooled = [0.0] * dims
    for row in rows:
        for index, item in enumerate(row):
            pooled[index] += float(item)
    count = float(len(rows))
    return [item / count for item in pooled]


def _try_tensor_mean_pool(value: object) -> List[float] | None:
    pooled = _try_tensor_mean_pool_raw(value)
    if pooled is None:
        return None
    return as_float_vector(pooled)


def _try_tensor_mean_pool_raw(value: object) -> object | None:
    shape = getattr(value, "shape", None)
    mean = getattr(value, "mean", None)
    if shape is None or not callable(mean):
        return None
    try:
        if len(shape) != 2:
            return None
        if int(shape[0]) < 1:
            raise EmbeddingInputError("Cannot mean-pool an empty embedding matrix.")
        tensor: Any = value
        float_fn = getattr(tensor, "float", None)
        if callable(float_fn):
            tensor = cast(Any, float_fn())
        pooled: Any = tensor.mean(dim=0)
        detach = getattr(pooled, "detach", None)
        if callable(detach):
            pooled = cast(Any, detach())
        cpu = getattr(pooled, "cpu", None)
        if callable(cpu):
            pooled = cast(Any, cpu())
        return pooled
    except EmbeddingGenerationError:
        raise
    except Exception:
        return None


def _matrix_shape(matrix: Sequence[Sequence[float]]) -> tuple[int, int]:
    if not matrix or not matrix[0]:
        raise EmbeddingInputError("Embedding matrix is empty.")
    hidden_dim = len(matrix[0])
    if any(len(row) != hidden_dim for row in matrix):
        raise EmbeddingInputError("Embedding matrix rows do not share the same dimension.")
    return (len(matrix), hidden_dim)


__all__ = [
    "ClsPooler",
    "EmbeddingPooler",
    "IdentityPooler",
    "MeanPooler",
    "PoolerInput",
    "PoolerName",
    "as_mean_pooled_vector",
    "mean_pool_embedding_record",
    "mean_pool_matrix",
    "materialize_embedding_payload",
    "pooler_factory",
    "resolve_pooler",
]
