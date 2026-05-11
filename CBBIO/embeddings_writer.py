"""Writers for embedding generation jobs."""

from __future__ import annotations

from pathlib import Path
import pickle
from typing import Any, Dict, List, Literal, Sequence, Tuple, cast

from .embeddings import (
    EmbeddingDependencyError,
    EmbeddingInputError,
    EmbeddingPayload,
    EmbeddingRecord,
    PicklePayloadFormat,
    as_float_matrix,
    as_float_vector,
)

EmbeddingWriterFormat = Literal["memory", "pkl", "npy", "h5"]


class EmbeddingWriter:
    """Write embedding records to memory or file-backed formats."""

    def __init__(
        self,
        *,
        format: EmbeddingWriterFormat | str,
        path: str | Path | None = None,
        records_per_shard: int = 10_000,
        payload_format: PicklePayloadFormat = "records",
        compression: str | None = "gzip",
        write_batch_size: int = 1,
    ) -> None:
        self.format = str(format).strip().lower()
        if self.format not in {"memory", "pkl", "npy", "h5"}:
            raise EmbeddingInputError("format must be one of: 'memory', 'pkl', 'npy', 'h5'.")

        self.path = Path(path) if path is not None else None
        if self.format == "memory" and self.path is not None:
            raise EmbeddingInputError("path must be omitted when format='memory'.")
        if self.format != "memory" and self.path is None:
            raise EmbeddingInputError(f"path is required when format={self.format!r}.")

        self.records_per_shard = _validate_positive("records_per_shard", records_per_shard)
        if payload_format not in {"records", "mapping"}:
            raise EmbeddingInputError("payload_format must be one of: 'records', 'mapping'.")
        self.payload_format: PicklePayloadFormat = payload_format
        self.compression = compression
        self.write_batch_size = _validate_positive("write_batch_size", write_batch_size)

        self.records: List[EmbeddingRecord] = []
        self.paths: List[Path] = []
        self.id_paths: List[Path] = []
        self.record_count = 0
        self._pending: List[EmbeddingRecord] = []
        self._shard_index = 1
        self._h5_writer: _H5EmbeddingWriter | None = None

        if self.format == "h5":
            self._h5_writer = _H5EmbeddingWriter(cast(Path, self.path), compression=self.compression)

    def write(self, records: Sequence[EmbeddingRecord]) -> None:
        normalized = [_normalize_embedding_record(record) for record in records]
        if not normalized:
            return

        if self.format == "memory":
            self.records.extend(normalized)
            self.record_count += len(normalized)
            return

        if self.format in {"pkl", "npy"}:
            self._pending.extend(normalized)
            while len(self._pending) >= self.records_per_shard:
                self._write_pending_shard(self.records_per_shard)
            return

        if self.format == "h5":
            assert self._h5_writer is not None
            pending = list(normalized)
            while pending:
                chunk = pending[: self.write_batch_size]
                pending = pending[self.write_batch_size :]
                self._h5_writer.append(chunk)
                self.record_count += len(chunk)
            if self.path is not None and self.path not in self.paths:
                self.paths.append(self.path)
            return

        raise AssertionError("unreachable writer format")

    def close(self) -> None:
        if self.format in {"pkl", "npy"} and self._pending:
            self._write_pending_shard(len(self._pending))
        if self._h5_writer is not None:
            self._h5_writer.close()
            self._h5_writer = None

    def to_matrix(self) -> object:
        """Return memory records as a NumPy matrix."""
        try:
            import numpy as np
        except ModuleNotFoundError as exc:
            raise EmbeddingDependencyError(
                "NumPy is required for matrix conversion. Install with: pip install numpy"
            ) from exc
        return np.array([as_float_vector(record.embedding) for record in self.records], dtype=np.float32)

    def _write_pending_shard(self, count: int) -> None:
        shard_records = self._pending[:count]
        self._pending = self._pending[count:]
        assert self.path is not None

        if self.format == "pkl":
            shard_path = _pickle_shard_path(self.path, self._shard_index)
            _save_pickle(shard_path, shard_records, payload_format=self.payload_format)
            self.paths.append(shard_path)
        elif self.format == "npy":
            shard_path, ids_path = _write_npy_shard(self.path, shard_records, self._shard_index)
            self.paths.append(shard_path)
            self.id_paths.append(ids_path)
        else:
            raise AssertionError("sharded writer used for non-sharded format")

        self.record_count += len(shard_records)
        self._shard_index += 1


def _validate_positive(name: str, value: int) -> int:
    resolved = int(value)
    if resolved < 1:
        raise EmbeddingInputError(f"{name} must be >= 1.")
    return resolved


def _normalize_embedding_record(record: object) -> EmbeddingRecord:
    if not isinstance(record, EmbeddingRecord):
        raise EmbeddingInputError(f"Expected EmbeddingRecord, got {type(record).__name__}.")
    rec_id = str(record.id).strip()
    if not rec_id:
        raise EmbeddingInputError("EmbeddingRecord has empty id.")
    payload, shape = _normalize_payload(record.embedding)
    return EmbeddingRecord(
        id=rec_id,
        embedding=payload,
        layer_index=int(record.layer_index),
        model_reference=str(record.model_reference),
        shape=shape,
        metadata=record.metadata,
    )


def _normalize_payload(value: object) -> tuple[EmbeddingPayload, Tuple[int, ...]]:
    tolist = getattr(value, "tolist", None)
    if callable(tolist):
        value = tolist()
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)) and value:
        values = cast(Sequence[object], value)
        first = values[0]
        if isinstance(first, Sequence) and not isinstance(first, (str, bytes, bytearray)):
            matrix = as_float_matrix(values)
            if not matrix or not matrix[0]:
                raise EmbeddingInputError("Embedding matrix is empty.")
            width = len(matrix[0])
            if any(len(row) != width for row in matrix):
                raise EmbeddingInputError("Embedding matrix rows must have the same length.")
            return matrix, (len(matrix), width)
    vector = as_float_vector(cast(object, value))
    if not vector:
        raise EmbeddingInputError("Embedding payload is empty.")
    return vector, (len(vector),)


def _vector_records(records: Sequence[EmbeddingRecord]) -> List[EmbeddingRecord]:
    normalized: List[EmbeddingRecord] = []
    for index, record in enumerate(records):
        vector = as_float_vector(record.embedding)
        if tuple(record.shape) != (len(vector),):
            raise EmbeddingInputError(
                f"Record at index {index} must contain one vector for this writer, got shape {record.shape}."
            )
        normalized.append(record)
    return normalized


def _save_pickle(
    path: str | Path,
    records: Sequence[EmbeddingRecord],
    *,
    payload_format: PicklePayloadFormat,
) -> None:
    file_path = Path(path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    if payload_format == "records":
        payload: Dict[str, Any] = {
            "records": [
                {
                    "id": record.id,
                    "embedding": list(record.embedding),
                    "layer_index": int(record.layer_index),
                    "model_reference": record.model_reference,
                    "shape": tuple(record.shape),
                    "metadata": record.metadata,
                }
                for record in records
            ]
        }
    elif payload_format == "mapping":
        payload = {record.id: list(record.embedding) for record in records}
    else:
        raise EmbeddingInputError("payload_format must be one of: 'records', 'mapping'.")
    with file_path.open("wb") as handle:
        pickle.dump(payload, handle)


def _save_npy(path: str | Path, records: Sequence[EmbeddingRecord]) -> None:
    try:
        import numpy as np
    except ModuleNotFoundError as exc:
        raise EmbeddingDependencyError(
            "NumPy is required for .npy saving. Install with: pip install numpy"
        ) from exc

    normalized = _vector_records(records)
    dims = {len(cast(Sequence[float], record.embedding)) for record in normalized}
    if len(dims) != 1:
        raise EmbeddingInputError("All embeddings must have the same length for .npy output.")
    matrix = np.array([list(record.embedding) for record in normalized], dtype=np.float32)
    file_path = Path(path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(file_path, matrix)


def _pickle_shard_path(path: str | Path, shard_index: int) -> Path:
    output_path = Path(path)
    suffix = output_path.suffix or ".pkl"
    stem = output_path.stem if output_path.suffix else output_path.name
    return output_path.with_name(f"{stem}.shard_{int(shard_index):06d}{suffix}")


def _npy_shard_path(path: str | Path, shard_index: int) -> Path:
    output_path = Path(path)
    stem = output_path.stem if output_path.suffix else output_path.name
    return output_path.with_name(f"{stem}.shard_{int(shard_index):06d}.npy")


def _npy_ids_path(npy_path: str | Path) -> Path:
    return Path(npy_path).with_suffix(".ids.txt")


def _write_npy_shard(
    path: str | Path,
    records: Sequence[EmbeddingRecord],
    shard_index: int,
) -> tuple[Path, Path]:
    shard_path = _npy_shard_path(path, shard_index)
    ids_path = _npy_ids_path(shard_path)
    _save_npy(shard_path, records)
    ids_path.write_text("".join(f"{record.id}\n" for record in records), encoding="utf-8")
    return shard_path, ids_path


class _H5EmbeddingWriter:
    def __init__(self, path: str | Path, *, compression: str | None, append: bool = False) -> None:
        try:
            import h5py  # type: ignore
        except ModuleNotFoundError as exc:
            raise EmbeddingDependencyError(
                "h5py is required for HDF5 embedding output. Install with: pip install h5py"
            ) from exc

        h5py_module = cast(Any, h5py)
        self._h5py: Any = h5py_module
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.compression = compression
        self.handle: Any = h5py_module.File(self.path, "a" if append else "w")
        self.record_count = int(self.handle.attrs.get("record_count", 0))

    def append(self, records: Sequence[EmbeddingRecord]) -> None:
        try:
            import numpy as np
        except ModuleNotFoundError as exc:
            raise EmbeddingDependencyError(
                "NumPy is required for HDF5 embedding output. Install with: pip install numpy"
            ) from exc

        normalized = _vector_records(records)
        if not normalized:
            return

        matrix = np.array([list(record.embedding) for record in normalized], dtype=np.float32)
        ids = [record.id for record in normalized]
        layer_indices = np.array([int(record.layer_index) for record in normalized], dtype=np.int32)
        model_refs = [record.model_reference for record in normalized]

        if "embeddings" not in self.handle:
            dim = int(matrix.shape[1])
            string_dtype = self._h5py.string_dtype(encoding="utf-8")
            self.handle.create_dataset(
                "embeddings",
                data=matrix,
                maxshape=(None, dim),
                chunks=True,
                compression=self.compression,
            )
            self.handle.create_dataset("ids", data=ids, maxshape=(None,), dtype=string_dtype, chunks=True)
            self.handle.create_dataset("layer_index", data=layer_indices, maxshape=(None,), chunks=True)
            self.handle.create_dataset(
                "model_reference",
                data=model_refs,
                maxshape=(None,),
                dtype=string_dtype,
                chunks=True,
            )
            self.record_count = int(matrix.shape[0])
        else:
            embeddings = self.handle["embeddings"]
            if int(embeddings.shape[1]) != int(matrix.shape[1]):
                raise EmbeddingInputError(
                    f"HDF5 embedding dimension mismatch: file has {embeddings.shape[1]}, new batch has {matrix.shape[1]}."
                )
            old_size = int(embeddings.shape[0])
            new_size = old_size + int(matrix.shape[0])
            for dataset_name in ("embeddings", "ids", "layer_index", "model_reference"):
                shape = (new_size, matrix.shape[1]) if dataset_name == "embeddings" else (new_size,)
                self.handle[dataset_name].resize(shape)
            self.handle["embeddings"][old_size:new_size] = matrix
            self.handle["ids"][old_size:new_size] = ids
            self.handle["layer_index"][old_size:new_size] = layer_indices
            self.handle["model_reference"][old_size:new_size] = model_refs
            self.record_count = new_size

        self.handle.attrs["record_count"] = self.record_count
        self.handle.flush()

    def close(self) -> None:
        self.handle.close()


__all__ = ["EmbeddingWriter", "EmbeddingWriterFormat"]
