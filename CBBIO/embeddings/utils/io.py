"""Embedding persistence and file-backed streaming helpers."""

from __future__ import annotations

from collections.abc import Iterable, Mapping as MappingABC
from dataclasses import dataclass
import pickle
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple, cast

from .. import (
    EmbeddingDependencyError,
    EmbeddingInputError,
    EmbeddingPayload,
    EmbeddingRecord,
    H5WriteResult,
    NpyShardWriteResult,
    PicklePayloadFormat,
    PickleShardWriteResult,
    as_float_vector,
)
from .pooler import mean_pool_embedding_record
from .writer import _H5EmbeddingWriter, _normalize_embedding_record  # pyright: ignore[reportPrivateUsage]


def load_embedding_records(
    path: str | Path,
    *,
    model_reference: str = "unknown",
    layer_index: int = 0,
    ids: Sequence[str] | None = None,
) -> List[EmbeddingRecord]:
    """Load embedding records from supported file formats."""
    file_path = Path(path)
    suffix = file_path.suffix.lower()
    if suffix in {".pkl", ".pickle"}:
        return load_embedding_records_pickle(
            file_path,
            model_reference=model_reference,
            layer_index=layer_index,
        )
    if suffix in {".npy", ".npz"}:
        return load_embedding_records_npy(
            file_path,
            model_reference=model_reference,
            layer_index=layer_index,
            ids=ids,
        )
    if suffix in {".h5", ".hdf5"}:
        return load_embedding_records_h5(file_path)
    raise EmbeddingInputError(f"Unsupported embedding file extension: {suffix!r}")


def save_embedding_records_pickle(
    path: str | Path,
    records: Sequence[EmbeddingRecord],
    *,
    payload_format: PicklePayloadFormat = "records",
) -> Path:
    """Save embedding records to a pickle file."""
    file_path = Path(path)
    normalized = _normalize_embedding_records(records)

    file_path.parent.mkdir(parents=True, exist_ok=True)
    if payload_format == "records":
        payload = {
            "records": [
                {
                    "id": record.id,
                    "embedding": list(record.embedding),
                    "layer_index": int(record.layer_index),
                    "model_reference": record.model_reference,
                    "shape": tuple(record.shape),
                    "metadata": record.metadata,
                }
                for record in normalized
            ]
        }
    elif payload_format == "mapping":
        payload = {record.id: list(record.embedding) for record in normalized}
    else:
        raise EmbeddingInputError("payload_format must be one of: 'records', 'mapping'.")

    with file_path.open("wb") as handle:
        pickle.dump(payload, handle)
    return file_path


def save_embedding_records_pickle_shards(
    path: str | Path,
    records: Iterable[EmbeddingRecord],
    *,
    records_per_shard: int,
    payload_format: PicklePayloadFormat = "records",
    start_index: int = 1,
) -> PickleShardWriteResult:
    """Stream embedding records to numbered pickle shards."""
    resolved_records_per_shard = _validate_records_per_shard(records_per_shard)
    shard_index = int(start_index)
    if shard_index < 1:
        raise EmbeddingInputError("start_index must be >= 1.")

    output_path = Path(path)
    pending: List[EmbeddingRecord] = []
    paths: List[Path] = []
    record_count = 0

    for record in records:
        pending.append(record)
        if len(pending) >= resolved_records_per_shard:
            shard_path = _write_pickle_shard(
                output_path,
                pending,
                shard_index,
                payload_format=payload_format,
            )
            paths.append(shard_path)
            record_count += len(pending)
            pending = []
            shard_index += 1

    if pending:
        shard_path = _write_pickle_shard(
            output_path,
            pending,
            shard_index,
            payload_format=payload_format,
        )
        paths.append(shard_path)
        record_count += len(pending)

    return PickleShardWriteResult(paths=paths, record_count=record_count)


def save_embedding_records_npy(
    path: str | Path,
    records: Sequence[EmbeddingRecord],
) -> Path:
    """Save embedding records to a numeric ``.npy`` matrix."""
    try:
        import numpy as np
    except ModuleNotFoundError as exc:
        raise EmbeddingDependencyError(
            "NumPy is required for .npy saving. Install with: pip install numpy"
        ) from exc

    normalized = _normalize_embedding_records(records)
    if not normalized:
        raise EmbeddingInputError("Cannot save empty embedding record list to .npy.")

    dims = {len(record.embedding) for record in normalized}
    if len(dims) != 1:
        raise EmbeddingInputError("All embeddings must have the same length for .npy output.")

    matrix = np.array([list(record.embedding) for record in normalized], dtype=np.float32)
    file_path = Path(path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(file_path, matrix)
    return file_path


def save_embedding_records_npy_shards(
    path: str | Path,
    records: Iterable[EmbeddingRecord],
    *,
    records_per_shard: int,
    start_index: int = 1,
) -> NpyShardWriteResult:
    """Stream vector embeddings to numbered ``.npy`` matrix shards."""
    resolved_records_per_shard = _validate_records_per_shard(records_per_shard)
    shard_index = int(start_index)
    if shard_index < 1:
        raise EmbeddingInputError("start_index must be >= 1.")

    output_path = Path(path)
    pending: List[EmbeddingRecord] = []
    paths: List[Path] = []
    id_paths: List[Path] = []
    record_count = 0

    for record in records:
        pending.append(record)
        if len(pending) >= resolved_records_per_shard:
            shard_path, ids_path = _write_npy_shard(output_path, pending, shard_index)
            paths.append(shard_path)
            id_paths.append(ids_path)
            record_count += len(pending)
            pending = []
            shard_index += 1

    if pending:
        shard_path, ids_path = _write_npy_shard(output_path, pending, shard_index)
        paths.append(shard_path)
        id_paths.append(ids_path)
        record_count += len(pending)

    return NpyShardWriteResult(paths=paths, id_paths=id_paths, record_count=record_count)


def save_embedding_records_h5(
    path: str | Path,
    records: Sequence[EmbeddingRecord],
    *,
    append: bool = False,
    compression: str | None = "gzip",
) -> H5WriteResult:
    """Write or append vector/matrix embeddings to one extendable HDF5 file."""
    normalized = [_normalize_embedding_record(record) for record in records]
    if not normalized:
        raise EmbeddingInputError("Cannot save empty embedding record list to HDF5.")

    file_path = Path(path)
    writer = _H5EmbeddingWriter(file_path, append=append, compression=compression)
    try:
        writer.append(normalized)
        return H5WriteResult(path=file_path, record_count=writer.record_count)
    finally:
        writer.close()


def load_embedding_records_pickle(
    path: str | Path,
    *,
    model_reference: str = "unknown",
    layer_index: int = 0,
) -> List[EmbeddingRecord]:
    """Load embeddings from pickle and normalize to ``EmbeddingRecord`` values."""
    file_path = Path(path)
    with file_path.open("rb") as handle:
        payload = pickle.load(handle)
    return _records_from_payload(payload, model_reference=model_reference, layer_index=layer_index)


@dataclass(frozen=True)
class _H5RecordIndex:
    ids: List[str]
    layer_values: List[int]
    model_values: List[str]
    pool_methods: List[str]
    payload_kinds: List[str] | None
    vector_index: List[int]
    matrix_index: List[int]


class H5EmbeddingReader:
    """Indexed reader for HDF5 embedding files.

    The reader keeps only record metadata in memory. Matrix payloads are returned
    as lazy row-slice objects, so reading one protein segment does not materialize
    unrelated proteins, layers, or residues.
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        try:
            import h5py  # type: ignore
        except ModuleNotFoundError as exc:
            raise EmbeddingDependencyError(
                "h5py is required for HDF5 loading. Install with: pip install h5py"
            ) from exc
        with h5py.File(self.path, "r") as handle:
            self._index = _read_h5_record_index(handle)

    def read(
        self,
        record_id: str,
        *,
        layer_index: int | None = None,
        pool_method: str | None = None,
        residue_start: int | None = None,
        residue_end: int | None = None,
        residue_slice: slice | tuple[int | None, int | None] | None = None,
    ) -> EmbeddingRecord:
        """Read one record, optionally restricted to a residue row range."""
        records = self.read_many(
            ids=[record_id],
            layer_index=layer_index,
            pool_method=pool_method,
            residue_start=residue_start,
            residue_end=residue_end,
            residue_slice=residue_slice,
        )
        if not records:
            detail = f"id={record_id!r}"
            if layer_index is not None:
                detail += f", layer_index={int(layer_index)}"
            if pool_method is not None:
                detail += f", pool_method={pool_method!r}"
            raise EmbeddingInputError(f"No HDF5 embedding record found for {detail}.")
        if len(records) > 1:
            raise EmbeddingInputError(
                f"Multiple HDF5 embedding records found for id={record_id!r}; pass layer_index and/or pool_method to select one."
            )
        return records[0]

    def read_many(
        self,
        *,
        ids: str | Sequence[str] | None = None,
        layer_index: int | Sequence[int] | None = None,
        pool_method: str | Sequence[str] | None = None,
        residue_start: int | None = None,
        residue_end: int | None = None,
        residue_slice: slice | tuple[int | None, int | None] | None = None,
        eager: bool = True,
    ) -> List[EmbeddingRecord]:
        """Read records matching optional id/layer filters.

        eager=True (default): reads all matrix values within a single open file
        handle — avoids one h5py.File() open per record when iterating embeddings.
        Set eager=False to get lazy _LazyH5Matrix wrappers instead (useful when
        you only need a small fraction of a very large file).
        """
        selected = _select_h5_record_indices(
            self._index,
            ids=ids,
            layer_index=layer_index,
            pool_method=pool_method,
        )
        return _load_h5_records_by_index(
            self.path,
            self._index,
            selected,
            residue_start=residue_start,
            residue_end=residue_end,
            residue_slice=residue_slice,
            eager=eager,
        )


def load_embedding_records_h5(
    path: str | Path,
    *,
    ids: str | Sequence[str] | None = None,
    layer_index: int | Sequence[int] | None = None,
    pool_method: str | Sequence[str] | None = None,
    residue_start: int | None = None,
    residue_end: int | None = None,
    residue_slice: slice | tuple[int | None, int | None] | None = None,
    eager: bool = True,
) -> List[EmbeddingRecord]:
    """Load embedding records from HDF5 files with optional id/layer/row filters."""
    try:
        import h5py  # type: ignore
    except ModuleNotFoundError as exc:
        raise EmbeddingDependencyError(
            "h5py is required for HDF5 loading. Install with: pip install h5py"
        ) from exc

    file_path = Path(path)
    with h5py.File(file_path, "r") as handle:
        record_index = _read_h5_record_index(handle)
    selected = _select_h5_record_indices(record_index, ids=ids, layer_index=layer_index, pool_method=pool_method)
    return _load_h5_records_by_index(
        file_path,
        record_index,
        selected,
        residue_start=residue_start,
        residue_end=residue_end,
        residue_slice=residue_slice,
        eager=eager,
    )


def _read_h5_record_index(handle: Any) -> _H5RecordIndex:
    if "ids" not in handle:
        raise EmbeddingInputError("HDF5 file is missing required dataset 'ids'.")
    ids = handle["ids"].asstr()[:].tolist()
    total = len(ids)

    if "layer_index" not in handle or "model_reference" not in handle:
        raise EmbeddingInputError("HDF5 file is missing required datasets 'layer_index' or 'model_reference'.")

    layer_values = handle["layer_index"][:].astype(int).tolist()
    model_values = handle["model_reference"].asstr()[:].tolist()
    if len(layer_values) != total or len(model_values) != total:
        raise EmbeddingInputError("HDF5 index datasets are inconsistent in length.")

    payload_kinds: List[str] | None = None
    if "payload_kind" in handle:
        payload_kinds = [v.strip().lower() for v in handle["payload_kind"].asstr()[:].tolist()]
        if len(payload_kinds) != total:
            raise EmbeddingInputError("HDF5 payload_kind index length mismatch.")

    if "pool_method" in handle:
        pool_methods = [_normalize_pool_method(v) for v in handle["pool_method"].asstr()[:].tolist()]
        if len(pool_methods) != total:
            raise EmbeddingInputError("HDF5 pool_method index length mismatch.")
    elif payload_kinds is None:
        pool_methods = ["unknown"] * total
    else:
        pool_methods = ["none" if kind == "matrix" else "unknown" for kind in payload_kinds]

    vector_index = handle["vector_index"][:].astype(int).tolist() if "vector_index" in handle else list(range(total))
    matrix_index = handle["matrix_index"][:].astype(int).tolist() if "matrix_index" in handle else [-1] * total
    if len(vector_index) != total or len(matrix_index) != total:
        raise EmbeddingInputError("HDF5 vector/matrix index datasets are inconsistent in length.")

    return _H5RecordIndex(
        ids=ids,
        layer_values=layer_values,
        model_values=model_values,
        pool_methods=pool_methods,
        payload_kinds=payload_kinds,
        vector_index=vector_index,
        matrix_index=matrix_index,
    )


def _select_h5_record_indices(
    record_index: _H5RecordIndex,
    *,
    ids: str | Sequence[str] | None,
    layer_index: int | Sequence[int] | None,
    pool_method: str | Sequence[str] | None,
) -> List[int]:
    id_set = _string_filter_set(ids)
    layer_set = _int_filter_set(layer_index)
    pool_set = _pool_filter_set(pool_method)
    selected: List[int] = []
    for index, rec_id in enumerate(record_index.ids):
        if id_set is not None and rec_id not in id_set:
            continue
        if layer_set is not None and int(record_index.layer_values[index]) not in layer_set:
            continue
        if pool_set is not None and record_index.pool_methods[index] not in pool_set:
            continue
        selected.append(index)
    return selected


def _load_h5_records_by_index(
    file_path: Path,
    record_index: _H5RecordIndex,
    selected: Sequence[int],
    *,
    residue_start: int | None,
    residue_end: int | None,
    residue_slice: slice | tuple[int | None, int | None] | None,
    eager: bool = True,
) -> List[EmbeddingRecord]:
    if not selected:
        return []
    row_start, row_end = _normalize_residue_slice(
        residue_start=residue_start,
        residue_end=residue_end,
        residue_slice=residue_slice,
    )
    try:
        import h5py  # type: ignore
    except ModuleNotFoundError as exc:
        raise EmbeddingDependencyError(
            "h5py is required for HDF5 loading. Install with: pip install h5py"
        ) from exc

    result: Dict[int, EmbeddingRecord] = {}
    with h5py.File(file_path, "r") as handle:
        if record_index.payload_kinds is None:
            if row_start is not None or row_end is not None:
                raise EmbeddingInputError("Residue slicing is only supported for matrix HDF5 payloads.")
            if "embeddings" not in handle:
                raise EmbeddingInputError("HDF5 file is missing required dataset 'embeddings'.")
            embeddings = cast(Any, handle["embeddings"])
            if int(embeddings.shape[0]) != len(record_index.ids):
                raise EmbeddingInputError("HDF5 ids and embeddings lengths do not match.")
            # Bulk read: one contiguous HDF5 slice instead of one per vector
            sorted_sel = sorted(selected)
            bulk = embeddings[sorted_sel[0] : sorted_sel[-1] + 1].astype("float32", copy=False)
            sel_set = {v: i for i, v in enumerate(sorted_sel)}
            for pos, index in enumerate(selected):
                row = bulk[sel_set[index]]
                result[pos] = EmbeddingRecord(
                    id=record_index.ids[index],
                    embedding=as_float_vector(row.tolist()),
                    layer_index=record_index.layer_values[index],
                    model_reference=record_index.model_values[index],
                    shape=(int(row.shape[0]),),
                    metadata={"pooling": record_index.pool_methods[index]},
                )
            return [result[pos] for pos in range(len(selected))]

        embeddings = cast(Any, handle["embeddings"]) if "embeddings" in handle else None
        matrix_offsets = cast(Any, handle["matrix_offsets"])[:] if "matrix_offsets" in handle else None
        matrix_rows = cast(Any, handle["matrix_rows"])[:] if "matrix_rows" in handle else None
        matrix_cols = cast(Any, handle["matrix_cols"])[:] if "matrix_cols" in handle else None

        # Separate vector and matrix records; use a positional dict to preserve order
        # when emitting the final list (vectors and matrices are processed independently).
        result: Dict[int, EmbeddingRecord] = {}   # position-in-selected → record
        vector_selected: List[int] = []           # (position, record_index) pairs
        matrix_selected: List[int] = []
        for pos, index in enumerate(selected):
            kind = record_index.payload_kinds[index]
            if kind == "vector":
                vector_selected.append(pos)
            elif kind == "matrix":
                matrix_selected.append(pos)
            else:
                raise EmbeddingInputError(f"Unsupported payload_kind {kind!r} at record index {index}.")

        for pos in vector_selected:
            index = selected[pos]
            if row_start is not None or row_end is not None:
                raise EmbeddingInputError("Residue slicing is only supported for matrix HDF5 payloads.")
            if embeddings is None:
                raise EmbeddingInputError("HDF5 vector payload requested but 'embeddings' dataset is missing.")
            row_index = int(record_index.vector_index[index])
            if row_index < 0 or row_index >= int(embeddings.shape[0]):
                raise EmbeddingInputError(f"Invalid vector_index {row_index} for record {index}.")
            vector = as_float_vector(embeddings[row_index].tolist())
            result[pos] = EmbeddingRecord(
                id=record_index.ids[index],
                embedding=vector,
                layer_index=record_index.layer_values[index],
                model_reference=record_index.model_values[index],
                shape=(len(vector),),
                metadata={"pooling": record_index.pool_methods[index]},
            )

        if matrix_selected:
            if matrix_offsets is None or matrix_rows is None or matrix_cols is None:
                raise EmbeddingInputError("HDF5 matrix payload requested but matrix datasets are missing.")

            if eager:
                # Group records by contiguous runs of matrix_index.
                # In protein-major files (all layers per protein stored together) this
                # collapses 24 individual reads per protein into 1 bulk read per protein,
                # giving up to N_layers× fewer HDF5 I/O calls.
                matrix_values_ds = cast(Any, handle["matrix_values"])

                # Sort positions by matrix_index to form contiguous groups
                sorted_pos = sorted(
                    matrix_selected,
                    key=lambda pos: int(record_index.matrix_index[selected[pos]]),
                )

                # Walk sorted list, emit one bulk HDF5 read per contiguous run
                i = 0
                mat_arrays: Dict[int, Any] = {}  # pos → numpy array
                while i < len(sorted_pos):
                    run = [sorted_pos[i]]
                    first_mat_idx = int(record_index.matrix_index[selected[sorted_pos[i]]])
                    j = i + 1
                    while j < len(sorted_pos):
                        next_mat_idx = int(record_index.matrix_index[selected[sorted_pos[j]]])
                        if next_mat_idx == first_mat_idx + (j - i):
                            run.append(sorted_pos[j])
                            j += 1
                        else:
                            break

                    first_mat = int(record_index.matrix_index[selected[run[0]]])
                    last_mat = int(record_index.matrix_index[selected[run[-1]]])
                    bulk_start = int(matrix_offsets[first_mat])
                    bulk_end = int(matrix_offsets[last_mat + 1])
                    bulk = matrix_values_ds[bulk_start:bulk_end].astype("float32", copy=False)

                    for pos in run:
                        index = selected[pos]
                        mat_idx = int(record_index.matrix_index[index])
                        local_start = int(matrix_offsets[mat_idx]) - bulk_start
                        local_end = int(matrix_offsets[mat_idx + 1]) - bulk_start
                        rows = int(matrix_rows[mat_idx])
                        cols = int(matrix_cols[mat_idx])
                        if local_end - local_start != rows * cols:
                            raise EmbeddingInputError(
                                f"Matrix payload length mismatch for record {index}."
                            )
                        start_row, end_row = _resolve_row_bounds(row_start, row_end, rows=rows)
                        s = local_start + start_row * cols
                        e = local_start + end_row * cols
                        mat_arrays[pos] = bulk[s:e].reshape(end_row - start_row, cols)

                    i = j

                for pos in matrix_selected:
                    index = selected[pos]
                    arr = mat_arrays[pos]
                    result[pos] = EmbeddingRecord(
                        id=record_index.ids[index],
                        embedding=cast(EmbeddingPayload, arr),
                        layer_index=record_index.layer_values[index],
                        model_reference=record_index.model_values[index],
                        shape=arr.shape,
                        metadata={"pooling": record_index.pool_methods[index]},
                    )
            else:
                # Lazy path: defers the read to first .embedding access
                for pos in matrix_selected:
                    index = selected[pos]
                    mat_index = int(record_index.matrix_index[index])
                    if mat_index < 0 or mat_index >= int(len(matrix_rows)):
                        raise EmbeddingInputError(f"Invalid matrix_index {mat_index} for record {index}.")
                    matrix = _lazy_h5_matrix_for_record(
                        file_path,
                        record_index=index,
                        matrix_index=mat_index,
                        matrix_offsets=matrix_offsets,
                        matrix_rows=matrix_rows,
                        matrix_cols=matrix_cols,
                        row_start=row_start,
                        row_end=row_end,
                    )
                    result[pos] = EmbeddingRecord(
                        id=record_index.ids[index],
                        embedding=cast(Any, matrix),
                        layer_index=record_index.layer_values[index],
                        model_reference=record_index.model_values[index],
                        shape=matrix.shape,
                        metadata={"pooling": record_index.pool_methods[index]},
                    )

    return [result[pos] for pos in range(len(selected))]


def _lazy_h5_matrix_for_record(
    file_path: Path,
    *,
    record_index: int,
    matrix_index: int,
    matrix_offsets: Any,
    matrix_rows: Any,
    matrix_cols: Any,
    row_start: int | None,
    row_end: int | None,
) -> "_LazyH5Matrix":
    full_start = int(matrix_offsets[matrix_index])
    full_end = int(matrix_offsets[matrix_index + 1])
    rows = int(matrix_rows[matrix_index])
    cols = int(matrix_cols[matrix_index])
    value_count = full_end - full_start
    if value_count != rows * cols:
        raise EmbeddingInputError(
            f"Matrix payload length mismatch for record {record_index}: expected {rows * cols}, got {value_count}."
        )
    start_row, end_row = _resolve_row_bounds(row_start, row_end, rows=rows)
    start = full_start + start_row * cols
    end = full_start + end_row * cols
    return _LazyH5Matrix(file_path, start=start, end=end, rows=end_row - start_row, cols=cols)


def _normalize_residue_slice(
    *,
    residue_start: int | None,
    residue_end: int | None,
    residue_slice: slice | tuple[int | None, int | None] | None,
) -> tuple[int | None, int | None]:
    if residue_slice is None:
        return residue_start, residue_end
    if residue_start is not None or residue_end is not None:
        raise EmbeddingInputError("Pass either residue_slice or residue_start/residue_end, not both.")
    if isinstance(residue_slice, slice):
        if residue_slice.step not in {None, 1}:
            raise EmbeddingInputError("HDF5 residue_slice step must be None or 1.")
        return residue_slice.start, residue_slice.stop
    if len(residue_slice) != 2:
        raise EmbeddingInputError("residue_slice must be a slice or (start, end) tuple.")
    return residue_slice[0], residue_slice[1]


def _resolve_row_bounds(start: int | None, end: int | None, *, rows: int) -> tuple[int, int]:
    resolved_start = 0 if start is None else int(start)
    resolved_end = int(rows) if end is None else int(end)
    if resolved_start < 0:
        resolved_start += int(rows)
    if resolved_end < 0:
        resolved_end += int(rows)
    if resolved_start < 0 or resolved_end < 0 or resolved_start > int(rows) or resolved_end > int(rows):
        raise EmbeddingInputError(f"Residue slice [{resolved_start}:{resolved_end}] is out of range for {rows} rows.")
    if resolved_end < resolved_start:
        raise EmbeddingInputError("residue_end must be greater than or equal to residue_start.")
    return resolved_start, resolved_end


def _string_filter_set(values: str | Sequence[str] | None) -> set[str] | None:
    if values is None:
        return None
    if isinstance(values, str):
        return {values}
    return {str(value) for value in values}


def _int_filter_set(values: int | Sequence[int] | None) -> set[int] | None:
    if values is None:
        return None
    if isinstance(values, int):
        return {int(values)}
    return {int(value) for value in values}


def _pool_filter_set(values: str | Sequence[str] | None) -> set[str] | None:
    if values is None:
        return None
    if isinstance(values, str):
        return {_normalize_pool_method(values)}
    return {_normalize_pool_method(value) for value in values}


def _normalize_pool_method(value: object) -> str:
    normalized = str(value).strip().lower()
    if normalized == "identity":
        return "none"
    if normalized == "bos":
        return "cls"
    return normalized or "unknown"


class _LazyH5Matrix:
    """Array-like HDF5 matrix payload that reads values only when materialized.

    The first call to any accessor loads the full slice from HDF5 and caches it
    as a float32 numpy array.  Subsequent calls return the cached array without
    reopening the file, which avoids one file-open per training epoch when the
    same protein appears in multiple probe batches.
    """

    def __init__(self, path: Path, *, start: int, end: int, rows: int, cols: int) -> None:
        self.path = Path(path)
        self.start = int(start)
        self.end = int(end)
        self.shape = (int(rows), int(cols))
        self._cached: Any = None

    def __len__(self) -> int:
        return self.shape[0]

    def __getitem__(self, index: int) -> Any:
        return self._array()[index]

    def __array__(self, dtype: Any = None, copy: bool = False) -> Any:
        array = self._array()
        if dtype is not None:
            return array.astype(dtype, copy=copy)
        return array.copy() if copy else array

    def tolist(self) -> list[list[float]]:
        return cast(list[list[float]], self._array().tolist())

    def _array(self) -> Any:
        if self._cached is not None:
            return self._cached
        try:
            import h5py  # type: ignore
        except ModuleNotFoundError as exc:
            raise EmbeddingDependencyError(
                "h5py is required for HDF5 loading. Install with: pip install h5py"
            ) from exc
        with h5py.File(self.path, "r") as handle:
            flat = cast(Any, handle["matrix_values"])[self.start:self.end]
        self._cached = flat.astype("float32", copy=False).reshape(self.shape)
        return self._cached


def load_embedding_records_npy(
    path: str | Path,
    *,
    model_reference: str = "unknown",
    layer_index: int = 0,
    ids: Sequence[str] | None = None,
) -> List[EmbeddingRecord]:
    """Load embeddings from ``.npy`` or ``.npz`` and normalize records."""
    try:
        import numpy as np
    except ModuleNotFoundError as exc:
        raise EmbeddingDependencyError(
            "NumPy is required for .npy/.npz loading. Install with: pip install numpy"
        ) from exc

    file_path = Path(path)
    arr_obj = np.load(file_path, allow_pickle=False)

    if hasattr(arr_obj, "files"):
        npz_obj = arr_obj
        if not npz_obj.files:
            return []
        if len(npz_obj.files) > 1:
            raise EmbeddingInputError(
                "NPZ loading currently expects one array. "
                "Provide a single-array .npz or use .npy."
            )
        matrix = npz_obj[npz_obj.files[0]]
    else:
        matrix = arr_obj

    return _records_from_numpy(matrix, model_reference=model_reference, layer_index=layer_index, ids=ids)


def _validate_records_per_shard(records_per_shard: int) -> int:
    resolved = int(records_per_shard)
    if resolved <= 0:
        raise EmbeddingInputError("records_per_shard must be a positive integer.")
    return resolved


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
    file_path = Path(npy_path)
    return file_path.with_suffix(".ids.txt")


def _write_npy_shard(
    path: str | Path,
    records: Sequence[EmbeddingRecord],
    shard_index: int,
) -> Tuple[Path, Path]:
    shard_path = _npy_shard_path(path, shard_index)
    ids_path = _npy_ids_path(shard_path)
    save_embedding_records_npy(shard_path, records)
    ids_path.write_text("".join(f"{record.id}\n" for record in records), encoding="utf-8")
    return shard_path, ids_path


def _write_pickle_shard(
    path: str | Path,
    records: Sequence[EmbeddingRecord],
    shard_index: int,
    *,
    payload_format: PicklePayloadFormat,
) -> Path:
    shard_path = _pickle_shard_path(path, shard_index)
    save_embedding_records_pickle(shard_path, records, payload_format=payload_format)
    return shard_path


def _records_from_payload(
    payload: object,
    *,
    model_reference: str,
    layer_index: int,
) -> List[EmbeddingRecord]:
    if isinstance(payload, MappingABC):
        payload_map = cast(MappingABC[object, object], payload)
        if "records" in payload_map:
            records_raw = payload_map["records"]
            if not isinstance(records_raw, Sequence) or isinstance(records_raw, (str, bytes, bytearray)):
                raise EmbeddingInputError("Pickle payload 'records' must be a sequence.")
            record_items = cast(Sequence[object], records_raw)
            return [
                _record_from_item(item, index=index, model_reference=model_reference, layer_index=layer_index)
                for index, item in enumerate(record_items)
            ]

        if all(isinstance(key, str) for key in payload_map.keys()):
            records: List[EmbeddingRecord] = []
            for rec_id, emb_value in payload_map.items():
                vector = as_float_vector(emb_value)
                records.append(
                    EmbeddingRecord(
                        id=str(rec_id),
                        embedding=vector,
                        layer_index=int(layer_index),
                        model_reference=model_reference,
                        shape=(len(vector),),
                        metadata=None,
                    )
                )
            return records

    if isinstance(payload, Sequence) and not isinstance(payload, (str, bytes, bytearray)):
        payload_items = cast(Sequence[object], payload)
        return [
            _record_from_item(item, index=index, model_reference=model_reference, layer_index=layer_index)
            for index, item in enumerate(payload_items)
        ]

    raise EmbeddingInputError(
        "Unsupported pickle payload format. Expected list-like records, {'records': [...]}, or {id: embedding}."
    )


def _record_from_item(
    item: object,
    *,
    index: int,
    model_reference: str,
    layer_index: int,
) -> EmbeddingRecord:
    if isinstance(item, EmbeddingRecord):
        return item

    if not isinstance(item, MappingABC):
        raise EmbeddingInputError(f"Record at index {index} must be a dict or EmbeddingRecord.")

    item_map = cast(MappingABC[object, object], item)
    rec_id = str(item_map.get("id", "")).strip()
    if not rec_id:
        raise EmbeddingInputError(f"Record at index {index} has empty 'id'.")

    vector = as_float_vector(item_map.get("embedding"))
    rec_layer = _int_or_default(item_map.get("layer_index"), default=layer_index)
    rec_model_ref = str(item_map.get("model_reference", model_reference)).strip() or model_reference

    shape_raw = item_map.get("shape")
    shape: Tuple[int, ...]
    if shape_raw is None:
        shape = (len(vector),)
    else:
        if not isinstance(shape_raw, Sequence) or isinstance(shape_raw, (str, bytes, bytearray)):
            raise EmbeddingInputError(f"Record at index {index} has invalid 'shape'.")
        shape_values = cast(Sequence[object], shape_raw)
        shape = tuple(int(cast(Any, dim)) for dim in shape_values)
        if shape != (len(vector),):
            raise EmbeddingInputError(
                f"Record at index {index} has shape {shape} inconsistent with vector length {len(vector)}."
            )

    metadata = item_map.get("metadata")
    if metadata is not None and not isinstance(metadata, MappingABC):
        raise EmbeddingInputError(f"Record at index {index} has non-dict metadata.")

    return EmbeddingRecord(
        id=rec_id,
        embedding=vector,
        layer_index=rec_layer,
        model_reference=rec_model_ref,
        shape=shape,
        metadata=_metadata_dict_or_none(cast(object, metadata)),
    )


def _records_from_numpy(
    matrix: Any,
    *,
    model_reference: str,
    layer_index: int,
    ids: Sequence[str] | None,
) -> List[EmbeddingRecord]:
    try:
        ndim = int(matrix.ndim)
    except Exception as exc:
        raise EmbeddingInputError("Could not determine NumPy array dimensions.") from exc

    try:
        import numpy as _np
        arr = _np.asarray(matrix, dtype=_np.float32)
    except Exception as exc:
        raise EmbeddingInputError("Could not convert matrix to float32 NumPy array.") from exc

    if ndim == 1:
        numpy_rows = [arr]
        row_shapes = [(int(arr.shape[0]),)]
    elif ndim == 2:
        numpy_rows = [arr[idx] for idx in range(int(arr.shape[0]))]
        row_shapes = [(int(arr.shape[1]),)] * int(arr.shape[0])
    else:
        raise EmbeddingInputError(f"Expected 1D or 2D array, got ndim={ndim}.")

    total = len(numpy_rows)
    if ids is None:
        resolved_ids = [f"row_{idx}" for idx in range(total)]
    else:
        resolved_ids = [str(value) for value in ids]
        if len(resolved_ids) != total:
            raise EmbeddingInputError(
                f"ids length ({len(resolved_ids)}) does not match number of rows ({total})."
            )

    return [
        EmbeddingRecord(
            id=resolved_ids[idx],
            embedding=cast(EmbeddingPayload, vector),
            layer_index=int(layer_index),
            model_reference=model_reference,
            shape=row_shapes[idx],
            metadata={"source": "npy"},
        )
        for idx, vector in enumerate(numpy_rows)
    ]


def _normalize_embedding_records(records: Sequence[object]) -> List[EmbeddingRecord]:
    if not records:
        return []
    normalized: List[EmbeddingRecord] = []
    for index, record in enumerate(records):
        if not isinstance(record, EmbeddingRecord):
            raise EmbeddingInputError(
                f"Expected EmbeddingRecord at index {index}, got {type(record).__name__}."
            )
        rec_id = str(record.id).strip()
        if not rec_id:
            raise EmbeddingInputError(f"EmbeddingRecord at index {index} has empty id.")
        vector = as_float_vector(record.embedding)
        if not vector:
            raise EmbeddingInputError(f"EmbeddingRecord at index {index} has empty embedding.")
        expected_shape = (len(vector),)
        if tuple(record.shape) != expected_shape:
            raise EmbeddingInputError(
                f"EmbeddingRecord at index {index} has shape {record.shape} inconsistent with embedding length {len(vector)}."
            )
        normalized.append(
            EmbeddingRecord(
                id=rec_id,
                embedding=vector,
                layer_index=int(record.layer_index),
                model_reference=str(record.model_reference),
                shape=expected_shape,
                metadata=record.metadata,
            )
        )
    return normalized


def _metadata_dict_or_none(value: object) -> Dict[str, Any] | None:
    if value is None:
        return None
    if not isinstance(value, MappingABC):
        raise EmbeddingInputError("metadata must be a mapping when provided.")
    value_map = cast(MappingABC[object, Any], value)
    return {str(key): item for key, item in value_map.items()}


def _int_or_default(value: object, *, default: int) -> int:
    if value is None:
        return int(default)
    return int(cast(Any, value))


__all__ = [
    "load_embedding_records",
    "H5EmbeddingReader",
    "load_embedding_records_pickle",
    "load_embedding_records_h5",
    "load_embedding_records_npy",
    "save_embedding_records_pickle",
    "save_embedding_records_pickle_shards",
    "save_embedding_records_npy",
    "save_embedding_records_npy_shards",
    "save_embedding_records_h5",
    "mean_pool_embedding_record",
]
