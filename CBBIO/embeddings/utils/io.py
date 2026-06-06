"""Embedding persistence and file-backed streaming helpers."""

from __future__ import annotations

from collections.abc import Iterable, Mapping as MappingABC
import pickle
from pathlib import Path
from typing import Any, Callable, Dict, List, Literal, Sequence, Tuple, cast
import warnings

from .. import (
    EmbeddingDependencyError,
    EmbeddingGenerator,
    EmbeddingInputError,
    EmbeddingRecord,
    FastaEmbeddingH5Result,
    FastaEmbeddingNpyShardResult,
    FastaEmbeddingPickleShardResult,
    H5WriteResult,
    LayerSelection,
    NpyShardWriteResult,
    PicklePayloadFormat,
    PickleShardWriteResult,
    as_float_vector,
)
from .pooler import mean_pool_embedding_record
from .writer import _H5EmbeddingWriter, _normalize_embedding_record  # pyright: ignore[reportPrivateUsage]


def _warn_deprecated(name: str, replacement: str) -> None:
    warnings.warn(
        f"{name} is deprecated and will be removed soon. Use {replacement} instead.",
        DeprecationWarning,
        stacklevel=3,
    )


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


def generate_fasta_pickle_shards(
    path: str | Path,
    generator: EmbeddingGenerator,
    output_path: str | Path,
    *,
    batch_size: int,
    records_per_shard: int,
    layer_index: LayerSelection = 0,
    fail_fast: bool = False,
    id_from: Literal["record_id", "description"] = "record_id",
    payload_format: PicklePayloadFormat = "records",
    max_batch_tokens: int | None = None,
    max_records: int | None = None,
    max_sequence_length: int | None = None,
    length_sort_window: int | None = None,
    skipped_path: str | Path | None = None,
    pool: Literal["mean", "none"] = "mean",
    progress_callback: Callable[[Dict[str, Any]], None] | None = None,
) -> FastaEmbeddingPickleShardResult:
    """Deprecated wrapper around the embedding job API."""
    _warn_deprecated("generate_fasta_pickle_shards", "FastaBatcher + EmbeddingWriter(format='pkl') + run_embedding_generation")
    from .batcher import FastaBatcher
    from .jobs import run_embedding_generation
    from .pooler import pooler_factory
    from .writer import EmbeddingWriter

    batcher = FastaBatcher(
        path,
        batch_size=batch_size,
        max_batch_tokens=max_batch_tokens,
        limit=max_records,
        length_sort_window=length_sort_window,
        max_sequence_length=max_sequence_length,
        skipped_path=skipped_path,
        id_from=id_from,
    )
    writer = EmbeddingWriter(
        format="pkl",
        path=output_path,
        records_per_shard=records_per_shard,
        payload_format=payload_format,
    )
    state = run_embedding_generation(
        generator,
        batcher,
        writer,
        layer_index=layer_index,
        pooler=pooler_factory(pool),
        fail_fast=fail_fast,
        progress_callback=progress_callback,
    )
    return FastaEmbeddingPickleShardResult(
        paths=state.paths,
        record_count=state.record_count,
        error_count=state.error_count,
        skipped_count=state.skipped_count,
        errors=state.errors,
        skipped=state.skipped,
    )


def generate_fasta_npy_shards(
    path: str | Path,
    generator: EmbeddingGenerator,
    output_path: str | Path,
    *,
    batch_size: int,
    records_per_shard: int,
    layer_index: LayerSelection = 0,
    fail_fast: bool = False,
    id_from: Literal["record_id", "description"] = "record_id",
    max_batch_tokens: int | None = None,
    max_records: int | None = None,
    max_sequence_length: int | None = None,
    length_sort_window: int | None = None,
    skipped_path: str | Path | None = None,
    pool: Literal["mean", "none"] = "mean",
    progress_callback: Callable[[Dict[str, Any]], None] | None = None,
) -> FastaEmbeddingNpyShardResult:
    """Deprecated wrapper around the embedding job API."""
    _warn_deprecated("generate_fasta_npy_shards", "FastaBatcher + EmbeddingWriter(format='npy') + run_embedding_generation")
    from .batcher import FastaBatcher
    from .jobs import run_embedding_generation
    from .pooler import pooler_factory
    from .writer import EmbeddingWriter

    batcher = FastaBatcher(
        path,
        batch_size=batch_size,
        max_batch_tokens=max_batch_tokens,
        limit=max_records,
        length_sort_window=length_sort_window,
        max_sequence_length=max_sequence_length,
        skipped_path=skipped_path,
        id_from=id_from,
    )
    writer = EmbeddingWriter(format="npy", path=output_path, records_per_shard=records_per_shard)
    state = run_embedding_generation(
        generator,
        batcher,
        writer,
        layer_index=layer_index,
        pooler=pooler_factory(pool),
        fail_fast=fail_fast,
        progress_callback=progress_callback,
    )
    return FastaEmbeddingNpyShardResult(
        paths=state.paths,
        id_paths=state.id_paths,
        record_count=state.record_count,
        error_count=state.error_count,
        skipped_count=state.skipped_count,
        errors=state.errors,
        skipped=state.skipped,
    )


def generate_fasta_h5(
    path: str | Path,
    generator: EmbeddingGenerator,
    output_path: str | Path,
    *,
    batch_size: int,
    layer_index: LayerSelection = 0,
    fail_fast: bool = False,
    id_from: Literal["record_id", "description"] = "record_id",
    max_batch_tokens: int | None = None,
    max_records: int | None = None,
    max_sequence_length: int | None = None,
    length_sort_window: int | None = None,
    skipped_path: str | Path | None = None,
    pool: Literal["mean", "none"] = "mean",
    compression: str | None = "gzip",
    write_batch_size: int = 1,
    progress_callback: Callable[[Dict[str, Any]], None] | None = None,
) -> FastaEmbeddingH5Result:
    """Deprecated wrapper around the embedding job API."""
    _warn_deprecated("generate_fasta_h5", "FastaBatcher + EmbeddingWriter(format='h5') + run_embedding_generation")
    from .batcher import FastaBatcher
    from .jobs import run_embedding_generation
    from .pooler import pooler_factory
    from .writer import EmbeddingWriter

    batcher = FastaBatcher(
        path,
        batch_size=batch_size,
        max_batch_tokens=max_batch_tokens,
        limit=max_records,
        length_sort_window=length_sort_window,
        max_sequence_length=max_sequence_length,
        skipped_path=skipped_path,
        id_from=id_from,
    )
    writer = EmbeddingWriter(
        format="h5",
        path=output_path,
        compression=compression,
        write_batch_size=write_batch_size,
    )
    state = run_embedding_generation(
        generator,
        batcher,
        writer,
        layer_index=layer_index,
        pooler=pooler_factory(pool),
        fail_fast=fail_fast,
        progress_callback=progress_callback,
    )
    return FastaEmbeddingH5Result(
        path=Path(output_path),
        record_count=state.record_count,
        error_count=state.error_count,
        skipped_count=state.skipped_count,
        errors=state.errors,
        skipped=state.skipped,
    )


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


def load_embedding_records_h5(path: str | Path) -> List[EmbeddingRecord]:
    """Load embedding records from HDF5 files with vector, matrix, or mixed payloads."""
    try:
        import h5py  # type: ignore
    except ModuleNotFoundError as exc:
        raise EmbeddingDependencyError(
            "h5py is required for HDF5 loading. Install with: pip install h5py"
        ) from exc

    file_path = Path(path)
    with h5py.File(file_path, "r") as handle:
        if "ids" not in handle:
            raise EmbeddingInputError("HDF5 file is missing required dataset 'ids'.")
        ids = handle["ids"].asstr()[:].tolist()
        total = len(ids)
        if total == 0:
            return []

        if "layer_index" not in handle or "model_reference" not in handle:
            raise EmbeddingInputError("HDF5 file is missing required datasets 'layer_index' or 'model_reference'.")

        layer_values = [int(value) for value in handle["layer_index"][:].tolist()]
        model_values = handle["model_reference"].asstr()[:].tolist()
        if len(layer_values) != total or len(model_values) != total:
            raise EmbeddingInputError("HDF5 index datasets are inconsistent in length.")

        if "payload_kind" not in handle:
            if "embeddings" not in handle:
                raise EmbeddingInputError("HDF5 file is missing required dataset 'embeddings'.")
            embeddings = handle["embeddings"]
            if int(embeddings.shape[0]) != total:
                raise EmbeddingInputError("HDF5 ids and embeddings lengths do not match.")
            def _vector_record(index: int) -> EmbeddingRecord:
                vector = as_float_vector(embeddings[index])
                return EmbeddingRecord(
                    id=ids[index],
                    embedding=vector,
                    layer_index=layer_values[index],
                    model_reference=model_values[index],
                    shape=(len(vector),),
                    metadata=None,
                )
            return [_vector_record(i) for i in range(total)]

        payload_kinds = handle["payload_kind"].asstr()[:].tolist()
        if len(payload_kinds) != total:
            raise EmbeddingInputError("HDF5 payload_kind index length mismatch.")

        vector_index = handle["vector_index"][:].tolist() if "vector_index" in handle else list(range(total))
        matrix_index = handle["matrix_index"][:].tolist() if "matrix_index" in handle else [-1] * total
        embeddings = handle["embeddings"] if "embeddings" in handle else None
        matrix_values = handle["matrix_values"] if "matrix_values" in handle else None
        matrix_offsets = handle["matrix_offsets"][:] if "matrix_offsets" in handle else None
        matrix_rows = handle["matrix_rows"][:] if "matrix_rows" in handle else None
        matrix_cols = handle["matrix_cols"][:] if "matrix_cols" in handle else None

        records: List[EmbeddingRecord] = []
        for index in range(total):
            kind = str(payload_kinds[index]).strip().lower()
            if kind == "vector":
                if embeddings is None:
                    raise EmbeddingInputError("HDF5 vector payload requested but 'embeddings' dataset is missing.")
                row_index = int(vector_index[index])
                if row_index < 0 or row_index >= int(embeddings.shape[0]):
                    raise EmbeddingInputError(f"Invalid vector_index {row_index} for record {index}.")
                vector = as_float_vector(embeddings[row_index].tolist())
                records.append(
                    EmbeddingRecord(
                        id=ids[index],
                        embedding=vector,
                        layer_index=layer_values[index],
                        model_reference=model_values[index],
                        shape=(len(vector),),
                        metadata=None,
                    )
                )
                continue

            if kind == "matrix":
                if (
                    matrix_values is None
                    or matrix_offsets is None
                    or matrix_rows is None
                    or matrix_cols is None
                ):
                    raise EmbeddingInputError("HDF5 matrix payload requested but matrix datasets are missing.")
                mat_index = int(matrix_index[index])
                if mat_index < 0 or mat_index >= int(len(matrix_rows)):
                    raise EmbeddingInputError(f"Invalid matrix_index {mat_index} for record {index}.")
                start = int(matrix_offsets[mat_index])
                end = int(matrix_offsets[mat_index + 1])
                rows = int(matrix_rows[mat_index])
                cols = int(matrix_cols[mat_index])
                value_count = end - start
                if value_count != rows * cols:
                    raise EmbeddingInputError(
                        f"Matrix payload length mismatch for record {index}: expected {rows * cols}, got {value_count}."
                    )
                matrix = _LazyH5Matrix(file_path, start=start, end=end, rows=rows, cols=cols)
                records.append(
                    EmbeddingRecord(
                        id=ids[index],
                        embedding=matrix,
                        layer_index=layer_values[index],
                        model_reference=model_values[index],
                        shape=(rows, cols),
                        metadata=None,
                    )
                )
                continue

            raise EmbeddingInputError(f"Unsupported payload_kind {kind!r} at record index {index}.")

        return records


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

    def __array__(self, dtype: Any = None) -> Any:
        array = self._array()
        if dtype is not None:
            return array.astype(dtype, copy=False)
        return array

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
            flat = handle["matrix_values"][self.start:self.end]
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

    rows: List[List[float]] = []
    if ndim == 1:
        rows = [as_float_vector(matrix)]
    elif ndim == 2:
        row_count = int(matrix.shape[0])
        rows = [as_float_vector(matrix[idx]) for idx in range(row_count)]
    else:
        raise EmbeddingInputError(f"Expected 1D or 2D array, got ndim={ndim}.")

    total = len(rows)
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
            embedding=vector,
            layer_index=int(layer_index),
            model_reference=model_reference,
            shape=(len(vector),),
            metadata={"source": "npy"},
        )
        for idx, vector in enumerate(rows)
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
    "load_embedding_records_pickle",
    "load_embedding_records_h5",
    "load_embedding_records_npy",
    "save_embedding_records_pickle",
    "save_embedding_records_pickle_shards",
    "save_embedding_records_npy",
    "save_embedding_records_npy_shards",
    "save_embedding_records_h5",
    "generate_fasta_pickle_shards",
    "generate_fasta_npy_shards",
    "generate_fasta_h5",
    "mean_pool_embedding_record",
]
