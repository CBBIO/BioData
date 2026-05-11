"""Embedding generation abstractions and orchestration utilities.

This module is intentionally independent from database access code.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Literal, Sequence, Tuple, TypeAlias, cast
import warnings


_VALID_AA = set("ACDEFGHIKLMNPQRSTVWYBXZJUO")
EmbeddingPayload: TypeAlias = Sequence[float] | Sequence[Sequence[float]]
LayerSelection: TypeAlias = int | Sequence[int] | None
PicklePayloadFormat: TypeAlias = Literal["records", "mapping"]


def _warn_deprecated(name: str, replacement: str) -> None:
    warnings.warn(
        f"{name} is deprecated and will be removed soon. Use {replacement} instead.",
        DeprecationWarning,
        stacklevel=3,
    )


class EmbeddingGenerationError(Exception):
    """Base exception for embedding generation workflows."""


class EmbeddingInputError(EmbeddingGenerationError):
    """Raised when input records/sequences are invalid."""


class EmbeddingDependencyError(EmbeddingGenerationError):
    """Raised when an optional dependency required by a path is missing."""


class EmbeddingBackendError(EmbeddingGenerationError):
    """Raised when a backend adapter fails while generating embeddings."""


@dataclass(frozen=True)
class GenerationInput:
    id: str
    sequence: str
    description: str | None = None
    metadata: Dict[str, Any] | None = None


@dataclass(frozen=True)
class EmbeddingRecord:
    id: str
    embedding: EmbeddingPayload
    layer_index: int
    model_reference: str
    shape: Tuple[int, ...]
    metadata: Dict[str, Any] | None = None


@dataclass(frozen=True)
class ModelMetadata:
    provider: str
    model_name: str
    model_reference: str
    model_revision: str | None = None
    tokenizer_name: str | None = None
    tokenizer_revision: str | None = None
    device: str | None = None
    framework_versions: Dict[str, str] | None = None
    parameters: Dict[str, Any] | None = None


@dataclass(frozen=True)
class RunMetadata:
    run_id: str
    created_at_utc: str
    sequence_count: int
    requested_layers: List[int] | None = None
    resolved_layers: List[int] | None = None
    failure_count: int = 0
    parameters: Dict[str, Any] | None = None


def _embedding_record_list() -> List[EmbeddingRecord]:
    return []


def _error_dict_list() -> List[Dict[str, Any]]:
    return []


@dataclass(frozen=True)
class GenerationResult:
    records: List[EmbeddingRecord] = field(default_factory=_embedding_record_list)
    errors: List[Dict[str, Any]] = field(default_factory=_error_dict_list)
    skipped: List[Dict[str, Any]] = field(default_factory=_error_dict_list)
    model_metadata: ModelMetadata | None = None
    run_metadata: RunMetadata | None = None


@dataclass(frozen=True)
class PickleShardWriteResult:
    paths: List[Path]
    record_count: int


@dataclass(frozen=True)
class NpyShardWriteResult:
    paths: List[Path]
    id_paths: List[Path]
    record_count: int


@dataclass(frozen=True)
class H5WriteResult:
    path: Path
    record_count: int


@dataclass(frozen=True)
class FastaEmbeddingPickleShardResult:
    paths: List[Path]
    record_count: int
    error_count: int = 0
    skipped_count: int = 0
    errors: List[Dict[str, Any]] = field(default_factory=_error_dict_list)
    skipped: List[Dict[str, Any]] = field(default_factory=_error_dict_list)


@dataclass(frozen=True)
class FastaEmbeddingNpyShardResult:
    paths: List[Path]
    id_paths: List[Path]
    record_count: int
    error_count: int = 0
    skipped_count: int = 0
    errors: List[Dict[str, Any]] = field(default_factory=_error_dict_list)
    skipped: List[Dict[str, Any]] = field(default_factory=_error_dict_list)


@dataclass(frozen=True)
class FastaEmbeddingH5Result:
    path: Path
    record_count: int
    error_count: int = 0
    skipped_count: int = 0
    errors: List[Dict[str, Any]] = field(default_factory=_error_dict_list)
    skipped: List[Dict[str, Any]] = field(default_factory=_error_dict_list)


class PreprocessorAdapter(ABC):
    @abstractmethod
    def preprocess(self, raw_sequence: str) -> str: ...


class TokenizerAdapter(ABC):
    @abstractmethod
    def tokenize(self, sequence: str) -> Any: ...

    def tokenize_many(self, sequences: Sequence[str]) -> Any:
        """Tokenize a batch of preprocessed sequences.

        True batched generators override this method so one generated batch
        becomes one padded token payload and one model forward pass.
        """
        return [self.tokenize(sequence) for sequence in sequences]


class ModelAdapter(ABC):
    @abstractmethod
    def infer(self, tokens: Any, *, layer_index: int | Sequence[int] | None = 0) -> Any: ...

    def available_layers(self) -> List[int] | None:
        """Return available layer indices when discoverable by this adapter."""
        return None


class PostprocessorAdapter(ABC):
    @abstractmethod
    def postprocess(self, model_output: Any) -> EmbeddingPayload: ...


class EmbeddingGenerator:
    """Adapter-orchestrated embedding generator without persistence side effects."""

    def __init__(
        self,
        *,
        model_reference: str,
        preprocessor: PreprocessorAdapter,
        tokenizer: TokenizerAdapter,
        model: ModelAdapter,
        postprocessor: PostprocessorAdapter,
    ) -> None:
        model_ref = str(model_reference).strip()
        if not model_ref:
            raise EmbeddingInputError("model_reference must be a non-empty string.")
        self.model_reference = model_ref
        self.preprocessor = preprocessor
        self.tokenizer = tokenizer
        self.model = model
        self.postprocessor = postprocessor

    def available_layers(self) -> List[int]:
        """Return layer indices exposed by the underlying model adapter."""
        layers = self.model.available_layers()
        if layers is None:
            raise EmbeddingBackendError(
                "This generator cannot determine available layers from the current model adapter."
            )
        if not layers:
            raise EmbeddingBackendError("Model adapter returned an empty available layer list.")
        return [int(value) for value in layers]

    def num_layers(self) -> int:
        """Return total number of available layers."""
        return len(self.available_layers())

    def family_models(self) -> List[str]:
        """Return known model identifiers in this generator's family."""
        values = getattr(self, "FAMILY_MODELS", None)
        if isinstance(values, Sequence) and not isinstance(values, (str, bytes, bytearray)):
            family_values = cast(Sequence[object], values)
            models = [str(value) for value in family_values if str(value).strip()]
            if models:
                return models
        return [self.model_reference]

    def generate(
        self,
        records: Sequence[GenerationInput],
        *,
        layer_index: LayerSelection = 0,
        pooler: Any | None = None,
        fail_fast: bool = False,
    ) -> GenerationResult:
        if not isinstance(layer_index, int):
            raise EmbeddingInputError("Base EmbeddingGenerator.generate requires an integer layer_index.")
        resolved_layer_index = int(layer_index)
        result = GenerationResult()
        resolved_pooler: Any | None = None
        mean_pooler_type: type[Any] | None = None
        if pooler is not None:
            from .embeddings_pooler import MeanPooler, resolve_pooler

            resolved_pooler = resolve_pooler(pooler)
            mean_pooler_type = MeanPooler

        for index, record in enumerate(records):
            try:
                normalized = _validate_generation_input(record, index=index)
                prepared = self.preprocessor.preprocess(normalized.sequence)
                _validate_sequence(prepared, context=f"record index={index} id={normalized.id!r}")
                tokens = self.tokenizer.tokenize(prepared)
                model_output = self.model.infer(tokens, layer_index=resolved_layer_index)
                vector_out = self.postprocessor.postprocess(model_output)
                vector_candidate: object = _as_float_vector(vector_out)
                if resolved_pooler is not None:
                    try:
                        vector_candidate = resolved_pooler(vector_candidate)
                    except EmbeddingGenerationError:
                        if mean_pooler_type is None or not isinstance(resolved_pooler, mean_pooler_type):
                            raise

                try:
                    vector = _as_float_vector(vector_candidate)
                except EmbeddingBackendError:
                    if mean_pooler_type is not None and isinstance(resolved_pooler, mean_pooler_type):
                        # Backward compatibility: mean pooler is a no-op for already vector outputs.
                        vector = _as_float_vector(vector_out)
                    else:
                        raise
                if not vector:
                    raise EmbeddingBackendError("postprocessor returned an empty embedding vector.")

                record_out = EmbeddingRecord(
                    id=normalized.id,
                    embedding=vector,
                    layer_index=resolved_layer_index,
                    model_reference=self.model_reference,
                    shape=(len(vector),),
                    metadata=normalized.metadata,
                )

                result.records.append(
                    record_out
                )
            except Exception as exc:
                normalized_exc = _normalize_generation_exception(exc)
                if fail_fast:
                    raise normalized_exc
                result.errors.append(
                    {
                        "index": index,
                        "id": getattr(record, "id", None),
                        "error_type": normalized_exc.__class__.__name__,
                        "message": str(normalized_exc),
                    }
                )

        return result

    def _generate_from_layer_output_map(
        self,
        records: Sequence[GenerationInput],
        *,
        layer_index: LayerSelection,
        pooler: Any | None,
        fail_fast: bool,
        missing_layers_error: str,
        requested_layers: List[int] | None,
        run_parameters: Dict[str, Any] | None,
    ) -> GenerationResult:
        from .embeddings_pooler import materialize_embedding_payload, resolve_pooler

        result = GenerationResult()
        resolved_pooler = resolve_pooler(pooler)

        for index, record in enumerate(records):
            try:
                normalized = _validate_generation_input(record, index=index)
                prepared = self.preprocessor.preprocess(normalized.sequence)
                tokenized = self.tokenizer.tokenize(prepared)
                model_output = self.model.infer(tokenized, layer_index=layer_index)

                model_output_map = cast(Dict[str, Any], model_output) if isinstance(model_output, dict) else None
                layers_obj_raw = model_output_map.get("layers") if model_output_map is not None else None
                if not isinstance(layers_obj_raw, dict):
                    raise EmbeddingBackendError(missing_layers_error)
                layers_obj = cast(Dict[int, Any], layers_obj_raw)

                for layer_id, layer_tensor in sorted(layers_obj.items(), key=lambda item: int(item[0])):
                    payload = resolved_pooler(layer_tensor) if resolved_pooler is not None else layer_tensor
                    embedding, shape = materialize_embedding_payload(payload)
                    result.records.append(
                        EmbeddingRecord(
                            id=normalized.id,
                            embedding=embedding,
                            layer_index=int(layer_id),
                            model_reference=self.model_reference,
                            shape=shape,
                            metadata=normalized.metadata,
                        )
                    )
            except Exception as exc:
                normalized_exc: EmbeddingGenerationError = _normalize_generation_exception(exc)
                if fail_fast:
                    raise normalized_exc
                result.errors.append(
                    {
                        "index": index,
                        "id": getattr(record, "id", None),
                        "error_type": normalized_exc.__class__.__name__,
                        "message": str(normalized_exc),
                    }
                )

        resolved_layers = sorted({int(record.layer_index) for record in result.records}) or None
        model_metadata_obj = getattr(self, "model_metadata", None)
        model_metadata = model_metadata_obj if isinstance(model_metadata_obj, ModelMetadata) else None
        run_metadata = RunMetadata(
            run_id=_uuid4(),
            created_at_utc=utc_now_iso(),
            sequence_count=len(records),
            requested_layers=requested_layers,
            resolved_layers=resolved_layers,
            failure_count=len(result.errors),
            parameters=run_parameters,
        )
        return GenerationResult(
            records=result.records,
            errors=result.errors,
            skipped=result.skipped,
            model_metadata=model_metadata,
            run_metadata=run_metadata,
        )

    def _generate_from_batched_layer_output_map(
        self,
        records: Sequence[GenerationInput],
        *,
        layer_index: LayerSelection,
        pooler: Any | None,
        fail_fast: bool,
        missing_layers_error: str,
        requested_layers: List[int] | None,
        run_parameters: Dict[str, Any] | None,
    ) -> GenerationResult:
        """Generate per-residue records with one tokenization/inference batch.

        Model adapters return ``{"layers": {layer: tensor}, ...}`` where each
        layer tensor is shaped as ``[batch, tokens_or_residues, hidden]``. They
        must also include either ``sample_spans=[(start, end), ...]`` for
        token tensors or ``residue_lens=[length, ...]`` for residue-aligned
        tensors.
        """
        result = GenerationResult()
        prepared_records: List[Tuple[int, GenerationInput, str]] = []

        for index, record in enumerate(records):
            try:
                normalized = _validate_generation_input(record, index=index)
                prepared = self.preprocessor.preprocess(normalized.sequence)
                prepared_records.append((index, normalized, prepared))
            except Exception as exc:
                normalized_exc = _normalize_generation_exception(exc)
                if fail_fast:
                    raise normalized_exc
                result.errors.append(
                    {
                        "index": index,
                        "id": getattr(record, "id", None),
                        "error_type": normalized_exc.__class__.__name__,
                        "message": str(normalized_exc),
                    }
                )

        if prepared_records:
            try:
                prepared_sequences = [prepared for _, _, prepared in prepared_records]
                tokenized = self.tokenizer.tokenize_many(prepared_sequences)
                model_output = self.model.infer(tokenized, layer_index=layer_index)
                self._append_batched_layer_output_records(
                    result,
                    model_output=model_output,
                    prepared_records=prepared_records,
                    pooler=pooler,
                    missing_layers_error=missing_layers_error,
                )
            except Exception as exc:
                normalized_exc = _normalize_generation_exception(exc)
                if fail_fast:
                    raise normalized_exc
                for index, normalized, _prepared in prepared_records:
                    result.errors.append(
                        {
                            "index": index,
                            "id": normalized.id,
                            "error_type": normalized_exc.__class__.__name__,
                            "message": str(normalized_exc),
                        }
                    )

        resolved_layers = sorted({int(record.layer_index) for record in result.records}) or None
        model_metadata_obj = getattr(self, "model_metadata", None)
        model_metadata = model_metadata_obj if isinstance(model_metadata_obj, ModelMetadata) else None
        run_metadata = RunMetadata(
            run_id=_uuid4(),
            created_at_utc=utc_now_iso(),
            sequence_count=len(records),
            requested_layers=requested_layers,
            resolved_layers=resolved_layers,
            failure_count=len(result.errors),
            parameters=run_parameters,
        )
        return GenerationResult(
            records=result.records,
            errors=result.errors,
            skipped=result.skipped,
            model_metadata=model_metadata,
            run_metadata=run_metadata,
        )

    def _append_batched_layer_output_records(
        self,
        result: GenerationResult,
        *,
        model_output: Any,
        prepared_records: Sequence[Tuple[int, GenerationInput, str]],
        pooler: Any | None,
        missing_layers_error: str,
    ) -> None:
        from .embeddings_pooler import materialize_embedding_payload, resolve_pooler

        resolved_pooler = resolve_pooler(pooler)
        model_output_map = cast(Dict[str, Any], model_output) if isinstance(model_output, dict) else None
        layers_obj_raw = model_output_map.get("layers") if model_output_map is not None else None
        if not isinstance(layers_obj_raw, dict):
            raise EmbeddingBackendError(missing_layers_error)
        layers_obj = cast(Dict[int, Any], layers_obj_raw)
        spans, explicit_spans = _sample_spans_from_model_output(
            model_output_map,
            sample_count=len(prepared_records),
        )

        for row_index, (_source_index, normalized, _prepared) in enumerate(prepared_records):
            start, end = spans[row_index]
            for layer_id, layer_tensor in sorted(layers_obj.items(), key=lambda item: int(item[0])):
                sample_tensor = _slice_batched_layer_tensor(
                    layer_tensor,
                    row_index=row_index,
                    start=start,
                    end=end,
                    singleton_count=len(prepared_records),
                    explicit_spans=explicit_spans,
                )
                payload = resolved_pooler(sample_tensor) if resolved_pooler is not None else sample_tensor
                embedding, shape = materialize_embedding_payload(payload)
                result.records.append(
                    EmbeddingRecord(
                        id=normalized.id,
                        embedding=embedding,
                        layer_index=int(layer_id),
                        model_reference=self.model_reference,
                        shape=shape,
                        metadata=normalized.metadata,
                    )
                )

    def generate_batches(
        self,
        records: Iterable[GenerationInput],
        *,
        batch_size: int,
        max_batch_tokens: int | None = None,
        layer_index: LayerSelection = 0,
        pooler: Any | None = None,
        fail_fast: bool = False,
    ) -> Iterator[GenerationResult]:
        """Yield one ``GenerationResult`` per input batch."""
        resolved_batch_size = _validate_batch_size(batch_size)
        for batch in batch_generation_inputs(
            records,
            batch_size=resolved_batch_size,
            max_batch_tokens=max_batch_tokens,
        ):
            yield self.generate(batch, layer_index=cast(Any, layer_index), pooler=pooler, fail_fast=fail_fast)

    def iter_records(
        self,
        records: Iterable[GenerationInput],
        *,
        batch_size: int,
        max_batch_tokens: int | None = None,
        layer_index: LayerSelection = 0,
        pooler: Any | None = None,
        fail_fast: bool = False,
    ) -> Iterator[EmbeddingRecord]:
        """Iterate over records from ``generate_batches(...)`` results."""
        for result in self.generate_batches(
            records,
            batch_size=batch_size,
            max_batch_tokens=max_batch_tokens,
            layer_index=layer_index,
            pooler=pooler,
            fail_fast=fail_fast,
        ):
            yield from result.records


def load_fasta_inputs(
    path: str | Path,
    *,
    id_from: Literal["record_id", "description"] = "record_id",
) -> List[GenerationInput]:
    """Load FASTA records into ``GenerationInput`` values."""
    return list(iter_fasta_inputs(path, id_from=id_from))


def iter_fasta_inputs(
    path: str | Path,
    *,
    id_from: Literal["record_id", "description"] = "record_id",
) -> Iterator[GenerationInput]:
    """Yield ``GenerationInput`` values lazily from a FASTA file."""
    try:
        from Bio import SeqIO  # type: ignore
    except ModuleNotFoundError as exc:
        raise EmbeddingDependencyError(
            "Biopython is required for FASTA loading. Install with: pip install biopython"
        ) from exc

    file_path = Path(path)
    seqio_module = cast(Any, SeqIO)
    parsed_records = cast(Iterable[object], seqio_module.parse(str(file_path), "fasta"))
    for index, record_obj in enumerate(parsed_records):
        yield _generation_input_from_fasta_record(record_obj, index=index, id_from=id_from)


def batch_generation_inputs(
    records: Iterable[GenerationInput],
    *,
    batch_size: int,
    max_batch_tokens: int | None = None,
) -> Iterator[List[GenerationInput]]:
    """Group generation inputs into size-capped lists.

    ``max_batch_tokens`` caps the padded token budget estimated as
    ``batch_size * max(sequence_length + 1)``. A single over-budget record is
    still yielded as a singleton because a record cannot be split here.
    """
    resolved_batch_size = _validate_batch_size(batch_size)
    resolved_max_batch_tokens = _validate_optional_positive_int("max_batch_tokens", max_batch_tokens)
    batch: List[GenerationInput] = []
    batch_max_tokens = 0
    for record in records:
        record_tokens = len(str(record.sequence)) + 1
        candidate_max_tokens = max(batch_max_tokens, record_tokens)
        if batch and (
            len(batch) >= resolved_batch_size
            or (
                resolved_max_batch_tokens is not None
                and (len(batch) + 1) * candidate_max_tokens > resolved_max_batch_tokens
            )
        ):
            yield batch
            batch = []
            batch_max_tokens = 0

        batch.append(record)
        batch_max_tokens = max(batch_max_tokens, record_tokens)
    if batch:
        yield batch


def generate_from_fasta(
    path: str | Path,
    generator: EmbeddingGenerator,
    *,
    layer_index: LayerSelection = 0,
    pooler: Any | None = None,
    fail_fast: bool = False,
    id_from: Literal["record_id", "description"] = "record_id",
    batch_size: int | None = None,
    max_batch_tokens: int | None = None,
) -> GenerationResult:
    """Deprecated wrapper around ``load_fasta_inputs`` and generator methods."""
    _warn_deprecated(
        "generate_from_fasta",
        "load_fasta_inputs(...) with generator.generate(...) or generator.generate_batches(...)",
    )
    if batch_size is None:
        records = load_fasta_inputs(path, id_from=id_from)
        return generator.generate(records, layer_index=layer_index, pooler=pooler, fail_fast=fail_fast)

    results = generator.generate_batches(
        iter_fasta_inputs(path, id_from=id_from),
        batch_size=int(batch_size),
        max_batch_tokens=max_batch_tokens,
        layer_index=layer_index,
        pooler=pooler,
        fail_fast=fail_fast,
    )
    return collect_generation_results(results)


def generate_from_fasta_batches(
    path: str | Path,
    generator: EmbeddingGenerator,
    *,
    batch_size: int,
    max_batch_tokens: int | None = None,
    layer_index: LayerSelection = 0,
    pooler: Any | None = None,
    fail_fast: bool = False,
    id_from: Literal["record_id", "description"] = "record_id",
) -> Iterator[GenerationResult]:
    """Deprecated wrapper around ``iter_fasta_inputs`` and ``generate_batches``."""
    _warn_deprecated(
        "generate_from_fasta_batches",
        "generator.generate_batches(iter_fasta_inputs(...), ...)",
    )
    yield from generator.generate_batches(
        iter_fasta_inputs(path, id_from=id_from),
        batch_size=batch_size,
        max_batch_tokens=max_batch_tokens,
        layer_index=layer_index,
        pooler=pooler,
        fail_fast=fail_fast,
    )


def iter_embedding_records_from_fasta(
    path: str | Path,
    generator: EmbeddingGenerator,
    *,
    batch_size: int,
    max_batch_tokens: int | None = None,
    layer_index: LayerSelection = 0,
    pooler: Any | None = None,
    fail_fast: bool = False,
    id_from: Literal["record_id", "description"] = "record_id",
) -> Iterator[EmbeddingRecord]:
    """Deprecated wrapper around ``iter_fasta_inputs`` and ``generate_batches``."""
    _warn_deprecated(
        "iter_embedding_records_from_fasta",
        "generator.generate_batches(iter_fasta_inputs(...), ...)",
    )
    for result in generator.generate_batches(
        iter_fasta_inputs(path, id_from=id_from),
        batch_size=batch_size,
        max_batch_tokens=max_batch_tokens,
        layer_index=layer_index,
        pooler=pooler,
        fail_fast=fail_fast,
    ):
        yield from result.records


def collect_generation_results(results: Iterable[GenerationResult]) -> GenerationResult:
    """Merge batch results into one aggregate ``GenerationResult``."""
    merged_records: List[EmbeddingRecord] = []
    merged_errors: List[Dict[str, Any]] = []
    merged_skipped: List[Dict[str, Any]] = []
    model_metadata: ModelMetadata | None = None
    requested_layers: List[int] | None = None
    resolved_layers: set[int] = set()
    failure_count = 0
    sequence_count = 0
    parameter_values: List[Dict[str, Any] | None] = []

    for result in results:
        merged_records.extend(result.records)
        merged_errors.extend(result.errors)
        merged_skipped.extend(result.skipped)
        if model_metadata is None and result.model_metadata is not None:
            model_metadata = result.model_metadata

        run_metadata = result.run_metadata
        if run_metadata is not None:
            sequence_count += int(run_metadata.sequence_count)
            failure_count += int(run_metadata.failure_count)
            if run_metadata.requested_layers is not None:
                if requested_layers is None:
                    requested_layers = list(run_metadata.requested_layers)
                elif list(run_metadata.requested_layers) != requested_layers:
                    requested_layers = None
            if run_metadata.resolved_layers is not None:
                resolved_layers.update(int(value) for value in run_metadata.resolved_layers)
            parameter_values.append(run_metadata.parameters)
        else:
            sequence_count += _result_sequence_count(result)
            failure_count += len(result.errors)

        if run_metadata is None and result.records:
            resolved_layers.update(int(record.layer_index) for record in result.records)

    aggregate_run_metadata: RunMetadata | None = None
    if sequence_count or merged_records or merged_errors or merged_skipped:
        aggregate_run_metadata = RunMetadata(
            run_id=_uuid4(),
            created_at_utc=utc_now_iso(),
            sequence_count=sequence_count,
            requested_layers=requested_layers,
            resolved_layers=sorted(resolved_layers) or None,
            failure_count=failure_count,
            parameters=_shared_parameters(parameter_values),
        )

    return GenerationResult(
        records=merged_records,
        errors=merged_errors,
        skipped=merged_skipped,
        model_metadata=model_metadata,
        run_metadata=aggregate_run_metadata,
    )


def _validate_generation_input(record: object, *, index: int) -> GenerationInput:
    if not isinstance(record, GenerationInput):
        raise EmbeddingInputError(
            f"Expected GenerationInput at index {index}, got {type(record).__name__}."
        )

    rec_id = str(record.id).strip()
    if not rec_id:
        raise EmbeddingInputError(f"Empty record id at index {index}.")

    sequence = str(record.sequence).strip().upper()
    _validate_sequence(sequence, context=f"record index={index} id={rec_id!r}")

    return GenerationInput(
        id=rec_id,
        sequence=sequence,
        description=record.description,
        metadata=record.metadata,
    )


def _validate_sequence(sequence: str, *, context: str) -> None:
    if not sequence:
        raise EmbeddingInputError(f"Empty amino-acid sequence ({context}).")
    invalid = sorted({char for char in sequence if char not in _VALID_AA})
    if invalid:
        invalid_text = "".join(invalid)
        raise EmbeddingInputError(
            f"Invalid amino-acid characters [{invalid_text}] in sequence ({context})."
        )


def _as_float_vector(value: object) -> List[float]:
    tolist = getattr(value, "tolist", None)
    if callable(tolist):
        value = tolist()

    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise EmbeddingBackendError("Embedding output must be a sequence of numeric values.")

    values = cast(Sequence[object], value)
    vector: List[float] = []
    for item in values:
        try:
            vector.append(float(cast(Any, item)))
        except (TypeError, ValueError) as exc:
            raise EmbeddingBackendError(f"Embedding element is not numeric: {item!r}") from exc
    return vector


def _as_float_matrix(value: object) -> List[List[float]]:
    tolist = getattr(value, "tolist", None)
    if callable(tolist):
        value = tolist()

    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise EmbeddingBackendError("Embedding output must be a row-major sequence of numeric vectors.")

    rows = cast(Sequence[object], value)
    return [_as_float_vector(row) for row in rows]


def _normalize_generation_exception(exc: Exception) -> EmbeddingGenerationError:
    if isinstance(exc, EmbeddingGenerationError):
        return exc
    return EmbeddingBackendError(f"Backend failure: {exc}")


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _generation_input_from_fasta_record(
    record_obj: object,
    *,
    index: int,
    id_from: Literal["record_id", "description"],
) -> GenerationInput:
    record = cast(Any, record_obj)
    record_id = str(record.id or "").strip()
    description = str(record.description or "").strip() or None
    selected_id = record_id if id_from == "record_id" else str(description or "").strip()
    if not selected_id:
        raise EmbeddingInputError(f"Empty FASTA identifier at record index {index}.")

    sequence = str(record.seq or "").strip().upper()
    _validate_sequence(sequence, context=f"FASTA record index={index} id={selected_id!r}")
    return GenerationInput(
        id=selected_id,
        sequence=sequence,
        description=description,
        metadata={"source": "fasta", "record_id": record_id},
    )


def _validate_batch_size(batch_size: int) -> int:
    resolved = int(batch_size)
    if resolved <= 0:
        raise EmbeddingInputError("batch_size must be a positive integer.")
    return resolved


def _validate_optional_positive_int(name: str, value: int | None) -> int | None:
    if value is None:
        return None
    resolved = int(value)
    if resolved <= 0:
        raise EmbeddingInputError(f"{name} must be a positive integer when provided.")
    return resolved


def _sample_spans_from_model_output(
    model_output_map: Dict[str, Any] | None,
    *,
    sample_count: int,
) -> Tuple[List[Tuple[int, int]], bool]:
    if model_output_map is None:
        raise EmbeddingBackendError("Batched model output must be a dictionary.")

    sample_spans_raw = model_output_map.get("sample_spans")
    if sample_spans_raw is not None:
        if not isinstance(sample_spans_raw, Sequence) or isinstance(sample_spans_raw, (str, bytes, bytearray)):
            raise EmbeddingBackendError("Batched model output 'sample_spans' must be a sequence.")
        spans: List[Tuple[int, int]] = []
        for item in cast(Sequence[object], sample_spans_raw):
            if not isinstance(item, Sequence) or isinstance(item, (str, bytes, bytearray)):
                raise EmbeddingBackendError("Each sample span must be a (start, end) pair.")
            pair = cast(Sequence[object], item)
            if len(pair) != 2:
                raise EmbeddingBackendError("Each sample span must be a (start, end) pair.")
            start = int(cast(Any, pair[0]))
            end = int(cast(Any, pair[1]))
            if start < 0 or end < start:
                raise EmbeddingBackendError(f"Invalid sample span: {(start, end)!r}.")
            spans.append((start, end))
        if len(spans) != sample_count:
            raise EmbeddingBackendError(
                f"Batched model output sample span count {len(spans)} does not match batch size {sample_count}."
            )
        return spans, True

    residue_lens_raw = model_output_map.get("residue_lens")
    if residue_lens_raw is None and sample_count == 1:
        residue_len_raw = model_output_map.get("residue_len")
        if residue_len_raw is not None:
            residue_lens_raw = [residue_len_raw]

    if residue_lens_raw is None:
        raise EmbeddingBackendError("Batched model output must include sample_spans or residue_lens.")
    if not isinstance(residue_lens_raw, Sequence) or isinstance(residue_lens_raw, (str, bytes, bytearray)):
        raise EmbeddingBackendError("Batched model output 'residue_lens' must be a sequence.")

    lengths = [int(cast(Any, value)) for value in cast(Sequence[object], residue_lens_raw)]
    if len(lengths) != sample_count:
        raise EmbeddingBackendError(
            f"Batched model output residue length count {len(lengths)} does not match batch size {sample_count}."
        )
    spans = []
    for length in lengths:
        if length < 0:
            raise EmbeddingBackendError(f"Invalid residue length: {length}.")
        spans.append((0, length))
    return spans, False


def _slice_batched_layer_tensor(
    layer_tensor: Any,
    *,
    row_index: int,
    start: int,
    end: int,
    singleton_count: int,
    explicit_spans: bool,
) -> Any:
    if singleton_count == 1 and not explicit_spans:
        return layer_tensor[start:end]
    return layer_tensor[row_index, start:end]


def _result_sequence_count(result: GenerationResult) -> int:
    return len(result.records) + len(result.errors) + len(result.skipped)


def _shared_parameters(values: Sequence[Dict[str, Any] | None]) -> Dict[str, Any] | None:
    filtered = [value for value in values if value is not None]
    if not filtered:
        return None
    first = filtered[0]
    if all(value == first for value in filtered[1:]):
        return dict(first)
    return None


def _uuid4() -> str:
    import uuid

    return str(uuid.uuid4())


validate_generation_input = _validate_generation_input
validate_sequence = _validate_sequence
as_float_vector = _as_float_vector
as_float_matrix = _as_float_matrix
normalize_generation_exception = _normalize_generation_exception


__all__ = [
    "EmbeddingGenerationError",
    "EmbeddingInputError",
    "EmbeddingDependencyError",
    "EmbeddingBackendError",
    "EmbeddingPayload",
    "LayerSelection",
    "PicklePayloadFormat",
    "GenerationInput",
    "EmbeddingRecord",
    "ModelMetadata",
    "RunMetadata",
    "GenerationResult",
    "PickleShardWriteResult",
    "NpyShardWriteResult",
    "H5WriteResult",
    "FastaEmbeddingPickleShardResult",
    "FastaEmbeddingNpyShardResult",
    "FastaEmbeddingH5Result",
    "PreprocessorAdapter",
    "TokenizerAdapter",
    "ModelAdapter",
    "PostprocessorAdapter",
    "EmbeddingGenerator",
    "iter_fasta_inputs",
    "batch_generation_inputs",
    "load_fasta_inputs",
    "generate_from_fasta",
    "generate_from_fasta_batches",
    "iter_embedding_records_from_fasta",
    "collect_generation_results",
    "validate_generation_input",
    "validate_sequence",
    "as_float_vector",
    "as_float_matrix",
    "normalize_generation_exception",
    "utc_now_iso",
]
