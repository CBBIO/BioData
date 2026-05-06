"""Embedding generation abstractions and orchestration utilities.

This module is intentionally independent from database access code.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable, Iterator, Mapping as MappingABC
from dataclasses import dataclass, field
from datetime import datetime, timezone
import pickle
from pathlib import Path
from typing import Any, Dict, List, Literal, Sequence, Tuple, Type, TypeAlias, cast


_VALID_AA = set("ACDEFGHIKLMNPQRSTVWYBXZJUO")
EmbeddingPayload: TypeAlias = Sequence[float] | Sequence[Sequence[float]]
LayerSelection: TypeAlias = int | Sequence[int] | None


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


class PreprocessorAdapter(ABC):
    @abstractmethod
    def preprocess(self, raw_sequence: str) -> str: ...


class TokenizerAdapter(ABC):
    @abstractmethod
    def tokenize(self, sequence: str) -> Any: ...


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
        layer_index: int = 0,
        fail_fast: bool = False,
    ) -> GenerationResult:
        if not isinstance(layer_index, int):
            raise EmbeddingInputError("Base EmbeddingGenerator.generate requires an integer layer_index.")
        result = GenerationResult()

        for index, record in enumerate(records):
            try:
                normalized = _validate_generation_input(record, index=index)
                prepared = self.preprocessor.preprocess(normalized.sequence)
                _validate_sequence(prepared, context=f"record index={index} id={normalized.id!r}")
                tokens = self.tokenizer.tokenize(prepared)
                model_output = self.model.infer(tokens, layer_index=layer_index)
                vector_out = self.postprocessor.postprocess(model_output)
                vector = _as_float_vector(vector_out)
                if not vector:
                    raise EmbeddingBackendError("postprocessor returned an empty embedding vector.")

                result.records.append(
                    EmbeddingRecord(
                        id=normalized.id,
                        embedding=vector,
                        layer_index=int(layer_index),
                        model_reference=self.model_reference,
                        shape=(len(vector),),
                        metadata=normalized.metadata,
                    )
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

    def generate_batches(
        self,
        records: Iterable[GenerationInput],
        *,
        batch_size: int,
        layer_index: LayerSelection = 0,
        fail_fast: bool = False,
    ) -> Iterator[GenerationResult]:
        """Yield one ``GenerationResult`` per input batch."""
        resolved_batch_size = _validate_batch_size(batch_size)
        for batch in batch_generation_inputs(records, batch_size=resolved_batch_size):
            yield self.generate(batch, layer_index=cast(Any, layer_index), fail_fast=fail_fast)

    def iter_records(
        self,
        records: Iterable[GenerationInput],
        *,
        batch_size: int,
        layer_index: LayerSelection = 0,
        fail_fast: bool = False,
    ) -> Iterator[EmbeddingRecord]:
        """Yield embedding records incrementally from batched generation."""
        for result in self.generate_batches(
            records,
            batch_size=batch_size,
            layer_index=layer_index,
            fail_fast=fail_fast,
        ):
            yield from result.records


def Generator(
    *,
    name: str | None = None,
    model_class: str | None = None,
    class_: str | None = None,
    device: str = "cpu",
    **kwargs: Any,
) -> EmbeddingGenerator:
    """Convenience factory for model-specific embedding generators.

    Examples:
    - ``Generator(model_class="protT5")``  # uses family default model
    - ``Generator(model_class="protT5", name="Rostlab/prot_t5_xl_uniref50")``
    - ``Generator(class_="prostT5", name="Rostlab/ProstT5")``
    - ``Generator(model_class="ankh3", name="ElnaggarLab/ankh3-large", prefix="[S2S]")``
    - ``Generator(model_class="esmc", name="esmc_300m", use_flash_attention=True)``
    - ``Generator(model_class="esm2", name="esm2_t33_650M_UR50D")``
    - ``Generator(model_class="esm1b", name="esm1b_t33_650M_UR50S")``
    - ``Generator(**{"class": "protT5", "name": "Rostlab/prot_t5_xl_uniref50"})``
    """
    resolved_class = model_class or class_ or cast(str | None, kwargs.pop("class", None))
    if resolved_class is None or not str(resolved_class).strip():
        raise EmbeddingInputError("Generator requires model_class/class_/class (e.g. 'protT5').")

    registry, aliases = _generator_registry()
    canonical = _normalize_model_class(resolved_class, aliases)
    if canonical is None or canonical not in registry:
        supported = ", ".join(registry.keys())
        raise EmbeddingInputError(
            f"Unknown model class: {resolved_class!r}. Supported values: {supported}."
        )

    generator_cls = registry[canonical]
    resolved_name = str(name).strip() if name is not None else ""
    if not resolved_name:
        default_name = getattr(generator_cls, "DEFAULT_MODEL_NAME", None)
        if isinstance(default_name, str) and default_name.strip():
            resolved_name = default_name.strip()
        else:
            raise EmbeddingInputError(
                f"Generator requires a model name for class {canonical!r}; no DEFAULT_MODEL_NAME is defined."
            )
    generator_factory = cast(Any, generator_cls)
    return generator_factory(model_name=resolved_name, device=device, **kwargs)


def available_generator_classes() -> List[str]:
    """Return canonical generator class names supported by ``Generator``."""
    registry, _ = _generator_registry()
    return list(registry.keys())


def available_generator_models(model_class: str | None = None) -> Dict[str, List[str]] | List[str]:
    """Return available model identifiers by family.

    - When ``model_class`` is ``None``: returns ``{class_name: [models...]}``.
    - When ``model_class`` is provided: returns ``[models...]`` for that family.
    """
    registry, aliases = _generator_registry()
    if model_class is None:
        return {name: _family_models_for_class(name, registry=registry) for name in registry.keys()}

    canonical = _normalize_model_class(model_class, aliases)
    if canonical is None or canonical not in registry:
        supported = ", ".join(registry.keys())
        raise EmbeddingInputError(
            f"Unknown model class: {model_class!r}. Supported values: {supported}."
        )
    return _family_models_for_class(canonical, registry=registry)


def _normalize_model_class(value: str | None, aliases: Dict[str, str]) -> str | None:
    if value is None:
        return None
    key = str(value).strip().lower()
    if not key:
        return None
    return aliases.get(key)


def _family_models_for_class(
    canonical_class: str,
    *,
    registry: Dict[str, Type["EmbeddingGenerator"]],
) -> List[str]:
    generator_cls = registry.get(canonical_class)
    if generator_cls is None:
        supported = ", ".join(registry.keys())
        raise EmbeddingInputError(
            f"Unknown model class: {canonical_class!r}. Supported values: {supported}."
        )

    values = getattr(generator_cls, "FAMILY_MODELS", None)
    if isinstance(values, Sequence) and not isinstance(values, (str, bytes, bytearray)):
        family_values = cast(Sequence[object], values)
        models = [str(value) for value in family_values if str(value).strip()]
        if models:
            return models

    default_name = getattr(generator_cls, "DEFAULT_MODEL_NAME", None)
    if isinstance(default_name, str) and default_name.strip():
        return [default_name]
    return []


def _generator_registry() -> Tuple[Dict[str, Type["EmbeddingGenerator"]], Dict[str, str]]:
    classes = _load_generator_classes()
    registry: Dict[str, Type["EmbeddingGenerator"]] = {}
    aliases: Dict[str, str] = {}

    for generator_cls in classes:
        canonical = str(getattr(generator_cls, "GENERATOR_CLASS", "")).strip()
        if not canonical:
            continue
        if canonical in registry:
            raise EmbeddingBackendError(f"Duplicate generator class registration for {canonical!r}.")
        registry[canonical] = generator_cls

        raw_aliases = getattr(generator_cls, "GENERATOR_ALIASES", ())
        alias_values = [canonical]
        if isinstance(raw_aliases, Sequence) and not isinstance(raw_aliases, (str, bytes, bytearray)):
            alias_values.extend(str(value) for value in cast(Sequence[object], raw_aliases))

        for alias in alias_values:
            key = str(alias).strip().lower()
            if not key:
                continue
            existing = aliases.get(key)
            if existing is not None and existing != canonical:
                raise EmbeddingBackendError(
                    f"Alias {alias!r} is defined by multiple generator classes: {existing!r}, {canonical!r}."
                )
            aliases[key] = canonical

    return registry, aliases


def _load_generator_classes() -> Tuple[Type["EmbeddingGenerator"], ...]:
    from .embeddings_prott5 import ProtT5EmbeddingGenerator
    from .embeddings_prostt5 import ProstT5EmbeddingGenerator
    from .embeddings_ankh3 import Ankh3EmbeddingGenerator
    from .embeddings_esmc import EsmcEmbeddingGenerator
    from .embeddings_esm2 import Esm2EmbeddingGenerator
    from .embeddings_esm1b import Esm1bEmbeddingGenerator

    return (
        ProtT5EmbeddingGenerator,
        ProstT5EmbeddingGenerator,
        Ankh3EmbeddingGenerator,
        EsmcEmbeddingGenerator,
        Esm2EmbeddingGenerator,
        Esm1bEmbeddingGenerator,
    )


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
) -> Iterator[List[GenerationInput]]:
    """Group generation inputs into fixed-size lists."""
    resolved_batch_size = _validate_batch_size(batch_size)
    batch: List[GenerationInput] = []
    for record in records:
        batch.append(record)
        if len(batch) >= resolved_batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


def generate_from_fasta(
    path: str | Path,
    generator: EmbeddingGenerator,
    *,
    layer_index: LayerSelection = 0,
    fail_fast: bool = False,
    id_from: Literal["record_id", "description"] = "record_id",
    batch_size: int | None = None,
) -> GenerationResult:
    """Load FASTA records and generate embeddings through ``generator``."""
    if batch_size is None:
        records = load_fasta_inputs(path, id_from=id_from)
        return generator.generate(records, layer_index=layer_index, fail_fast=fail_fast)

    results = generate_from_fasta_batches(
        path,
        generator,
        batch_size=batch_size,
        layer_index=layer_index,
        fail_fast=fail_fast,
        id_from=id_from,
    )
    return collect_generation_results(results)


def generate_from_fasta_batches(
    path: str | Path,
    generator: EmbeddingGenerator,
    *,
    batch_size: int,
    layer_index: LayerSelection = 0,
    fail_fast: bool = False,
    id_from: Literal["record_id", "description"] = "record_id",
) -> Iterator[GenerationResult]:
    """Yield one ``GenerationResult`` per FASTA batch."""
    yield from generator.generate_batches(
        iter_fasta_inputs(path, id_from=id_from),
        batch_size=batch_size,
        layer_index=layer_index,
        fail_fast=fail_fast,
    )


def iter_embedding_records_from_fasta(
    path: str | Path,
    generator: EmbeddingGenerator,
    *,
    batch_size: int,
    layer_index: LayerSelection = 0,
    fail_fast: bool = False,
    id_from: Literal["record_id", "description"] = "record_id",
) -> Iterator[EmbeddingRecord]:
    """Yield embedding records incrementally from a FASTA file."""
    yield from generator.iter_records(
        iter_fasta_inputs(path, id_from=id_from),
        batch_size=batch_size,
        layer_index=layer_index,
        fail_fast=fail_fast,
    )


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


def load_embedding_records(
    path: str | Path,
    *,
    model_reference: str = "unknown",
    layer_index: int = 0,
    ids: Sequence[str] | None = None,
) -> List[EmbeddingRecord]:
    """Load embedding records from supported file formats.

    Supported extensions:
    - ``.pkl`` / ``.pickle``: pickled Python objects
    - ``.npy`` / ``.npz``: NumPy arrays (optional dependency)
    """
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
    raise EmbeddingInputError(f"Unsupported embedding file extension: {suffix!r}")


def save_embedding_records_pickle(
    path: str | Path,
    records: Sequence[EmbeddingRecord],
    *,
    payload_format: Literal["records", "mapping"] = "records",
) -> Path:
    """Save embedding records to a pickle file.

    ``payload_format='records'`` stores:
    ``{"records": [{"id": ..., "embedding": ..., ...}, ...]}``

    ``payload_format='mapping'`` stores:
    ``{id: embedding_vector}``
    """
    file_path = Path(path)
    normalized = _normalize_embedding_records(records)

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
    np.save(file_path, matrix)
    return file_path


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

        # Common compact shape: {id: embedding_vector}
        if all(isinstance(key, str) for key in payload_map.keys()):
            records: List[EmbeddingRecord] = []
            for rec_id, emb_value in payload_map.items():
                vector = _as_float_vector(emb_value)
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

    vector = _as_float_vector(item_map.get("embedding"))
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
        rows = [_as_float_vector(matrix)]
    elif ndim == 2:
        row_count = int(matrix.shape[0])
        rows = [_as_float_vector(matrix[idx]) for idx in range(row_count)]
    else:
        raise EmbeddingInputError(f"Expected 1D or 2D array, got ndim={ndim}.")

    total = len(rows)
    resolved_ids: List[str]
    if ids is None:
        resolved_ids = [f"row_{idx}" for idx in range(total)]
    else:
        resolved_ids = [str(value) for value in ids]
        if len(resolved_ids) != total:
            raise EmbeddingInputError(
                f"ids length ({len(resolved_ids)}) does not match number of rows ({total})."
            )

    records: List[EmbeddingRecord] = []
    for idx, vector in enumerate(rows):
        records.append(
            EmbeddingRecord(
                id=resolved_ids[idx],
                embedding=vector,
                layer_index=int(layer_index),
                model_reference=model_reference,
                shape=(len(vector),),
                metadata={"source": "npy"},
            )
        )
    return records


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
        vector = _as_float_vector(record.embedding)
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
    "GenerationInput",
    "EmbeddingRecord",
    "ModelMetadata",
    "RunMetadata",
    "GenerationResult",
    "PreprocessorAdapter",
    "TokenizerAdapter",
    "ModelAdapter",
    "PostprocessorAdapter",
    "EmbeddingGenerator",
    "Generator",
    "available_generator_classes",
    "available_generator_models",
    "iter_fasta_inputs",
    "batch_generation_inputs",
    "load_fasta_inputs",
    "generate_from_fasta",
    "generate_from_fasta_batches",
    "iter_embedding_records_from_fasta",
    "collect_generation_results",
    "load_embedding_records",
    "load_embedding_records_pickle",
    "load_embedding_records_npy",
    "save_embedding_records_pickle",
    "save_embedding_records_npy",
    "validate_generation_input",
    "validate_sequence",
    "as_float_vector",
    "as_float_matrix",
    "normalize_generation_exception",
    "utc_now_iso",
]
