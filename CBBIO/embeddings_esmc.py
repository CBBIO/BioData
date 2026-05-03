"""ESM-C specific embedding adapters and generator (protein -> embedding)."""

from __future__ import annotations

from importlib import metadata as importlib_metadata
import re
import uuid
from typing import Any, Dict, List, Sequence, cast

from .embeddings import (
    EmbeddingBackendError,
    EmbeddingDependencyError,
    EmbeddingPayload,
    EmbeddingRecord,
    EmbeddingGenerator,
    EmbeddingGenerationError,
    EmbeddingInputError,
    GenerationInput,
    GenerationResult,
    ModelMetadata,
    ModelAdapter,
    PostprocessorAdapter,
    PreprocessorAdapter,
    RunMetadata,
    TokenizerAdapter,
    as_float_matrix,
    normalize_generation_exception,
    utc_now_iso,
    validate_generation_input,
    validate_sequence,
)


ESMC_LAYER_SPECS: Dict[str, int] = {
    "esmc-6b-2024-12": 80,
    "esmc-600m-2024-12": 36,
    "esmc-300m-2024-12": 30,
    "esmc_6b": 80,
    "esmc_600m": 36,
    "esmc_300m": 30,
}


class EsmcPreprocessor(PreprocessorAdapter):
    """Preprocessing for ESM-C protein inputs."""

    def preprocess(self, raw_sequence: str) -> str:
        sequence = str(raw_sequence).strip().upper()
        validate_sequence(sequence, context="ESM-C preprocessing")
        return re.sub(r"[UZOB]", "X", sequence)


class EsmcTokenizerAdapter(TokenizerAdapter):
    """Tokenizer adapter for ESM-C SDK encode path."""

    def __init__(self, client: Any, *, protein_cls: Any | None = None) -> None:
        self.client = client
        self.protein_cls = protein_cls

    def tokenize(self, sequence: str) -> Any:
        protein_cls = self.protein_cls
        if protein_cls is None:
            try:
                from esm.sdk.api import ESMProtein  # type: ignore
            except ModuleNotFoundError as exc:
                raise EmbeddingDependencyError(
                    "ESM SDK is required for ESM-C tokenization. Install package providing esm.sdk.api."
                ) from exc
            protein_cls = ESMProtein

        protein = protein_cls(sequence=sequence)
        return self.client.encode(protein)


class EsmcModelAdapter(ModelAdapter):
    """Model adapter for ESM-C SDK logits path."""

    def __init__(
        self,
        client: Any,
        *,
        logits_config_cls: Any | None = None,
        available_layer_count: int | None = None,
    ) -> None:
        self.client = client
        self.logits_config_cls = logits_config_cls
        self.available_layer_count = available_layer_count

    def infer(self, tokens: Any, *, layer_index: int | Sequence[int] | None = None) -> Any:
        logits_config_cls = self.logits_config_cls
        if logits_config_cls is None:
            try:
                from esm.sdk.api import LogitsConfig  # type: ignore
            except ModuleNotFoundError as exc:
                raise EmbeddingDependencyError(
                    "ESM SDK is required for ESM-C logits config. Install package providing esm.sdk.api."
                ) from exc
            logits_config_cls = LogitsConfig

        output = self.client.logits(
            tokens,
            logits_config_cls(sequence=True, return_embeddings=True),
        )
        embeddings = getattr(output, "embeddings", None)
        if embeddings is None:
            raise EmbeddingBackendError("ESM-C logits output did not include embeddings.")

        layers = _layers_from_embeddings(embeddings)
        selected = _select_layers(layers, layer_index=layer_index)
        return {"layers": selected}

    def available_layers(self) -> List[int] | None:
        if self.available_layer_count is None:
            return None
        if self.available_layer_count < 1:
            return None
        return list(range(int(self.available_layer_count)))


class EsmcPostprocessor(PostprocessorAdapter):
    """Postprocessing for ESM-C outputs without pooling."""

    def postprocess(self, model_output: Any) -> EmbeddingPayload:
        if not isinstance(model_output, dict):
            raise EmbeddingBackendError("EsmcPostprocessor expects a dict payload from model adapter.")
        model_output_map = cast(Dict[str, Any], model_output)
        layers_obj_raw = model_output_map.get("layers")
        if not isinstance(layers_obj_raw, dict):
            raise EmbeddingBackendError("EsmcPostprocessor expects a 'layers' dict in model output.")
        layers_obj = cast(Dict[int, Any], layers_obj_raw)
        if not layers_obj:
            raise EmbeddingBackendError("EsmcPostprocessor received no layers.")

        first_key = sorted(layers_obj.keys())[0]
        layer_tensor = layers_obj[first_key]
        return as_float_matrix(layer_tensor)


class EsmcEmbeddingGenerator(EmbeddingGenerator):
    """Concrete embedding generator for ESM-C family models."""

    GENERATOR_CLASS = "esmc"
    GENERATOR_ALIASES = ("esm-c", "esmc3", "esm3c","ESM3c")
    DEFAULT_MODEL_NAME = "esmc_600m"
    FAMILY_MODELS = sorted(str(value) for value in ESMC_LAYER_SPECS.keys())

    def __init__(
        self,
        *,
        model_name: str = DEFAULT_MODEL_NAME,
        device: str = "cpu",
        use_flash_attention: bool | None = None,
        client: Any | None = None,
        from_pretrained_kwargs: Dict[str, Any] | None = None,
    ) -> None:
        resolved_client = client
        kwargs = dict(from_pretrained_kwargs or {})
        if use_flash_attention is not None:
            kwargs.setdefault("use_flash_attention", bool(use_flash_attention))

        if resolved_client is None:
            try:
                from esm.models.esmc import ESMC  # type: ignore
            except ModuleNotFoundError as exc:
                raise EmbeddingDependencyError(
                    "ESM SDK is required for ESM-C loading. Install package providing esm.models.esmc."
                ) from exc

            resolved_client = ESMC.from_pretrained(model_name, **kwargs).to(device)

        layer_count = _resolve_layer_count(model_name)
        super().__init__(
            model_reference=model_name,
            preprocessor=EsmcPreprocessor(),
            tokenizer=EsmcTokenizerAdapter(resolved_client),
            model=EsmcModelAdapter(resolved_client, available_layer_count=layer_count),
            postprocessor=EsmcPostprocessor(),
        )
        self.model_metadata = ModelMetadata(
            provider="esm-sdk",
            model_name=model_name,
            model_reference=model_name,
            tokenizer_name="esm.sdk.api.ESMProtein",
            device=str(device),
            framework_versions=_framework_versions(),
            parameters={
                "mode": "protein_to_embedding_only",
                "representation": "per-residue",
                "pooling": "none",
                "layer_indexing": "hf_native_0_is_first_hidden",
                "use_flash_attention": use_flash_attention,
                "available_layer_count_hint": layer_count,
            },
        )

    def generate(
        self,
        records: Sequence[GenerationInput],
        *,
        layer_index: int | Sequence[int] | None = None,
        fail_fast: bool = False,
    ) -> GenerationResult:
        result = GenerationResult()

        for index, record in enumerate(records):
            try:
                normalized = validate_generation_input(record, index=index)
                prepared = self.preprocessor.preprocess(normalized.sequence)
                tokenized = self.tokenizer.tokenize(prepared)
                model_output = self.model.infer(tokenized, layer_index=layer_index)

                model_output_map = cast(Dict[str, Any], model_output) if isinstance(model_output, dict) else None
                layers_obj_raw = model_output_map.get("layers") if model_output_map is not None else None
                if not isinstance(layers_obj_raw, dict):
                    raise EmbeddingBackendError("ESM-C model output missing layers dictionary.")
                layers_obj = cast(Dict[int, Any], layers_obj_raw)

                for layer_id, layer_tensor in sorted(layers_obj.items(), key=lambda item: int(item[0])):
                    matrix = _as_matrix(layer_tensor)
                    if not matrix:
                        raise EmbeddingBackendError(f"ESM-C returned empty matrix for layer {layer_id}.")
                    hidden_dim = len(matrix[0]) if matrix else 0
                    for row in matrix:
                        if len(row) != hidden_dim:
                            raise EmbeddingBackendError(
                                f"Inconsistent row length in layer {layer_id}: expected {hidden_dim}."
                            )

                    result.records.append(
                        EmbeddingRecord(
                            id=normalized.id,
                            embedding=matrix,
                            layer_index=int(layer_id),
                            model_reference=self.model_reference,
                            shape=(len(matrix), hidden_dim),
                            metadata=normalized.metadata,
                        )
                    )
            except Exception as exc:
                normalized_exc: EmbeddingGenerationError = normalize_generation_exception(exc)
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
        run_metadata = RunMetadata(
            run_id=str(uuid.uuid4()),
            created_at_utc=utc_now_iso(),
            sequence_count=len(records),
            requested_layers=_normalize_requested_layers(layer_index),
            resolved_layers=resolved_layers,
            failure_count=len(result.errors),
            parameters={
                "model_reference": self.model_reference,
                "mode": "protein_to_embedding_only",
            },
        )
        return GenerationResult(
            records=result.records,
            errors=result.errors,
            skipped=result.skipped,
            model_metadata=self.model_metadata,
            run_metadata=run_metadata,
        )


def _layers_from_embeddings(embeddings: Any) -> Dict[int, Any]:
    # Accept common tensor-like outputs and normalize to {layer_idx: residue_tensor}.
    shape = getattr(embeddings, "shape", None)
    shape_seq = cast(Sequence[Any], shape) if shape is not None else None
    ndim = len(shape_seq) if shape_seq is not None else None

    if ndim == 4:
        # [batch, layers, residues, hidden]
        assert shape_seq is not None
        return {int(i): embeddings[0, i] for i in range(int(shape_seq[1]))}
    if ndim == 3:
        assert shape_seq is not None
        first_dim = int(shape_seq[0])
        # Heuristic: if first dim is 1, assume [batch, residues, hidden] final-layer only.
        if first_dim == 1:
            return {0: embeddings[0]}
        # otherwise assume [layers, residues, hidden]
        return {int(i): embeddings[i] for i in range(first_dim)}
    if ndim == 2:
        # [residues, hidden] final-layer only.
        return {0: embeddings}

    # Fallback for list-like objects.
    tolist = getattr(embeddings, "tolist", None)
    values = tolist() if callable(tolist) else embeddings
    if isinstance(values, list):
        list_values = cast(List[Any], values)
        if list_values and isinstance(list_values[0], list) and list_values[0] and isinstance(list_values[0][0], list):
            return {int(i): list_values[i] for i in range(len(list_values))}
        return {0: list_values}
    raise EmbeddingBackendError("Could not normalize ESM-C embeddings output.")


def _select_layers(
    layers: Dict[int, Any],
    *,
    layer_index: int | Sequence[int] | None,
) -> Dict[int, Any]:
    available = sorted(int(key) for key in layers.keys())
    if not available:
        raise EmbeddingBackendError("No layers available in ESM-C output.")
    if layer_index is None:
        return {idx: layers[idx] for idx in available}

    if isinstance(layer_index, int):
        requested = [layer_index]
    else:
        requested = [int(v) for v in layer_index]
        if not requested:
            raise EmbeddingInputError("layer_index sequence cannot be empty.")

    selected: Dict[int, Any] = {}
    for idx in requested:
        if idx < 0:
            idx = available[-1] + 1 + idx
        if idx not in layers:
            raise EmbeddingInputError(f"Requested layer index {idx} not available in ESM-C output.")
        selected[idx] = layers[idx]
    return {idx: selected[idx] for idx in sorted(selected)}


def _as_matrix(layer_tensor: Any) -> List[List[float]]:
    return as_float_matrix(layer_tensor)


def _resolve_layer_count(model_name: str) -> int | None:
    key = str(model_name).strip()
    if key in ESMC_LAYER_SPECS:
        return int(ESMC_LAYER_SPECS[key])
    return None


def _normalize_requested_layers(layer_index: int | Sequence[int] | None) -> List[int] | None:
    if layer_index is None:
        return None
    if isinstance(layer_index, int):
        return [int(layer_index)]
    return [int(value) for value in layer_index]


def _framework_versions() -> Dict[str, str]:
    versions: Dict[str, str] = {}
    for package_name in ("esm", "torch"):
        try:
            versions[package_name] = importlib_metadata.version(package_name)
        except importlib_metadata.PackageNotFoundError:
            continue
    return versions


__all__ = [
    "ESMC_LAYER_SPECS",
    "EsmcPreprocessor",
    "EsmcTokenizerAdapter",
    "EsmcModelAdapter",
    "EsmcPostprocessor",
    "EsmcEmbeddingGenerator",
]
