"""ESM2-specific embedding adapters and generator (protein -> embedding)."""

from __future__ import annotations

from importlib import metadata as importlib_metadata
import re
import uuid
from typing import Any, Dict, List, Sequence, Tuple

from .embeddings import (
    EmbeddingBackendError,
    EmbeddingDependencyError,
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
    _as_float_vector,
    _normalize_generation_exception,
    _validate_generation_input,
    _validate_sequence,
    utc_now_iso,
)


ESM2_PRETRAINED_LOADERS: Dict[str, str] = {
    "esm2_t33_650m_ur50d": "esm2_t33_650M_UR50D",
    "esm2_t36_3b_ur50d": "esm2_t36_3B_UR50D",
    "esm2_t48_15b_ur50d": "esm2_t48_15B_UR50D",
    "esm2_t30_150m_ur50d": "esm2_t30_150M_UR50D",
    "esm2_t12_35m_ur50d": "esm2_t12_35M_UR50D",
    "esm2_t6_8m_ur50d": "esm2_t6_8M_UR50D",
}


class Esm2Preprocessor(PreprocessorAdapter):
    """Preprocessing for ESM2 protein sequences."""

    def preprocess(self, raw_sequence: str) -> str:
        sequence = str(raw_sequence).strip().upper().replace(" ", "")
        _validate_sequence(sequence, context="ESM2 preprocessing")
        return re.sub(r"[UZOB]", "X", sequence)


class Esm2TokenizerAdapter(TokenizerAdapter):
    """Tokenizer adapter using ESM alphabet batch converter."""

    def __init__(self, batch_converter: Any, padding_idx: int) -> None:
        self.batch_converter = batch_converter
        self.padding_idx = int(padding_idx)

    def tokenize(self, sequence: str) -> Any:
        labels, strs, tokens = self.batch_converter([("query", sequence)])
        _ = labels, strs
        lens = (tokens != self.padding_idx).sum(1)
        return {"tokens": tokens, "lens": lens}


class Esm2ModelAdapter(ModelAdapter):
    """Model adapter for ESM2 models."""

    def __init__(self, model: Any, *, available_layer_count: int | None = None) -> None:
        self.model = model
        self.available_layer_count = available_layer_count

    def infer(self, tokens: Any, *, layer_index: int | Sequence[int] | None = None) -> Any:
        if not isinstance(tokens, dict):
            raise EmbeddingInputError("Esm2ModelAdapter expects tokenized input as a dict.")
        if "tokens" not in tokens or "lens" not in tokens:
            raise EmbeddingInputError("Token dict must contain 'tokens' and 'lens'.")

        try:
            import torch  # type: ignore
        except ModuleNotFoundError as exc:
            raise EmbeddingDependencyError(
                "PyTorch is required for ESM2 inference. Install with: pip install torch"
            ) from exc

        total = self._total_layers()
        requested = _resolve_layer_indices(layer_index, total_layers=total)
        # ESM2 repr_layers uses native indices, where top layer is `num_layers`.
        repr_layers = requested

        with torch.no_grad():
            out = self.model(tokens["tokens"], repr_layers=repr_layers, return_contacts=False)

        reps = out.get("representations")
        if not isinstance(reps, dict):
            raise EmbeddingBackendError("ESM2 output missing 'representations' dictionary.")

        layers: Dict[int, Any] = {}
        tokens_len = int(tokens["lens"][0].item())
        start = 1
        end = max(tokens_len - 1, 1)  # remove BOS and EOS
        for idx in requested:
            tensor = reps.get(idx)
            if tensor is None:
                raise EmbeddingBackendError(f"ESM2 output does not include requested layer {idx}.")
            layers[idx] = tensor[0, start:end]
        return {"layers": layers, "token_span": (start, end)}

    def available_layers(self) -> List[int] | None:
        total = self._total_layers()
        if total < 0:
            return None
        return list(range(total + 1))

    def _total_layers(self) -> int:
        if self.available_layer_count is not None:
            return int(self.available_layer_count)

        num_layers = getattr(self.model, "num_layers", None)
        if isinstance(num_layers, int):
            return int(num_layers)

        args = getattr(self.model, "args", None)
        if args is not None:
            value = getattr(args, "layers", None)
            if isinstance(value, int):
                return int(value)

        raise EmbeddingBackendError("Could not infer ESM2 total layer count from model.")


class Esm2Postprocessor(PostprocessorAdapter):
    """Postprocessing for ESM2 outputs without pooling."""

    def postprocess(self, model_output: Any) -> Sequence[float]:
        if not isinstance(model_output, dict):
            raise EmbeddingBackendError("Esm2Postprocessor expects a dict payload from model adapter.")
        layers_obj = model_output.get("layers")
        if not isinstance(layers_obj, dict):
            raise EmbeddingBackendError("Esm2Postprocessor expects a 'layers' dict in model output.")
        if not layers_obj:
            raise EmbeddingBackendError("Esm2Postprocessor received no layers.")

        first_key = sorted(layers_obj.keys())[0]
        layer_tensor = layers_obj[first_key]
        if hasattr(layer_tensor, "tolist") and callable(layer_tensor.tolist):
            values = layer_tensor.tolist()
            if isinstance(values, list):
                return [[float(col) for col in row] for row in values if isinstance(row, list)]

        rows: List[List[float]] = []
        for row in layer_tensor:
            rows.append(_as_float_vector(row))
        return rows


class Esm2EmbeddingGenerator(EmbeddingGenerator):
    """Concrete embedding generator for ESM2 family models."""

    GENERATOR_CLASS = "esm2"
    GENERATOR_ALIASES = ("esm-2", "ESM")
    DEFAULT_MODEL_NAME = "esm2_t33_650M_UR50D"
    FAMILY_MODELS = sorted(str(value) for value in ESM2_PRETRAINED_LOADERS.keys())

    def __init__(
        self,
        *,
        model_name: str = DEFAULT_MODEL_NAME,
        device: str = "cpu",
        model: Any | None = None,
        alphabet: Any | None = None,
        batch_converter: Any | None = None,
    ) -> None:
        resolved_model = model
        resolved_alphabet = alphabet
        resolved_converter = batch_converter
        resolved_name = str(model_name).strip()
        if not resolved_name:
            raise EmbeddingInputError("ESM2 model_name must be non-empty.")

        if resolved_model is None or resolved_alphabet is None:
            try:
                import esm  # type: ignore
            except ModuleNotFoundError as exc:
                raise EmbeddingDependencyError("ESM package is required for ESM2 loading. Install with: pip install esm") from exc

            loader_name = ESM2_PRETRAINED_LOADERS.get(resolved_name.lower())
            if loader_name is None:
                raise EmbeddingInputError(
                    f"Unsupported ESM2 model name: {model_name!r}. "
                    f"Supported: {sorted(ESM2_PRETRAINED_LOADERS.keys())}"
                )
            loader = getattr(esm.pretrained, loader_name, None)
            if loader is None or not callable(loader):
                raise EmbeddingDependencyError(f"ESM pretrained loader not available: {loader_name}")
            resolved_model, resolved_alphabet = loader()

        if resolved_converter is None:
            get_converter = getattr(resolved_alphabet, "get_batch_converter", None)
            if get_converter is None or not callable(get_converter):
                raise EmbeddingBackendError("ESM2 alphabet does not expose get_batch_converter().")
            resolved_converter = get_converter()

        to_fn = getattr(resolved_model, "to", None)
        if callable(to_fn):
            to_fn(device)
        eval_fn = getattr(resolved_model, "eval", None)
        if callable(eval_fn):
            eval_fn()

        padding_idx = getattr(resolved_alphabet, "padding_idx", None)
        if padding_idx is None:
            raise EmbeddingBackendError("ESM2 alphabet missing padding_idx.")

        available_count = _resolve_esm2_layer_count(resolved_model)
        super().__init__(
            model_reference=resolved_name,
            preprocessor=Esm2Preprocessor(),
            tokenizer=Esm2TokenizerAdapter(resolved_converter, padding_idx=int(padding_idx)),
            model=Esm2ModelAdapter(resolved_model, available_layer_count=available_count),
            postprocessor=Esm2Postprocessor(),
        )
        self.model_metadata = ModelMetadata(
            provider="esm-pretrained",
            model_name=resolved_name,
            model_reference=resolved_name,
            device=str(device),
            framework_versions=_framework_versions(),
            parameters={
                "mode": "protein_to_embedding_only",
                "representation": "per-residue",
                "pooling": "none",
                "layer_indexing": "esm2_native_0_is_embedding_and_top_is_num_layers",
                "available_layer_count_hint": available_count + 1 if available_count >= 0 else None,
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
                normalized = _validate_generation_input(record, index=index)
                prepared = self.preprocessor.preprocess(normalized.sequence)
                tokenized = self.tokenizer.tokenize(prepared)
                model_output = self.model.infer(tokenized, layer_index=layer_index)

                layers_obj = model_output.get("layers") if isinstance(model_output, dict) else None
                if not isinstance(layers_obj, dict):
                    raise EmbeddingBackendError("ESM2 model output missing layers dictionary.")

                for layer_id, layer_tensor in sorted(layers_obj.items(), key=lambda item: int(item[0])):
                    matrix = _as_matrix(layer_tensor)
                    if not matrix:
                        raise EmbeddingBackendError(f"ESM2 returned empty matrix for layer {layer_id}.")
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


def _resolve_layer_indices(layer_index: int | Sequence[int] | None, *, total_layers: int) -> List[int]:
    if total_layers < 0:
        raise EmbeddingBackendError("ESM2 reported invalid total layer count.")

    if layer_index is None:
        return list(range(total_layers + 1))

    if isinstance(layer_index, int):
        requested = [layer_index]
    else:
        requested = [int(v) for v in layer_index]
        if not requested:
            raise EmbeddingInputError("layer_index sequence cannot be empty.")

    resolved: List[int] = []
    for idx in requested:
        if idx < 0:
            idx = (total_layers + 1) + idx
        if idx < 0 or idx > total_layers:
            raise EmbeddingInputError(
                f"Requested layer index {idx} out of range for ESM2 total layers={total_layers}."
            )
        resolved.append(idx)
    return sorted(set(resolved))


def _as_matrix(layer_tensor: Any) -> List[List[float]]:
    if hasattr(layer_tensor, "tolist") and callable(layer_tensor.tolist):
        as_list = layer_tensor.tolist()
    else:
        as_list = layer_tensor
    if not isinstance(as_list, list):
        raise EmbeddingBackendError("Layer tensor could not be converted to a row-major list.")
    return [_as_float_vector(row) for row in as_list]


def _resolve_esm2_layer_count(model: Any) -> int:
    num_layers = getattr(model, "num_layers", None)
    if isinstance(num_layers, int):
        return int(num_layers)
    args = getattr(model, "args", None)
    if args is not None:
        value = getattr(args, "layers", None)
        if isinstance(value, int):
            return int(value)
    raise EmbeddingBackendError("Could not infer ESM2 num_layers from model.")


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
    "ESM2_PRETRAINED_LOADERS",
    "Esm2Preprocessor",
    "Esm2TokenizerAdapter",
    "Esm2ModelAdapter",
    "Esm2Postprocessor",
    "Esm2EmbeddingGenerator",
]
