"""ESM1b-specific embedding adapters and generator (protein -> embedding)."""

from __future__ import annotations

import importlib
from importlib import metadata as importlib_metadata
import re
import uuid
from typing import Any, Dict, List, Sequence

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


class Esm1bPreprocessor(PreprocessorAdapter):
    """Preprocessing for ESM1b protein sequences."""

    def preprocess(self, raw_sequence: str) -> str:
        sequence = str(raw_sequence).strip().upper().replace(" ", "")
        _validate_sequence(sequence, context="ESM1b preprocessing")
        return re.sub(r"[UZOB]", "X", sequence)


class Esm1bTokenizerAdapter(TokenizerAdapter):
    """Tokenizer adapter using ESM alphabet batch converter."""

    def __init__(
        self,
        batch_converter: Any | None = None,
        padding_idx: int | None = None,
        *,
        tokenizer: Any | None = None,
        device: str = "cpu",
    ) -> None:
        self.batch_converter = batch_converter
        self.padding_idx = int(padding_idx) if padding_idx is not None else None
        self.tokenizer = tokenizer
        self.device = str(device)

    def tokenize(self, sequence: str) -> Any:
        if self.tokenizer is not None:
            encoded = self.tokenizer(sequence, add_special_tokens=True, return_tensors="pt")
            return {
                "input_ids": encoded["input_ids"].to(self.device),
                "attention_mask": encoded["attention_mask"].to(self.device),
            }

        if self.batch_converter is None or self.padding_idx is None:
            raise EmbeddingBackendError("ESM1b tokenizer adapter is missing batch converter configuration.")
        _, _, tokens = self.batch_converter([("query", sequence)])
        lens = (tokens != self.padding_idx).sum(1)
        return {"tokens": tokens, "lens": lens}


class Esm1bModelAdapter(ModelAdapter):
    """Model adapter for ESM1b model."""

    def __init__(self, model: Any, *, available_layer_count: int | None = 33) -> None:
        self.model = model
        self.available_layer_count = available_layer_count

    def infer(self, tokens: Any, *, layer_index: int | Sequence[int] | None = None) -> Any:
        if not isinstance(tokens, dict):
            raise EmbeddingInputError("Esm1bModelAdapter expects tokenized input as a dict.")

        try:
            import torch  # type: ignore
        except ModuleNotFoundError as exc:
            raise EmbeddingDependencyError(
                "PyTorch is required for ESM1b inference. Install with: pip install torch"
            ) from exc

        total = self._total_layers()
        requested = _resolve_layer_indices(layer_index, total_layers=total)
        if "tokens" in tokens and "lens" in tokens:
            with torch.no_grad():
                out = self.model(tokens["tokens"], repr_layers=requested, return_contacts=False)

            reps = out.get("representations")
            if not isinstance(reps, dict):
                raise EmbeddingBackendError("ESM1b output missing 'representations' dictionary.")

            layers: Dict[int, Any] = {}
            tokens_len = int(tokens["lens"][0].item())
            start = 1
            end = max(tokens_len - 1, 1)  # remove BOS/EOS
            for idx in requested:
                tensor = reps.get(idx)
                if tensor is None:
                    raise EmbeddingBackendError(f"ESM1b output does not include requested layer {idx}.")
                layers[idx] = tensor[0, start:end]
            return {"layers": layers, "token_span": (start, end)}

        if "input_ids" in tokens and "attention_mask" in tokens:
            with torch.no_grad():
                out = self.model(
                    input_ids=tokens["input_ids"],
                    attention_mask=tokens["attention_mask"],
                    output_hidden_states=True,
                    return_dict=True,
                )
            hidden_states = getattr(out, "hidden_states", None)
            if hidden_states is None:
                raise EmbeddingBackendError("ESM1b HF output missing hidden_states.")
            valid_len = int(tokens["attention_mask"][0].sum().item())
            start = 1
            end = max(valid_len - 1, 1)
            return {"layers": {idx: hidden_states[idx][0, start:end] for idx in requested}, "token_span": (start, end)}

        raise EmbeddingInputError("Token dict must contain either 'tokens'/'lens' or 'input_ids'/'attention_mask'.")

    def available_layers(self) -> List[int] | None:
        total = self._total_layers()
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
        raise EmbeddingBackendError("Could not infer ESM1b num_layers from model.")


class Esm1bPostprocessor(PostprocessorAdapter):
    """Postprocessing for ESM1b outputs without pooling."""

    def postprocess(self, model_output: Any) -> Sequence[float]:
        if not isinstance(model_output, dict):
            raise EmbeddingBackendError("Esm1bPostprocessor expects a dict payload from model adapter.")
        layers_obj = model_output.get("layers")
        if not isinstance(layers_obj, dict):
            raise EmbeddingBackendError("Esm1bPostprocessor expects a 'layers' dict in model output.")
        if not layers_obj:
            raise EmbeddingBackendError("Esm1bPostprocessor received no layers.")

        first_key = sorted(layers_obj.keys())[0]
        layer_tensor = layers_obj[first_key]
        if hasattr(layer_tensor, "tolist") and callable(layer_tensor.tolist):
            values = layer_tensor.tolist()
            if isinstance(values, list):
                return [[float(col) for col in row] for row in values if isinstance(row, list)]

        return [_as_float_vector(row) for row in layer_tensor]


class Esm1bEmbeddingGenerator(EmbeddingGenerator):
    """Concrete embedding generator for ESM1b."""

    GENERATOR_CLASS = "esm1b"
    GENERATOR_ALIASES = ("esm-1b",)
    DEFAULT_MODEL_NAME = "esm1b_t33_650M_UR50S"
    FAMILY_MODELS = [DEFAULT_MODEL_NAME]

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

        if resolved_model is None or resolved_alphabet is None:
            try:
                import esm  # type: ignore
            except ModuleNotFoundError as exc:
                raise EmbeddingDependencyError("ESM package is required for ESM1b loading. Install with: pip install esm") from exc

            pretrained_module = None
            try:
                pretrained_module = importlib.import_module("esm.pretrained")
            except Exception:
                pretrained_module = getattr(esm, "pretrained", None)
            loader = getattr(pretrained_module, "esm1b_t33_650M_UR50S", None) if pretrained_module is not None else None
            if loader is not None and callable(loader):
                resolved_model, resolved_alphabet = loader()
            else:
                try:
                    from transformers import AutoModel, EsmTokenizer  # type: ignore
                except ModuleNotFoundError as exc:
                    raise EmbeddingDependencyError(
                        "Neither esm.pretrained nor transformers fallback is available for ESM1b loading."
                    ) from exc
                hf_name = _resolve_esm1b_hf_model_name(model_name)
                resolved_model = AutoModel.from_pretrained(hf_name)
                resolved_converter = EsmTokenizer.from_pretrained(hf_name)

        if resolved_converter is None:
            if resolved_alphabet is not None:
                get_converter = getattr(resolved_alphabet, "get_batch_converter", None)
                if get_converter is None or not callable(get_converter):
                    raise EmbeddingBackendError("ESM1b alphabet does not expose get_batch_converter().")
                resolved_converter = get_converter()

        to_fn = getattr(resolved_model, "to", None)
        if callable(to_fn):
            to_fn(device)
        eval_fn = getattr(resolved_model, "eval", None)
        if callable(eval_fn):
            eval_fn()

        super().__init__(
            model_reference=model_name,
            preprocessor=Esm1bPreprocessor(),
            tokenizer=Esm1bTokenizerAdapter(
                resolved_converter if resolved_alphabet is not None else None,
                int(getattr(resolved_alphabet, "padding_idx", None)) if resolved_alphabet is not None else None,
                tokenizer=resolved_converter if resolved_alphabet is None else None,
                device=device,
            ),
            model=Esm1bModelAdapter(resolved_model, available_layer_count=33),
            postprocessor=Esm1bPostprocessor(),
        )
        self.model_metadata = ModelMetadata(
            provider="esm-pretrained",
            model_name=model_name,
            model_reference=model_name,
            device=str(device),
            framework_versions=_framework_versions(),
            parameters={
                "mode": "protein_to_embedding_only",
                "representation": "per-residue",
                "pooling": "none",
                "layer_indexing": "esm1b_native_0_is_embedding_and_top_is_33",
                "available_layer_count_hint": 34,
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
                    raise EmbeddingBackendError("ESM1b model output missing layers dictionary.")

                for layer_id, layer_tensor in sorted(layers_obj.items(), key=lambda item: int(item[0])):
                    matrix = _as_matrix(layer_tensor)
                    hidden_dim = len(matrix[0]) if matrix else 0
                    if not matrix:
                        raise EmbeddingBackendError(f"ESM1b returned empty matrix for layer {layer_id}.")
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
            parameters={"model_reference": self.model_reference, "mode": "protein_to_embedding_only"},
        )
        return GenerationResult(
            records=result.records,
            errors=result.errors,
            skipped=result.skipped,
            model_metadata=self.model_metadata,
            run_metadata=run_metadata,
        )


def _resolve_layer_indices(layer_index: int | Sequence[int] | None, *, total_layers: int) -> List[int]:
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
            raise EmbeddingInputError(f"Requested layer index {idx} out of range for ESM1b total layers={total_layers}.")
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


def _resolve_esm1b_hf_model_name(model_name: str) -> str:
    normalized = str(model_name).strip().lower()
    if normalized.startswith("facebook/"):
        return str(model_name).strip()
    if normalized == "esm1b_t33_650m_ur50s":
        return "facebook/esm-1b"
    raise EmbeddingInputError(f"Unsupported ESM1b Hugging Face fallback model name: {model_name!r}.")


__all__ = [
    "Esm1bPreprocessor",
    "Esm1bTokenizerAdapter",
    "Esm1bModelAdapter",
    "Esm1bPostprocessor",
    "Esm1bEmbeddingGenerator",
]
