"""ANKH3-specific embedding adapters and generator (protein -> embedding)."""

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


class Ankh3Preprocessor(PreprocessorAdapter):
    """Preprocessing for ANKH3 protein inputs."""

    ALLOWED_PREFIXES = {"[NLU]", "[S2S]"}

    def __init__(self, *, prefix: str = "[NLU]") -> None:
        value = str(prefix).strip()
        if value not in self.ALLOWED_PREFIXES:
            raise EmbeddingInputError(
                f"Unsupported ANKH3 prefix: {prefix!r}. Use one of: [NLU], [S2S]."
            )
        self.prefix = value

    def preprocess(self, raw_sequence: str) -> str:
        sequence = str(raw_sequence).strip().upper()
        validate_sequence(sequence, context="ANKH3 preprocessing")
        replaced = re.sub(r"[UZOB]", "X", sequence)
        return f"{self.prefix}{replaced}"


class Ankh3TokenizerAdapter(TokenizerAdapter):
    """Tokenizer adapter for ANKH3 tokenizers."""

    def __init__(self, tokenizer: Any, *, device: str = "cpu") -> None:
        self.tokenizer = tokenizer
        self.device = str(device)

    def tokenize(self, sequence: str) -> Any:
        encoded = self.tokenizer(
            sequence,
            add_special_tokens=True,
            return_tensors="pt",
            is_split_into_words=False,
        )
        encoded_map = cast(Dict[str, Any], encoded)
        input_ids = encoded_map["input_ids"].to(self.device)
        attention_mask = encoded_map["attention_mask"].to(self.device)
        return {"input_ids": input_ids, "attention_mask": attention_mask}


class Ankh3ModelAdapter(ModelAdapter):
    """Model adapter for ANKH3 encoder models."""

    def __init__(self, model: Any, *, device: str = "cpu") -> None:
        self.model = model
        self.device = str(device)
        to_fn = getattr(self.model, "to", None)
        if callable(to_fn):
            to_fn(self.device)
        eval_fn = getattr(self.model, "eval", None)
        if callable(eval_fn):
            eval_fn()

    def infer(self, tokens: Any, *, layer_index: int | Sequence[int] | None = None) -> Any:
        if not isinstance(tokens, dict):
            raise EmbeddingInputError("Ankh3ModelAdapter expects tokenized input as a dict.")
        if "input_ids" not in tokens or "attention_mask" not in tokens:
            raise EmbeddingInputError("Token dict must contain 'input_ids' and 'attention_mask'.")
        token_map = cast(Dict[str, Any], tokens)

        try:
            import torch  # type: ignore
        except ModuleNotFoundError as exc:
            raise EmbeddingDependencyError(
                "PyTorch is required for ANKH3 inference. Install with: pip install torch"
            ) from exc

        with torch.no_grad():
            model_output = self.model(
                input_ids=token_map["input_ids"],
                attention_mask=token_map["attention_mask"],
                output_hidden_states=True,
                return_dict=True,
            )

        hidden_states = getattr(model_output, "hidden_states", None)
        if hidden_states is None:
            raise EmbeddingBackendError("ANKH3 model output did not include hidden_states.")

        total_layers = len(hidden_states)
        layer_indices = _resolve_layer_indices(layer_index, total_layers=total_layers)
        start, end = _protein_token_span_from_mask(token_map["attention_mask"])

        selected_layers: Dict[int, Any] = {}
        for idx in layer_indices:
            selected_layers[idx] = hidden_states[idx][0, start:end]
        return {"layers": selected_layers, "token_span": (start, end)}

    def available_layers(self) -> List[int] | None:
        total_layers = _infer_total_layers_from_model(self.model)
        if total_layers is None:
            return None
        return list(range(total_layers))


class Ankh3Postprocessor(PostprocessorAdapter):
    """Postprocessing for ANKH3 outputs without pooling."""

    def postprocess(self, model_output: Any) -> EmbeddingPayload:
        if not isinstance(model_output, dict):
            raise EmbeddingBackendError("Ankh3Postprocessor expects a dict payload from model adapter.")
        model_output_map = cast(Dict[str, Any], model_output)
        layers_obj_raw = model_output_map.get("layers")
        if not isinstance(layers_obj_raw, dict):
            raise EmbeddingBackendError("Ankh3Postprocessor expects a 'layers' dict in model output.")
        layers_obj = cast(Dict[int, Any], layers_obj_raw)
        if not layers_obj:
            raise EmbeddingBackendError("Ankh3Postprocessor received no layers.")

        first_key = sorted(layers_obj.keys())[0]
        layer_tensor = layers_obj[first_key]
        return as_float_matrix(layer_tensor)


class Ankh3EmbeddingGenerator(EmbeddingGenerator):
    """Concrete embedding generator for ANKH3 models."""

    GENERATOR_CLASS = "ankh3"
    GENERATOR_ALIASES = ("Ankh3-Large", "ankh3_large")
    DEFAULT_MODEL_NAME = "ElnaggarLab/ankh3-large"
    # Published model family members from ANKH/ANKH3 plus the HF checkpoint used by default.
    FAMILY_MODELS = [
        "Ankh Large",
        "Ankh Base",
        "Ankh3 Large",
        "Ankh3 XL",
        DEFAULT_MODEL_NAME,
    ]

    def __init__(
        self,
        *,
        model_name: str = DEFAULT_MODEL_NAME,
        prefix: str = "[NLU]",
        device: str = "cpu",
        tokenizer: Any | None = None,
        model: Any | None = None,
    ) -> None:
        resolved_tokenizer: Any | None = tokenizer
        resolved_model: Any | None = model

        if resolved_tokenizer is None or resolved_model is None:
            try:
                from transformers import AutoConfig, T5EncoderModel, T5Tokenizer  # type: ignore
            except ModuleNotFoundError as exc:
                raise EmbeddingDependencyError(
                    "transformers is required for ANKH3 loading. Install with: pip install transformers"
                ) from exc

            if resolved_tokenizer is None:
                resolved_tokenizer = cast(Any, T5Tokenizer).from_pretrained(model_name, do_lower_case=False)
            if resolved_model is None:
                config = cast(Any, AutoConfig).from_pretrained(model_name)
                setattr(config, "tie_word_embeddings", False)
                resolved_model = cast(Any, T5EncoderModel).from_pretrained(model_name, config=config)

        super().__init__(
            model_reference=model_name,
            preprocessor=Ankh3Preprocessor(prefix=prefix),
            tokenizer=Ankh3TokenizerAdapter(resolved_tokenizer, device=device),
            model=Ankh3ModelAdapter(resolved_model, device=device),
            postprocessor=Ankh3Postprocessor(),
        )
        self.model_metadata = ModelMetadata(
            provider="huggingface-transformers",
            model_name=model_name,
            model_reference=model_name,
            model_revision=_extract_revision(resolved_model),
            tokenizer_name=_extract_name_or_path(resolved_tokenizer),
            tokenizer_revision=_extract_revision(resolved_tokenizer),
            device=str(device),
            framework_versions=_framework_versions(),
            parameters={
                "mode": "protein_to_embedding_only",
                "prefix": str(prefix),
                "representation": "per-residue",
                "pooling": "none",
                "layer_indexing": "hf_native_0_is_first_hidden",
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
                    raise EmbeddingBackendError("ANKH3 model output missing layers dictionary.")
                layers_obj = cast(Dict[int, Any], layers_obj_raw)

                for layer_id, layer_tensor in sorted(layers_obj.items(), key=lambda item: int(item[0])):
                    matrix = _as_matrix(layer_tensor)
                    if not matrix:
                        raise EmbeddingBackendError(f"ANKH3 returned empty matrix for layer {layer_id}.")
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
                "prefix": cast(Ankh3Preprocessor, self.preprocessor).prefix,
            },
        )
        return GenerationResult(
            records=result.records,
            errors=result.errors,
            skipped=result.skipped,
            model_metadata=self.model_metadata,
            run_metadata=run_metadata,
        )


def _resolve_layer_indices(
    layer_index: int | Sequence[int] | None,
    *,
    total_layers: int,
) -> List[int]:
    if total_layers < 1:
        raise EmbeddingBackendError("ANKH3 output reported zero layers.")

    if layer_index is None:
        return list(range(total_layers))

    if isinstance(layer_index, int):
        indices = [layer_index]
    else:
        indices = [int(value) for value in layer_index]
        if not indices:
            raise EmbeddingInputError("layer_index sequence cannot be empty.")

    resolved: List[int] = []
    for idx in indices:
        if idx < 0:
            idx = total_layers + idx
        if idx < 0 or idx >= total_layers:
            raise EmbeddingInputError(
                f"Requested layer index {idx} out of range for total layers={total_layers}."
            )
        resolved.append(idx)
    return sorted(set(resolved))


def _protein_token_span_from_mask(attention_mask: Any) -> tuple[int, int]:
    try:
        valid_len = int(attention_mask[0].sum().item())
    except Exception as exc:
        raise EmbeddingBackendError("Could not infer valid token length from attention_mask.") from exc
    if valid_len < 3:
        raise EmbeddingBackendError("Expected at least one residue plus special tokens in ANKH3 input.")
    # Keep residues and remove start/end special tokens.
    return 1, valid_len - 1


def _as_matrix(layer_tensor: Any) -> List[List[float]]:
    return as_float_matrix(layer_tensor)


def _normalize_requested_layers(layer_index: int | Sequence[int] | None) -> List[int] | None:
    if layer_index is None:
        return None
    if isinstance(layer_index, int):
        return [int(layer_index)]
    return [int(value) for value in layer_index]


def _framework_versions() -> Dict[str, str]:
    versions: Dict[str, str] = {}
    for package_name in ("transformers", "torch"):
        try:
            versions[package_name] = importlib_metadata.version(package_name)
        except importlib_metadata.PackageNotFoundError:
            continue
    return versions


def _extract_name_or_path(obj: Any) -> str | None:
    value = getattr(obj, "name_or_path", None)
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _extract_revision(obj: Any) -> str | None:
    config = getattr(obj, "config", None)
    if config is None:
        return None
    revision = getattr(config, "_commit_hash", None)
    if revision is None:
        return None
    text = str(revision).strip()
    return text or None


def _infer_total_layers_from_model(model: Any) -> int | None:
    config = getattr(model, "config", None)
    if config is None:
        return None

    num_layers = getattr(config, "num_layers", None)
    if isinstance(num_layers, int) and num_layers >= 1:
        return int(num_layers) + 1

    num_hidden = getattr(config, "num_hidden_layers", None)
    if isinstance(num_hidden, int) and num_hidden >= 1:
        return int(num_hidden) + 1
    return None


__all__ = [
    "Ankh3Preprocessor",
    "Ankh3TokenizerAdapter",
    "Ankh3ModelAdapter",
    "Ankh3Postprocessor",
    "Ankh3EmbeddingGenerator",
]
