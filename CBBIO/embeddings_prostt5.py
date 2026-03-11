"""ProstT5-specific embedding adapters and generator (protein -> embedding only)."""

from __future__ import annotations

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


class ProstT5Preprocessor(PreprocessorAdapter):
    """Preprocessing for ProstT5 protein inputs.

    This adapter is intentionally limited to protein sequences (AA -> embedding),
    not 3Di inputs.
    """

    PREFIX = "<AA2fold>"

    def preprocess(self, raw_sequence: str) -> str:
        sequence = str(raw_sequence).strip().upper()
        _validate_sequence(sequence, context="ProstT5 preprocessing (protein mode)")
        replaced = re.sub(r"[UZOB]", "X", sequence)
        return f"{self.PREFIX} " + " ".join(list(replaced))


class ProstT5TokenizerAdapter(TokenizerAdapter):
    """Tokenizer adapter for ProstT5 tokenizers."""

    def __init__(self, tokenizer: Any, *, device: str = "cpu") -> None:
        self.tokenizer = tokenizer
        self.device = str(device)

    def tokenize(self, sequence: str) -> Any:
        try:
            import torch  # type: ignore
        except ModuleNotFoundError as exc:
            raise EmbeddingDependencyError(
                "PyTorch is required for ProstT5 tokenization. Install with: pip install torch"
            ) from exc

        if callable(self.tokenizer):
            encoded = self.tokenizer(
                [sequence],
                add_special_tokens=True,
                padding="longest",
                return_tensors="pt",
            )
            input_ids = encoded["input_ids"].to(self.device)
            attention_mask = encoded["attention_mask"].to(self.device)
            return {"input_ids": input_ids, "attention_mask": attention_mask}

        batch_encode_plus = getattr(self.tokenizer, "batch_encode_plus", None)
        if callable(batch_encode_plus):
            ids = batch_encode_plus(
                [sequence],
                add_special_tokens=True,
                padding="longest",
                return_tensors="pt",
            )
            input_ids = ids["input_ids"].to(self.device)
            attention_mask = ids["attention_mask"].to(self.device)
            return {"input_ids": input_ids, "attention_mask": attention_mask}

        raise EmbeddingBackendError("Tokenizer does not support call(...) or batch_encode_plus(...).")


class ProstT5ModelAdapter(ModelAdapter):
    """Model adapter for ProstT5 encoder models."""

    def __init__(self, model: Any, *, device: str = "cpu") -> None:
        self.model = model
        self.device = str(device)
        to_fn = getattr(self.model, "to", None)
        if callable(to_fn):
            to_fn(self.device)

        # ProstT5 recommendation: float on CPU, half on GPU.
        if self.device == "cpu":
            float_fn = getattr(self.model, "float", None)
            if callable(float_fn):
                float_fn()
        else:
            half_fn = getattr(self.model, "half", None)
            if callable(half_fn):
                half_fn()

        eval_fn = getattr(self.model, "eval", None)
        if callable(eval_fn):
            eval_fn()

    def infer(self, tokens: Any, *, layer_index: int | Sequence[int] | None = None) -> Any:
        if not isinstance(tokens, dict):
            raise EmbeddingInputError("ProstT5ModelAdapter expects tokenized input as a dict.")
        if "input_ids" not in tokens or "attention_mask" not in tokens:
            raise EmbeddingInputError("Token dict must contain 'input_ids' and 'attention_mask'.")

        try:
            import torch  # type: ignore
        except ModuleNotFoundError as exc:
            raise EmbeddingDependencyError(
                "PyTorch is required for ProstT5 inference. Install with: pip install torch"
            ) from exc

        with torch.no_grad():
            model_output = self.model(
                input_ids=tokens["input_ids"],
                attention_mask=tokens["attention_mask"],
                output_hidden_states=True,
                return_dict=True,
            )

        hidden_states = getattr(model_output, "hidden_states", None)
        if hidden_states is None:
            raise EmbeddingBackendError("ProstT5 model output did not include hidden_states.")

        total_layers = len(hidden_states)
        layer_indices = _resolve_layer_indices(layer_index, total_layers=total_layers)
        start, end = _protein_token_span_from_mask(tokens["attention_mask"])

        selected_layers: Dict[int, Any] = {}
        # BioData convention: layer 0 means last hidden layer.
        for user_idx, hf_idx in layer_indices:
            selected_layers[user_idx] = hidden_states[hf_idx][0, start:end]
        return {"layers": selected_layers, "token_span": (start, end)}

    def available_layers(self) -> List[int] | None:
        total_layers = _infer_total_layers_from_model(self.model)
        if total_layers is None:
            return None
        return list(range(total_layers))


class ProstT5Postprocessor(PostprocessorAdapter):
    """Postprocessing for ProstT5 outputs without pooling."""

    def postprocess(self, model_output: Any) -> Sequence[float]:
        if not isinstance(model_output, dict):
            raise EmbeddingBackendError("ProstT5Postprocessor expects a dict payload from model adapter.")
        layers_obj = model_output.get("layers")
        if not isinstance(layers_obj, dict):
            raise EmbeddingBackendError("ProstT5Postprocessor expects a 'layers' dict in model output.")
        if not layers_obj:
            raise EmbeddingBackendError("ProstT5Postprocessor received no layers.")

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


class ProstT5EmbeddingGenerator(EmbeddingGenerator):
    """Concrete embedding generator for ProstT5 (protein mode only)."""

    GENERATOR_CLASS = "prostT5"
    GENERATOR_ALIASES = ("prostt5", "prost_t5", "prost-t5")
    DEFAULT_MODEL_NAME = "Rostlab/ProstT5"
    FAMILY_MODELS = [DEFAULT_MODEL_NAME]

    def __init__(
        self,
        *,
        model_name: str = DEFAULT_MODEL_NAME,
        device: str = "cpu",
        tokenizer: Any | None = None,
        model: Any | None = None,
    ) -> None:
        resolved_tokenizer = tokenizer
        resolved_model = model

        if resolved_tokenizer is None or resolved_model is None:
            try:
                from transformers import AutoConfig, AutoTokenizer, T5EncoderModel  # type: ignore
            except ModuleNotFoundError as exc:
                raise EmbeddingDependencyError(
                    "transformers is required for ProstT5 loading. Install with: pip install transformers"
                ) from exc

            if resolved_tokenizer is None:
                resolved_tokenizer = AutoTokenizer.from_pretrained(model_name, do_lower_case=False)
            if resolved_model is None:
                config = AutoConfig.from_pretrained(model_name)
                resolved_model = T5EncoderModel.from_pretrained(model_name, config=config)

        super().__init__(
            model_reference=model_name,
            preprocessor=ProstT5Preprocessor(),
            tokenizer=ProstT5TokenizerAdapter(resolved_tokenizer, device=device),
            model=ProstT5ModelAdapter(resolved_model, device=device),
            postprocessor=ProstT5Postprocessor(),
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
                "prefix": "<AA2fold>",
                "representation": "per-residue",
                "pooling": "none",
                "precision_policy": "float_on_cpu_half_on_gpu",
                "layer_indexing": "biodata_reversed_0_is_last_hidden",
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
                    raise EmbeddingBackendError("ProstT5 model output missing layers dictionary.")

                for layer_id, layer_tensor in sorted(layers_obj.items(), key=lambda item: int(item[0])):
                    matrix = _as_matrix(layer_tensor)
                    if not matrix:
                        raise EmbeddingBackendError(f"ProstT5 returned empty matrix for layer {layer_id}.")
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
            parameters={"model_reference": self.model_reference, "mode": "protein_to_embedding_only"},
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
) -> List[tuple[int, int]]:
    if total_layers < 1:
        raise EmbeddingBackendError("ProstT5 output reported zero layers.")

    if layer_index is None:
        return [(idx, _to_hf_layer_index(idx, total_layers=total_layers)) for idx in range(total_layers)]

    if isinstance(layer_index, int):
        indices = [layer_index]
    else:
        indices = [int(value) for value in layer_index]
        if not indices:
            raise EmbeddingInputError("layer_index sequence cannot be empty.")

    resolved_user: List[int] = []
    for idx in indices:
        if idx < 0:
            idx = total_layers + idx
        if idx < 0 or idx >= total_layers:
            raise EmbeddingInputError(
                f"Requested layer index {idx} out of range for total layers={total_layers}."
            )
        resolved_user.append(idx)

    unique_sorted_user = sorted(set(resolved_user))
    return [(user_idx, _to_hf_layer_index(user_idx, total_layers=total_layers)) for user_idx in unique_sorted_user]


def _to_hf_layer_index(user_layer_index: int, *, total_layers: int) -> int:
    return (total_layers - 1) - int(user_layer_index)


def _protein_token_span_from_mask(attention_mask: Any) -> tuple[int, int]:
    try:
        valid_len = int(attention_mask[0].sum().item())
    except Exception as exc:
        raise EmbeddingBackendError("Could not infer valid token length from attention_mask.") from exc
    if valid_len < 3:
        raise EmbeddingBackendError(
            "Expected at least prefix token + residue token + end token in ProstT5 input."
        )
    # token 0 is task prefix (<AA2fold>), final valid token is end token.
    return 1, valid_len - 1


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
    "ProstT5Preprocessor",
    "ProstT5TokenizerAdapter",
    "ProstT5ModelAdapter",
    "ProstT5Postprocessor",
    "ProstT5EmbeddingGenerator",
]
