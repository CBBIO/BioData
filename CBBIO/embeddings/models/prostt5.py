"""ProstT5-specific embedding adapters and generator (protein -> embedding only)."""

from __future__ import annotations

from typing import Any, Dict, List, Sequence, cast

from .. import (
    EmbeddingBackendError,
    EmbeddingDependencyError,
    EmbeddingGenerator,
    EmbeddingInputError,
    GenerationInput,
    GenerationResult,
    ModelMetadata,
    ModelAdapter,
    TokenizerAdapter,
)
from ..utils.pooler import PoolerInput
from ..utils.torch import (
    BasePreprocessor,
    DefaultPostprocessor,
    extract_name_or_path,
    extract_revision,
    framework_versions,
    infer_total_layers_from_model,
    move_model_to_device,
    normalize_requested_layers,
    normalize_torch_dtype_name,
    resolve_torch_dtype,
)


class ProstT5Preprocessor(BasePreprocessor):
    """Preprocessing for ProstT5 protein inputs.

    Intentionally limited to protein sequences (AA -> embedding), not 3Di inputs.
    """

    PREFIX = "<AA2fold>"

    def __init__(self) -> None:
        super().__init__(
            context="ProstT5 preprocessing (protein mode)",
            spacing=True,
            prefix=self.PREFIX,
            prefix_space=True,
        )


class ProstT5TokenizerAdapter(TokenizerAdapter):
    """Tokenizer adapter for ProstT5 tokenizers."""

    def __init__(self, tokenizer: Any, *, device: str = "cpu") -> None:
        self.tokenizer = tokenizer
        self.device = str(device)

    def tokenize(self, sequence: str) -> Any:
        """Tokenize one sequence for ProstT5."""
        return self.tokenize_many([sequence])

    def tokenize_many(self, sequences: Sequence[str]) -> Any:
        """Tokenize a batch of sequences for ProstT5."""
        try:
            import torch  # type: ignore
        except ModuleNotFoundError as exc:
            raise EmbeddingDependencyError(
                "PyTorch is required for ProstT5 tokenization. Install with: pip install torch"
            ) from exc
        if not sequences:
            raise EmbeddingInputError("ProstT5 tokenization requires at least one sequence.")

        if callable(self.tokenizer):
            encoded = self.tokenizer(
                list(sequences),
                add_special_tokens=True,
                padding="longest",
                return_tensors="pt",
            )
            encoded_map = cast(Dict[str, Any], encoded)
            input_ids = encoded_map["input_ids"].to(self.device)
            attention_mask = encoded_map["attention_mask"].to(self.device)
            return {"input_ids": input_ids, "attention_mask": attention_mask}

        batch_encode_plus = getattr(self.tokenizer, "batch_encode_plus", None)
        if callable(batch_encode_plus):
            ids = batch_encode_plus(
                list(sequences),
                add_special_tokens=True,
                padding="longest",
                return_tensors="pt",
            )
            ids_map = cast(Dict[str, Any], ids)
            input_ids = ids_map["input_ids"].to(self.device)
            attention_mask = ids_map["attention_mask"].to(self.device)
            return {"input_ids": input_ids, "attention_mask": attention_mask}

        raise EmbeddingBackendError("Tokenizer does not support call(...) or batch_encode_plus(...).")


class ProstT5ModelAdapter(ModelAdapter):
    """Model adapter for ProstT5 encoder models."""

    def __init__(
        self,
        model: Any,
        *,
        device: str = "cpu",
        dtype: Any | None = None,
        dtype_name: str | None = None,
    ) -> None:
        self.model = model
        self.device = str(device)
        self.dtype = dtype
        self.dtype_name = dtype_name

        if self.dtype is not None:
            move_model_to_device(self.model, device=self.device, dtype=self.dtype, dtype_name=self.dtype_name)
        else:
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
        """Run ProstT5 inference and return hidden states."""
        if not isinstance(tokens, dict):
            raise EmbeddingInputError("ProstT5ModelAdapter expects tokenized input as a dict.")
        if "input_ids" not in tokens or "attention_mask" not in tokens:
            raise EmbeddingInputError("Token dict must contain 'input_ids' and 'attention_mask'.")
        token_map = cast(Dict[str, Any], tokens)

        try:
            import torch  # type: ignore
        except ModuleNotFoundError as exc:
            raise EmbeddingDependencyError(
                "PyTorch is required for ProstT5 inference. Install with: pip install torch"
            ) from exc

        with torch.inference_mode():
            model_output = self.model(
                input_ids=token_map["input_ids"],
                attention_mask=token_map["attention_mask"],
                output_hidden_states=True,
                return_dict=True,
            )

        hidden_states = getattr(model_output, "hidden_states", None)
        if hidden_states is None:
            raise EmbeddingBackendError("ProstT5 model output did not include hidden_states.")

        total_layers = len(hidden_states)
        layer_indices = _resolve_layer_indices(layer_index, total_layers=total_layers)
        sample_spans = _protein_token_spans_from_mask(token_map["attention_mask"])

        selected_layers: Dict[int, Any] = {}
        for user_idx, hf_idx in layer_indices:
            selected_layers[user_idx] = hidden_states[hf_idx]
        return {"layers": selected_layers, "sample_spans": sample_spans}

    def available_layers(self) -> List[int] | None:
        """Return layer indices exposed by the ProstT5 model."""
        total_layers = infer_total_layers_from_model(self.model)
        if total_layers is None:
            return None
        return list(range(total_layers))


class ProstT5Postprocessor(DefaultPostprocessor):
    """Postprocessing for ProstT5 outputs without pooling."""


class ProstT5EmbeddingGenerator(EmbeddingGenerator):
    """Concrete embedding generator for ProstT5 (protein mode only)."""

    GENERATOR_CLASS = "prostt5"
    GENERATOR_ALIASES = ("prost_t5", "prost-t5")
    MODEL_ALIASES: Dict[str, str] = {}
    DEFAULT_MODEL_NAME = "Rostlab/ProstT5"
    FAMILY_MODELS = [DEFAULT_MODEL_NAME]
    SUPPORTED_POOLERS = ("none", "mean")

    def __init__(
        self,
        *,
        model_name: str = DEFAULT_MODEL_NAME,
        device: str = "cpu",
        dtype: str | None = None,
        tokenizer: Any | None = None,
        model: Any | None = None,
    ) -> None:
        resolved_tokenizer: Any | None = tokenizer
        resolved_model: Any | None = model
        resolved_dtype_name = normalize_torch_dtype_name(dtype)
        resolved_torch_dtype = resolve_torch_dtype(resolved_dtype_name) if resolved_dtype_name is not None else None

        if resolved_tokenizer is None or resolved_model is None:
            try:
                from transformers import AutoConfig, T5EncoderModel, T5Tokenizer  # type: ignore
            except ModuleNotFoundError as exc:
                raise EmbeddingDependencyError(
                    "transformers is required for ProstT5 loading. Install with: pip install transformers"
                ) from exc

            if resolved_tokenizer is None:
                resolved_tokenizer = cast(Any, T5Tokenizer).from_pretrained(model_name, do_lower_case=False)
            if resolved_model is None:
                config = cast(Any, AutoConfig).from_pretrained(model_name)
                model_kwargs: Dict[str, Any] = {"config": config}
                if resolved_torch_dtype is not None:
                    model_kwargs["torch_dtype"] = resolved_torch_dtype
                resolved_model = cast(Any, T5EncoderModel).from_pretrained(model_name, **model_kwargs)

        parameters: Dict[str, Any] = {
            "mode": "protein_to_embedding_only",
            "prefix": "<AA2fold>",
            "representation": "per-residue",
            "pooling": "none",
            "precision_policy": "float_on_cpu_half_on_gpu",
            "layer_indexing": "hf_native_0_is_first_hidden",
        }
        if resolved_dtype_name is not None:
            parameters["torch_dtype"] = resolved_dtype_name
            parameters["precision_policy"] = "explicit_torch_dtype"

        super().__init__(
            model_reference=model_name,
            preprocessor=ProstT5Preprocessor(),
            tokenizer=ProstT5TokenizerAdapter(resolved_tokenizer, device=device),
            model=ProstT5ModelAdapter(
                resolved_model,
                device=device,
                dtype=resolved_torch_dtype,
                dtype_name=resolved_dtype_name,
            ),
            postprocessor=ProstT5Postprocessor(),
        )
        self.model_metadata = ModelMetadata(
            provider="huggingface-transformers",
            model_name=model_name,
            model_reference=model_name,
            model_revision=extract_revision(resolved_model),
            tokenizer_name=extract_name_or_path(resolved_tokenizer),
            tokenizer_revision=extract_revision(resolved_tokenizer),
            device=str(device),
            framework_versions=framework_versions("transformers", "torch"),
            parameters=parameters,
        )

    def generate(
        self,
        records: Sequence[GenerationInput],
        *,
        layer_index: int | Sequence[int] | None = None,
        pooler: PoolerInput = None,
        fail_fast: bool = False,
    ) -> GenerationResult:
        """Generate embeddings with the ProstT5 adapter."""
        return self._generate_from_batched_layer_output_map(
            records,
            layer_index=layer_index,
            pooler=pooler,
            fail_fast=fail_fast,
            missing_layers_error="ProstT5 model output missing layers dictionary.",
            requested_layers=normalize_requested_layers(layer_index),
            run_parameters={"model_reference": self.model_reference, "mode": "protein_to_embedding_only"},
        )


def _resolve_layer_indices(
    layer_index: int | Sequence[int] | None,
    *,
    total_layers: int,
) -> List[tuple[int, int]]:
    if total_layers < 1:
        raise EmbeddingBackendError("ProstT5 output reported zero layers.")

    if layer_index is None:
        return [(idx, idx) for idx in range(total_layers)]

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
    return [(user_idx, user_idx) for user_idx in unique_sorted_user]


def _protein_token_spans_from_mask(attention_mask: Any) -> List[tuple[int, int]]:
    try:
        rows = attention_mask.tolist()
    except Exception:
        try:
            rows = [attention_mask[0].tolist()]
        except Exception as exc:
            raise EmbeddingBackendError("Could not infer valid token lengths from attention_mask.") from exc

    if rows and isinstance(rows[0], (int, float)):
        rows = [rows]

    spans: List[tuple[int, int]] = []
    for row in cast(Sequence[Sequence[object]], rows):
        valid_len = int(sum(int(cast(Any, value)) for value in row))
        if valid_len < 3:
            raise EmbeddingBackendError(
                "Expected at least prefix token + residue token + end token in ProstT5 input."
            )
        # token 0 is task prefix (<AA2fold>), final valid token is end token.
        spans.append((1, valid_len - 1))
    if not spans:
        raise EmbeddingBackendError("No token rows produced by ProstT5 tokenizer.")
    return spans

__all__ = [
    "ProstT5Preprocessor",
    "ProstT5TokenizerAdapter",
    "ProstT5ModelAdapter",
    "ProstT5Postprocessor",
    "ProstT5EmbeddingGenerator",
]
