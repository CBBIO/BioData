"""Shared Hugging Face ESM embedding adapters."""

from __future__ import annotations

from typing import Any, Dict, List, Sequence, cast

from .. import (
    EmbeddingBackendError,
    EmbeddingDependencyError,
    EmbeddingGenerator,
    EmbeddingInputError,
    GenerationInput,
    GenerationResult,
    ModelAdapter,
    ModelMetadata,
    TokenizerAdapter,
)
from ..utils.pooler import PoolerInput
from ..utils.torch import (
    BasePreprocessor,
    DefaultPostprocessor,
    extract_name_or_path,
    extract_revision,
    framework_versions,
    move_model_to_device,
    normalize_requested_layers,
    normalize_torch_dtype_name,
    resolve_torch_dtype,
)
from ..utils.transformers import load_esm_tokenizer


class HfEsmPreprocessor(BasePreprocessor):
    """Shared preprocessing for ESM-family protein inputs."""

    def __init__(self, *, context: str) -> None:
        super().__init__(context=context)


class HfEsmTokenizerAdapter(TokenizerAdapter):
    """Tokenizer adapter for Hugging Face ESM tokenizers."""

    def __init__(self, tokenizer: Any, *, device: str = "cpu") -> None:
        self.tokenizer = tokenizer
        self.device = str(device)

    def tokenize(self, sequence: str) -> Any:
        """Tokenize one sequence for a HuggingFace ESM model."""
        return self.tokenize_many([sequence])

    def tokenize_many(self, sequences: Sequence[str]) -> Any:
        """Tokenize a batch of sequences for a HuggingFace ESM model."""
        if not sequences:
            raise EmbeddingInputError("ESM tokenization requires at least one sequence.")
        encoded = self.tokenizer(
            list(sequences),
            add_special_tokens=True,
            padding=True,
            return_tensors="pt",
        )
        encoded_map = cast(Dict[str, Any], encoded)
        if "input_ids" not in encoded_map or "attention_mask" not in encoded_map:
            raise EmbeddingBackendError("ESM tokenizer output must include input_ids and attention_mask.")
        return {
            "input_ids": _maybe_to_device(encoded_map["input_ids"], self.device),
            "attention_mask": _maybe_to_device(encoded_map["attention_mask"], self.device),
        }


class HfEsmModelAdapter(ModelAdapter):
    """Model adapter for Hugging Face ESM-like encoders."""

    def __init__(self, model: Any, *, available_layer_count: int | None = None) -> None:
        self.model = model
        self.available_layer_count = available_layer_count

    def infer(self, tokens: Any, *, layer_index: int | Sequence[int] | None = None) -> Any:
        """Run HuggingFace ESM inference and return hidden states."""
        if not isinstance(tokens, dict):
            raise EmbeddingInputError("HfEsmModelAdapter expects tokenized input as a dict.")
        token_map = cast(Dict[str, Any], tokens)
        if "input_ids" not in token_map or "attention_mask" not in token_map:
            raise EmbeddingInputError("Token dict must contain input_ids and attention_mask.")

        try:
            import torch  # type: ignore
        except ModuleNotFoundError as exc:
            raise EmbeddingDependencyError(
                "PyTorch is required for ESM inference. Install with: pip install torch"
            ) from exc

        total = self._total_layers()
        requested = resolve_hf_esm_layer_indices(layer_index, total_layers=total)
        with torch.inference_mode():
            out = self.model(
                input_ids=token_map["input_ids"],
                attention_mask=token_map["attention_mask"],
                output_hidden_states=True,
                return_dict=True,
            )
        hidden_states = getattr(out, "hidden_states", None)
        if hidden_states is None:
            raise EmbeddingBackendError("Hugging Face ESM output missing hidden_states.")
        if len(hidden_states) <= total:
            raise EmbeddingBackendError(
                f"Hugging Face ESM returned {len(hidden_states)} hidden states; expected at least {total + 1}."
            )
        return {
            "layers": {idx: hidden_states[idx] for idx in requested},
            "sample_spans": esm_sample_spans_from_attention_mask(token_map["attention_mask"]),
        }

    def available_layers(self) -> List[int] | None:
        """Return layer indices exposed by the HuggingFace ESM model."""
        total = self._total_layers()
        return list(range(total + 1))

    def _total_layers(self) -> int:
        if self.available_layer_count is not None:
            return int(self.available_layer_count)
        total = infer_hf_esm_transformer_layers(self.model)
        if total is None:
            raise EmbeddingBackendError("Could not infer ESM num_hidden_layers from model config.")
        return total


class HfEsmPostprocessor(DefaultPostprocessor):
    """Postprocessing for ESM-family outputs without pooling."""


class HfEsmEmbeddingGenerator(EmbeddingGenerator):
    """Shared Hugging Face ESM embedding generator."""

    SUPPORTED_POOLERS = ("none", "mean", "cls")

    def __init__(
        self,
        *,
        model_name: str,
        model_reference: str,
        context: str,
        provider: str,
        device: str = "cpu",
        dtype: str | None = None,
        model: Any | None = None,
        tokenizer: Any | None = None,
        tokenizer_trust_remote_code: bool = False,
        from_pretrained_kwargs: Dict[str, Any] | None = None,
        auto_model_class: str = "AutoModel",
        tokenizer_from_pretrained_kwargs: Dict[str, Any] | None = None,
        layer_count_hint: int | None = None,
        max_sequence_length: int | None = None,
        preprocessor_adapter_cls: Any | None = None,
        tokenizer_adapter_cls: Any | None = None,
        model_adapter_cls: Any | None = None,
        postprocessor_adapter_cls: Any | None = None,
    ) -> None:
        resolved_dtype_name = normalize_torch_dtype_name(dtype)
        resolved_torch_dtype = resolve_torch_dtype(resolved_dtype_name) if resolved_dtype_name is not None else None
        resolved_model = model
        resolved_tokenizer = tokenizer
        model_kwargs = dict(from_pretrained_kwargs or {})
        if resolved_torch_dtype is not None:
            model_kwargs.setdefault("torch_dtype", resolved_torch_dtype)

        if resolved_model is None:
            try:
                import transformers  # type: ignore
            except ModuleNotFoundError as exc:
                raise EmbeddingDependencyError(
                    "transformers is required for ESM loading. Install with: pip install transformers"
                ) from exc
            auto_model = getattr(transformers, auto_model_class, None)
            if auto_model is None:
                raise EmbeddingDependencyError(
                    f"transformers does not expose {auto_model_class}; install a compatible transformers version."
                )
            try:
                resolved_model = auto_model.from_pretrained(model_reference, **model_kwargs)
            except (ImportError, RuntimeError) as exc:
                raise EmbeddingDependencyError(
                    f"Failed to load Hugging Face model {model_reference!r}. This checkpoint may require "
                    f"optional model-specific dependencies that are not installed or not usable: {exc}"
                ) from exc

        if resolved_tokenizer is None:
            if tokenizer_trust_remote_code:
                try:
                    from transformers import AutoTokenizer  # type: ignore
                except ModuleNotFoundError as exc:
                    raise EmbeddingDependencyError(
                        "transformers is required for ESM tokenization. Install with: pip install transformers"
                    ) from exc
                try:
                    tokenizer_kwargs = {"trust_remote_code": True}
                    tokenizer_kwargs.update(tokenizer_from_pretrained_kwargs or {})
                    resolved_tokenizer = cast(Any, AutoTokenizer).from_pretrained(
                        model_reference,
                        **tokenizer_kwargs,
                    )
                except Exception as exc:
                    raise EmbeddingBackendError(f"Failed to load ESM tokenizer from transformers: {exc}") from exc
            else:
                try:
                    resolved_tokenizer = load_esm_tokenizer(model_reference)
                except Exception as exc:
                    raise EmbeddingBackendError(f"Failed to load ESM tokenizer from transformers: {exc}") from exc

        move_model_to_device(
            resolved_model,
            device=str(device),
            dtype=resolved_torch_dtype,
            dtype_name=resolved_dtype_name,
        )
        eval_fn = getattr(resolved_model, "eval", None)
        if callable(eval_fn):
            eval_fn()

        available_layer_count = layer_count_hint
        if available_layer_count is None:
            available_layer_count = infer_hf_esm_transformer_layers(resolved_model)

        preprocessor_cls = preprocessor_adapter_cls or HfEsmPreprocessor
        tokenizer_cls = tokenizer_adapter_cls or HfEsmTokenizerAdapter
        model_cls = model_adapter_cls or HfEsmModelAdapter
        postprocessor_cls = postprocessor_adapter_cls or HfEsmPostprocessor
        super().__init__(
            model_reference=model_reference,
            preprocessor=preprocessor_cls(context=context),
            tokenizer=tokenizer_cls(resolved_tokenizer, device=device),
            model=model_cls(resolved_model, available_layer_count=available_layer_count),
            postprocessor=postprocessor_cls(),
        )
        parameters: Dict[str, Any] = {
            "mode": "protein_to_embedding_only",
            "representation": "per-residue",
            "pooling": "none",
            "layer_indexing": "hf_hidden_states_0_is_embedding_and_top_is_num_hidden_layers",
            "available_layer_count_hint": available_layer_count + 1 if available_layer_count is not None else None,
        }
        if max_sequence_length is not None:
            parameters["max_sequence_length"] = int(max_sequence_length)
        if resolved_dtype_name is not None:
            parameters["torch_dtype"] = resolved_dtype_name

        self.model_metadata = ModelMetadata(
            provider=provider,
            model_name=model_name,
            model_reference=model_reference,
            model_revision=extract_revision(resolved_model),
            tokenizer_name=extract_name_or_path(resolved_tokenizer) or model_reference,
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
        """Generate embeddings with a HuggingFace ESM adapter."""
        return self._generate_from_batched_layer_output_map(
            records,
            layer_index=layer_index,
            pooler=pooler,
            fail_fast=fail_fast,
            missing_layers_error="ESM model output missing layers dictionary.",
            requested_layers=normalize_requested_layers(layer_index),
            run_parameters={
                "model_reference": self.model_reference,
                "mode": "protein_to_embedding_only",
            },
        )


def resolve_hf_esm_layer_indices(layer_index: int | Sequence[int] | None, *, total_layers: int) -> List[int]:
    """Resolve requested HuggingFace ESM layers to concrete indices."""
    if total_layers < 0:
        raise EmbeddingBackendError("ESM reported invalid total layer count.")
    if layer_index is None:
        return list(range(total_layers + 1))
    if isinstance(layer_index, int):
        requested = [int(layer_index)]
    else:
        requested = [int(value) for value in layer_index]
        if not requested:
            raise EmbeddingInputError("layer_index sequence cannot be empty.")

    resolved: List[int] = []
    layer_count = total_layers + 1
    for index in requested:
        original = index
        if index < 0:
            index = layer_count + index
        if index < 0 or index > total_layers:
            raise EmbeddingInputError(
                f"Requested layer index {original} out of range for ESM available layers=0..{total_layers}."
            )
        resolved.append(index)
    return sorted(set(resolved))


def esm_sample_spans_from_attention_mask(attention_mask: Any) -> List[tuple[int, int]]:
    """Return residue-token spans from an ESM attention mask."""
    try:
        rows = attention_mask.tolist()
    except Exception:
        rows = [attention_mask[0].tolist()]
    if rows and isinstance(rows[0], (int, float)):
        rows = [rows]
    spans: List[tuple[int, int]] = []
    for row in cast(Sequence[Sequence[object]], rows):
        valid_len = int(sum(int(cast(Any, value)) for value in row))
        spans.append((1, max(valid_len - 1, 1)))
    return spans


def infer_hf_esm_transformer_layers(model: Any) -> int | None:
    """Infer the number of transformer layers in a HuggingFace ESM model."""
    config = getattr(model, "config", None)
    if config is not None:
        num_hidden_layers = getattr(config, "num_hidden_layers", None)
        if isinstance(num_hidden_layers, int):
            return int(num_hidden_layers)
        num_layers = getattr(config, "num_layers", None)
        if isinstance(num_layers, int):
            return int(num_layers)
    num_layers = getattr(model, "num_layers", None)
    if isinstance(num_layers, int):
        return int(num_layers)
    args = getattr(model, "args", None)
    if args is not None:
        value = getattr(args, "layers", None)
        if isinstance(value, int):
            return int(value)
    return None


def resolve_model_name(value: str, aliases: Dict[str, str], *, family: str) -> str:
    """Resolve a model name or alias against known family models."""
    resolved = str(value).strip()
    if not resolved:
        raise EmbeddingInputError(f"{family} model_name must be non-empty.")
    return aliases.get(resolved.lower(), resolved)


def _maybe_to_device(value: Any, device: str) -> Any:
    to_fn = getattr(value, "to", None)
    if callable(to_fn):
        return to_fn(device)
    return value


__all__ = [
    "HfEsmEmbeddingGenerator",
    "HfEsmModelAdapter",
    "HfEsmPostprocessor",
    "HfEsmPreprocessor",
    "HfEsmTokenizerAdapter",
    "esm_sample_spans_from_attention_mask",
    "infer_hf_esm_transformer_layers",
    "resolve_hf_esm_layer_indices",
    "resolve_model_name",
]
