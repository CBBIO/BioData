"""ESM-1b embedding generator backed by the original torch.hub checkpoint."""

from __future__ import annotations

from typing import Any, Dict, Sequence, cast

from .. import EmbeddingBackendError, EmbeddingDependencyError, EmbeddingInputError

from ._esm_hf import (
    HfEsmEmbeddingGenerator,
    HfEsmModelAdapter,
    HfEsmPostprocessor,
    HfEsmPreprocessor,
    HfEsmTokenizerAdapter,
    esm_sample_spans_from_attention_mask,
    resolve_hf_esm_layer_indices,
    resolve_model_name,
)


ESM1B_MODEL_ALIASES: Dict[str, str] = {
    "facebook/esm-1b": "esm1b_t33_650M_UR50S",
}


class Esm1bPreprocessor(HfEsmPreprocessor):
    """Preprocessing for ESM1b protein sequences."""

    def __init__(self, *, context: str = "ESM1b preprocessing") -> None:
        super().__init__(context=context)


class Esm1bTokenizerAdapter(HfEsmTokenizerAdapter):
    """Tokenizer adapter for the original ESM-1b alphabet."""

    def tokenize_many(self, sequences: Sequence[str]) -> Any:
        """Tokenize a batch with the torch.hub ESM alphabet."""
        get_batch_converter = getattr(self.tokenizer, "get_batch_converter", None)
        if not callable(get_batch_converter):
            return super().tokenize_many(sequences)
        if not sequences:
            raise EmbeddingInputError("ESM-1b tokenization requires at least one sequence.")

        batch_converter = cast(Any, get_batch_converter)()
        _labels, _sequences, tokens = cast(
            tuple[Any, Any, Any],
            batch_converter(
                [(str(index), sequence) for index, sequence in enumerate(sequences)]
            ),
        )
        tokens = _maybe_to_device(tokens, self.device)
        padding_index = int(getattr(self.tokenizer, "padding_idx", 1))
        return {
            "tokens": tokens,
            "attention_mask": tokens != padding_index,
        }


class Esm1bModelAdapter(HfEsmModelAdapter):
    """Model adapter for the original ESM-1b model."""

    def __init__(self, model: Any, *, available_layer_count: int | None = 33) -> None:
        super().__init__(model, available_layer_count=available_layer_count)

    def infer(self, tokens: Any, *, layer_index: int | Sequence[int] | None = None) -> Any:
        """Run ESM-1b inference and return selected representations."""
        if not isinstance(tokens, dict):
            raise EmbeddingInputError("Esm1bModelAdapter expects tokenized input as a dict.")
        token_map = cast(Dict[str, Any], tokens)
        if "tokens" not in token_map:
            return super().infer(tokens, layer_index=layer_index)
        if "attention_mask" not in token_map:
            raise EmbeddingInputError("Token dict must contain tokens and attention_mask.")

        try:
            import torch  # type: ignore
        except ModuleNotFoundError as exc:
            raise EmbeddingDependencyError(
                "PyTorch is required for ESM-1b inference. Install with: pip install torch"
            ) from exc

        requested = resolve_hf_esm_layer_indices(layer_index, total_layers=33)
        with torch.inference_mode():
            output = self.model(
                token_map["tokens"],
                repr_layers=requested,
                return_contacts=False,
            )
        if not isinstance(output, dict) or "representations" not in output:
            raise EmbeddingBackendError("ESM-1b output missing representations.")
        representations = cast(Dict[int, Any], output["representations"])
        missing = [index for index in requested if index not in representations]
        if missing:
            raise EmbeddingBackendError(f"ESM-1b output missing requested layers: {missing}.")
        return {
            "layers": {index: representations[index] for index in requested},
            "sample_spans": esm_sample_spans_from_attention_mask(token_map["attention_mask"]),
        }


class Esm1bPostprocessor(HfEsmPostprocessor):
    """Postprocessing for ESM1b outputs without pooling."""


class Esm1bEmbeddingGenerator(HfEsmEmbeddingGenerator):
    """Concrete embedding generator for ESM1b."""

    GENERATOR_CLASS = "esm1b"
    GENERATOR_ALIASES = ("esm-1b",)
    MODEL_ALIASES = ESM1B_MODEL_ALIASES
    DEFAULT_MODEL_NAME = "esm1b_t33_650M_UR50S"
    FAMILY_MODELS = [DEFAULT_MODEL_NAME, "facebook/esm-1b"]
    MAX_SEQUENCE_LENGTH = 1022
    SUPPORTED_POOLERS = ("none", "mean", "cls")

    def __init__(
        self,
        *,
        model_name: str = DEFAULT_MODEL_NAME,
        device: str = "cpu",
        dtype: str | None = None,
        model: Any | None = None,
        tokenizer: Any | None = None,
        from_pretrained_kwargs: Dict[str, Any] | None = None,
    ) -> None:
        model_reference = resolve_model_name(model_name, self.MODEL_ALIASES, family="ESM1b")
        if (model is None) != (tokenizer is None):
            raise EmbeddingInputError(
                "ESM-1b requires model and tokenizer together when wrapping loaded objects."
            )
        if model is None and tokenizer is None:
            try:
                import torch  # type: ignore
            except ModuleNotFoundError as exc:
                raise EmbeddingDependencyError(
                    "PyTorch is required for ESM-1b loading. Install with: pip install torch"
                ) from exc
            try:
                torch_hub = cast(Any, torch.hub)
                model, tokenizer = cast(
                    tuple[Any, Any],
                    torch_hub.load(
                        "facebookresearch/esm:main",
                        "esm1b_t33_650M_UR50S",
                        **dict(from_pretrained_kwargs or {}),
                    ),
                )
            except Exception as exc:
                raise EmbeddingDependencyError(
                    "Failed to load ESM-1b from torch.hub. Ensure network access is available and "
                    "install the fair-esm dependencies required by facebookresearch/esm."
                ) from exc
        super().__init__(
            model_name=model_name,
            model_reference=model_reference,
            context="ESM1b preprocessing",
            provider="torch-hub",
            device=device,
            dtype=dtype,
            model=model,
            tokenizer=tokenizer,
            layer_count_hint=33,
            max_sequence_length=self.MAX_SEQUENCE_LENGTH,
            preprocessor_adapter_cls=Esm1bPreprocessor,
            tokenizer_adapter_cls=Esm1bTokenizerAdapter,
            model_adapter_cls=Esm1bModelAdapter,
            postprocessor_adapter_cls=Esm1bPostprocessor,
        )


def _maybe_to_device(value: Any, device: str) -> Any:
    to_fn = getattr(value, "to", None)
    if callable(to_fn):
        return to_fn(device)
    return value


__all__ = [
    "ESM1B_MODEL_ALIASES",
    "Esm1bPreprocessor",
    "Esm1bTokenizerAdapter",
    "Esm1bModelAdapter",
    "Esm1bPostprocessor",
    "Esm1bEmbeddingGenerator",
]
