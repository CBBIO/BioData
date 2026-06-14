"""ESM1b-specific Hugging Face embedding generator."""

from __future__ import annotations

from typing import Any, Dict

from ._esm_hf import (
    HfEsmEmbeddingGenerator,
    HfEsmModelAdapter,
    HfEsmPostprocessor,
    HfEsmPreprocessor,
    HfEsmTokenizerAdapter,
    resolve_model_name,
)


ESM1B_MODEL_ALIASES: Dict[str, str] = {
    "esm1b_t33_650m_ur50s": "facebook/esm-1b",
}


class Esm1bPreprocessor(HfEsmPreprocessor):
    """Preprocessing for ESM1b protein sequences."""

    def __init__(self) -> None:
        super().__init__(context="ESM1b preprocessing")


class Esm1bTokenizerAdapter(HfEsmTokenizerAdapter):
    """Tokenizer adapter for Hugging Face ESM1b."""


class Esm1bModelAdapter(HfEsmModelAdapter):
    """Model adapter for Hugging Face ESM1b."""

    def __init__(self, model: Any, *, available_layer_count: int | None = 33) -> None:
        super().__init__(model, available_layer_count=available_layer_count)


class Esm1bPostprocessor(HfEsmPostprocessor):
    """Postprocessing for ESM1b outputs without pooling."""


class Esm1bEmbeddingGenerator(HfEsmEmbeddingGenerator):
    """Concrete embedding generator for ESM1b."""

    GENERATOR_CLASS = "esm1b"
    GENERATOR_ALIASES = ("esm-1b",)
    DEFAULT_MODEL_NAME = "facebook/esm-1b"
    FAMILY_MODELS = ["facebook/esm-1b", "esm1b_t33_650M_UR50S"]
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
        model_reference = resolve_model_name(model_name, ESM1B_MODEL_ALIASES, family="ESM1b")
        super().__init__(
            model_name=model_name,
            model_reference=model_reference,
            context="ESM1b preprocessing",
            provider="huggingface",
            device=device,
            dtype=dtype,
            model=model,
            tokenizer=tokenizer,
            from_pretrained_kwargs=from_pretrained_kwargs,
            layer_count_hint=33,
            max_sequence_length=self.MAX_SEQUENCE_LENGTH,
        )


__all__ = [
    "ESM1B_MODEL_ALIASES",
    "Esm1bPreprocessor",
    "Esm1bTokenizerAdapter",
    "Esm1bModelAdapter",
    "Esm1bPostprocessor",
    "Esm1bEmbeddingGenerator",
]
