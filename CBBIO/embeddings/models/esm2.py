"""ESM2-specific Hugging Face embedding generators."""

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


ESM2_HF_MODEL_NAMES: Dict[str, str] = {
    "esm2_8m": "facebook/esm2_t6_8M_UR50D",
    "esm2_35m": "facebook/esm2_t12_35M_UR50D",
    "esm2_150m": "facebook/esm2_t30_150M_UR50D",
    "esm2_650m": "facebook/esm2_t33_650M_UR50D",
    "esm2_3b": "facebook/esm2_t36_3B_UR50D",
    "esm2_15b": "facebook/esm2_t48_15B_UR50D",
    "esm2_t6_8m_ur50d": "facebook/esm2_t6_8M_UR50D",
    "esm2_t12_35m_ur50d": "facebook/esm2_t12_35M_UR50D",
    "esm2_t30_150m_ur50d": "facebook/esm2_t30_150M_UR50D",
    "esm2_t33_650m_ur50d": "facebook/esm2_t33_650M_UR50D",
    "esm2_t36_3b_ur50d": "facebook/esm2_t36_3B_UR50D",
    "esm2_t48_15b_ur50d": "facebook/esm2_t48_15B_UR50D",
}

ESM2_LAYER_COUNTS: Dict[str, int] = {
    "facebook/esm2_t6_8M_UR50D": 6,
    "facebook/esm2_t12_35M_UR50D": 12,
    "facebook/esm2_t30_150M_UR50D": 30,
    "facebook/esm2_t33_650M_UR50D": 33,
    "facebook/esm2_t36_3B_UR50D": 36,
    "facebook/esm2_t48_15B_UR50D": 48,
}


class Esm2Preprocessor(HfEsmPreprocessor):
    """Preprocessing for ESM2 protein sequences."""

    def __init__(self) -> None:
        super().__init__(context="ESM2 preprocessing")


class Esm2TokenizerAdapter(HfEsmTokenizerAdapter):
    """Tokenizer adapter for Hugging Face ESM2."""


class Esm2ModelAdapter(HfEsmModelAdapter):
    """Model adapter for Hugging Face ESM2."""


class Esm2Postprocessor(HfEsmPostprocessor):
    """Postprocessing for ESM2 outputs without pooling."""


class Esm2EmbeddingGenerator(HfEsmEmbeddingGenerator):
    """Concrete embedding generator for ESM2 family models."""

    GENERATOR_CLASS = "esm2"
    GENERATOR_ALIASES = ("esm-2", "esm")
    MODEL_ALIASES = ESM2_HF_MODEL_NAMES
    DEFAULT_MODEL_NAME = "facebook/esm2_t33_650M_UR50D"
    FAMILY_MODELS = [
        "facebook/esm2_t6_8M_UR50D",
        "facebook/esm2_t12_35M_UR50D",
        "facebook/esm2_t30_150M_UR50D",
        "facebook/esm2_t33_650M_UR50D",
        "facebook/esm2_t36_3B_UR50D",
        "facebook/esm2_t48_15B_UR50D",
    ]
    MAX_SEQUENCE_LENGTH = 1024
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
        model_reference = resolve_model_name(model_name, self.MODEL_ALIASES, family="ESM2")
        super().__init__(
            model_name=model_name,
            model_reference=model_reference,
            context="ESM2 preprocessing",
            provider="huggingface-transformers",
            device=device,
            dtype=dtype,
            model=model,
            tokenizer=tokenizer,
            from_pretrained_kwargs=from_pretrained_kwargs,
            layer_count_hint=ESM2_LAYER_COUNTS.get(model_reference),
            max_sequence_length=self.MAX_SEQUENCE_LENGTH,
        )


__all__ = [
    "ESM2_HF_MODEL_NAMES",
    "ESM2_LAYER_COUNTS",
    "Esm2Preprocessor",
    "Esm2TokenizerAdapter",
    "Esm2ModelAdapter",
    "Esm2Postprocessor",
    "Esm2EmbeddingGenerator",
]
