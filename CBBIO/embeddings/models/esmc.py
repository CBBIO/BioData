"""ESM-C-specific Hugging Face embedding generator."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Sequence, cast

from .. import (
    EmbeddingBackendError,
    EmbeddingDependencyError,
    EmbeddingInputError,
    GenerationInput,
    GenerationResult,
    ModelAdapter,
    ModelDownloadResult,
)
from ..utils.download import download_huggingface_snapshot
from ..utils.pooler import PoolerInput
from ..utils.torch import BasePreprocessor, normalize_requested_layers
from ._esm_hf import (
    HfEsmEmbeddingGenerator,
    HfEsmPostprocessor,
    HfEsmTokenizerAdapter,
    esm_sample_spans_from_attention_mask,
    infer_hf_esm_transformer_layers,
)


ESMC_HF_MODEL_NAMES: Dict[str, str] = {
    "esmc_300m": "biohub/ESMC-300M",
    "esmc300m": "biohub/ESMC-300M",
    "esmc-300m": "biohub/ESMC-300M",
    "esm_c_300m": "biohub/ESMC-300M",
    "esm-c-300m": "biohub/ESMC-300M",
    "esmc-300m-2024-12": "biohub/ESMC-300M",
    "biohub/esmc-300m-2024-12": "biohub/ESMC-300M",
    "biohub/esmc-300m": "biohub/ESMC-300M",
    "esmc_600m": "biohub/ESMC-600M",
    "esmc600m": "biohub/ESMC-600M",
    "esmc-600m": "biohub/ESMC-600M",
    "esm_c_600m": "biohub/ESMC-600M",
    "esm-c-600m": "biohub/ESMC-600M",
    "esmc-600m-2024-12": "biohub/ESMC-600M",
    "biohub/esmc-600m-2024-12": "biohub/ESMC-600M",
    "biohub/esmc-600m": "biohub/ESMC-600M",
    "esmc_6b": "biohub/ESMC-6B",
    "esmc6b": "biohub/ESMC-6B",
    "esmc-6b": "biohub/ESMC-6B",
    "esm_c_6b": "biohub/ESMC-6B",
    "esm-c-6b": "biohub/ESMC-6B",
    "esmc-6b-2024-12": "biohub/ESMC-6B",
    "biohub/esmc-6b-2024-12": "biohub/ESMC-6B",
    "biohub/esmc-6b": "biohub/ESMC-6B",
}

ESMC_SDK_MODEL_NAMES: Dict[str, str] = {
    "esmc_300m": "esmc_300m",
    "esmc300m": "esmc_300m",
    "esmc-300m": "esmc_300m",
    "esm_c_300m": "esmc_300m",
    "esm-c-300m": "esmc_300m",
    "esmc-300m-2024-12": "esmc_300m",
    "biohub/esmc-300m-2024-12": "esmc_300m",
    "biohub/esmc-300m": "esmc_300m",
    "biohub/ESMC-300M": "esmc_300m",
    "esmc_600m": "esmc_600m",
    "esmc600m": "esmc_600m",
    "esmc-600m": "esmc_600m",
    "esm_c_600m": "esmc_600m",
    "esm-c-600m": "esmc_600m",
    "esmc-600m-2024-12": "esmc_600m",
    "biohub/esmc-600m-2024-12": "esmc_600m",
    "biohub/esmc-600m": "esmc_600m",
    "biohub/ESMC-600M": "esmc_600m",
    "esmc_6b": "esmc_6b",
    "esmc6b": "esmc_6b",
    "esmc-6b": "esmc_6b",
    "esm_c_6b": "esmc_6b",
    "esm-c-6b": "esmc_6b",
    "esmc-6b-2024-12": "esmc_6b",
    "biohub/esmc-6b-2024-12": "esmc_6b",
    "biohub/esmc-6b": "esmc_6b",
    "biohub/ESMC-6B": "esmc_6b",
}

_EsmcArchitectureCache = tuple[type[Any], type[Any], type[Any], type[Any]]
_esmc_architecture_cache: _EsmcArchitectureCache | None = None


class EsmcPreprocessor(BasePreprocessor):
    """Preprocessing for ESM-C protein inputs."""

    def __init__(self, *, context: str = "ESM-C preprocessing") -> None:
        super().__init__(context=context)


class EsmcTokenizerAdapter(HfEsmTokenizerAdapter):
    """Tokenizer adapter for Biohub ESM-C tokenizers."""


class EsmcModelAdapter(ModelAdapter):
    """Model adapter for Hugging Face ESM-C encoders."""

    def __init__(self, model: Any, *, available_layer_count: int | None = None) -> None:
        self.model = model
        self.available_layer_count = available_layer_count

    def infer(self, tokens: Any, *, layer_index: int | Sequence[int] | None = None) -> Any:
        """Run Hugging Face ESM-C inference and return transformer-layer states."""
        if not isinstance(tokens, dict):
            raise EmbeddingInputError("EsmcModelAdapter expects tokenized input as a dict.")
        token_map = cast(Dict[str, Any], tokens)
        if "input_ids" not in token_map or "attention_mask" not in token_map:
            raise EmbeddingInputError("Token dict must contain input_ids and attention_mask.")

        try:
            import torch  # type: ignore
        except ModuleNotFoundError as exc:
            raise EmbeddingDependencyError("PyTorch is required for ESM-C inference. Install with: pip install torch") from exc

        total = self._total_layers()
        requested = _resolve_requested_layers(layer_index, available_layer_count=total)
        with torch.inference_mode():
            output = self.model(
                input_ids=token_map["input_ids"],
                attention_mask=token_map["attention_mask"],
                output_hidden_states=True,
                return_dict=True,
            )
        hidden_states = getattr(output, "hidden_states", None)
        if hidden_states is None:
            raise EmbeddingBackendError("ESM-C output missing hidden_states.")
        if len(hidden_states) <= total:
            raise EmbeddingBackendError(f"ESM-C returned {len(hidden_states)} hidden states; expected at least {total + 1}.")
        return {
            "layers": {idx: hidden_states[idx + 1] for idx in requested},
            "sample_spans": esm_sample_spans_from_attention_mask(token_map["attention_mask"]),
        }

    def available_layers(self) -> List[int] | None:
        """Return layer indices exposed by the ESM-C model."""
        if self.available_layer_count is None:
            return None
        return list(range(int(self.available_layer_count)))

    def _total_layers(self) -> int:
        if self.available_layer_count is not None:
            return int(self.available_layer_count)
        total = infer_hf_esm_transformer_layers(self.model)
        if total is None:
            raise EmbeddingBackendError("Could not infer ESM-C num_hidden_layers from model config.")
        return total


class EsmcPostprocessor(HfEsmPostprocessor):
    """Postprocessing for ESM-C outputs without pooling."""


class EsmcEmbeddingGenerator(HfEsmEmbeddingGenerator):
    """Concrete embedding generator for ESM-C family models."""

    GENERATOR_CLASS = "esmc"
    GENERATOR_ALIASES = ("esmC", "esm-c", "esmc3", "esm3-C", "esm3c")
    MODEL_ALIASES = ESMC_HF_MODEL_NAMES
    DEFAULT_MODEL_NAME = "esmc_600m"
    FAMILY_MODELS = ["esmc_300m", "esmc_600m", "biohub/ESMC-300M", "biohub/ESMC-600M", "biohub/ESMC-6B"]
    SUPPORTED_POOLERS = ("none", "mean", "cls")

    @classmethod
    def download(
        cls,
        model_name: str | None = None,
        *,
        revision: str | None = None,
        cache_dir: str | Path | None = None,
        local_dir: str | Path | None = None,
        token: str | bool | None = None,
        allow_patterns: str | Sequence[str] | None = None,
        ignore_patterns: str | Sequence[str] | None = None,
        **kwargs: Any,
    ) -> ModelDownloadResult:
        """Download an ESM-C checkpoint into the local Hugging Face cache."""
        model_reference = _resolve_model_reference(model_name or cls.DEFAULT_MODEL_NAME)
        return download_huggingface_snapshot(
            model_reference,
            revision=revision,
            cache_dir=cache_dir,
            local_dir=local_dir,
            token=token,
            allow_patterns=allow_patterns,
            ignore_patterns=ignore_patterns,
            **kwargs,
        )

    def __init__(
        self,
        *,
        model_name: str = DEFAULT_MODEL_NAME,
        device: str = "cpu",
        dtype: str | None = None,
        model: Any | None = None,
        tokenizer: Any | None = None,
        client: Any | None = None,
        use_flash_attention: bool | None = None,
        from_pretrained_kwargs: Dict[str, Any] | None = None,
        tokenizer_from_pretrained_kwargs: Dict[str, Any] | None = None,
    ) -> None:
        model_reference = _resolve_model_reference(model_name)
        if client is not None:
            raise EmbeddingInputError("ESM-C now loads Biohub checkpoints through transformers; pass model= and tokenizer= instead of client=.")
        if use_flash_attention is not None:
            raise EmbeddingInputError("use_flash_attention is only supported by the ESM SDK loader, not transformers.")
        if model is None:
            try:
                register_hf_esmc_architecture()
            except ModuleNotFoundError as exc:
                raise EmbeddingDependencyError(
                    "transformers is required for ESM-C loading. Install with: pip install transformers"
                ) from exc

        super().__init__(
            model_name=model_name,
            model_reference=model_reference,
            context="ESM-C preprocessing",
            provider="huggingface-transformers",
            device=device,
            dtype=dtype,
            model=model,
            tokenizer=tokenizer,
            from_pretrained_kwargs=from_pretrained_kwargs,
            tokenizer_from_pretrained_kwargs=tokenizer_from_pretrained_kwargs,
            preprocessor_adapter_cls=EsmcPreprocessor,
            tokenizer_adapter_cls=EsmcTokenizerAdapter,
            model_adapter_cls=EsmcModelAdapter,
            postprocessor_adapter_cls=EsmcPostprocessor,
        )
        if self.model_metadata.parameters is not None:
            self.model_metadata.parameters["layer_indexing"] = "hf_hidden_states_0_is_embedding_esmc_layers_skip_embedding"

    def generate(
        self,
        records: Sequence[GenerationInput],
        *,
        layer_index: int | Sequence[int] | None = None,
        pooler: PoolerInput = None,
        fail_fast: bool = False,
    ) -> GenerationResult:
        """Generate embeddings with the ESM-C adapter."""
        return self._generate_from_batched_layer_output_map(
            records,
            layer_index=layer_index,
            pooler=pooler,
            fail_fast=fail_fast,
            missing_layers_error="ESM-C model output missing layers dictionary.",
            requested_layers=normalize_requested_layers(layer_index),
            run_parameters={
                "model_reference": self.model_reference,
                "mode": "protein_to_embedding_only",
            },
        )


def register_hf_esmc_architecture() -> None:
    """Register Biohub ESM-C config metadata with Transformers' built-in ESM config."""

    global _esmc_architecture_cache

    try:
        from transformers import AutoConfig, AutoModel, EsmConfig, EsmModel  # type: ignore
    except ModuleNotFoundError:
        raise

    if (
        _esmc_architecture_cache is not None
        and _esmc_architecture_cache[0] is EsmConfig
        and _esmc_architecture_cache[1] is EsmModel
    ):
        EsmcConfig = _esmc_architecture_cache[2]
        EsmcModel = _esmc_architecture_cache[3]
    else:

        class EsmcConfig(EsmConfig):  # type: ignore[misc]
            model_type = "esmc"

            def __init__(
                self,
                *,
                d_model: int | None = None,
                n_layers: int | None = None,
                n_heads: int | None = None,
                dtype: str | None = None,
                classifier_dropout: float | None = None,
                **kwargs: Any,
            ) -> None:
                if d_model is not None:
                    kwargs.setdefault("hidden_size", int(d_model))
                    kwargs.setdefault("intermediate_size", int(d_model) * 4)
                if n_layers is not None:
                    kwargs.setdefault("num_hidden_layers", int(n_layers))
                if n_heads is not None:
                    kwargs.setdefault("num_attention_heads", int(n_heads))
                super().__init__(**kwargs)  # pyright: ignore[reportUnknownMemberType]
                self.d_model = int(d_model) if d_model is not None else self.hidden_size
                self.n_layers = int(n_layers) if n_layers is not None else self.num_hidden_layers
                self.n_heads = int(n_heads) if n_heads is not None else self.num_attention_heads
                self.dtype = dtype
                self.classifier_dropout = classifier_dropout

        class EsmcModel(EsmModel):  # type: ignore[misc]
            config_class = EsmcConfig

        _esmc_architecture_cache = (EsmConfig, EsmModel, EsmcConfig, EsmcModel)

    cast(Any, AutoConfig).register("esmc", EsmcConfig, exist_ok=True)
    cast(Any, AutoModel).register(EsmcConfig, EsmcModel, exist_ok=True)


def _resolve_model_reference(model_name: str) -> str:
    key = str(model_name).strip()
    if not key:
        raise EmbeddingInputError("ESM-C model_name must be non-empty.")
    return ESMC_HF_MODEL_NAMES.get(key.lower(), key)


def _resolve_requested_layers(
    layer_index: int | Sequence[int] | None,
    *,
    available_layer_count: int,
) -> List[int]:
    if layer_index is None:
        return list(range(int(available_layer_count)))
    if isinstance(layer_index, int):
        requested = [int(layer_index)]
    else:
        requested = [int(value) for value in layer_index]
        if not requested:
            raise EmbeddingInputError("layer_index sequence cannot be empty.")

    resolved: List[int] = []
    upper = int(available_layer_count)
    for index in requested:
        original = index
        if index < 0:
            index = upper + index
        if index < 0 or index >= upper:
            raise EmbeddingInputError(f"Requested ESM-C layer index {original} out of range; expected 0..{upper - 1}.")
        resolved.append(index)
    return sorted(set(resolved))


__all__ = [
    "ESMC_HF_MODEL_NAMES",
    "ESMC_SDK_MODEL_NAMES",
    "EsmcPreprocessor",
    "EsmcTokenizerAdapter",
    "EsmcModelAdapter",
    "EsmcPostprocessor",
    "EsmcEmbeddingGenerator",
    "register_hf_esmc_architecture",
]
