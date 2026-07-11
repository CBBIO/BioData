"""ESM-C-specific embedding generator backed by the ESM SDK."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Sequence, cast

from .. import (
    EmbeddingBackendError,
    EmbeddingDependencyError,
    EmbeddingGenerator,
    EmbeddingInputError,
    GenerationInput,
    GenerationResult,
    ModelAdapter,
    ModelDownloadResult,
    ModelMetadata,
    TokenizerAdapter,
)
from ..utils.download import download_huggingface_snapshot
from ..utils.pooler import PoolerInput
from ..utils.torch import BasePreprocessor, DefaultPostprocessor, framework_versions, normalize_requested_layers


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

ESMC_LAYER_SPECS: Dict[str, int] = {
    "biohub/ESMC-300M": 30,
    "biohub/ESMC-600M": 36,
    "biohub/ESMC-6B": 80,
    "esmc_300m": 30,
    "esmc_600m": 36,
    "esmc_6b": 80,
}

_EsmcArchitectureCache = tuple[type[Any], type[Any], type[Any], type[Any]]
_esmc_architecture_cache: _EsmcArchitectureCache | None = None


class EsmcPreprocessor(BasePreprocessor):
    """Preprocessing for ESM-C protein inputs."""

    def __init__(self) -> None:
        super().__init__(context="ESM-C preprocessing")


class EsmcTokenizerAdapter(TokenizerAdapter):
    """Tokenizer adapter for the ESM SDK ESM-C model."""

    def __init__(self, client: Any, *, device: str = "cpu") -> None:
        self.client = client
        self.device = str(device)

    def tokenize(self, sequence: str) -> Any:
        """Tokenize one sequence for ESM-C."""
        return self.tokenize_many([sequence])

    def tokenize_many(self, sequences: Sequence[str]) -> Any:
        """Tokenize a batch of sequences for ESM-C."""
        if not sequences:
            raise EmbeddingInputError("ESM-C tokenization requires at least one sequence.")
        tokenize_fn = getattr(self.client, "_tokenize", None)
        if not callable(tokenize_fn):
            raise EmbeddingBackendError("ESM-C SDK model does not expose the expected _tokenize method.")
        sequence_tokens = tokenize_fn(list(sequences))
        sequence_tokens = _maybe_to_device(sequence_tokens, self.device)
        pad_token_id = _pad_token_id(self.client)
        attention_mask = sequence_tokens != pad_token_id
        return {"sequence_tokens": sequence_tokens, "attention_mask": attention_mask}


class EsmcModelAdapter(ModelAdapter):
    """Model adapter for the ESM SDK ESM-C forward path."""

    def __init__(self, client: Any, *, available_layer_count: int | None = None) -> None:
        self.client = client
        self.available_layer_count = available_layer_count

    def infer(self, tokens: Any, *, layer_index: int | Sequence[int] | None = None) -> Any:
        """Run ESM-C inference and return hidden states."""
        if not isinstance(tokens, dict):
            raise EmbeddingInputError("EsmcModelAdapter expects tokenized input as a dict.")
        token_map = cast(Dict[str, Any], tokens)
        if "sequence_tokens" not in token_map or "attention_mask" not in token_map:
            raise EmbeddingInputError("Token dict must contain sequence_tokens and attention_mask.")

        try:
            import torch  # type: ignore
        except ModuleNotFoundError as exc:
            raise EmbeddingDependencyError("PyTorch is required for ESM-C inference. Install with: pip install torch") from exc

        total = self._total_layers()
        requested = _resolve_requested_layers(layer_index, available_layer_count=total)
        forward_fn = getattr(self.client, "forward", None)
        if not callable(forward_fn):
            forward_fn = self.client
        with torch.inference_mode():
            output = forward_fn(sequence_tokens=token_map["sequence_tokens"])
        hidden_states = getattr(output, "hidden_states", None)
        if hidden_states is None:
            raise EmbeddingBackendError("ESM-C output missing hidden_states.")
        if len(hidden_states) < total:
            raise EmbeddingBackendError(f"ESM-C returned {len(hidden_states)} hidden states; expected at least {total}.")
        return {
            "layers": {idx: hidden_states[idx] for idx in requested},
            "sample_spans": _sample_spans_from_attention_mask(token_map["attention_mask"]),
        }

    def available_layers(self) -> List[int] | None:
        """Return layer indices exposed by the ESM-C model."""
        if self.available_layer_count is None:
            return None
        return list(range(int(self.available_layer_count)))

    def _total_layers(self) -> int:
        if self.available_layer_count is not None:
            return int(self.available_layer_count)
        transformer = getattr(self.client, "transformer", None)
        blocks = getattr(transformer, "blocks", None)
        if isinstance(blocks, Sequence):
            return len(cast(Sequence[object], blocks))
        raise EmbeddingBackendError("Could not infer ESM-C layer count from SDK model.")


class EsmcPostprocessor(DefaultPostprocessor):
    """Postprocessing for ESM-C outputs without pooling."""


class EsmcEmbeddingGenerator(EmbeddingGenerator):
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
    ) -> None:
        model_reference = _resolve_model_reference(model_name)
        sdk_model_name = _resolve_sdk_model_name(model_name)
        resolved_client = client if client is not None else model

        _ = dtype, tokenizer

        if resolved_client is None:
            try:
                from esm.models.esmc import ESMC  # type: ignore
                import torch  # type: ignore
            except ModuleNotFoundError as exc:
                raise EmbeddingDependencyError(
                    "ESM SDK is required for ESM-C loading. Install package providing esm.models.esmc."
                ) from exc

            kwargs = dict(from_pretrained_kwargs or {})
            if use_flash_attention is not None:
                kwargs.setdefault("use_flash_attention", bool(use_flash_attention))
            if kwargs:
                try:
                    resolved_client = ESMC.from_pretrained(sdk_model_name, device=torch.device(device), **kwargs)
                except TypeError:
                    resolved_client = ESMC.from_pretrained(sdk_model_name, device=torch.device(device))
            else:
                resolved_client = ESMC.from_pretrained(sdk_model_name, device=torch.device(device))
            resolved_client = _maybe_to_device(resolved_client, device)

        layer_count = ESMC_LAYER_SPECS.get(model_reference) or ESMC_LAYER_SPECS.get(sdk_model_name)
        EmbeddingGenerator.__init__(
            self,
            model_reference=model_reference,
            preprocessor=EsmcPreprocessor(),
            tokenizer=EsmcTokenizerAdapter(resolved_client, device=device),
            model=EsmcModelAdapter(resolved_client, available_layer_count=layer_count),
            postprocessor=EsmcPostprocessor(),
        )
        self.model_metadata = ModelMetadata(
            provider="esm-sdk",
            model_name=model_name,
            model_reference=model_reference,
            tokenizer_name="esm.tokenization.EsmSequenceTokenizer",
            device=str(device),
            framework_versions=framework_versions("esm", "torch"),
            parameters={
                "mode": "protein_to_embedding_only",
                "representation": "per-residue",
                "pooling": "none",
                "layer_indexing": "esm_sdk_hidden_states_0_is_first_transformer_layer",
                "sdk_model_name": sdk_model_name,
                "use_flash_attention": use_flash_attention,
                "available_layer_count_hint": layer_count,
            },
        )

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


def _resolve_sdk_model_name(model_name: str) -> str:
    key = str(model_name).strip()
    if not key:
        raise EmbeddingInputError("ESM-C model_name must be non-empty.")
    return ESMC_SDK_MODEL_NAMES.get(key, ESMC_SDK_MODEL_NAMES.get(key.lower(), key))


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


def _sample_spans_from_attention_mask(attention_mask: Any) -> List[tuple[int, int]]:
    try:
        rows = attention_mask.tolist()
    except Exception:
        rows = [attention_mask[0].tolist()]
    if rows and isinstance(rows[0], (int, float, bool)):
        rows = [rows]
    spans: List[tuple[int, int]] = []
    for row in cast(Sequence[Sequence[object]], rows):
        valid_len = int(sum(int(cast(Any, value)) for value in row))
        spans.append((1, max(valid_len - 1, 1)))
    return spans


def _pad_token_id(client: Any) -> int:
    tokenizer = getattr(client, "tokenizer", None)
    pad_token_id = getattr(tokenizer, "pad_token_id", None)
    if isinstance(pad_token_id, int):
        return int(pad_token_id)
    return 1


def _maybe_to_device(value: Any, device: str) -> Any:
    to_fn = getattr(value, "to", None)
    if callable(to_fn):
        return to_fn(device)
    return value


__all__ = [
    "ESMC_HF_MODEL_NAMES",
    "ESMC_LAYER_SPECS",
    "ESMC_SDK_MODEL_NAMES",
    "EsmcPreprocessor",
    "EsmcTokenizerAdapter",
    "EsmcModelAdapter",
    "EsmcPostprocessor",
    "EsmcEmbeddingGenerator",
    "register_hf_esmc_architecture",
]
