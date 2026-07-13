"""ANKH3-specific embedding adapters and generator (protein -> embedding)."""

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
    ModelDownloadResult,
    ModelMetadata,
    ModelAdapter,
    TokenizerAdapter,
)
from ..utils.download import download_huggingface_snapshot
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
from ._esm_hf import resolve_model_name


ANKH3_MODEL_NAMES: Dict[str, str] = {
    "ankh_large": "ElnaggarLab/ankh-large",
    "ankh_base": "ElnaggarLab/ankh-base",
    "ankh3_base": "ElnaggarLab/ankh-base",
    "ankh3-base": "ElnaggarLab/ankh-base",
    "ankh3-large": "ElnaggarLab/ankh3-large",
    "ankh3_large": "ElnaggarLab/ankh3-large",
    "ankh3 xl": "ElnaggarLab/ankh3-xl",
    "ankh3-xl": "ElnaggarLab/ankh3-xl",
    "ankh3_xl": "ElnaggarLab/ankh3-xl",
}


class Ankh3Preprocessor(BasePreprocessor):
    """Preprocessing for ANKH3 protein inputs."""

    ALLOWED_PREFIXES = {"[NLU]", "[S2S]"}

    def __init__(self, *, prefix: str = "[NLU]") -> None:
        value = str(prefix).strip()
        if value not in self.ALLOWED_PREFIXES:
            raise EmbeddingInputError(
                f"Unsupported ANKH3 prefix: {prefix!r}. Use one of: [NLU], [S2S]."
            )
        super().__init__(context="ANKH3 preprocessing", prefix=value, prefix_space=False)

    @property
    def prefix(self) -> str:
        """Return the ANKH3 prefix token for a sequence."""
        return str(self._prefix)


class Ankh3TokenizerAdapter(TokenizerAdapter):
    """Tokenizer adapter for ANKH3 tokenizers."""

    def __init__(self, tokenizer: Any, *, device: str = "cpu") -> None:
        self.tokenizer = tokenizer
        self.device = str(device)

    def tokenize(self, sequence: str) -> Any:
        """Tokenize one sequence for ANKH3."""
        return self.tokenize_many([sequence])

    def tokenize_many(self, sequences: Sequence[str]) -> Any:
        """Tokenize a batch of sequences for ANKH3."""
        if not sequences:
            raise EmbeddingInputError("ANKH3 tokenization requires at least one sequence.")
        encoded = self.tokenizer(
            list(sequences),
            add_special_tokens=True,
            padding="longest",
            return_tensors="pt",
            is_split_into_words=False,
        )
        encoded_map = cast(Dict[str, Any], encoded)
        input_ids = encoded_map["input_ids"].to(self.device)
        attention_mask = encoded_map["attention_mask"].to(self.device)
        return {"input_ids": input_ids, "attention_mask": attention_mask}


class Ankh3ModelAdapter(ModelAdapter):
    """Model adapter for ANKH3 encoder models."""

    def __init__(self, model: Any, *, device: str = "cpu", dtype: Any | None = None, dtype_name: str | None = None) -> None:
        self.model = model
        self.device = str(device)
        self.dtype = dtype
        self.dtype_name = dtype_name
        move_model_to_device(self.model, device=self.device, dtype=self.dtype, dtype_name=self.dtype_name)
        eval_fn = getattr(self.model, "eval", None)
        if callable(eval_fn):
            eval_fn()

    def infer(self, tokens: Any, *, layer_index: int | Sequence[int] | None = None) -> Any:
        """Run ANKH3 inference and return hidden states."""
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

        with torch.inference_mode():
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
        sample_spans = _protein_token_spans_from_mask(token_map["attention_mask"])

        selected_layers: Dict[int, Any] = {}
        for idx in layer_indices:
            selected_layers[idx] = hidden_states[idx]
        return {"layers": selected_layers, "sample_spans": sample_spans}

    def available_layers(self) -> List[int] | None:
        """Return layer indices exposed by the ANKH3 model."""
        total_layers = infer_total_layers_from_model(self.model)
        if total_layers is None:
            return None
        return list(range(total_layers))


class Ankh3Postprocessor(DefaultPostprocessor):
    """Postprocessing for ANKH3 outputs without pooling."""


class Ankh3EmbeddingGenerator(EmbeddingGenerator):
    """Concrete embedding generator for ANKH3 models."""

    GENERATOR_CLASS = "ankh3"
    GENERATOR_ALIASES = ("ankh", "ankh3-large", "ankh3_large")
    MODEL_ALIASES = ANKH3_MODEL_NAMES
    DEFAULT_MODEL_NAME = "ElnaggarLab/ankh3-large"
    # Published model family members from ANKH/ANKH3 plus the HF checkpoint used by default.
    FAMILY_MODELS = [
        "Ankh Large",
        "Ankh Base",
        "Ankh3 Large",
        "Ankh3 XL",
        "ElnaggarLab/ankh3-xl",
        DEFAULT_MODEL_NAME,
    ]
    SUPPORTED_POOLERS = ("none", "mean")

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
        """Download an ANKH/ANKH3 checkpoint into the local Hugging Face cache."""
        raw_model_name = model_name or cls.DEFAULT_MODEL_NAME
        model_reference = resolve_model_name(raw_model_name, cls.MODEL_ALIASES, family="ANKH3")
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
        prefix: str = "[NLU]",
        device: str = "cpu",
        dtype: str | None = None,
        tokenizer: Any | None = None,
        model: Any | None = None,
    ) -> None:
        model_reference = resolve_model_name(model_name, self.MODEL_ALIASES, family="ANKH3")
        resolved_tokenizer: Any | None = tokenizer
        resolved_model: Any | None = model
        resolved_dtype_name = normalize_torch_dtype_name(dtype)
        resolved_torch_dtype = resolve_torch_dtype(resolved_dtype_name) if resolved_dtype_name is not None else None

        if resolved_tokenizer is None or resolved_model is None:
            try:
                from transformers import AutoConfig, T5EncoderModel, T5Tokenizer  # type: ignore
            except ModuleNotFoundError as exc:
                raise EmbeddingDependencyError(
                    "transformers is required for ANKH3 loading. Install with: pip install transformers"
                ) from exc

            if resolved_tokenizer is None:
                resolved_tokenizer = cast(Any, T5Tokenizer).from_pretrained(
                    model_reference,
                    do_lower_case=False,
                )
            if resolved_model is None:
                config = cast(Any, AutoConfig).from_pretrained(model_reference)
                setattr(config, "tie_word_embeddings", False)
                model_kwargs: Dict[str, Any] = {"config": config}
                if resolved_torch_dtype is not None:
                    model_kwargs["torch_dtype"] = resolved_torch_dtype
                resolved_model = cast(Any, T5EncoderModel).from_pretrained(
                    model_reference,
                    **model_kwargs,
                )

        parameters: Dict[str, Any] = {
            "mode": "protein_to_embedding_only",
            "prefix": str(prefix),
            "representation": "per-residue",
            "pooling": "none",
            "layer_indexing": "hf_native_0_is_first_hidden",
        }
        if resolved_dtype_name is not None:
            parameters["torch_dtype"] = resolved_dtype_name

        super().__init__(
            model_reference=model_reference,
            preprocessor=Ankh3Preprocessor(prefix=prefix),
            tokenizer=Ankh3TokenizerAdapter(resolved_tokenizer, device=device),
            model=Ankh3ModelAdapter(
                resolved_model,
                device=device,
                dtype=resolved_torch_dtype,
                dtype_name=resolved_dtype_name,
            ),
            postprocessor=Ankh3Postprocessor(),
        )
        self.model_metadata = ModelMetadata(
            provider="huggingface-transformers",
            model_name=model_name,
            model_reference=model_reference,
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
        """Generate embeddings with the ANKH3 adapter."""
        return self._generate_from_batched_layer_output_map(
            records,
            layer_index=layer_index,
            pooler=pooler,
            fail_fast=fail_fast,
            missing_layers_error="ANKH3 model output missing layers dictionary.",
            requested_layers=normalize_requested_layers(layer_index),
            run_parameters={
                "model_reference": self.model_reference,
                "mode": "protein_to_embedding_only",
                "prefix": cast(Ankh3Preprocessor, self.preprocessor).prefix,
            },
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
            raise EmbeddingBackendError("Expected at least one residue plus special tokens in ANKH3 input.")
        # Keep residues and remove start/end special tokens.
        spans.append((1, valid_len - 1))
    if not spans:
        raise EmbeddingBackendError("No token rows produced by ANKH3 tokenizer.")
    return spans

__all__ = [
    "ANKH3_MODEL_NAMES",
    "Ankh3Preprocessor",
    "Ankh3TokenizerAdapter",
    "Ankh3ModelAdapter",
    "Ankh3Postprocessor",
    "Ankh3EmbeddingGenerator",
]
