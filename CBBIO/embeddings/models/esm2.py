"""ESM2-specific embedding adapters and generator (protein -> embedding)."""

from __future__ import annotations

import importlib
from typing import Any, Dict, List, Sequence, Tuple, cast

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
from ..utils.transformers import load_esm_tokenizer
from ..utils.pooler import PoolerInput
from ..utils.torch import (
    BasePreprocessor,
    DefaultPostprocessor,
    framework_versions,
    move_model_to_device,
    normalize_requested_layers,
    normalize_torch_dtype_name,
    resolve_torch_dtype,
)


ESM2_PRETRAINED_LOADERS: Dict[str, str] = {
    "esm2_t33_650m_ur50d": "esm2_t33_650M_UR50D",
    "esm2_t36_3b_ur50d": "esm2_t36_3B_UR50D",
    "esm2_t48_15b_ur50d": "esm2_t48_15B_UR50D",
    "esm2_t30_150m_ur50d": "esm2_t30_150M_UR50D",
    "esm2_t12_35m_ur50d": "esm2_t12_35M_UR50D",
    "esm2_t6_8m_ur50d": "esm2_t6_8M_UR50D",
}

ESM2_HF_MODEL_NAMES: Dict[str, str] = {
    "esm2_t33_650m_ur50d": "facebook/esm2_t33_650M_UR50D",
    "esm2_t36_3b_ur50d": "facebook/esm2_t36_3B_UR50D",
    "esm2_t48_15b_ur50d": "facebook/esm2_t48_15B_UR50D",
    "esm2_t30_150m_ur50d": "facebook/esm2_t30_150M_UR50D",
    "esm2_t12_35m_ur50d": "facebook/esm2_t12_35M_UR50D",
    "esm2_t6_8m_ur50d": "facebook/esm2_t6_8M_UR50D",
}


class Esm2Preprocessor(BasePreprocessor):
    """Preprocessing for ESM2 protein sequences."""

    def __init__(self) -> None:
        super().__init__(context="ESM2 preprocessing")


class Esm2TokenizerAdapter(TokenizerAdapter):
    """Tokenizer adapter using ESM alphabet batch converter."""

    def __init__(
        self,
        batch_converter: Any | None = None,
        padding_idx: int | None = None,
        *,
        tokenizer: Any | None = None,
        device: str = "cpu",
    ) -> None:
        self.batch_converter = batch_converter
        self.padding_idx = int(padding_idx) if padding_idx is not None else None
        self.tokenizer = tokenizer
        self.device = str(device)

    def tokenize(self, sequence: str) -> Any:
        return self.tokenize_many([sequence])

    def tokenize_many(self, sequences: Sequence[str]) -> Any:
        if not sequences:
            raise EmbeddingInputError("ESM2 tokenization requires at least one sequence.")
        if self.tokenizer is not None:
            encoded = self.tokenizer(
                list(sequences),
                add_special_tokens=True,
                padding=True,
                return_tensors="pt",
            )
            encoded_map = cast(Dict[str, Any], encoded)
            return {
                "input_ids": encoded_map["input_ids"].to(self.device),
                "attention_mask": encoded_map["attention_mask"].to(self.device),
            }

        if self.batch_converter is None or self.padding_idx is None:
            raise EmbeddingBackendError("ESM2 tokenizer adapter is missing batch converter configuration.")
        labeled = [(f"query_{index}", sequence) for index, sequence in enumerate(sequences)]
        labels, strs, tokens = cast(Tuple[Any, Any, Any], self.batch_converter(labeled))
        _ = labels, strs
        tokens = tokens.to(self.device)
        lens = (tokens != self.padding_idx).sum(1)
        return {"tokens": tokens, "lens": lens}


class Esm2ModelAdapter(ModelAdapter):
    """Model adapter for ESM2 models."""

    def __init__(self, model: Any, *, available_layer_count: int | None = None) -> None:
        self.model = model
        self.available_layer_count = available_layer_count

    def infer(self, tokens: Any, *, layer_index: int | Sequence[int] | None = None) -> Any:
        if not isinstance(tokens, dict):
            raise EmbeddingInputError("Esm2ModelAdapter expects tokenized input as a dict.")
        token_map = cast(Dict[str, Any], tokens)

        try:
            import torch  # type: ignore
        except ModuleNotFoundError as exc:
            raise EmbeddingDependencyError(
                "PyTorch is required for ESM2 inference. Install with: pip install torch"
            ) from exc

        total = self._total_layers()
        requested = _resolve_layer_indices(layer_index, total_layers=total)
        if "tokens" in token_map and "lens" in token_map:
            with torch.no_grad():
                out = self.model(token_map["tokens"], repr_layers=requested, return_contacts=False)

            reps = out.get("representations")
            if not isinstance(reps, dict):
                raise EmbeddingBackendError("ESM2 output missing 'representations' dictionary.")
            reps_map = cast(Dict[int, Any], reps)

            layers: Dict[int, Any] = {}
            for idx in requested:
                tensor = reps_map.get(idx)
                if tensor is None:
                    raise EmbeddingBackendError(f"ESM2 output does not include requested layer {idx}.")
                layers[idx] = tensor
            return {"layers": layers, "sample_spans": _esm_sample_spans_from_lens(token_map["lens"])}

        if "input_ids" in token_map and "attention_mask" in token_map:
            with torch.no_grad():
                out = self.model(
                    input_ids=token_map["input_ids"],
                    attention_mask=token_map["attention_mask"],
                    output_hidden_states=True,
                    return_dict=True,
                )
            hidden_states = getattr(out, "hidden_states", None)
            if hidden_states is None:
                raise EmbeddingBackendError("ESM2 HF output missing hidden_states.")
            return {
                "layers": {idx: hidden_states[idx] for idx in requested},
                "sample_spans": _esm_sample_spans_from_attention_mask(token_map["attention_mask"]),
            }

        raise EmbeddingInputError("Token dict must contain either 'tokens'/'lens' or 'input_ids'/'attention_mask'.")

    def available_layers(self) -> List[int] | None:
        total = self._total_layers()
        if total < 0:
            return None
        return list(range(total + 1))

    def _total_layers(self) -> int:
        if self.available_layer_count is not None:
            return int(self.available_layer_count)

        num_layers = getattr(self.model, "num_layers", None)
        if isinstance(num_layers, int):
            return int(num_layers)

        args = getattr(self.model, "args", None)
        if args is not None:
            value = getattr(args, "layers", None)
            if isinstance(value, int):
                return int(value)

        raise EmbeddingBackendError("Could not infer ESM2 total layer count from model.")


class Esm2Postprocessor(DefaultPostprocessor):
    """Postprocessing for ESM2 outputs without pooling."""


class Esm2EmbeddingGenerator(EmbeddingGenerator):
    """Concrete embedding generator for ESM2 family models."""

    GENERATOR_CLASS = "esm2"
    GENERATOR_ALIASES = ("esm-2", "ESM")
    DEFAULT_MODEL_NAME = "esm2_t33_650M_UR50D"
    FAMILY_MODELS = sorted(str(value) for value in ESM2_PRETRAINED_LOADERS.keys())
    MAX_SEQUENCE_LENGTH = 1024
    SUPPORTED_POOLERS = ("none", "mean", "cls")

    def __init__(
        self,
        *,
        model_name: str = DEFAULT_MODEL_NAME,
        device: str = "cpu",
        dtype: str | None = None,
        model: Any | None = None,
        alphabet: Any | None = None,
        batch_converter: Any | None = None,
    ) -> None:
        resolved_model: Any | None = model
        resolved_alphabet: Any | None = alphabet
        resolved_converter: Any | None = batch_converter
        resolved_name = str(model_name).strip()
        if not resolved_name:
            raise EmbeddingInputError("ESM2 model_name must be non-empty.")
        resolved_dtype_name = normalize_torch_dtype_name(dtype)
        resolved_torch_dtype = resolve_torch_dtype(resolved_dtype_name) if resolved_dtype_name is not None else None

        if resolved_model is None or resolved_alphabet is None:
            try:
                import esm  # type: ignore
            except ModuleNotFoundError as exc:
                raise EmbeddingDependencyError("ESM package is required for ESM2 loading. Install with: pip install esm") from exc

            loader_name = ESM2_PRETRAINED_LOADERS.get(resolved_name.lower())
            if loader_name is None:
                raise EmbeddingInputError(
                    f"Unsupported ESM2 model name: {model_name!r}. "
                    f"Supported: {sorted(ESM2_PRETRAINED_LOADERS.keys())}"
                )
            pretrained_module = None
            try:
                pretrained_module = importlib.import_module("esm.pretrained")
            except Exception:
                pretrained_module = getattr(esm, "pretrained", None)
            loader = getattr(pretrained_module, loader_name, None) if pretrained_module is not None else None
            if loader is not None and callable(loader):
                resolved_model, resolved_alphabet = cast(tuple[Any, Any], loader())
            else:
                try:
                    from transformers import AutoModel  # type: ignore
                except ModuleNotFoundError as exc:
                    raise EmbeddingDependencyError(
                        "Neither esm.pretrained nor transformers fallback is available for ESM2 loading."
                    ) from exc
                hf_name = _resolve_esm2_hf_model_name(resolved_name)
                model_kwargs: Dict[str, Any] = {}
                if resolved_torch_dtype is not None:
                    model_kwargs["torch_dtype"] = resolved_torch_dtype
                resolved_model = cast(Any, AutoModel).from_pretrained(hf_name, **model_kwargs)
                try:
                    resolved_converter = load_esm_tokenizer(hf_name)
                except Exception as exc:
                    raise EmbeddingBackendError(
                        f"Failed to load ESM2 tokenizer from transformers fallback: {exc}"
                    ) from exc

        if resolved_converter is None:
            if resolved_alphabet is not None:
                get_converter = getattr(resolved_alphabet, "get_batch_converter", None)
                if get_converter is None or not callable(get_converter):
                    raise EmbeddingBackendError("ESM2 alphabet does not expose get_batch_converter().")
                resolved_converter = get_converter()

        to_fn = getattr(resolved_model, "to", None)
        if resolved_torch_dtype is not None:
            move_model_to_device(
                resolved_model,
                device=str(device),
                dtype=resolved_torch_dtype,
                dtype_name=resolved_dtype_name,
            )
        elif callable(to_fn):
            to_fn(device)
        eval_fn = getattr(resolved_model, "eval", None)
        if callable(eval_fn):
            eval_fn()

        available_count = _resolve_esm2_layer_count(resolved_model)
        padding_idx = getattr(resolved_alphabet, "padding_idx", None) if resolved_alphabet is not None else None
        super().__init__(
            model_reference=resolved_name,
            preprocessor=Esm2Preprocessor(),
            tokenizer=Esm2TokenizerAdapter(
                resolved_converter if resolved_alphabet is not None else None,
                int(padding_idx) if padding_idx is not None else None,
                tokenizer=resolved_converter if resolved_alphabet is None else None,
                device=device,
            ),
            model=Esm2ModelAdapter(resolved_model, available_layer_count=available_count),
            postprocessor=Esm2Postprocessor(),
        )
        parameters: Dict[str, Any] = {
            "mode": "protein_to_embedding_only",
            "representation": "per-residue",
            "pooling": "none",
            "layer_indexing": "esm2_native_0_is_embedding_and_top_is_num_layers",
            "available_layer_count_hint": available_count + 1 if available_count >= 0 else None,
            "max_sequence_length": self.MAX_SEQUENCE_LENGTH,
        }
        if resolved_dtype_name is not None:
            parameters["torch_dtype"] = resolved_dtype_name

        self.model_metadata = ModelMetadata(
            provider="esm-pretrained",
            model_name=resolved_name,
            model_reference=resolved_name,
            device=str(device),
            framework_versions=framework_versions("esm", "torch"),
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
        return self._generate_from_batched_layer_output_map(
            records,
            layer_index=layer_index,
            pooler=pooler,
            fail_fast=fail_fast,
            missing_layers_error="ESM2 model output missing layers dictionary.",
            requested_layers=normalize_requested_layers(layer_index),
            run_parameters={
                "model_reference": self.model_reference,
                "mode": "protein_to_embedding_only",
            },
        )


def _resolve_layer_indices(layer_index: int | Sequence[int] | None, *, total_layers: int) -> List[int]:
    if total_layers < 0:
        raise EmbeddingBackendError("ESM2 reported invalid total layer count.")

    if layer_index is None:
        return list(range(total_layers + 1))

    if isinstance(layer_index, int):
        requested = [layer_index]
    else:
        requested = [int(v) for v in layer_index]
        if not requested:
            raise EmbeddingInputError("layer_index sequence cannot be empty.")

    resolved: List[int] = []
    for idx in requested:
        if idx < 0:
            idx = (total_layers + 1) + idx
        if idx < 0 or idx > total_layers:
            raise EmbeddingInputError(
                f"Requested layer index {idx} out of range for ESM2 total layers={total_layers}."
            )
        resolved.append(idx)
    return sorted(set(resolved))


def _esm_sample_spans_from_lens(lens: Any) -> List[tuple[int, int]]:
    try:
        values = lens.tolist()
    except Exception:
        values = [lens[0].item()]
    if isinstance(values, (int, float)):
        values = [values]
    spans: List[tuple[int, int]] = []
    for value in cast(Sequence[object], values):
        tokens_len = int(cast(Any, value))
        spans.append((1, max(tokens_len - 1, 1)))
    return spans


def _esm_sample_spans_from_attention_mask(attention_mask: Any) -> List[tuple[int, int]]:
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


def _resolve_esm2_layer_count(model: Any) -> int:
    num_layers = getattr(model, "num_layers", None)
    if isinstance(num_layers, int):
        return int(num_layers)
    config = getattr(model, "config", None)
    if config is not None:
        num_hidden_layers = getattr(config, "num_hidden_layers", None)
        if isinstance(num_hidden_layers, int):
            return int(num_hidden_layers)
    args = getattr(model, "args", None)
    if args is not None:
        value = getattr(args, "layers", None)
        if isinstance(value, int):
            return int(value)
    raise EmbeddingBackendError("Could not infer ESM2 num_layers from model.")


def _resolve_esm2_hf_model_name(model_name: str) -> str:
    normalized = str(model_name).strip().lower()
    if normalized.startswith("facebook/"):
        return str(model_name).strip()
    resolved = ESM2_HF_MODEL_NAMES.get(normalized)
    if resolved is None:
        raise EmbeddingInputError(f"Unsupported ESM2 Hugging Face fallback model name: {model_name!r}.")
    return resolved



__all__ = [
    "ESM2_PRETRAINED_LOADERS",
    "Esm2Preprocessor",
    "Esm2TokenizerAdapter",
    "Esm2ModelAdapter",
    "Esm2Postprocessor",
    "Esm2EmbeddingGenerator",
]
