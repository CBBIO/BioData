"""AMPLIFY-specific Hugging Face embedding generators."""

from __future__ import annotations

import importlib
from typing import Any, Dict, List, Sequence, cast

from .. import EmbeddingBackendError, EmbeddingDependencyError, EmbeddingInputError
from ._esm_hf import (
    esm_sample_spans_from_attention_mask,
    HfEsmEmbeddingGenerator,
    HfEsmModelAdapter,
    HfEsmPostprocessor,
    HfEsmPreprocessor,
    HfEsmTokenizerAdapter,
    resolve_model_name,
)


AMPLIFY_HF_MODEL_NAMES: Dict[str, str] = {
    "amplify_120m": "nvidia/AMPLIFY_120M",
    "amplify-120m": "nvidia/AMPLIFY_120M",
    "amplify120m": "nvidia/AMPLIFY_120M",
    "120m": "nvidia/AMPLIFY_120M",
    "amplify_350m": "nvidia/AMPLIFY_350M",
    "amplify-350m": "nvidia/AMPLIFY_350M",
    "amplify350m": "nvidia/AMPLIFY_350M",
    "350m": "nvidia/AMPLIFY_350M",
    "amplify_120m_chandar": "chandar-lab/AMPLIFY_120M",
    "chandar_amplify_120m": "chandar-lab/AMPLIFY_120M",
    "amplify_350m_chandar": "chandar-lab/AMPLIFY_350M",
    "chandar_amplify_350m": "chandar-lab/AMPLIFY_350M",
}


class AmplifyPreprocessor(HfEsmPreprocessor):
    """Preprocessing for AMPLIFY protein sequences."""

    def __init__(self, *, context: str = "AMPLIFY preprocessing") -> None:
        super().__init__(context=context)


class AmplifyTokenizerAdapter(HfEsmTokenizerAdapter):
    """Tokenizer adapter for Hugging Face AMPLIFY."""


class AmplifyModelAdapter(HfEsmModelAdapter):
    """Model adapter for Hugging Face AMPLIFY."""

    def infer(self, tokens: Any, *, layer_index: int | Sequence[int] | None = None) -> Any:
        """Run AMPLIFY model inference and return hidden states."""
        if not isinstance(tokens, dict):
            raise EmbeddingInputError("AmplifyModelAdapter expects tokenized input as a dict.")
        token_map = cast(Dict[str, Any], tokens)
        if "input_ids" not in token_map or "attention_mask" not in token_map:
            raise EmbeddingInputError("Token dict must contain input_ids and attention_mask.")

        try:
            import torch  # type: ignore
        except ModuleNotFoundError as exc:
            raise EmbeddingDependencyError(
                "PyTorch is required for AMPLIFY inference. Install with: pip install torch"
            ) from exc

        total = self._total_layers()
        requested = _resolve_amplify_layer_indices(layer_index, total_layers=total)
        attention_mask = token_map["attention_mask"]

        if _should_use_unpadded_amplify_forward(attention_mask):
            return _infer_unpadded_amplify_batch(
                self.model,
                input_ids=token_map["input_ids"],
                attention_mask=attention_mask,
                requested_layers=requested,
                total_layers=total,
                torch=torch,
            )

        additive_attention_mask = _amplify_additive_attention_mask(attention_mask, torch=torch)

        with torch.inference_mode():
            out = self.model(
                input_ids=token_map["input_ids"],
                attention_mask=additive_attention_mask,
                output_hidden_states=True,
                return_dict=True,
            )
        hidden_states = getattr(out, "hidden_states", None)
        if hidden_states is None:
            raise EmbeddingBackendError("AMPLIFY output missing hidden_states.")
        if len(hidden_states) < total:
            raise EmbeddingBackendError(
                f"AMPLIFY returned {len(hidden_states)} hidden states; expected at least {total}."
            )
        return {
            "layers": {idx: hidden_states[idx] for idx in requested},
            "sample_spans": esm_sample_spans_from_attention_mask(attention_mask),
        }

    def available_layers(self) -> List[int] | None:
        """Return layer indices exposed by the AMPLIFY model."""
        total = self._total_layers()
        return list(range(total))


class AmplifyPostprocessor(HfEsmPostprocessor):
    """Postprocessing for AMPLIFY outputs without pooling."""


class AmplifyEmbeddingGenerator(HfEsmEmbeddingGenerator):
    """Concrete embedding generator for AMPLIFY family models."""

    GENERATOR_CLASS = "amplify"
    GENERATOR_ALIASES = ("AMPLIFY", "amplify_120m", "amplify_350m")
    DEFAULT_MODEL_NAME = "nvidia/AMPLIFY_120M"
    FAMILY_MODELS = [
        "amplify_120m",
        "amplify_350m",
        "nvidia/AMPLIFY_120M",
        "nvidia/AMPLIFY_350M",
        "amplify_120m_chandar",
        "amplify_350m_chandar",
        "chandar-lab/AMPLIFY_120M",
        "chandar-lab/AMPLIFY_350M",
    ]
    MAX_SEQUENCE_LENGTH = 2048
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
        model_reference = resolve_model_name(model_name, AMPLIFY_HF_MODEL_NAMES, family="AMPLIFY")
        if model is None and _is_nvidia_amplify_model(model_reference):
            _ensure_transformer_engine_pytorch_available(model_reference)
        resolved_dtype = _resolve_amplify_dtype(dtype=dtype, device=device)
        model_kwargs = {"trust_remote_code": True}
        model_kwargs.update(from_pretrained_kwargs or {})
        super().__init__(
            model_name=model_name,
            model_reference=model_reference,
            context="AMPLIFY preprocessing",
            provider="huggingface-transformers",
            device=device,
            dtype=resolved_dtype,
            model=model,
            tokenizer=tokenizer,
            tokenizer_trust_remote_code=True,
            from_pretrained_kwargs=model_kwargs,
            max_sequence_length=self.MAX_SEQUENCE_LENGTH,
            preprocessor_adapter_cls=AmplifyPreprocessor,
            tokenizer_adapter_cls=AmplifyTokenizerAdapter,
            model_adapter_cls=AmplifyModelAdapter,
            postprocessor_adapter_cls=AmplifyPostprocessor,
        )
        if self.model_metadata.parameters is not None:
            self.model_metadata.parameters["trust_remote_code"] = True
            self.model_metadata.parameters["layer_indexing"] = "hf_native_0_is_first_hidden"
            self.model_metadata.parameters["available_layer_count_hint"] = len(self.available_layers())
            if dtype is None and resolved_dtype is not None:
                self.model_metadata.parameters["precision_policy"] = "amplify_cuda_default_bfloat16"


def _is_nvidia_amplify_model(model_reference: str) -> bool:
    return str(model_reference).strip().lower().startswith("nvidia/amplify_")


def _ensure_transformer_engine_pytorch_available(model_reference: str) -> None:
    try:
        importlib.import_module("transformer_engine.pytorch")
    except Exception:
        raise EmbeddingDependencyError(
            f"{model_reference} is NVIDIA's TransformerEngine-optimized AMPLIFY checkpoint, "
            "but transformer_engine.pytorch is not importable. A bare `transformer-engine` install is only "
            "the meta package. Install the PyTorch extension in this environment, for example "
            "`poetry run pip install --no-build-isolation 'transformer-engine[pytorch,core-cu13]==2.16.0'`, "
            "or use `amplify_350m_chandar` / `amplify_120m_chandar` for the upstream checkpoints."
        ) from None


def _resolve_amplify_dtype(*, dtype: str | None, device: str) -> str | None:
    device_text = str(device).strip().lower()
    if not device_text.startswith("cuda"):
        return dtype
    if dtype is None:
        return "bfloat16"
    dtype_text = str(dtype).strip().lower()
    if dtype_text in {"float32", "fp32", "float"}:
        raise EmbeddingInputError(
            "AMPLIFY on CUDA requires dtype='float16' or dtype='bfloat16' because its xFormers attention "
            "kernels do not support float32 on GPU."
        )
    return dtype


def _resolve_amplify_layer_indices(layer_index: int | Sequence[int] | None, *, total_layers: int) -> List[int]:
    if total_layers < 1:
        raise EmbeddingBackendError("AMPLIFY reported invalid total layer count.")
    if layer_index is None:
        return list(range(total_layers))
    if isinstance(layer_index, int):
        requested = [int(layer_index)]
    else:
        requested = [int(value) for value in layer_index]
        if not requested:
            raise EmbeddingInputError("layer_index sequence cannot be empty.")

    resolved: List[int] = []
    for index in requested:
        original = index
        if index < 0:
            index = total_layers + index
        if index < 0 or index >= total_layers:
            raise EmbeddingInputError(
                f"Requested layer index {original} out of range for AMPLIFY available layers=0..{total_layers - 1}."
            )
        resolved.append(index)
    return sorted(set(resolved))


def _amplify_additive_attention_mask(attention_mask: Any, *, torch: Any) -> Any:
    bool_mask = attention_mask.to(dtype=torch.bool)
    return torch.where(
        bool_mask,
        torch.tensor(0.0, device=attention_mask.device),
        torch.tensor(float("-inf"), device=attention_mask.device),
    )


def _should_use_unpadded_amplify_forward(attention_mask: Any) -> bool:
    is_cuda = bool(getattr(attention_mask, "is_cuda", False))
    if not is_cuda:
        return False
    try:
        rows = attention_mask.tolist()
    except Exception:
        return True
    if rows and isinstance(rows[0], (int, float, bool)):
        rows = [rows]
    for row in cast(Sequence[Sequence[object]], rows):
        valid_len = sum(int(cast(Any, value)) for value in row)
        if valid_len < len(row):
            return True
    return False


def _infer_unpadded_amplify_batch(
    model: Any,
    *,
    input_ids: Any,
    attention_mask: Any,
    requested_layers: Sequence[int],
    total_layers: int,
    torch: Any,
) -> Dict[str, Any]:
    valid_lengths = _valid_lengths_from_attention_mask(attention_mask)
    if not valid_lengths:
        raise EmbeddingBackendError("AMPLIFY token batch is empty.")

    layer_rows: Dict[int, List[Any]] = {int(index): [] for index in requested_layers}
    with torch.inference_mode():
        for row_index, valid_len in enumerate(valid_lengths):
            sample_input_ids = input_ids[row_index : row_index + 1, :valid_len]
            out = model(
                input_ids=sample_input_ids,
                attention_mask=None,
                output_hidden_states=True,
                return_dict=True,
            )
            hidden_states = getattr(out, "hidden_states", None)
            if hidden_states is None:
                raise EmbeddingBackendError("AMPLIFY output missing hidden_states.")
            if len(hidden_states) < total_layers:
                raise EmbeddingBackendError(
                    f"AMPLIFY returned {len(hidden_states)} hidden states; expected at least {total_layers}."
                )
            for layer_index in requested_layers:
                layer_rows[int(layer_index)].append(hidden_states[int(layer_index)][0])

    layers = {
        layer_index: torch.nn.utils.rnn.pad_sequence(rows, batch_first=True)
        for layer_index, rows in layer_rows.items()
    }
    return {
        "layers": layers,
        "sample_spans": [(1, max(valid_len - 1, 1)) for valid_len in valid_lengths],
    }


def _valid_lengths_from_attention_mask(attention_mask: Any) -> List[int]:
    try:
        rows = attention_mask.tolist()
    except Exception as exc:
        raise EmbeddingBackendError("Could not read AMPLIFY attention_mask lengths.") from exc
    if rows and isinstance(rows[0], (int, float, bool)):
        rows = [rows]
    lengths = [int(sum(int(cast(Any, value)) for value in row)) for row in cast(Sequence[Sequence[object]], rows)]
    for length in lengths:
        if length <= 0:
            raise EmbeddingBackendError(f"Invalid AMPLIFY attention mask length: {length}.")
    return lengths


__all__ = [
    "AMPLIFY_HF_MODEL_NAMES",
    "AmplifyPreprocessor",
    "AmplifyTokenizerAdapter",
    "AmplifyModelAdapter",
    "AmplifyPostprocessor",
    "AmplifyEmbeddingGenerator",
]
