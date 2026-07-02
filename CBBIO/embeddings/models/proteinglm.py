"""ProteinGLM MLM Hugging Face embedding generators."""

from __future__ import annotations

from contextlib import contextmanager
import importlib
import sys
import types
from typing import Any, Dict, List, Sequence, cast

from .. import (
    EmbeddingBackendError,
    EmbeddingDependencyError,
    EmbeddingInputError,
    GenerationInput,
    GenerationResult,
)
from ..utils.pooler import PoolerInput
from ._esm_hf import (
    HfEsmEmbeddingGenerator,
    HfEsmModelAdapter,
    HfEsmPostprocessor,
    HfEsmPreprocessor,
    HfEsmTokenizerAdapter,
    infer_hf_esm_transformer_layers,
    resolve_hf_esm_layer_indices,
    resolve_model_name,
)


PROTEINGLM_HF_MODEL_NAMES: Dict[str, str] = {
    "proteinglm_1b": "biomap-research/proteinglm-1b-mlm",
    "proteinglm_1b_mlm": "biomap-research/proteinglm-1b-mlm",
    "proteinglm-1b-mlm": "biomap-research/proteinglm-1b-mlm",
    "pglm_1b": "biomap-research/proteinglm-1b-mlm",
    "pglm_1b_mlm": "biomap-research/proteinglm-1b-mlm",
    "pglm-1b-mlm": "biomap-research/proteinglm-1b-mlm",
    "1b": "biomap-research/proteinglm-1b-mlm",
    "1b_mlm": "biomap-research/proteinglm-1b-mlm",
    "bo1015/proteinglm-1b-mlm": "biomap-research/proteinglm-1b-mlm",
    "proteinglm_3b": "biomap-research/proteinglm-3b-mlm",
    "proteinglm_3b_mlm": "biomap-research/proteinglm-3b-mlm",
    "proteinglm-3b-mlm": "biomap-research/proteinglm-3b-mlm",
    "pglm_3b": "biomap-research/proteinglm-3b-mlm",
    "pglm_3b_mlm": "biomap-research/proteinglm-3b-mlm",
    "pglm-3b-mlm": "biomap-research/proteinglm-3b-mlm",
    "3b": "biomap-research/proteinglm-3b-mlm",
    "3b_mlm": "biomap-research/proteinglm-3b-mlm",
    "bo1015/proteinglm-3b-mlm": "biomap-research/proteinglm-3b-mlm",
    "proteinglm_10b": "biomap-research/proteinglm-10b-mlm",
    "proteinglm_10b_mlm": "biomap-research/proteinglm-10b-mlm",
    "proteinglm-10b-mlm": "biomap-research/proteinglm-10b-mlm",
    "pglm_10b": "biomap-research/proteinglm-10b-mlm",
    "pglm_10b_mlm": "biomap-research/proteinglm-10b-mlm",
    "pglm-10b-mlm": "biomap-research/proteinglm-10b-mlm",
    "10b": "biomap-research/proteinglm-10b-mlm",
    "10b_mlm": "biomap-research/proteinglm-10b-mlm",
    "bo1015/proteinglm-10b-mlm": "biomap-research/proteinglm-10b-mlm",
}


class ProteinGlmPreprocessor(HfEsmPreprocessor):
    """Preprocessing for ProteinGLM MLM protein sequences."""

    def __init__(self, *, context: str = "ProteinGLM preprocessing") -> None:
        super().__init__(context=context)


class ProteinGlmTokenizerAdapter(HfEsmTokenizerAdapter):
    """Tokenizer adapter for ProteinGLM MLM tokenizers."""


class ProteinGlmModelAdapter(HfEsmModelAdapter):
    """Model adapter for ProteinGLM MLM checkpoints."""

    def infer(self, tokens: Any, *, layer_index: int | Sequence[int] | None = None) -> Any:
        """Run ProteinGLM inference and return hidden states."""
        if not isinstance(tokens, dict):
            raise EmbeddingInputError("ProteinGlmModelAdapter expects tokenized input as a dict.")
        token_map = cast(Dict[str, Any], tokens)
        if "input_ids" not in token_map or "attention_mask" not in token_map:
            raise EmbeddingInputError("Token dict must contain input_ids and attention_mask.")

        try:
            import torch  # type: ignore
        except ModuleNotFoundError as exc:
            raise EmbeddingDependencyError(
                "PyTorch is required for ProteinGLM inference. Install with: pip install torch"
            ) from exc

        total = self._total_layers()
        requested = resolve_hf_esm_layer_indices(layer_index, total_layers=total)
        final_layer_index = total if total > 0 else 0
        return_last_hidden_state = sorted(set(requested)) == [final_layer_index]
        with torch.inference_mode():
            out = self.model(
                input_ids=token_map["input_ids"],
                attention_mask=token_map["attention_mask"],
                output_hidden_states=True,
                return_last_hidden_state=return_last_hidden_state,
                return_dict=True,
            )
        hidden_states = getattr(out, "hidden_states", None)
        layers = _proteinglm_hidden_state_layers(
            hidden_states,
            requested=requested,
            total_layers=total,
            attention_mask=token_map["attention_mask"],
        )
        return {
            "layers": layers,
            "sample_spans": proteinglm_sample_spans_from_attention_mask(token_map["attention_mask"]),
        }

    def _total_layers(self) -> int:
        if self.available_layer_count is not None:
            return int(self.available_layer_count)
        total = infer_hf_esm_transformer_layers(self.model)
        if total is None:
            hidden_size = getattr(getattr(self.model, "config", None), "hidden_size", None)
            if isinstance(hidden_size, int):
                return 0
            raise EmbeddingBackendError("Could not infer ProteinGLM num_hidden_layers from model config.")
        return total


class ProteinGlmPostprocessor(HfEsmPostprocessor):
    """Postprocessing for ProteinGLM outputs without pooling."""


class ProteinGlmEmbeddingGenerator(HfEsmEmbeddingGenerator):
    """Concrete embedding generator for ProteinGLM MLM family models."""

    GENERATOR_CLASS = "proteinglm"
    GENERATOR_ALIASES = ("proteinpglm", "pglm", "proteinglm_mlm")
    MODEL_ALIASES = PROTEINGLM_HF_MODEL_NAMES
    DEFAULT_MODEL_NAME = "biomap-research/proteinglm-1b-mlm"
    FAMILY_MODELS = [
        "proteinglm_1b_mlm",
        "proteinglm_3b_mlm",
        "proteinglm_10b_mlm",
        "biomap-research/proteinglm-1b-mlm",
        "biomap-research/proteinglm-3b-mlm",
        "biomap-research/proteinglm-10b-mlm",
        "Bo1015/proteinglm-1b-mlm",
        "Bo1015/proteinglm-3b-mlm",
        "Bo1015/proteinglm-10b-mlm",
    ]
    SUPPORTED_POOLERS = ("none", "mean")

    def __init__(
        self,
        *,
        model_name: str = DEFAULT_MODEL_NAME,
        device: str = "cpu",
        dtype: str | None = None,
        model: Any | None = None,
        tokenizer: Any | None = None,
        from_pretrained_kwargs: Dict[str, Any] | None = None,
        tokenizer_from_pretrained_kwargs: Dict[str, Any] | None = None,
    ) -> None:
        model_reference = resolve_model_name(
            model_name,
            self.MODEL_ALIASES,
            family="ProteinGLM",
        )
        model_kwargs = {"trust_remote_code": True}
        model_kwargs.update(from_pretrained_kwargs or {})
        tokenizer_kwargs = {"use_fast": True}
        tokenizer_kwargs.update(tokenizer_from_pretrained_kwargs or {})
        with _proteinglm_deepspeed_checkpointing_stub():
            super().__init__(
                model_name=model_name,
                model_reference=model_reference,
                context="ProteinGLM preprocessing",
                provider="huggingface-transformers",
                device=device,
                dtype=dtype,
                model=model,
                tokenizer=tokenizer,
                tokenizer_trust_remote_code=True,
                from_pretrained_kwargs=model_kwargs,
                auto_model_class="AutoModelForMaskedLM",
                tokenizer_from_pretrained_kwargs=tokenizer_kwargs,
                preprocessor_adapter_cls=ProteinGlmPreprocessor,
                tokenizer_adapter_cls=ProteinGlmTokenizerAdapter,
                model_adapter_cls=ProteinGlmModelAdapter,
                postprocessor_adapter_cls=ProteinGlmPostprocessor,
            )
        if self.model_metadata.parameters is not None:
            self.model_metadata.parameters["trust_remote_code"] = True
            self.model_metadata.parameters["auto_model_class"] = "AutoModelForMaskedLM"
            self.model_metadata.parameters["tokenizer_use_fast"] = bool(tokenizer_kwargs.get("use_fast", False))
            self.model_metadata.parameters["special_tokens"] = "trailing_eos"
            self.model_metadata.parameters["pooling"] = "none"

    def generate(
        self,
        records: Sequence[GenerationInput],
        *,
        layer_index: int | Sequence[int] | None = -1,
        pooler: PoolerInput = None,
        fail_fast: bool = False,
    ) -> GenerationResult:
        """Generate embeddings with the ProteinGLM adapter."""
        return super().generate(
            records,
            layer_index=layer_index,
            pooler=pooler,
            fail_fast=fail_fast,
        )

def proteinglm_sample_spans_from_attention_mask(attention_mask: Any) -> List[tuple[int, int]]:
    """Return residue-token spans from a ProteinGLM attention mask."""
    try:
        rows = attention_mask.tolist()
    except Exception:
        rows = [attention_mask[0].tolist()]
    if rows and isinstance(rows[0], (int, float)):
        rows = [rows]
    spans: List[tuple[int, int]] = []
    for row in cast(Sequence[Sequence[object]], rows):
        valid_len = int(sum(int(cast(Any, value)) for value in row))
        spans.append((0, max(valid_len - 1, 0)))
    return spans


def _proteinglm_hidden_state_layers(
    hidden_states: Any,
    *,
    requested: Sequence[int],
    total_layers: int,
    attention_mask: Any,
) -> Dict[int, Any]:
    if hidden_states is None:
        raise EmbeddingBackendError("ProteinGLM output missing hidden_states.")

    if isinstance(hidden_states, (tuple, list)):
        hidden_state_layers = cast(Sequence[Any], hidden_states)
        if len(hidden_state_layers) <= total_layers:
            raise EmbeddingBackendError(
                f"ProteinGLM returned {len(hidden_state_layers)} hidden states; expected at least {total_layers + 1}."
            )
        return {idx: _batch_first_proteinglm_tensor(hidden_state_layers[idx], attention_mask) for idx in requested}

    shape = getattr(hidden_states, "shape", None)
    if shape is not None and len(shape) == 3:
        if list(requested) not in ([0], [total_layers]):
            raise EmbeddingInputError(
                "This ProteinGLM output exposes only the final hidden state; request layer_index=-1."
            )
        return {int(requested[0]): _batch_first_proteinglm_tensor(hidden_states, attention_mask)}

    raise EmbeddingBackendError("ProteinGLM hidden_states must be a layer sequence or a rank-3 tensor.")


def _batch_first_proteinglm_tensor(value: Any, attention_mask: Any) -> Any:
    shape = getattr(value, "shape", None)
    mask_shape = getattr(attention_mask, "shape", None)
    if shape is None or mask_shape is None or len(shape) != 3 or len(mask_shape) != 2:
        return value

    batch_size = int(mask_shape[0])
    token_count = int(mask_shape[1])
    if int(shape[0]) == token_count and int(shape[1]) == batch_size:
        transpose = getattr(value, "transpose", None)
        if callable(transpose):
            return transpose(0, 1)
        permute = getattr(value, "permute", None)
        if callable(permute):
            return permute(1, 0, 2)
        raise EmbeddingBackendError("ProteinGLM returned token-major hidden states that cannot be transposed.")
    return value


@contextmanager
def _proteinglm_deepspeed_checkpointing_stub() -> Any:
    try:
        importlib.import_module("deepspeed")
    except ImportError:
        installed_stub = False
    else:
        yield
        return

    if "deepspeed" in sys.modules:
        yield
        return

    checkpointing = types.SimpleNamespace(
        is_configured=lambda: False,
        checkpoint=None,
    )
    module = types.ModuleType("deepspeed")
    module.checkpointing = checkpointing  # type: ignore[attr-defined]
    sys.modules["deepspeed"] = module
    installed_stub = True
    try:
        yield
    finally:
        if installed_stub and sys.modules.get("deepspeed") is module:
            del sys.modules["deepspeed"]


__all__ = [
    "PROTEINGLM_HF_MODEL_NAMES",
    "ProteinGlmPreprocessor",
    "ProteinGlmTokenizerAdapter",
    "ProteinGlmModelAdapter",
    "ProteinGlmPostprocessor",
    "ProteinGlmEmbeddingGenerator",
    "proteinglm_sample_spans_from_attention_mask",
]
