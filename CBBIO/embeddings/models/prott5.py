"""ProtT5-specific embedding adapters and generator."""

from __future__ import annotations

from typing import Any, Dict, List, Sequence, cast

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


class ProtT5Preprocessor(BasePreprocessor):
    """Preprocessing used by ProtT5 models.

    Uppercases the sequence, replaces uncommon amino acids (U, Z, O, B)
    with ``X``, and inserts whitespace between residues.
    """

    def __init__(self) -> None:
        super().__init__(context="ProtT5 preprocessing", spacing=True)


class ProtT5TokenizerAdapter(TokenizerAdapter):
    """Tokenizer adapter for ProtT5-like tokenizers."""

    def __init__(self, tokenizer: Any, *, device: str = "cpu") -> None:
        self.tokenizer = tokenizer
        self.device = str(device)

    def tokenize(self, sequence: str) -> Any:
        return self.tokenize_many([sequence])

    def tokenize_many(self, sequences: Sequence[str]) -> Any:
        try:
            import torch  # type: ignore
        except ModuleNotFoundError as exc:
            raise EmbeddingDependencyError(
                "PyTorch is required for ProtT5 tokenization. Install with: pip install torch"
            ) from exc
        if not sequences:
            raise EmbeddingInputError("ProtT5 tokenization requires at least one sequence.")

        # transformers>=5 tokenizers commonly use __call__(..., return_tensors="pt")
        # while older versions expose batch_encode_plus.
        if callable(self.tokenizer):
            encoded = self.tokenizer(
                list(sequences),
                add_special_tokens=True,
                padding="longest",
                return_tensors="pt",
            )
            encoded_map = cast(Dict[str, Any], encoded)
            input_ids = encoded_map["input_ids"].to(self.device)
            attention_mask = encoded_map["attention_mask"].to(self.device)
            return {"input_ids": input_ids, "attention_mask": attention_mask}

        batch_encode_plus = getattr(self.tokenizer, "batch_encode_plus", None)
        if callable(batch_encode_plus):
            ids = batch_encode_plus(
                list(sequences),
                add_special_tokens=True,
                padding="longest",
            )
            ids_map = cast(Dict[str, Any], ids)
            tensor_fn = getattr(torch, "tensor")
            input_ids = tensor_fn(ids_map["input_ids"]).to(self.device)
            attention_mask = tensor_fn(ids_map["attention_mask"]).to(self.device)
            return {"input_ids": input_ids, "attention_mask": attention_mask}

        raise EmbeddingBackendError(
            "Tokenizer does not support call(...) or batch_encode_plus(...)."
        )


class ProtT5ModelAdapter(ModelAdapter):
    """Model adapter for ProtT5 encoder models."""

    def __init__(
        self,
        model: Any,
        *,
        device: str = "cpu",
        dtype: Any | None = None,
        dtype_name: str | None = None,
    ) -> None:
        self.model = model
        self.device = str(device)
        self.dtype = dtype
        self.dtype_name = dtype_name
        move_model_to_device(self.model, device=self.device, dtype=self.dtype, dtype_name=self.dtype_name)
        eval_fn = getattr(self.model, "eval", None)
        if callable(eval_fn):
            eval_fn()

    def infer(self, tokens: Any, *, layer_index: int | Sequence[int] | None = None) -> Any:
        if not isinstance(tokens, dict):
            raise EmbeddingInputError("ProtT5ModelAdapter expects tokenized input as a dict.")
        if "input_ids" not in tokens or "attention_mask" not in tokens:
            raise EmbeddingInputError("Token dict must contain 'input_ids' and 'attention_mask'.")
        token_map = cast(Dict[str, Any], tokens)

        try:
            import torch  # type: ignore
        except ModuleNotFoundError as exc:
            raise EmbeddingDependencyError(
                "PyTorch is required for ProtT5 inference. Install with: pip install torch"
            ) from exc

        context_factory = getattr(torch, "inference_mode", None)
        if not callable(context_factory):
            context_factory = getattr(torch, "no_grad")
        with cast(Any, context_factory)():
            model_output = self.model(
                input_ids=token_map["input_ids"],
                attention_mask=token_map["attention_mask"],
                output_hidden_states=True,
                return_dict=True,
            )
        hidden_states = getattr(model_output, "hidden_states", None)
        if hidden_states is None:
            raise EmbeddingBackendError("ProtT5 model output did not include hidden_states.")

        total_layers = len(hidden_states)
        layer_indices = _resolve_layer_indices(layer_index, total_layers=total_layers)
        residue_lens = _residue_lengths_from_mask(token_map["attention_mask"])

        selected_layers: Dict[int, Any] = {}
        if len(residue_lens) == 1:
            residue_len = residue_lens[0]
            for user_idx, hf_idx in layer_indices:
                selected_layers[user_idx] = hidden_states[hf_idx][0, :residue_len]
            return {"layers": selected_layers, "residue_len": residue_len}

        for user_idx, hf_idx in layer_indices:
            selected_layers[user_idx] = hidden_states[hf_idx]
        return {"layers": selected_layers, "residue_lens": residue_lens}

    def available_layers(self) -> List[int] | None:
        total_layers = infer_total_layers_from_model(self.model)
        if total_layers is None:
            return None
        return list(range(total_layers))


class ProtT5Postprocessor(DefaultPostprocessor):
    """Postprocessing for ProtT5 outputs without pooling."""


class ProtT5EmbeddingGenerator(EmbeddingGenerator):
    """Concrete embedding generator for ProtT5 models."""

    GENERATOR_CLASS = "protT5"
    GENERATOR_ALIASES = ("prott5", "prot_t5", "prot-t5", "Prot-T5")
    DEFAULT_MODEL_NAME = "Rostlab/prot_t5_xl_uniref50"
    FAMILY_MODELS = [DEFAULT_MODEL_NAME]
    SUPPORTED_POOLERS = ("none", "mean")

    def __init__(
        self,
        *,
        model_name: str = DEFAULT_MODEL_NAME,
        device: str = "cpu",
        dtype: str | None = None,
        tokenizer: Any | None = None,
        model: Any | None = None,
    ) -> None:
        resolved_tokenizer: Any | None = tokenizer
        resolved_model: Any | None = model
        resolved_dtype_name = normalize_torch_dtype_name(dtype)
        resolved_torch_dtype = resolve_torch_dtype(resolved_dtype_name) if resolved_dtype_name is not None else None

        if resolved_tokenizer is None or resolved_model is None:
            try:
                from transformers import AutoConfig, T5EncoderModel, T5Tokenizer  # type: ignore
            except ModuleNotFoundError as exc:
                raise EmbeddingDependencyError(
                    "transformers is required for ProtT5 loading. Install with: pip install transformers"
                ) from exc

            if resolved_tokenizer is None:
                tokenizer_cls = cast(Any, T5Tokenizer)
                resolved_tokenizer = tokenizer_cls.from_pretrained(model_name, do_lower_case=False)
            if resolved_model is None:
                config = cast(Any, AutoConfig).from_pretrained(model_name)
                # Silence tied-weights warning for ProtT5 checkpoints with both shared and encoder embeds present.
                setattr(config, "tie_word_embeddings", False)
                model_kwargs: Dict[str, Any] = {"config": config}
                if resolved_torch_dtype is not None:
                    model_kwargs["torch_dtype"] = resolved_torch_dtype
                resolved_model = cast(Any, T5EncoderModel).from_pretrained(model_name, **model_kwargs)

        super().__init__(
            model_reference=model_name,
            preprocessor=ProtT5Preprocessor(),
            tokenizer=ProtT5TokenizerAdapter(resolved_tokenizer, device=device),
            model=ProtT5ModelAdapter(
                resolved_model,
                device=device,
                dtype=resolved_torch_dtype,
                dtype_name=resolved_dtype_name,
            ),
            postprocessor=ProtT5Postprocessor(),
        )
        parameters: Dict[str, Any] = {
            "representation": "per-residue",
            "pooling": "none",
            "layer_indexing": "hf_native_0_is_first_hidden",
        }
        if resolved_dtype_name is not None:
            parameters["torch_dtype"] = resolved_dtype_name

        self.model_metadata = ModelMetadata(
            provider="huggingface-transformers",
            model_name=model_name,
            model_reference=model_name,
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
        """Generate per-residue embeddings for selected layers.

        - ``layer_index=None`` returns all layers.
        - ``layer_index=int`` returns that layer.
        - ``layer_index=Sequence[int]`` returns those layers.
        """
        requested_layers: int | list[int] | None
        if layer_index is None or isinstance(layer_index, int):
            requested_layers = layer_index
        else:
            requested_layers = [int(value) for value in layer_index]
        return self._generate_from_batched_layer_output_map(
            records,
            layer_index=layer_index,
            pooler=pooler,
            fail_fast=fail_fast,
            missing_layers_error="ProtT5 model output missing layers dictionary.",
            requested_layers=normalize_requested_layers(requested_layers),
            run_parameters={"model_reference": self.model_reference},
        )

    def _generate(
        self,
        records: Sequence[GenerationInput],
        *,
        layer_index: int | Sequence[int] | None,
        fail_fast: bool,
        pooler: PoolerInput,
    ) -> GenerationResult:
        return self.generate(
            records,
            layer_index=layer_index,
            pooler=pooler,
            fail_fast=fail_fast,
        )


def _resolve_layer_indices(
    layer_index: int | Sequence[int] | None,
    *,
    total_layers: int,
) -> List[tuple[int, int]]:
    if total_layers < 1:
        raise EmbeddingBackendError("ProtT5 output reported zero layers.")

    if layer_index is None:
        return [(idx, idx) for idx in range(total_layers)]

    if isinstance(layer_index, int):
        indices = [layer_index]
    else:
        indices = [int(value) for value in layer_index]
        if not indices:
            raise EmbeddingInputError("layer_index sequence cannot be empty.")

    resolved_user: List[int] = []
    for idx in indices:
        if idx < 0:
            idx = total_layers + idx
        if idx < 0 or idx >= total_layers:
            raise EmbeddingInputError(
                f"Requested layer index {idx} out of range for total layers={total_layers}."
            )
        resolved_user.append(idx)

    unique_sorted_user = sorted(set(resolved_user))
    return [(user_idx, user_idx) for user_idx in unique_sorted_user]


def _residue_lengths_from_mask(attention_mask: Any) -> List[int]:
    try:
        rows = attention_mask.tolist()
    except Exception:
        try:
            rows = [attention_mask[0].tolist()]
        except Exception as nested_exc:
            raise EmbeddingBackendError("Could not infer valid token lengths from attention_mask.") from nested_exc

    if rows and isinstance(rows[0], (int, float)):
        rows = [rows]

    lengths: List[int] = []
    for row in cast(Sequence[Sequence[object]], rows):
        valid_len = int(sum(int(cast(Any, value)) for value in row))
        if valid_len < 1:
            raise EmbeddingBackendError("No valid tokens produced by ProtT5 model.")
        # T5 tokenization adds an end token. Keep residues only.
        lengths.append(max(valid_len - 1, 1))
    if not lengths:
        raise EmbeddingBackendError("No token rows produced by ProtT5 tokenizer.")
    return lengths


__all__ = [
    "ProtT5Preprocessor",
    "ProtT5TokenizerAdapter",
    "ProtT5ModelAdapter",
    "ProtT5Postprocessor",
    "ProtT5EmbeddingGenerator",
]
