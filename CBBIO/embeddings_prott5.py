"""ProtT5-specific embedding adapters and generator."""

from __future__ import annotations

from importlib import metadata as importlib_metadata
import re
import uuid
from typing import Any, Dict, List, Sequence

from .embeddings import (
    EmbeddingBackendError,
    EmbeddingDependencyError,
    EmbeddingRecord,
    EmbeddingGenerator,
    EmbeddingGenerationError,
    EmbeddingInputError,
    GenerationInput,
    GenerationResult,
    ModelMetadata,
    ModelAdapter,
    PostprocessorAdapter,
    PreprocessorAdapter,
    RunMetadata,
    TokenizerAdapter,
    _as_float_vector,
    _normalize_generation_exception,
    _validate_generation_input,
    _validate_sequence,
    utc_now_iso,
)


class ProtT5Preprocessor(PreprocessorAdapter):
    """Preprocessing used by ProtT5 models.

    It uppercases the sequence, replaces uncommon amino acids (U, Z, O, B)
    with ``X``, and inserts whitespace between residues.
    """

    def preprocess(self, raw_sequence: str) -> str:
        sequence = str(raw_sequence).strip().upper()
        _validate_sequence(sequence, context="ProtT5 preprocessing")
        replaced = re.sub(r"[UZOB]", "X", sequence)
        return " ".join(list(replaced))


class ProtT5TokenizerAdapter(TokenizerAdapter):
    """Tokenizer adapter for ProtT5-like tokenizers."""

    def __init__(self, tokenizer: Any, *, device: str = "cpu") -> None:
        self.tokenizer = tokenizer
        self.device = str(device)

    def tokenize(self, sequence: str) -> Any:
        try:
            import torch  # type: ignore
        except ModuleNotFoundError as exc:
            raise EmbeddingDependencyError(
                "PyTorch is required for ProtT5 tokenization. Install with: pip install torch"
            ) from exc

        # transformers>=5 tokenizers commonly use __call__(..., return_tensors="pt")
        # while older versions expose batch_encode_plus.
        if callable(self.tokenizer):
            encoded = self.tokenizer(
                [sequence],
                add_special_tokens=True,
                padding="longest",
                return_tensors="pt",
            )
            input_ids = encoded["input_ids"].to(self.device)
            attention_mask = encoded["attention_mask"].to(self.device)
            return {"input_ids": input_ids, "attention_mask": attention_mask}

        batch_encode_plus = getattr(self.tokenizer, "batch_encode_plus", None)
        if callable(batch_encode_plus):
            ids = batch_encode_plus(
                [sequence],
                add_special_tokens=True,
                padding="longest",
            )
            input_ids = torch.tensor(ids["input_ids"]).to(self.device)
            attention_mask = torch.tensor(ids["attention_mask"]).to(self.device)
            return {"input_ids": input_ids, "attention_mask": attention_mask}

        raise EmbeddingBackendError(
            "Tokenizer does not support call(...) or batch_encode_plus(...)."
        )


class ProtT5ModelAdapter(ModelAdapter):
    """Model adapter for ProtT5 encoder models."""

    def __init__(self, model: Any, *, device: str = "cpu") -> None:
        self.model = model
        self.device = str(device)
        to_fn = getattr(self.model, "to", None)
        if callable(to_fn):
            to_fn(self.device)
        eval_fn = getattr(self.model, "eval", None)
        if callable(eval_fn):
            eval_fn()

    def infer(self, tokens: Any, *, layer_index: int | Sequence[int] | None = None) -> Any:
        if not isinstance(tokens, dict):
            raise EmbeddingInputError("ProtT5ModelAdapter expects tokenized input as a dict.")
        if "input_ids" not in tokens or "attention_mask" not in tokens:
            raise EmbeddingInputError("Token dict must contain 'input_ids' and 'attention_mask'.")

        try:
            import torch  # type: ignore
        except ModuleNotFoundError as exc:
            raise EmbeddingDependencyError(
                "PyTorch is required for ProtT5 inference. Install with: pip install torch"
            ) from exc

        with torch.no_grad():
            model_output = self.model(
                input_ids=tokens["input_ids"],
                attention_mask=tokens["attention_mask"],
                output_hidden_states=True,
                return_dict=True,
            )
        hidden_states = getattr(model_output, "hidden_states", None)
        if hidden_states is None:
            raise EmbeddingBackendError("ProtT5 model output did not include hidden_states.")

        total_layers = len(hidden_states)
        layer_indices = _resolve_layer_indices(layer_index, total_layers=total_layers)
        residue_len = _residue_length_from_mask(tokens["attention_mask"])

        selected_layers: Dict[int, Any] = {}
        # Important convention:
        # BioData stores Prot-T5 layers in reverse order where layer 0 is the
        # final hidden layer. This adapter follows that external convention:
        # user-facing layer index 0 -> HF hidden_states[-1].
        for user_idx, hf_idx in layer_indices:
            selected_layers[user_idx] = hidden_states[hf_idx][0, :residue_len]
        return {"layers": selected_layers, "residue_len": residue_len}

    def available_layers(self) -> List[int] | None:
        total_layers = _infer_total_layers_from_model(self.model)
        if total_layers is None:
            return None
        return list(range(total_layers))


class ProtT5Postprocessor(PostprocessorAdapter):
    """Postprocessing for ProtT5 outputs without pooling."""

    def postprocess(self, model_output: Any) -> Sequence[float]:
        if not isinstance(model_output, dict):
            raise EmbeddingBackendError("ProtT5Postprocessor expects a dict payload from model adapter.")
        layers_obj = model_output.get("layers")
        if not isinstance(layers_obj, dict):
            raise EmbeddingBackendError("ProtT5Postprocessor expects a 'layers' dict in model output.")
        if not layers_obj:
            raise EmbeddingBackendError("ProtT5Postprocessor received no layers.")

        first_key = sorted(layers_obj.keys())[0]
        layer_tensor = layers_obj[first_key]
        if hasattr(layer_tensor, "tolist") and callable(layer_tensor.tolist):
            values = layer_tensor.tolist()
            if isinstance(values, list):
                # Preserve per-residue structure and numeric coercion.
                return [[float(col) for col in row] for row in values if isinstance(row, list)]

        # Fallback path for non-tensor-compatible objects.
        rows: List[List[float]] = []
        for row in layer_tensor:
            rows.append(_as_float_vector(row))
        return rows


class ProtT5EmbeddingGenerator(EmbeddingGenerator):
    """Concrete embedding generator for ProtT5 models."""

    GENERATOR_CLASS = "protT5"
    GENERATOR_ALIASES = ("prott5", "prot_t5", "prot-t5", "Prot-T5")
    DEFAULT_MODEL_NAME = "Rostlab/prot_t5_xl_uniref50"
    FAMILY_MODELS = [DEFAULT_MODEL_NAME]

    def __init__(
        self,
        *,
        model_name: str = DEFAULT_MODEL_NAME,
        device: str = "cpu",
        tokenizer: Any | None = None,
        model: Any | None = None,
    ) -> None:
        resolved_tokenizer = tokenizer
        resolved_model = model

        if resolved_tokenizer is None or resolved_model is None:
            try:
                from transformers import AutoConfig, T5EncoderModel, T5Tokenizer  # type: ignore
            except ModuleNotFoundError as exc:
                raise EmbeddingDependencyError(
                    "transformers is required for ProtT5 loading. Install with: pip install transformers"
                ) from exc

            if resolved_tokenizer is None:
                resolved_tokenizer = T5Tokenizer.from_pretrained(model_name, do_lower_case=False)
            if resolved_model is None:
                config = AutoConfig.from_pretrained(model_name)
                # Silence tied-weights warning for ProtT5 checkpoints with both shared and encoder embeds present.
                setattr(config, "tie_word_embeddings", False)
                resolved_model = T5EncoderModel.from_pretrained(model_name, config=config)

        super().__init__(
            model_reference=model_name,
            preprocessor=ProtT5Preprocessor(),
            tokenizer=ProtT5TokenizerAdapter(resolved_tokenizer, device=device),
            model=ProtT5ModelAdapter(resolved_model, device=device),
            postprocessor=ProtT5Postprocessor(),
        )
        self.model_metadata = ModelMetadata(
            provider="huggingface-transformers",
            model_name=model_name,
            model_reference=model_name,
            model_revision=_extract_revision(resolved_model),
            tokenizer_name=_extract_name_or_path(resolved_tokenizer),
            tokenizer_revision=_extract_revision(resolved_tokenizer),
            device=str(device),
            framework_versions=_framework_versions(),
            parameters={
                "representation": "per-residue",
                "pooling": "none",
                "layer_indexing": "biodata_reversed_0_is_last_hidden",
            },
        )

    def generate(
        self,
        records: Sequence[GenerationInput],
        *,
        layer_index: int | Sequence[int] | None = None,
        fail_fast: bool = False,
    ) -> GenerationResult:
        """Generate per-residue embeddings for selected layers.

        - ``layer_index=None`` returns all layers.
        - ``layer_index=int`` returns that layer.
        - ``layer_index=Sequence[int]`` returns those layers.
        """
        result = GenerationResult()

        for index, record in enumerate(records):
            try:
                normalized = _validate_generation_input(record, index=index)
                prepared = self.preprocessor.preprocess(normalized.sequence)
                tokenized = self.tokenizer.tokenize(prepared)
                model_output = self.model.infer(tokenized, layer_index=layer_index)

                layers_obj = model_output.get("layers") if isinstance(model_output, dict) else None
                if not isinstance(layers_obj, dict):
                    raise EmbeddingBackendError("ProtT5 model output missing layers dictionary.")

                for layer_id, layer_tensor in sorted(layers_obj.items(), key=lambda item: int(item[0])):
                    matrix = _as_matrix(layer_tensor)
                    if not matrix:
                        raise EmbeddingBackendError(f"ProtT5 returned empty matrix for layer {layer_id}.")
                    hidden_dim = len(matrix[0]) if matrix else 0
                    for row in matrix:
                        if len(row) != hidden_dim:
                            raise EmbeddingBackendError(
                                f"Inconsistent row length in layer {layer_id}: expected {hidden_dim}."
                            )

                    result.records.append(
                        EmbeddingRecord(
                            id=normalized.id,
                            embedding=matrix,
                            layer_index=int(layer_id),
                            model_reference=self.model_reference,
                            shape=(len(matrix), hidden_dim),
                            metadata=normalized.metadata,
                        )
                    )
            except Exception as exc:
                normalized_exc: EmbeddingGenerationError = _normalize_generation_exception(exc)
                if fail_fast:
                    raise normalized_exc
                result.errors.append(
                    {
                        "index": index,
                        "id": getattr(record, "id", None),
                        "error_type": normalized_exc.__class__.__name__,
                        "message": str(normalized_exc),
                    }
                )

        resolved_layers = sorted({int(record.layer_index) for record in result.records}) or None
        run_metadata = RunMetadata(
            run_id=str(uuid.uuid4()),
            created_at_utc=utc_now_iso(),
            sequence_count=len(records),
            requested_layers=_normalize_requested_layers(layer_index),
            resolved_layers=resolved_layers,
            failure_count=len(result.errors),
            parameters={"model_reference": self.model_reference},
        )
        return GenerationResult(
            records=result.records,
            errors=result.errors,
            skipped=result.skipped,
            model_metadata=self.model_metadata,
            run_metadata=run_metadata,
        )


def _resolve_layer_indices(
    layer_index: int | Sequence[int] | None,
    *,
    total_layers: int,
) -> List[tuple[int, int]]:
    if total_layers < 1:
        raise EmbeddingBackendError("ProtT5 output reported zero layers.")

    if layer_index is None:
        return [(idx, _to_hf_layer_index(idx, total_layers=total_layers)) for idx in range(total_layers)]

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
    return [
        (user_idx, _to_hf_layer_index(user_idx, total_layers=total_layers))
        for user_idx in unique_sorted_user
    ]


def _to_hf_layer_index(user_layer_index: int, *, total_layers: int) -> int:
    # External (BioData) convention: 0 means last hidden layer.
    # Hugging Face convention: last hidden layer is index total_layers - 1.
    return (total_layers - 1) - int(user_layer_index)


def _residue_length_from_mask(attention_mask: Any) -> int:
    try:
        valid_len = int(attention_mask[0].sum().item())
    except Exception as exc:
        raise EmbeddingBackendError("Could not infer valid token length from attention_mask.") from exc
    if valid_len < 1:
        raise EmbeddingBackendError("No valid tokens produced by ProtT5 model.")
    # T5 tokenization adds an end token. Keep residues only.
    return max(valid_len - 1, 1)


def _as_matrix(layer_tensor: Any) -> List[List[float]]:
    if hasattr(layer_tensor, "tolist") and callable(layer_tensor.tolist):
        as_list = layer_tensor.tolist()
    else:
        as_list = layer_tensor

    if not isinstance(as_list, list):
        raise EmbeddingBackendError("Layer tensor could not be converted to a row-major list.")

    rows: List[List[float]] = []
    for row in as_list:
        rows.append(_as_float_vector(row))
    return rows


def _normalize_requested_layers(layer_index: int | Sequence[int] | None) -> List[int] | None:
    if layer_index is None:
        return None
    if isinstance(layer_index, int):
        return [int(layer_index)]
    return [int(value) for value in layer_index]


def _framework_versions() -> Dict[str, str]:
    versions: Dict[str, str] = {}
    for package_name in ("transformers", "torch"):
        try:
            versions[package_name] = importlib_metadata.version(package_name)
        except importlib_metadata.PackageNotFoundError:
            continue
    return versions


def _extract_name_or_path(obj: Any) -> str | None:
    value = getattr(obj, "name_or_path", None)
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _extract_revision(obj: Any) -> str | None:
    config = getattr(obj, "config", None)
    if config is None:
        return None
    revision = getattr(config, "_commit_hash", None)
    if revision is None:
        return None
    text = str(revision).strip()
    return text or None


def _infer_total_layers_from_model(model: Any) -> int | None:
    config = getattr(model, "config", None)
    if config is None:
        return None

    # T5 family commonly exposes encoder block count as num_layers.
    num_layers = getattr(config, "num_layers", None)
    if isinstance(num_layers, int) and num_layers >= 1:
        # hidden_states includes token embeddings + encoder block outputs.
        return int(num_layers) + 1

    num_hidden = getattr(config, "num_hidden_layers", None)
    if isinstance(num_hidden, int) and num_hidden >= 1:
        return int(num_hidden) + 1
    return None


__all__ = [
    "ProtT5Preprocessor",
    "ProtT5TokenizerAdapter",
    "ProtT5ModelAdapter",
    "ProtT5Postprocessor",
    "ProtT5EmbeddingGenerator",
]
