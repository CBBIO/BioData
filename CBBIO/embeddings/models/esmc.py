"""ESM-C specific embedding adapters and generator (protein -> embedding)."""

from __future__ import annotations

from typing import Any, Dict, List, Sequence, cast
import warnings

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
    framework_versions,
    normalize_requested_layers,
)


ESMC_LAYER_SPECS: Dict[str, int] = {
    "esmc-6b-2024-12": 80,
    "esmc-600m-2024-12": 36,
    "esmc-300m-2024-12": 30,
    "esmc_6b": 80,
    "esmc_600m": 36,
    "esmc_300m": 30,
}


class EsmcPreprocessor(BasePreprocessor):
    """Preprocessing for ESM-C protein inputs."""

    def __init__(self) -> None:
        super().__init__(context="ESM-C preprocessing")


class EsmcTokenizerAdapter(TokenizerAdapter):
    """Tokenizer adapter for ESM-C SDK encode path."""

    def __init__(self, client: Any, *, protein_cls: Any | None = None) -> None:
        self.client = client
        self.protein_cls = protein_cls
        self._warned_about_batched_sdk = False

    def tokenize(self, sequence: str) -> Any:
        return self.tokenize_many([sequence])

    def tokenize_many(self, sequences: Sequence[str]) -> Any:
        if not sequences:
            raise EmbeddingInputError("ESM-C tokenization requires at least one sequence.")
        if len(sequences) > 1 and not self._warned_about_batched_sdk:
            warnings.warn(
                "ESM-C batched generation passes multiple encoded proteins to the ESM SDK client. "
                "If your installed SDK does not support batched logits, use batch_size=1 for ESM-C.",
                RuntimeWarning,
                stacklevel=3,
            )
            self._warned_about_batched_sdk = True
        protein_cls = self.protein_cls
        if protein_cls is None:
            try:
                from esm.sdk.api import ESMProtein  # type: ignore
            except ModuleNotFoundError as exc:
                raise EmbeddingDependencyError(
                    "ESM SDK is required for ESM-C tokenization. Install package providing esm.sdk.api."
                ) from exc
            protein_cls = ESMProtein

        tokens = [self.client.encode(protein_cls(sequence=sequence)) for sequence in sequences]
        return {"tokens": tokens[0] if len(tokens) == 1 else tokens, "residue_lens": [len(sequence) for sequence in sequences]}


class EsmcModelAdapter(ModelAdapter):
    """Model adapter for ESM-C SDK logits path."""

    def __init__(
        self,
        client: Any,
        *,
        logits_config_cls: Any | None = None,
        available_layer_count: int | None = None,
    ) -> None:
        self.client = client
        self.logits_config_cls = logits_config_cls
        self.available_layer_count = available_layer_count

    def infer(self, tokens: Any, *, layer_index: int | Sequence[int] | None = None) -> Any:
        token_payload = tokens
        residue_lens: List[int] | None = None
        batch_size = 1
        if isinstance(tokens, dict):
            token_map = cast(Dict[str, Any], tokens)
            token_payload = token_map.get("tokens")
            residue_lens_raw = token_map.get("residue_lens")
            if isinstance(residue_lens_raw, Sequence) and not isinstance(residue_lens_raw, (str, bytes, bytearray)):
                residue_lens = [int(cast(Any, value)) for value in cast(Sequence[object], residue_lens_raw)]
                batch_size = len(residue_lens)

        logits_config_cls = self.logits_config_cls
        if logits_config_cls is None:
            try:
                from esm.sdk.api import LogitsConfig  # type: ignore
            except ModuleNotFoundError as exc:
                raise EmbeddingDependencyError(
                    "ESM SDK is required for ESM-C logits config. Install package providing esm.sdk.api."
                ) from exc
            logits_config_cls = LogitsConfig

        requested_layers = _resolve_requested_layers(layer_index, available_layer_count=self.available_layer_count)
        selected: Dict[int, Any] = {}
        for requested_layer in requested_layers:
            output = self.client.logits(
                token_payload,
                logits_config_cls(
                    sequence=False,
                    return_embeddings=True,
                    return_hidden_states=True,
                    ith_hidden_layer=requested_layer,
                ),
            )
            layer_tensor = _layer_tensor_from_logits_output(output, batch_size=batch_size)
            selected[int(requested_layer)] = layer_tensor
        if residue_lens is None:
            residue_lens = [_infer_residue_length_from_layers(selected)]
        return {"layers": selected, "residue_lens": residue_lens}

    def available_layers(self) -> List[int] | None:
        if self.available_layer_count is None:
            return None
        return list(range(int(self.available_layer_count)))


class EsmcPostprocessor(DefaultPostprocessor):
    """Postprocessing for ESM-C outputs without pooling."""


class EsmcEmbeddingGenerator(EmbeddingGenerator):
    """Concrete embedding generator for ESM-C family models."""

    GENERATOR_CLASS = "esmc"
    GENERATOR_ALIASES = ("esm-c", "esmc3", "esm3c","ESM3c")
    DEFAULT_MODEL_NAME = "esmc_600m"
    FAMILY_MODELS = sorted(str(value) for value in ESMC_LAYER_SPECS.keys())
    SUPPORTED_POOLERS = ("none", "mean")

    def __init__(
        self,
        *,
        model_name: str = DEFAULT_MODEL_NAME,
        device: str = "cpu",
        use_flash_attention: bool | None = None,
        client: Any | None = None,
        from_pretrained_kwargs: Dict[str, Any] | None = None,
    ) -> None:
        resolved_client = client
        kwargs = dict(from_pretrained_kwargs or {})
        if use_flash_attention is not None:
            kwargs.setdefault("use_flash_attention", bool(use_flash_attention))

        if resolved_client is None:
            try:
                from esm.models.esmc import ESMC  # type: ignore
            except ModuleNotFoundError as exc:
                raise EmbeddingDependencyError(
                    "ESM SDK is required for ESM-C loading. Install package providing esm.models.esmc."
                ) from exc

            resolved_client = ESMC.from_pretrained(model_name, **kwargs).to(device)

        layer_count = _resolve_layer_count(model_name)
        super().__init__(
            model_reference=model_name,
            preprocessor=EsmcPreprocessor(),
            tokenizer=EsmcTokenizerAdapter(resolved_client),
            model=EsmcModelAdapter(resolved_client, available_layer_count=layer_count),
            postprocessor=EsmcPostprocessor(),
        )
        self.model_metadata = ModelMetadata(
            provider="esm-sdk",
            model_name=model_name,
            model_reference=model_name,
            tokenizer_name="esm.sdk.api.ESMProtein",
            device=str(device),
            framework_versions=framework_versions("esm", "torch"),
            parameters={
                "mode": "protein_to_embedding_only",
                "representation": "per-residue",
                "pooling": "none",
                "layer_indexing": "hf_native_0_is_first_hidden",
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


def _layer_tensor_from_logits_output(output: Any, *, batch_size: int = 1) -> Any:
    hidden_states = getattr(output, "hidden_states", None)
    if hidden_states is not None:
        return _single_layer_tensor(hidden_states, batch_size=batch_size)
    embeddings = getattr(output, "embeddings", None)
    if embeddings is not None:
        return _single_layer_tensor(embeddings, batch_size=batch_size)
    raise EmbeddingBackendError("ESM-C logits output did not include hidden_states or embeddings.")


def _single_layer_tensor(embeddings: Any, *, batch_size: int = 1) -> Any:
    """Normalize ESM-C hidden state output to [batch, residues, hidden] or [residues, hidden]."""

    shape = getattr(embeddings, "shape", None)
    shape_seq = cast(Sequence[Any], shape) if shape is not None else None
    ndim = len(shape_seq) if shape_seq is not None else None

    if ndim == 4:
        assert shape_seq is not None
        # Common SDK shapes for a requested single layer include
        # [1, batch, residues, hidden] or [batch, 1, residues, hidden].
        if int(shape_seq[0]) == 1:
            return embeddings[0]
        if int(shape_seq[1]) == 1:
            return embeddings[:, 0]
        raise EmbeddingBackendError(f"ESM-C hidden_states has ambiguous 4D shape: {tuple(shape_seq)!r}.")
    if ndim == 3:
        assert shape_seq is not None
        first_dim = int(shape_seq[0])
        if first_dim == 1 and batch_size == 1:
            return embeddings[0]
        if first_dim == batch_size:
            return embeddings
        if first_dim == 1:
            return embeddings[0]
        # Some SDK versions return [layers, residues, hidden] even with a
        # requested layer. In that case the requested layer is the first item.
        return embeddings[0]
    if ndim == 2:
        return embeddings

    tolist = getattr(embeddings, "tolist", None)
    values = tolist() if callable(tolist) else embeddings
    if isinstance(values, list):
        list_values = cast(List[Any], values)
        return _single_layer_list(list_values, batch_size=batch_size)
    raise EmbeddingBackendError("Could not normalize ESM-C embeddings output.")


def _single_layer_list(values: List[Any], *, batch_size: int) -> Any:
    if not values:
        return values
    first = values[0]
    if isinstance(first, list) and first and isinstance(first[0], list):
        first_row = cast(List[Any], first)
        if first_row and isinstance(first_row[0], list):
            if len(values) == batch_size:
                return values
            return values[0]
        return values
    return values


def _infer_residue_length_from_layers(layers: Dict[int, Any]) -> int:
    if not layers:
        return 0
    first = next(iter(layers.values()))
    shape = getattr(first, "shape", None)
    if shape is not None:
        shape_seq = cast(Sequence[Any], shape)
        if len(shape_seq) >= 2:
            return int(shape_seq[-2])
    tolist = getattr(first, "tolist", None)
    values = tolist() if callable(tolist) else first
    if isinstance(values, list):
        values_list = cast(List[Any], values)
        if values_list and isinstance(values_list[0], list):
            first_row = cast(List[Any], values_list[0])
            if first_row and isinstance(first_row[0], list):
                return len(first_row)
        return len(values_list)
    return 0


def _resolve_requested_layers(
    layer_index: int | Sequence[int] | None,
    *,
    available_layer_count: int | None,
) -> List[int]:
    if layer_index is None:
        if available_layer_count is None:
            raise EmbeddingInputError(
                "ESM-C layer_index=None requires a known available_layer_count for this model name."
            )
        return list(range(int(available_layer_count)))
    if isinstance(layer_index, int):
        requested = [int(layer_index)]
    else:
        requested = [int(value) for value in layer_index]
        if not requested:
            raise EmbeddingInputError("layer_index sequence cannot be empty.")

    resolved: List[int] = []
    upper = int(available_layer_count) if available_layer_count is not None else None
    for index in requested:
        if index < 0:
            if upper is None:
                raise EmbeddingInputError("Negative ESM-C layer indices require a known available_layer_count.")
            index = upper + index
        if index < 0 or (upper is not None and index >= upper):
            limit = f"0..{upper - 1}" if upper is not None else ">=0"
            raise EmbeddingInputError(f"Requested ESM-C layer index {index} out of range; expected {limit}.")
        resolved.append(index)
    return sorted(set(resolved))


def _resolve_layer_count(model_name: str) -> int | None:
    key = str(model_name).strip()
    if key in ESMC_LAYER_SPECS:
        return int(ESMC_LAYER_SPECS[key])
    return None


__all__ = [
    "ESMC_LAYER_SPECS",
    "EsmcPreprocessor",
    "EsmcTokenizerAdapter",
    "EsmcModelAdapter",
    "EsmcPostprocessor",
    "EsmcEmbeddingGenerator",
]
