"""ESM-C-specific Hugging Face embedding generator."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict, List, Sequence, cast

import torch
import torch.nn as nn
import torch.nn.functional as F

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

_EsmcArchitectureCache = tuple[type[Any], type[Any], type[Any], type[Any], type[Any], type[Any] | None]
_esmc_architecture_cache: _EsmcArchitectureCache | None = None


class EsmcPreprocessor(BasePreprocessor):
    """Preprocessing for ESM-C protein inputs."""

    def __init__(self, *, context: str = "ESM-C preprocessing") -> None:
        super().__init__(context=context)


class EsmcSequenceTokenizer:
    """Minimal ESM-C sequence tokenizer with the Biohub checkpoint vocabulary."""

    name_or_path = "CBBIO.EsmcSequenceTokenizer"
    pad_token = "<pad>"
    cls_token = "<cls>"
    eos_token = "<eos>"
    unk_token = "<unk>"
    mask_token = "<mask>"
    pad_token_id = 1
    cls_token_id = 0
    eos_token_id = 2
    unk_token_id = 3
    mask_token_id = 32
    vocab_size = 33
    _TOKEN_IDS: Dict[str, int] = {
        "<cls>": 0,
        "<pad>": 1,
        "<eos>": 2,
        "<unk>": 3,
        "L": 4,
        "A": 5,
        "G": 6,
        "V": 7,
        "S": 8,
        "E": 9,
        "R": 10,
        "T": 11,
        "I": 12,
        "D": 13,
        "P": 14,
        "K": 15,
        "Q": 16,
        "N": 17,
        "F": 18,
        "Y": 19,
        "M": 20,
        "H": 21,
        "W": 22,
        "C": 23,
        "X": 24,
        "B": 25,
        "U": 26,
        "Z": 27,
        "O": 28,
        ".": 29,
        "-": 30,
        "|": 31,
        "<mask>": 32,
    }
    _ID_TOKENS: Dict[int, str] = {value: key for key, value in _TOKEN_IDS.items()}

    def __len__(self) -> int:
        return self.vocab_size

    def get_vocab(self) -> Dict[str, int]:
        return dict(self._TOKEN_IDS)

    def _tokenize(self, text: str) -> list[str]:
        return list(str(text))

    def _convert_token_to_id(self, token: str) -> int:
        return self._TOKEN_IDS.get(str(token), self.unk_token_id)

    def _convert_id_to_token(self, index: int) -> str:
        return self._ID_TOKENS.get(int(index), self.unk_token)

    def convert_tokens_to_string(self, tokens: Sequence[str]) -> str:
        return "".join(str(token) for token in tokens)

    def build_inputs_with_special_tokens(
        self,
        token_ids_0: list[int],
        token_ids_1: list[int] | None = None,
    ) -> list[int]:
        if token_ids_1 is not None:
            raise ValueError("ESM-C tokenizer only supports single protein sequences.")
        return [self.cls_token_id] + list(token_ids_0) + [self.eos_token_id]

    def save_vocabulary(self, save_directory: str, filename_prefix: str | None = None) -> tuple[str, ...]:
        _ = save_directory, filename_prefix
        return ()

    def __call__(
        self,
        sequences: str | Sequence[str],
        *,
        add_special_tokens: bool = True,
        padding: bool | str = True,
        return_tensors: str | None = None,
        **_: Any,
    ) -> Dict[str, Any]:
        try:
            import torch  # type: ignore
        except ModuleNotFoundError as exc:
            raise EmbeddingDependencyError("PyTorch is required for ESM-C tokenization.") from exc

        sequence_values = [sequences] if isinstance(sequences, str) else list(sequences)
        if not sequence_values:
            raise ValueError("ESM-C tokenizer requires at least one sequence.")

        rows: list[list[int]] = []
        for sequence in sequence_values:
            token_ids = [self._convert_token_to_id(char) for char in str(sequence)]
            if add_special_tokens:
                token_ids = self.build_inputs_with_special_tokens(token_ids)
            rows.append(token_ids)
        max_len = max(len(row) for row in rows)
        if padding:
            rows = [row + [self.pad_token_id] * (max_len - len(row)) for row in rows]
        attention_mask = [[int(value != self.pad_token_id) for value in row] for row in rows]
        if return_tensors == "pt":
            return {
                "input_ids": torch.tensor(rows, dtype=torch.long),
                "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            }
        return {"input_ids": rows, "attention_mask": attention_mask}


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


class _PureEsmcOutput:
    def __init__(self, *, embeddings: torch.Tensor, hidden_states: torch.Tensor) -> None:
        self.embeddings = embeddings
        self.hidden_states = hidden_states


class _ExtraStateMixin:
    def get_extra_state(self) -> torch.Tensor:
        return torch.empty(0)

    def set_extra_state(self, state: Any) -> None:
        _ = state
        return None


class _PackedLayerNormQkv(_ExtraStateMixin, nn.Module):
    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.layer_norm_weight = nn.Parameter(torch.ones(d_model))
        self.layer_norm_bias = nn.Parameter(torch.zeros(d_model))
        self.weight = nn.Parameter(torch.empty(d_model * 3, d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.layer_norm(
            x,
            normalized_shape=(x.shape[-1],),
            weight=self.layer_norm_weight,
            bias=self.layer_norm_bias,
        )
        return F.linear(x, self.weight)


class _LinearNoBias(_ExtraStateMixin, nn.Linear):
    def __init__(self, in_features: int, out_features: int) -> None:
        super().__init__(in_features, out_features, bias=False)  # pyright: ignore[reportUnknownMemberType]


class _EsmcSwiGluFfn(_ExtraStateMixin, nn.Module):
    def __init__(self, d_model: int, expansion_ratio: float = 8 / 3) -> None:
        super().__init__()
        hidden = int(((expansion_ratio * d_model) + 255) // 256 * 256)
        self.layer_norm_weight = nn.Parameter(torch.ones(d_model))
        self.layer_norm_bias = nn.Parameter(torch.zeros(d_model))
        self.fc1_weight = nn.Parameter(torch.empty(hidden * 2, d_model))
        self.fc2_weight = nn.Parameter(torch.empty(d_model, hidden))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.layer_norm(
            x,
            normalized_shape=(x.shape[-1],),
            weight=self.layer_norm_weight,
            bias=self.layer_norm_bias,
        )
        x = F.linear(x, self.fc1_weight)
        x1, x2 = x.chunk(2, dim=-1)
        x = F.silu(x1) * x2
        return F.linear(x, self.fc2_weight)


class _RotaryEmbedding(nn.Module):
    def __init__(self, dim: int, base: float = 10000.0) -> None:
        super().__init__()
        self.dim = int(dim)
        self.base = float(base)
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2, dtype=torch.float32) / self.dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._seq_len_cached = 0
        self._cos_cached: torch.Tensor | None = None
        self._sin_cached: torch.Tensor | None = None

    def forward(self, q: torch.Tensor, k: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        seq_len = int(q.shape[1])
        self._update_cache(seq_len, device=q.device, dtype=q.dtype)
        assert self._cos_cached is not None
        assert self._sin_cached is not None
        return _apply_rotary(q, self._cos_cached[:seq_len], self._sin_cached[:seq_len]), _apply_rotary(
            k,
            self._cos_cached[:seq_len],
            self._sin_cached[:seq_len],
        )

    def _update_cache(self, seq_len: int, *, device: torch.device, dtype: torch.dtype) -> None:
        if (
            seq_len <= self._seq_len_cached
            and self._cos_cached is not None
            and self._cos_cached.device == device
            and self._cos_cached.dtype == dtype
        ):
            return
        self._seq_len_cached = seq_len
        t = torch.arange(seq_len, device=device, dtype=torch.float32)
        inv_freq = cast(torch.Tensor, self.inv_freq).to(device=device, dtype=torch.float32)
        freqs = torch.outer(t, inv_freq)
        self._cos_cached = torch.cos(freqs).to(dtype=dtype)
        self._sin_cached = torch.sin(freqs).to(dtype=dtype)


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def _apply_rotary(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    rotary_dim = int(cos.shape[-1] * 2)
    cos_full = torch.cat((cos, cos), dim=-1)[None, :, None, :]
    sin_full = torch.cat((sin, sin), dim=-1)[None, :, None, :]
    rotated = x[..., :rotary_dim] * cos_full + _rotate_half(x[..., :rotary_dim]) * sin_full
    return torch.cat((rotated, x[..., rotary_dim:]), dim=-1)


class _EsmcAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int) -> None:
        super().__init__()
        self.d_model = int(d_model)
        self.n_heads = int(n_heads)
        self.d_head = self.d_model // self.n_heads
        self.layernorm_qkv = _PackedLayerNormQkv(self.d_model)
        self.out_proj = _LinearNoBias(self.d_model, self.d_model)
        self.q_ln = nn.LayerNorm(self.d_model, bias=False)
        self.k_ln = nn.LayerNorm(self.d_model, bias=False)
        self.rotary = _RotaryEmbedding(self.d_head)

    def forward(self, x: torch.Tensor, sequence_id: torch.Tensor | None) -> torch.Tensor:
        qkv = self.layernorm_qkv(x)
        query, key, value = torch.chunk(qkv, 3, dim=-1)
        query = self.q_ln(query).to(query.dtype)
        key = self.k_ln(key).to(query.dtype)
        query = query.reshape(*query.shape[:-1], self.n_heads, self.d_head)
        key = key.reshape(*key.shape[:-1], self.n_heads, self.d_head)
        query, key = self.rotary(query, key)
        value = value.reshape(*value.shape[:-1], self.n_heads, self.d_head)
        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)
        attn_mask = None
        if sequence_id is not None:
            attn_mask = (sequence_id.unsqueeze(-1) == sequence_id.unsqueeze(-2)).unsqueeze(1)
        context = F.scaled_dot_product_attention(query, key, value, attn_mask=attn_mask)
        context = context.transpose(1, 2).flatten(-2, -1)
        return self.out_proj(context)


class _EsmcBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, *, n_layers: int) -> None:
        super().__init__()
        self.attn = _EsmcAttention(d_model, n_heads)
        self.ffn = _EsmcSwiGluFfn(d_model)
        self.scaling_factor = math.sqrt(float(n_layers) / 36.0)

    def forward(self, x: torch.Tensor, sequence_id: torch.Tensor | None) -> torch.Tensor:
        x = x + self.attn(x, sequence_id) / self.scaling_factor
        x = x + self.ffn(x) / self.scaling_factor
        return x


class _EsmcTransformer(nn.Module):
    def __init__(self, d_model: int, n_heads: int, n_layers: int) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(
            [_EsmcBlock(d_model, n_heads, n_layers=n_layers) for _ in range(int(n_layers))]
        )
        self.norm = nn.LayerNorm(d_model, bias=False)

    def forward(self, x: torch.Tensor, sequence_id: torch.Tensor | None) -> tuple[torch.Tensor, list[torch.Tensor]]:
        hidden_states: list[torch.Tensor] = []
        for block in self.blocks:
            x = block(x, sequence_id)
            hidden_states.append(x)
        return self.norm(x), hidden_states


class _PureEsmcBackbone(nn.Module):
    def __init__(self, *, d_model: int, n_heads: int, n_layers: int) -> None:
        super().__init__()
        self.embed = nn.Embedding(64, d_model)
        self.transformer = _EsmcTransformer(d_model, n_heads, n_layers)

    def forward(self, *, input_ids: torch.Tensor, attention_mask: Any | None = None) -> _PureEsmcOutput:
        sequence_id = None
        if attention_mask is not None:
            bool_fn = getattr(attention_mask, "bool", None)
            sequence_id = bool_fn() if callable(bool_fn) else attention_mask
        else:
            sequence_id = input_ids != 1
        x = self.embed(input_ids)
        x, hidden_states = self.transformer(x, sequence_id)
        return _PureEsmcOutput(embeddings=x, hidden_states=torch.stack(hidden_states, dim=0))


class _EsmcLmHead(nn.Sequential):
    def __init__(self, d_model: int, vocab_size: int = 64) -> None:
        super().__init__(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model),
            nn.Linear(d_model, vocab_size),
        )


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
            raise EmbeddingInputError("use_flash_attention is not supported by the pure Transformers ESM-C loader.")
        if model is None:
            try:
                register_hf_esmc_architecture()
            except ModuleNotFoundError as exc:
                raise EmbeddingDependencyError(
                    "transformers is required for ESM-C loading. Install with: pip install transformers"
                ) from exc
        resolved_tokenizer = tokenizer if tokenizer is not None else EsmcSequenceTokenizer()

        super().__init__(
            model_name=model_name,
            model_reference=model_reference,
            context="ESM-C preprocessing",
            provider="huggingface-transformers",
            device=device,
            dtype=dtype,
            model=model,
            tokenizer=resolved_tokenizer,
            from_pretrained_kwargs=from_pretrained_kwargs,
            tokenizer_from_pretrained_kwargs=tokenizer_from_pretrained_kwargs,
            auto_model_class="AutoModelForMaskedLM",
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
    """Register Biohub ESM-C config, model, and tokenizer with Transformers Auto APIs.

    Biohub ESM-C checkpoints store weights under ``esmc.transformer...`` keys,
    which do not match the vanilla Hugging Face ESM encoder layout. This
    registers a local ``PreTrainedModel`` implementation that matches those
    keys, plus a local tokenizer for checkpoints that declare
    ``tokenizer_class = "ESMCTokenizer"``.
    """

    global _esmc_architecture_cache

    try:
        import transformers  # type: ignore
    except ModuleNotFoundError:
        raise
    AutoConfig = getattr(transformers, "AutoConfig")
    AutoModel = getattr(transformers, "AutoModel")
    AutoModelForMaskedLM = getattr(transformers, "AutoModelForMaskedLM", None)
    AutoTokenizer = getattr(transformers, "AutoTokenizer", None)
    PretrainedConfig = getattr(transformers, "PretrainedConfig", getattr(transformers, "EsmConfig", object))
    PreTrainedModel = getattr(transformers, "PreTrainedModel", getattr(transformers, "EsmModel", object))
    PreTrainedTokenizer = getattr(transformers, "PreTrainedTokenizer", None)
    try:
        from transformers.modeling_outputs import BaseModelOutput, MaskedLMOutput  # type: ignore
    except Exception:
        BaseModelOutput = _SimpleBaseModelOutput
        MaskedLMOutput = _SimpleMaskedLMOutput

    EsmcTokenizerClass: type[Any] | None
    if (
        _esmc_architecture_cache is not None
        and _esmc_architecture_cache[0] is PretrainedConfig
        and _esmc_architecture_cache[1] is PreTrainedModel
    ):
        EsmcConfig = _esmc_architecture_cache[2]
        EsmcModel = _esmc_architecture_cache[3]
        EsmcForMaskedLM = _esmc_architecture_cache[4]
        EsmcTokenizerClass = _esmc_architecture_cache[5]
    else:

        class EsmcConfig(PretrainedConfig):  # type: ignore[misc]
            model_type = "esmc"

            def __init__(
                self,
                *,
                d_model: int | None = None,
                n_layers: int | None = None,
                n_heads: int | None = None,
                dtype: str | None = None,
                classifier_dropout: float | None = None,
                use_flash_attn: bool = False,
                **kwargs: Any,
            ) -> None:
                hidden_size = int(d_model) if d_model is not None else int(kwargs.get("hidden_size", 960))
                layer_count = int(n_layers) if n_layers is not None else int(kwargs.get("num_hidden_layers", 30))
                head_count = int(n_heads) if n_heads is not None else int(kwargs.get("num_attention_heads", 15))
                kwargs.setdefault("hidden_size", hidden_size)
                kwargs.setdefault("num_hidden_layers", layer_count)
                kwargs.setdefault("num_attention_heads", head_count)
                super().__init__(**kwargs)  # pyright: ignore[reportUnknownMemberType]
                self.d_model = hidden_size
                self.n_layers = layer_count
                self.n_heads = head_count
                self.hidden_size = hidden_size
                self.num_hidden_layers = layer_count
                self.num_attention_heads = head_count
                self.dtype = dtype
                self.classifier_dropout = classifier_dropout
                self.use_flash_attn = bool(use_flash_attn)

        class EsmcModel(PreTrainedModel):  # type: ignore[misc]
            config_class = EsmcConfig
            base_model_prefix = "esmc"

            def __init__(self, config: Any) -> None:
                super().__init__(config)  # pyright: ignore[reportUnknownMemberType]
                self.esmc = _PureEsmcBackbone(
                    d_model=int(config.d_model),
                    n_heads=int(config.n_heads),
                    n_layers=int(config.n_layers),
                )

            def _init_weights(self, module: Any) -> None:
                return None

            def forward(
                self,
                input_ids: Any | None = None,
                attention_mask: Any | None = None,
                output_hidden_states: bool | None = None,
                return_dict: bool | None = None,
                **kwargs: Any,
            ) -> Any:
                sequence_tokens = input_ids if input_ids is not None else kwargs.get("sequence_tokens")
                if sequence_tokens is None:
                    raise ValueError("EsmcModel.forward requires input_ids or sequence_tokens.")

                output = self.esmc(input_ids=sequence_tokens, attention_mask=attention_mask)
                hidden_states = None
                if output_hidden_states:
                    hidden_states = (output.embeddings,) + _esmc_hidden_states_tuple(output.hidden_states)

                if return_dict is False:
                    values = (output.embeddings,)
                    if output_hidden_states:
                        values += (hidden_states,)
                    return values

                return cast(Any, BaseModelOutput)(
                    last_hidden_state=output.embeddings,
                    hidden_states=hidden_states,
                )

        class EsmcForMaskedLM(PreTrainedModel):  # type: ignore[misc]
            config_class = EsmcConfig
            base_model_prefix = "esmc"

            def __init__(self, config: Any) -> None:
                super().__init__(config)  # pyright: ignore[reportUnknownMemberType]
                self.esmc = _PureEsmcBackbone(
                    d_model=int(config.d_model),
                    n_heads=int(config.n_heads),
                    n_layers=int(config.n_layers),
                )
                self.lm_head = _EsmcLmHead(int(config.d_model), vocab_size=64)

            def _init_weights(self, module: Any) -> None:
                return None

            def get_output_embeddings(self) -> Any:
                return self.lm_head[3]

            def set_output_embeddings(self, new_embeddings: Any) -> None:
                self.lm_head[3] = new_embeddings

            def forward(
                self,
                input_ids: Any | None = None,
                attention_mask: Any | None = None,
                labels: Any | None = None,
                output_hidden_states: bool | None = None,
                return_dict: bool | None = None,
                **kwargs: Any,
            ) -> Any:
                sequence_tokens = input_ids if input_ids is not None else kwargs.get("sequence_tokens")
                if sequence_tokens is None:
                    raise ValueError("EsmcForMaskedLM.forward requires input_ids or sequence_tokens.")

                output = self.esmc(input_ids=sequence_tokens, attention_mask=attention_mask)
                logits = self.lm_head(output.embeddings)
                loss = None
                if labels is not None:
                    loss = F.cross_entropy(logits.view(-1, logits.shape[-1]), labels.view(-1))

                hidden_states = None
                if output_hidden_states:
                    hidden_states = (output.embeddings,) + _esmc_hidden_states_tuple(output.hidden_states)

                if return_dict is False:
                    values = (logits,)
                    if loss is not None:
                        values = (loss,) + values
                    if output_hidden_states:
                        values += (hidden_states,)
                    return values

                return cast(Any, MaskedLMOutput)(
                    loss=loss,
                    logits=logits,
                    hidden_states=hidden_states,
                )

        if PreTrainedTokenizer is None:
            EsmcTokenizerClass = None
        else:

            class ESMCTokenizer(EsmcSequenceTokenizer, PreTrainedTokenizer):  # type: ignore[misc]
                vocab_files_names: Dict[str, str] = {}
                model_input_names = ["input_ids", "attention_mask"]

                def __init__(self, **kwargs: Any) -> None:
                    kwargs.setdefault("pad_token", self.pad_token)
                    kwargs.setdefault("cls_token", self.cls_token)
                    kwargs.setdefault("eos_token", self.eos_token)
                    kwargs.setdefault("unk_token", self.unk_token)
                    kwargs.setdefault("mask_token", self.mask_token)
                    PreTrainedTokenizer.__init__(self, **kwargs)  # pyright: ignore[reportUnknownMemberType]

            EsmcTokenizerClass = ESMCTokenizer

        _esmc_architecture_cache = (
            PretrainedConfig,
            PreTrainedModel,
            EsmcConfig,
            EsmcModel,
            EsmcForMaskedLM,
            EsmcTokenizerClass,
        )

    AutoConfig.register("esmc", EsmcConfig, exist_ok=True)
    AutoModel.register(EsmcConfig, EsmcModel, exist_ok=True)
    if AutoModelForMaskedLM is not None:
        AutoModelForMaskedLM.register(EsmcConfig, EsmcForMaskedLM, exist_ok=True)
    if AutoTokenizer is not None and EsmcTokenizerClass is not None:
        setattr(transformers, "ESMCTokenizer", EsmcTokenizerClass)
        AutoTokenizer.register(
            EsmcConfig,
            slow_tokenizer_class=EsmcTokenizerClass,
            fast_tokenizer_class=None,
            exist_ok=True,
        )


def _esmc_hidden_states_tuple(hidden_states: Any) -> tuple[Any, ...]:
    try:
        import torch  # type: ignore

        if isinstance(hidden_states, torch.Tensor):
            return tuple(hidden_states[index] for index in range(int(hidden_states.shape[0])))
    except Exception:
        pass
    return tuple(hidden_states)


class _SimpleBaseModelOutput:
    def __init__(self, *, last_hidden_state: Any = None, hidden_states: Any = None, **kwargs: Any) -> None:
        self.last_hidden_state = last_hidden_state
        self.hidden_states = hidden_states
        for key, value in kwargs.items():
            setattr(self, key, value)


class _SimpleMaskedLMOutput:
    def __init__(
        self,
        *,
        loss: Any = None,
        logits: Any = None,
        hidden_states: Any = None,
        **kwargs: Any,
    ) -> None:
        self.loss = loss
        self.logits = logits
        self.hidden_states = hidden_states
        for key, value in kwargs.items():
            setattr(self, key, value)


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
    "EsmcSequenceTokenizer",
    "EsmcPreprocessor",
    "EsmcTokenizerAdapter",
    "EsmcModelAdapter",
    "EsmcPostprocessor",
    "EsmcEmbeddingGenerator",
    "register_hf_esmc_architecture",
]
