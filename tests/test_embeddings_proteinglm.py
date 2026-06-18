from __future__ import annotations

import sys
import types
from typing import Any

import pytest

from CBBIO.embeddings import EmbeddingInputError, GenerationInput
from CBBIO.embeddings.factory import Generator, available_generator_classes, available_generator_models
from CBBIO.embeddings.models import proteinglm as proteinglm_module
from CBBIO.embeddings_proteinglm import (
    PROTEINGLM_HF_MODEL_NAMES,
    ProteinGlmEmbeddingGenerator,
    ProteinGlmPreprocessor,
    ProteinGlmTokenizerAdapter,
)


def _as_list(value: Any) -> Any:
    tolist = getattr(value, "tolist", None)
    if callable(tolist):
        return tolist()
    return value


class _FakeTensor:
    def __init__(self, data: Any, *, device: str = "cpu") -> None:
        self.data = data
        self.device = device
        self.shape = _shape(data)

    def __getitem__(self, item: Any) -> "_FakeTensor":
        if isinstance(item, tuple):
            first = item[0]
            second = item[1] if len(item) > 1 else slice(None)
            base = self.data[first]
            return _FakeTensor(base[second], device=self.device)
        return _FakeTensor(self.data[item], device=self.device)

    def to(self, device: str) -> "_FakeTensor":
        return _FakeTensor(self.data, device=str(device))

    def tolist(self) -> Any:
        return self.data

    def transpose(self, dim0: int, dim1: int) -> "_FakeTensor":
        if (dim0, dim1) != (0, 1) or len(self.shape) != 3:
            raise NotImplementedError
        token_count, batch_size, _hidden_size = self.shape
        return _FakeTensor(
            [
                [self.data[token_index][batch_index] for token_index in range(token_count)]
                for batch_index in range(batch_size)
            ],
            device=self.device,
        )


class _FakeConfig:
    num_hidden_layers = 2


class _FakeHfOutput:
    def __init__(self, hidden_states: Any) -> None:
        self.hidden_states = hidden_states


class _FakeLayerSequenceModel:
    config = _FakeConfig()

    def __init__(self) -> None:
        self.to_kwargs: dict[str, Any] | None = None
        self.return_last_hidden_state_values: list[bool] = []

    def to(self, *args: Any, **kwargs: Any) -> "_FakeLayerSequenceModel":
        _ = args
        self.to_kwargs = kwargs
        return self

    def eval(self) -> None:
        return None

    def __call__(self, **kwargs: Any) -> _FakeHfOutput:
        token_rows = kwargs["input_ids"].tolist()
        seq_len = len(token_rows[0])
        hidden_states = []
        for layer in range(3):
            hidden_states.append(
                _FakeTensor(
                    [
                        [[float(layer), float(row_index * 100 + pos)] for pos in range(seq_len)]
                        for row_index, _row in enumerate(token_rows)
                    ]
                )
            )
        self.return_last_hidden_state_values.append(bool(kwargs["return_last_hidden_state"]))
        if kwargs["return_last_hidden_state"]:
            return _FakeHfOutput(hidden_states[-1])
        return _FakeHfOutput(hidden_states)


class _FakeFinalTensorModel(_FakeLayerSequenceModel):
    def __call__(self, **kwargs: Any) -> _FakeHfOutput:
        token_rows = kwargs["input_ids"].tolist()
        seq_len = len(token_rows[0])
        return _FakeHfOutput(
            _FakeTensor(
                [
                    [[2.0, float(row_index * 100 + pos)] for pos in range(seq_len)]
                    for row_index, _row in enumerate(token_rows)
                ]
            )
        )


class _FakeTokenMajorFinalTensorModel(_FakeLayerSequenceModel):
    def __call__(self, **kwargs: Any) -> _FakeHfOutput:
        token_rows = kwargs["input_ids"].tolist()
        seq_len = len(token_rows[0])
        batch_size = len(token_rows)
        return _FakeHfOutput(
            _FakeTensor(
                [
                    [[2.0, float(row_index * 100 + pos)] for row_index in range(batch_size)]
                    for pos in range(seq_len)
                ]
            )
        )


class _FakeTokenizer:
    name_or_path = "fake-proteinglm-tokenizer"

    def __call__(
        self,
        sequences: list[str],
        *,
        add_special_tokens: bool,
        padding: bool,
        return_tensors: str,
    ) -> dict[str, _FakeTensor]:
        assert add_special_tokens is True
        assert padding is True
        assert return_tensors == "pt"
        max_len = max(len(sequence) for sequence in sequences) + 1
        input_ids = []
        attention_mask = []
        for sequence in sequences:
            row = list(range(5, 5 + len(sequence))) + [2]
            mask = [1] * len(row)
            row.extend([1] * (max_len - len(row)))
            mask.extend([0] * (max_len - len(mask)))
            input_ids.append(row)
            attention_mask.append(mask)
        return {"input_ids": _FakeTensor(input_ids), "attention_mask": _FakeTensor(attention_mask)}


def test_proteinglm_preprocessor_normalizes_sequence() -> None:
    pre = ProteinGlmPreprocessor()
    assert pre.preprocess("acduzob") == "ACDXXXX"


def test_proteinglm_tokenizer_moves_inputs_to_device() -> None:
    tokenizer = ProteinGlmTokenizerAdapter(_FakeTokenizer(), device="cuda:0")

    result = tokenizer.tokenize_many(["ACDE"])

    assert result["input_ids"].device == "cuda:0"
    assert result["attention_mask"].device == "cuda:0"


def test_proteinglm_generate_trims_trailing_eos_without_leading_bos() -> None:
    model = _FakeLayerSequenceModel()
    generator = ProteinGlmEmbeddingGenerator(model=model, tokenizer=_FakeTokenizer())
    result = generator.generate(
        [
            GenerationInput(id="P1", sequence="ACDE"),
            GenerationInput(id="P2", sequence="AC"),
        ],
        layer_index=[2],
        fail_fast=True,
    )

    assert result.errors == []
    assert [record.id for record in result.records] == ["P1", "P2"]
    assert [record.shape for record in result.records] == [(4, 2), (2, 2)]
    assert _as_list(result.records[0].embedding) == [[2.0, 0.0], [2.0, 1.0], [2.0, 2.0], [2.0, 3.0]]
    assert _as_list(result.records[1].embedding) == [[2.0, 100.0], [2.0, 101.0]]
    assert model.return_last_hidden_state_values == [True]


def test_proteinglm_final_tensor_output_defaults_to_last_layer() -> None:
    generator = ProteinGlmEmbeddingGenerator(model=_FakeFinalTensorModel(), tokenizer=_FakeTokenizer())

    result = generator.generate([GenerationInput(id="P1", sequence="ACDE")], fail_fast=True)

    assert result.errors == []
    assert [record.layer_index for record in result.records] == [2]
    assert result.records[0].shape == (4, 2)
    assert _as_list(result.records[0].embedding) == [[2.0, 0.0], [2.0, 1.0], [2.0, 2.0], [2.0, 3.0]]


def test_proteinglm_token_major_final_tensor_is_transposed_before_slicing() -> None:
    generator = ProteinGlmEmbeddingGenerator(model=_FakeTokenMajorFinalTensorModel(), tokenizer=_FakeTokenizer())

    result = generator.generate(
        [
            GenerationInput(id="P1", sequence="ACDE"),
            GenerationInput(id="P2", sequence="AC"),
        ],
        fail_fast=True,
    )

    assert result.errors == []
    assert [record.id for record in result.records] == ["P1", "P2"]
    assert [record.shape for record in result.records] == [(4, 2), (2, 2)]
    assert _as_list(result.records[0].embedding) == [[2.0, 0.0], [2.0, 1.0], [2.0, 2.0], [2.0, 3.0]]
    assert _as_list(result.records[1].embedding) == [[2.0, 100.0], [2.0, 101.0]]


def test_proteinglm_all_layers_requests_hidden_state_sequence() -> None:
    model = _FakeLayerSequenceModel()
    generator = ProteinGlmEmbeddingGenerator(model=model, tokenizer=_FakeTokenizer())

    result = generator.generate([GenerationInput(id="P1", sequence="ACDE")], layer_index=None, fail_fast=True)

    assert result.errors == []
    assert [record.layer_index for record in result.records] == [0, 1, 2]
    assert [record.shape for record in result.records] == [(4, 2), (4, 2), (4, 2)]
    assert model.return_last_hidden_state_values == [False]


def test_proteinglm_generator_loads_masked_lm_and_tokenizer_with_remote_code(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed_models: list[tuple[str, dict[str, Any]]] = []
    observed_tokenizers: list[tuple[str, dict[str, Any]]] = []

    class _FakeAutoModelForMaskedLM:
        @staticmethod
        def from_pretrained(name: str, **kwargs: Any) -> _FakeLayerSequenceModel:
            observed_models.append((name, dict(kwargs)))
            return _FakeLayerSequenceModel()

    class _FakeAutoTokenizer:
        @staticmethod
        def from_pretrained(name: str, **kwargs: Any) -> _FakeTokenizer:
            observed_tokenizers.append((name, dict(kwargs)))
            return _FakeTokenizer()

    fake_transformers = types.SimpleNamespace(
        AutoModelForMaskedLM=_FakeAutoModelForMaskedLM,
        AutoTokenizer=_FakeAutoTokenizer,
    )
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)

    generator = ProteinGlmEmbeddingGenerator(model_name="proteinglm_3b_mlm", device="cpu")

    assert generator.model_metadata.model_name == "proteinglm_3b_mlm"
    assert generator.model_metadata.model_reference == "biomap-research/proteinglm-3b-mlm"
    assert generator.model_metadata.parameters is not None
    assert generator.model_metadata.parameters["auto_model_class"] == "AutoModelForMaskedLM"
    assert observed_models == [("biomap-research/proteinglm-3b-mlm", {"trust_remote_code": True})]
    assert observed_tokenizers == [
        ("biomap-research/proteinglm-3b-mlm", {"trust_remote_code": True, "use_fast": True})
    ]


def test_proteinglm_loader_stubs_missing_deepspeed_checkpointing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delitem(sys.modules, "deepspeed", raising=False)
    observed_checkpointing: list[bool] = []

    original_import_module = proteinglm_module.importlib.import_module

    def _fake_import_module(name: str) -> Any:
        if name == "deepspeed":
            raise ImportError("No module named 'deepspeed'")
        return original_import_module(name)

    class _FakeAutoModelForMaskedLM:
        @staticmethod
        def from_pretrained(name: str, **kwargs: Any) -> _FakeLayerSequenceModel:
            _ = name, kwargs
            import deepspeed

            observed_checkpointing.append(deepspeed.checkpointing.is_configured())
            return _FakeLayerSequenceModel()

    class _FakeAutoTokenizer:
        @staticmethod
        def from_pretrained(name: str, **kwargs: Any) -> _FakeTokenizer:
            _ = name, kwargs
            return _FakeTokenizer()

    fake_transformers = types.SimpleNamespace(
        AutoModelForMaskedLM=_FakeAutoModelForMaskedLM,
        AutoTokenizer=_FakeAutoTokenizer,
    )
    monkeypatch.setattr(proteinglm_module.importlib, "import_module", _fake_import_module)
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)

    ProteinGlmEmbeddingGenerator(model_name="proteinglm_1b_mlm", device="cpu")

    assert observed_checkpointing == [False]
    assert "deepspeed" not in sys.modules


def test_proteinglm_factory_registration() -> None:
    assert "proteinglm" in available_generator_classes()
    assert available_generator_models("proteinglm") == [
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
    assert PROTEINGLM_HF_MODEL_NAMES["bo1015/proteinglm-10b-mlm"] == "biomap-research/proteinglm-10b-mlm"

    generator = Generator(
        model_class="pglm",
        name="3b",
        model=_FakeLayerSequenceModel(),
        tokenizer=_FakeTokenizer(),
    )

    assert isinstance(generator, ProteinGlmEmbeddingGenerator)
    assert generator.model_metadata.model_reference == "biomap-research/proteinglm-3b-mlm"


def _shape(value: Any) -> tuple[int, ...]:
    dims = []
    current = value
    while isinstance(current, list):
        dims.append(len(current))
        current = current[0] if current else None
    return tuple(dims)
