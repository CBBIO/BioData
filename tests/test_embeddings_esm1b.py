from __future__ import annotations

import builtins
import sys
import types
from typing import Any

import pytest

from CBBIO.embeddings import EmbeddingDependencyError, GenerationInput
from CBBIO.embeddings_esm1b import Esm1bEmbeddingGenerator, Esm1bModelAdapter, Esm1bPreprocessor


class _FakeTensor:
    def __init__(self, data: Any, *, device: str = "cpu") -> None:
        self.data = data
        self.device = device

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


class _FakeConfig:
    num_hidden_layers = 33


class _FakeHfOutput:
    def __init__(self, hidden_states: Any) -> None:
        self.hidden_states = hidden_states


class _FakeHfModel:
    config = _FakeConfig()

    def __init__(self) -> None:
        self.to_args: tuple[Any, ...] | None = None
        self.to_kwargs: dict[str, Any] | None = None

    def to(self, *args: Any, **kwargs: Any) -> "_FakeHfModel":
        self.to_args = args
        self.to_kwargs = kwargs
        return self

    def eval(self) -> None:
        return None

    def __call__(self, **kwargs: Any) -> _FakeHfOutput:
        token_rows = kwargs["input_ids"].tolist()
        seq_len = len(token_rows[0])
        hidden_states = []
        for layer in range(34):
            hidden_states.append(
                _FakeTensor(
                    [
                        [[float(layer), float(row_index * 100 + pos)] for pos in range(seq_len)]
                        for row_index, _row in enumerate(token_rows)
                    ]
                )
            )
        return _FakeHfOutput(hidden_states)


class _FakeTokenizer:
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
        max_len = max(len(sequence) for sequence in sequences) + 2
        input_ids = []
        attention_mask = []
        for sequence in sequences:
            row = [0] + list(range(5, 5 + len(sequence))) + [2]
            mask = [1] * len(row)
            row.extend([1] * (max_len - len(row)))
            mask.extend([0] * (max_len - len(mask)))
            input_ids.append(row)
            attention_mask.append(mask)
        return {"input_ids": _FakeTensor(input_ids), "attention_mask": _FakeTensor(attention_mask)}


def test_esm1b_preprocessor_normalizes_sequence() -> None:
    pre = Esm1bPreprocessor()
    assert pre.preprocess("acduzob") == "ACDXXXX"


def test_esm1b_generator_raises_dependency_error_when_transformers_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_import = builtins.__import__

    def _raising_import(name: str, globals: Any = None, locals: Any = None, fromlist: Any = (), level: int = 0) -> Any:
        if name == "transformers" or name.startswith("transformers."):
            raise ModuleNotFoundError("No module named 'transformers'")
        return original_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", _raising_import)
    monkeypatch.delitem(sys.modules, "transformers", raising=False)

    with pytest.raises(EmbeddingDependencyError):
        Esm1bEmbeddingGenerator()


def test_esm1b_generate_returns_per_residue_matrices_without_pooling() -> None:
    generator = Esm1bEmbeddingGenerator(model=_FakeHfModel(), tokenizer=_FakeTokenizer())
    result = generator.generate([GenerationInput(id="P1", sequence="ACDE")], layer_index=[33], fail_fast=True)

    assert result.errors == []
    assert [record.layer_index for record in result.records] == [33]
    assert result.records[0].shape == (4, 2)
    assert result.records[0].embedding == [[33.0, 1.0], [33.0, 2.0], [33.0, 3.0], [33.0, 4.0]]


def test_esm1b_generate_slices_variable_length_batches_to_residues() -> None:
    generator = Esm1bEmbeddingGenerator(model=_FakeHfModel(), tokenizer=_FakeTokenizer())
    result = generator.generate(
        [
            GenerationInput(id="P1", sequence="ACDE"),
            GenerationInput(id="P2", sequence="AC"),
        ],
        layer_index=[33],
        fail_fast=True,
    )

    assert result.errors == []
    assert [record.id for record in result.records] == ["P1", "P2"]
    assert [record.shape for record in result.records] == [(4, 2), (2, 2)]
    assert result.records[0].embedding == [[33.0, 1.0], [33.0, 2.0], [33.0, 3.0], [33.0, 4.0]]
    assert result.records[1].embedding == [[33.0, 101.0], [33.0, 102.0]]


def test_esm1b_mean_pooler_excludes_special_tokens() -> None:
    generator = Esm1bEmbeddingGenerator(model=_FakeHfModel(), tokenizer=_FakeTokenizer())
    result = generator.generate(
        [GenerationInput(id="P1", sequence="ACDE")],
        layer_index=[33],
        pooler="mean",
        fail_fast=True,
    )

    assert result.errors == []
    assert result.records[0].shape == (2,)
    assert result.records[0].embedding == [33.0, 2.5]


def test_esm1b_cls_pooler_returns_leading_special_token() -> None:
    generator = Esm1bEmbeddingGenerator(model=_FakeHfModel(), tokenizer=_FakeTokenizer())
    result = generator.generate(
        [
            GenerationInput(id="P1", sequence="ACDE"),
            GenerationInput(id="P2", sequence="AC"),
        ],
        layer_index=[33],
        pooler="cls",
        fail_fast=True,
    )

    assert result.errors == []
    assert [record.id for record in result.records] == ["P1", "P2"]
    assert [record.shape for record in result.records] == [(2,), (2,)]
    assert result.records[0].embedding == [33.0, 0.0]
    assert result.records[1].embedding == [33.0, 100.0]


def test_esm1b_adapter_supports_all_layers_and_negative_indices() -> None:
    adapter = Esm1bModelAdapter(_FakeHfModel())
    tokens = _FakeTokenizer()(["AC"], add_special_tokens=True, padding=True, return_tensors="pt")

    all_layers = adapter.infer(tokens, layer_index=None)
    last_layer = adapter.infer(tokens, layer_index=[-1])

    assert list(all_layers["layers"].keys()) == list(range(34))
    assert list(last_layer["layers"].keys()) == [33]
    assert last_layer["sample_spans"] == [(1, 3)]


def test_esm1b_generator_records_requested_dtype(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_torch = types.SimpleNamespace(
        float32="float32",
        float16="float16",
        bfloat16="bfloat16",
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    model = _FakeHfModel()
    generator = Esm1bEmbeddingGenerator(
        model=model,
        tokenizer=_FakeTokenizer(),
        device="cuda:0",
        dtype="float16",
    )

    assert generator.model_metadata.parameters is not None
    assert generator.model_metadata.parameters["torch_dtype"] == "float16"
    assert model.to_kwargs == {"device": "cuda:0", "dtype": "float16"}


def test_esm1b_available_layers_and_count() -> None:
    generator = Esm1bEmbeddingGenerator(model=_FakeHfModel(), tokenizer=_FakeTokenizer())
    assert generator.available_layers() == list(range(34))
    assert generator.num_layers() == 34


def test_esm1b_generator_loads_hf_model_and_tokenizer_for_alias(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _FakeAutoModel:
        observed_name: str | None = None

        @staticmethod
        def from_pretrained(name: str, **kwargs: Any) -> _FakeHfModel:
            _ = kwargs
            _FakeAutoModel.observed_name = name
            return _FakeHfModel()

    class _FakeAutoTokenizer:
        observed_name: str | None = None

        @staticmethod
        def from_pretrained(name: str) -> _FakeTokenizer:
            _FakeAutoTokenizer.observed_name = name
            return _FakeTokenizer()

    fake_transformers = types.SimpleNamespace(
        AutoModel=_FakeAutoModel,
        AutoTokenizer=_FakeAutoTokenizer,
    )
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)

    generator = Esm1bEmbeddingGenerator(model_name="esm1b_t33_650M_UR50S", device="cpu")

    assert generator.model_metadata.model_name == "esm1b_t33_650M_UR50S"
    assert generator.model_metadata.model_reference == "facebook/esm-1b"
    assert _FakeAutoModel.observed_name == "facebook/esm-1b"
    assert _FakeAutoTokenizer.observed_name == "facebook/esm-1b"
