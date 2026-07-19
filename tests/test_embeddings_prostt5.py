from __future__ import annotations

import builtins
import sys
import types
from typing import Any

import pytest

from CBBIO.embeddings import EmbeddingDependencyError, EmbeddingInputError, GenerationInput
from CBBIO import ProstT5EmbeddingGenerator, ProstT5Preprocessor


class _FakeScalar:
    def __init__(self, value: float) -> None:
        self.value = value

    def item(self) -> float:
        return self.value


class _FakeTensor:
    def __init__(self, data: Any) -> None:
        self.data = data

    def to(self, _device: str) -> "_FakeTensor":
        return self

    def __getitem__(self, item: Any) -> "_FakeTensor":
        if isinstance(item, tuple):
            first = item[0]
            second = item[1] if len(item) > 1 else slice(None)
            base = self.data[first]
            return _FakeTensor(base[second])
        return _FakeTensor(self.data[item])

    def sum(self) -> _FakeScalar:
        if isinstance(self.data, list):
            return _FakeScalar(float(sum(self.data)))
        return _FakeScalar(float(self.data))

    def tolist(self) -> Any:
        return self.data


class _FakeNoGrad:
    def __enter__(self) -> None:
        return None

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        return None


class _FakeTokenizer:
    def __call__(
        self,
        sequences: list[str],
        add_special_tokens: bool,
        padding: str,
        return_tensors: str,
    ) -> dict[str, Any]:
        assert add_special_tokens is True
        assert padding == "longest"
        assert return_tensors == "pt"
        split_sequences = [sequence.split() for sequence in sequences]
        assert all(spaced[0] == "<AA2fold>" for spaced in split_sequences)
        # prefix + residues + end token
        lengths = [len(spaced) + 1 for spaced in split_sequences]
        max_len = max(lengths)
        return {
            "input_ids": _FakeTensor([[1] * length + [0] * (max_len - length) for length in lengths]),
            "attention_mask": _FakeTensor([[1] * length + [0] * (max_len - length) for length in lengths]),
        }


class _FakeModelOutput:
    def __init__(self, hidden_states: tuple[_FakeTensor, ...]) -> None:
        self.hidden_states = hidden_states


class _FakeModel:
    class _Config:
        num_layers = 2

    config = _Config()

    def __init__(self) -> None:
        self.did_half = False
        self.did_float = False

    def to(self, _device: str) -> "_FakeModel":
        return self

    def eval(self) -> None:
        return None

    def half(self) -> "_FakeModel":
        self.did_half = True
        return self

    def float(self) -> "_FakeModel":
        self.did_float = True
        return self

    def __call__(self, *, input_ids: Any, attention_mask: Any, output_hidden_states: bool, return_dict: bool) -> Any:
        assert output_hidden_states is True
        assert return_dict is True
        rows = input_ids.tolist()
        batch_size = len(rows)
        length = len(rows[0])  # includes prefix + residues + end
        hidden_states = []
        for layer in range(3):
            layer_values = [
                [[float(layer), float(row_index), float(pos)] for pos in range(length)]
                for row_index in range(batch_size)
            ]
            hidden_states.append(_FakeTensor(layer_values))
        return _FakeModelOutput(tuple(hidden_states))


def test_prostt5_preprocessor_adds_prefix_and_spacing() -> None:
    pre = ProstT5Preprocessor()
    processed = pre.preprocess("acduzob")
    assert processed == "<AA2fold> A C D X X X X"


def test_prostt5_generator_raises_dependency_error_when_transformers_missing(
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
        ProstT5EmbeddingGenerator()


def test_prostt5_generate_returns_per_residue_matrix_without_pooling(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_torch = types.SimpleNamespace(
        tensor=lambda values: _FakeTensor(values),
        no_grad=lambda: _FakeNoGrad(),
        inference_mode=lambda: _FakeNoGrad(),
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    model = _FakeModel()
    generator = ProstT5EmbeddingGenerator(
        tokenizer=_FakeTokenizer(),
        model=model,
        device="cpu",
    )
    result = generator.generate([GenerationInput(id="P1", sequence="ACDE")], layer_index=[0, 2])

    assert result.errors == []
    assert [record.layer_index for record in result.records] == [0, 2]
    assert result.records[0].embedding[0] == [0.0, 0.0, 1.0]
    assert result.records[1].embedding[0] == [2.0, 0.0, 1.0]
    # residues only: 4 rows, hidden dimension 3 in fake model
    assert all(record.shape == (4, 3) for record in result.records)
    assert isinstance(result.records[0].embedding[0], list)
    assert model.did_float is True
    assert model.did_half is False


def test_prostt5_rejects_cls_pooler(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_torch = types.SimpleNamespace(
        tensor=lambda values: _FakeTensor(values),
        no_grad=lambda: _FakeNoGrad(),
        inference_mode=lambda: _FakeNoGrad(),
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    generator = ProstT5EmbeddingGenerator(
        tokenizer=_FakeTokenizer(),
        model=_FakeModel(),
        device="cpu",
    )

    with pytest.raises(EmbeddingInputError, match="Supported poolers"):
        generator.generate([GenerationInput(id="P1", sequence="ACDE")], layer_index=[0], pooler="cls")


def test_prostt5_generator_records_requested_dtype(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_torch = types.SimpleNamespace(
        float32="float32",
        float16="float16",
        bfloat16="bfloat16",
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    model = _FakeModel()
    generator = ProstT5EmbeddingGenerator(
        tokenizer=_FakeTokenizer(),
        model=model,
        device="cuda:0",
        dtype="float16",
    )

    assert generator.model_metadata.parameters is not None
    assert generator.model_metadata.parameters["torch_dtype"] == "float16"
    assert generator.model_metadata.parameters["precision_policy"] == "explicit_torch_dtype"
    assert generator.model_metadata.parameters["layer_indexing"] == "hf_native_0_is_first_hidden"
    assert model.did_half is True


def test_prostt5_generator_available_layers_and_count(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_torch = types.SimpleNamespace(
        tensor=lambda values: _FakeTensor(values),
        no_grad=lambda: _FakeNoGrad(),
        inference_mode=lambda: _FakeNoGrad(),
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    generator = ProstT5EmbeddingGenerator(
        tokenizer=_FakeTokenizer(),
        model=_FakeModel(),
        device="cpu",
    )
    assert generator.available_layers() == [0, 1, 2]
    assert generator.num_layers() == 3


def test_prostt5_loader_resolves_alias_and_keeps_tied_embeddings_when_loading_hf_checkpoint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _FakeConfig:
        def __init__(self) -> None:
            self.tie_word_embeddings = True
            self.num_layers = 2

    class _FakeAutoConfig:
        observed_name: str | None = None

        @staticmethod
        def from_pretrained(name: str) -> _FakeConfig:
            _FakeAutoConfig.observed_name = name
            return _FakeConfig()

    class _FakeT5Tokenizer:
        observed_name: str | None = None

        @staticmethod
        def from_pretrained(name: str, do_lower_case: bool = False) -> _FakeTokenizer:
            _FakeT5Tokenizer.observed_name = name
            assert do_lower_case is False
            return _FakeTokenizer()

    class _FakeT5EncoderModel:
        observed_name: str | None = None
        observed_tie_flag: bool | None = None

        @staticmethod
        def from_pretrained(name: str, config: _FakeConfig) -> _FakeModel:
            _FakeT5EncoderModel.observed_name = name
            _FakeT5EncoderModel.observed_tie_flag = bool(config.tie_word_embeddings)
            return _FakeModel()

    fake_transformers = types.SimpleNamespace(
        AutoConfig=_FakeAutoConfig,
        T5Tokenizer=_FakeT5Tokenizer,
        T5EncoderModel=_FakeT5EncoderModel,
    )
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)

    generator = ProstT5EmbeddingGenerator(model_name="prostT5", device="cpu")

    assert generator.model_metadata.model_name == "prostT5"
    assert generator.model_reference == "Rostlab/ProstT5"
    assert _FakeAutoConfig.observed_name == "Rostlab/ProstT5"
    assert _FakeT5Tokenizer.observed_name == "Rostlab/ProstT5"
    assert _FakeT5EncoderModel.observed_name == "Rostlab/ProstT5"
    assert _FakeT5EncoderModel.observed_tie_flag is True
