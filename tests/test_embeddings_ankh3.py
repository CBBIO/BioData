from __future__ import annotations

import builtins
import sys
import types
from typing import Any

import pytest

from CBBIO.embeddings import EmbeddingDependencyError, EmbeddingInputError, GenerationInput
from CBBIO.embeddings_ankh3 import Ankh3EmbeddingGenerator, Ankh3Preprocessor


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
    def __call__(self, sequence: str, add_special_tokens: bool, return_tensors: str, is_split_into_words: bool) -> dict[str, Any]:
        assert add_special_tokens is True
        assert return_tensors == "pt"
        assert is_split_into_words is False
        assert sequence.startswith("[NLU]") or sequence.startswith("[S2S]")
        # sequence + start/end specials.
        length = len(sequence) + 2
        return {"input_ids": _FakeTensor([[1] * length]), "attention_mask": _FakeTensor([[1] * length])}


class _FakeModelOutput:
    def __init__(self, hidden_states: tuple[_FakeTensor, ...]) -> None:
        self.hidden_states = hidden_states


class _FakeModel:
    class _Config:
        num_layers = 2

    config = _Config()

    def to(self, _device: str) -> "_FakeModel":
        return self

    def eval(self) -> None:
        return None

    def __call__(self, *, input_ids: Any, attention_mask: Any, output_hidden_states: bool, return_dict: bool) -> Any:
        assert output_hidden_states is True
        assert return_dict is True
        length = len(input_ids.tolist()[0])
        hidden_states = []
        for layer in range(3):
            layer_values = [[[float(layer), float(pos)] for pos in range(length)]]
            hidden_states.append(_FakeTensor(layer_values))
        return _FakeModelOutput(tuple(hidden_states))


def test_ankh3_preprocessor_supports_configurable_prefix() -> None:
    pre = Ankh3Preprocessor(prefix="[S2S]")
    processed = pre.preprocess("acduzob")
    assert processed.startswith("[S2S]")
    assert processed == "[S2S]ACDXXXX"


def test_ankh3_preprocessor_rejects_invalid_prefix() -> None:
    with pytest.raises(EmbeddingInputError):
        Ankh3Preprocessor(prefix="[BAD]")


def test_ankh3_generator_raises_dependency_error_when_transformers_missing(
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
        Ankh3EmbeddingGenerator()


def test_ankh3_generate_returns_per_residue_matrix_without_pooling(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_torch = types.SimpleNamespace(
        no_grad=lambda: _FakeNoGrad(),
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    generator = Ankh3EmbeddingGenerator(
        tokenizer=_FakeTokenizer(),
        model=_FakeModel(),
        prefix="[NLU]",
        device="cpu",
    )
    result = generator.generate([GenerationInput(id="P1", sequence="ACDE")], layer_index=[0, 2])

    assert result.errors == []
    assert [record.layer_index for record in result.records] == [0, 2]
    assert all(record.shape[1] == 2 for record in result.records)
    assert all(record.shape[0] >= 1 for record in result.records)
    assert isinstance(result.records[0].embedding[0], list)
    assert result.model_metadata is not None
    assert result.model_metadata.parameters is not None
    assert result.model_metadata.parameters.get("prefix") == "[NLU]"


def test_ankh3_available_layers_and_count() -> None:
    generator = Ankh3EmbeddingGenerator(
        tokenizer=_FakeTokenizer(),
        model=_FakeModel(),
        device="cpu",
    )
    assert generator.available_layers() == [0, 1, 2]
    assert generator.num_layers() == 3
