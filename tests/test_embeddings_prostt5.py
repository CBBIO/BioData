from __future__ import annotations

import builtins
import sys
import types
from typing import Any

import pytest

from CBBIO.embeddings import EmbeddingDependencyError, GenerationInput
from CBBIO.embeddings_prostt5 import ProstT5EmbeddingGenerator, ProstT5Preprocessor


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
        spaced = sequences[0].split()
        assert spaced[0] == "<AA2fold>"
        # prefix + residues + end token
        length = len(spaced) + 1
        return {"input_ids": _FakeTensor([[1] * length]), "attention_mask": _FakeTensor([[1] * length])}


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
        length = len(input_ids.tolist()[0])  # includes prefix + residues + end
        hidden_states = []
        for layer in range(3):
            layer_values = [[[float(layer), float(pos)] for pos in range(length)]]
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
    # residues only: 4 rows, hidden dimension 2 in fake model
    assert all(record.shape == (4, 2) for record in result.records)
    assert isinstance(result.records[0].embedding[0], list)
    assert model.did_float is True
    assert model.did_half is False


def test_prostt5_generator_available_layers_and_count(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_torch = types.SimpleNamespace(
        tensor=lambda values: _FakeTensor(values),
        no_grad=lambda: _FakeNoGrad(),
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    generator = ProstT5EmbeddingGenerator(
        tokenizer=_FakeTokenizer(),
        model=_FakeModel(),
        device="cpu",
    )
    assert generator.available_layers() == [0, 1, 2]
    assert generator.num_layers() == 3


def test_prostt5_loader_keeps_tied_embeddings_when_loading_hf_checkpoint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _FakeConfig:
        def __init__(self) -> None:
            self.tie_word_embeddings = True
            self.num_layers = 2

    class _FakeAutoConfig:
        @staticmethod
        def from_pretrained(_name: str) -> _FakeConfig:
            return _FakeConfig()

    class _FakeAutoTokenizer:
        @staticmethod
        def from_pretrained(_name: str, do_lower_case: bool = False) -> _FakeTokenizer:
            assert do_lower_case is False
            return _FakeTokenizer()

    class _FakeT5EncoderModel:
        observed_tie_flag: bool | None = None

        @staticmethod
        def from_pretrained(_name: str, config: _FakeConfig) -> _FakeModel:
            _FakeT5EncoderModel.observed_tie_flag = bool(config.tie_word_embeddings)
            return _FakeModel()

    fake_transformers = types.SimpleNamespace(
        AutoConfig=_FakeAutoConfig,
        AutoTokenizer=_FakeAutoTokenizer,
        T5EncoderModel=_FakeT5EncoderModel,
    )
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)

    generator = ProstT5EmbeddingGenerator(model_name="Rostlab/ProstT5", device="cpu")

    assert generator.model_metadata.model_name == "Rostlab/ProstT5"
    assert _FakeT5EncoderModel.observed_tie_flag is True
