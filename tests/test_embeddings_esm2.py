from __future__ import annotations

import builtins
import sys
from typing import Any

import pytest

from CBBIO.embeddings import EmbeddingDependencyError, GenerationInput
from CBBIO.embeddings_esm2 import Esm2EmbeddingGenerator, Esm2Preprocessor


class _FakeScalar:
    def __init__(self, value: float) -> None:
        self.value = value

    def item(self) -> float:
        return self.value


class _FakeTensor:
    def __init__(self, data: Any) -> None:
        self.data = data

    def __getitem__(self, item: Any) -> "_FakeTensor":
        if isinstance(item, tuple):
            first = item[0]
            second = item[1] if len(item) > 1 else slice(None)
            base = self.data[first]
            return _FakeTensor(base[second])
        return _FakeTensor(self.data[item])

    def __ne__(self, other: Any) -> "_FakeTensor":
        _ = other
        converted = []
        for row in self.data:
            converted.append([1 if val != 0 else 0 for val in row])
        return _FakeTensor(converted)

    def sum(self, dim: int) -> "_FakeTensor":
        if dim != 1:
            raise ValueError("fake tensor supports dim=1 only")
        return _FakeTensor([sum(row) for row in self.data])

    def item(self) -> float:
        if isinstance(self.data, (int, float)):
            return float(self.data)
        raise TypeError("not a scalar")

    def tolist(self) -> Any:
        return self.data


class _FakeAlphabet:
    padding_idx = 0

    def get_batch_converter(self) -> Any:
        def _convert(data: Any) -> Any:
            _ = data
            # tokens: BOS=1, residues=2..5, EOS=1, PAD=0
            return (["query"], ["ACDE"], _FakeTensor([[1, 2, 3, 4, 5, 1, 0]]))

        return _convert


class _FakeModel:
    num_layers = 3

    def to(self, _device: str) -> "_FakeModel":
        return self

    def eval(self) -> None:
        return None

    def __call__(self, tokens: Any, repr_layers: list[int], return_contacts: bool) -> dict[str, Any]:
        _ = return_contacts
        seq_len = len(tokens.tolist()[0])
        reps = {}
        for layer in repr_layers:
            # [batch, seq_len, hidden]
            reps[layer] = _FakeTensor([[[float(layer), float(pos)] for pos in range(seq_len)]])
        return {"representations": reps}


def test_esm2_preprocessor_normalizes_sequence() -> None:
    pre = Esm2Preprocessor()
    assert pre.preprocess("acduzob") == "ACDXXXX"


def test_esm2_generator_raises_dependency_error_when_esm_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_import = builtins.__import__

    def _raising_import(name: str, globals: Any = None, locals: Any = None, fromlist: Any = (), level: int = 0) -> Any:
        if name == "esm" or name.startswith("esm."):
            raise ModuleNotFoundError("No module named 'esm'")
        return original_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", _raising_import)
    monkeypatch.delitem(sys.modules, "esm", raising=False)

    with pytest.raises(EmbeddingDependencyError):
        Esm2EmbeddingGenerator()


def test_esm2_generate_returns_per_residue_matrices_without_pooling() -> None:
    generator = Esm2EmbeddingGenerator(
        model=_FakeModel(),
        alphabet=_FakeAlphabet(),
    )
    result = generator.generate([GenerationInput(id="P1", sequence="ACDE")], layer_index=[3], fail_fast=True)

    assert result.errors == []
    assert [record.layer_index for record in result.records] == [3]
    # residues are 4 after removing BOS/EOS
    assert result.records[0].shape == (4, 2)
    assert isinstance(result.records[0].embedding[0], list)


def test_esm2_available_layers_and_count() -> None:
    generator = Esm2EmbeddingGenerator(
        model=_FakeModel(),
        alphabet=_FakeAlphabet(),
    )
    assert generator.available_layers() == [0, 1, 2, 3]
    assert generator.num_layers() == 4
