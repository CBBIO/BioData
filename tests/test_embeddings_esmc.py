from __future__ import annotations

import builtins
import sys
import types
from typing import Any

import pytest

from CBBIO.embeddings import EmbeddingDependencyError, GenerationInput
from CBBIO.embeddings_esmc import EsmcEmbeddingGenerator, EsmcPreprocessor


class _FakeScalar:
    def __init__(self, value: float) -> None:
        self.value = value

    def item(self) -> float:
        return self.value


class _FakeTensor:
    def __init__(self, data: Any) -> None:
        self.data = data
        self.shape = self._shape_of(data)

    def _shape_of(self, value: Any) -> tuple[int, ...]:
        if isinstance(value, list) and value:
            inner = self._shape_of(value[0])
            return (len(value),) + inner
        if isinstance(value, list):
            return (0,)
        return ()

    def __getitem__(self, item: Any) -> "_FakeTensor":
        if isinstance(item, tuple):
            first = item[0]
            second = item[1] if len(item) > 1 else slice(None)
            base = self.data[first]
            return _FakeTensor(base[second])
        return _FakeTensor(self.data[item])

    def tolist(self) -> Any:
        return self.data

    def sum(self) -> _FakeScalar:
        if isinstance(self.data, list):
            return _FakeScalar(float(sum(self.data)))
        return _FakeScalar(float(self.data))


class _FakeTokenizer:
    def __call__(self, sequence: str, add_special_tokens: bool, return_tensors: str, is_split_into_words: bool) -> Any:
        assert add_special_tokens is True
        assert return_tensors == "pt"
        assert is_split_into_words is False
        assert sequence.isupper()
        return {"input_ids": _FakeTensor([[1, 2, 3]]), "attention_mask": _FakeTensor([[1, 1, 1]])}


class _FakeESMProtein:
    def __init__(self, sequence: str) -> None:
        self.sequence = sequence


class _FakeLogitsConfig:
    def __init__(self, sequence: bool, return_embeddings: bool) -> None:
        self.sequence = sequence
        self.return_embeddings = return_embeddings


class _FakeLogitsOutput:
    def __init__(self, embeddings: Any) -> None:
        self.embeddings = embeddings


class _FakeClient:
    def __init__(self) -> None:
        self.device = "cpu"

    def to(self, device: str) -> "_FakeClient":
        self.device = device
        return self

    def encode(self, protein: _FakeESMProtein) -> Any:
        return protein.sequence

    def logits(self, protein_tensor: Any, _config: _FakeLogitsConfig) -> _FakeLogitsOutput:
        # Return [layers, residues, hidden] with two layers.
        residues = len(str(protein_tensor))
        layer0 = [[0.0, float(i)] for i in range(residues)]
        layer1 = [[1.0, float(i)] for i in range(residues)]
        return _FakeLogitsOutput(_FakeTensor([layer0, layer1]))


def test_esmc_preprocessor_replaces_ambiguous_amino_acids() -> None:
    pre = EsmcPreprocessor()
    assert pre.preprocess("acduzob") == "ACDXXXX"


def test_esmc_generator_raises_dependency_error_when_sdk_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_import = builtins.__import__

    def _raising_import(name: str, globals: Any = None, locals: Any = None, fromlist: Any = (), level: int = 0) -> Any:
        if name == "esm.models.esmc" or name.startswith("esm.models.esmc"):
            raise ModuleNotFoundError("No module named 'esm'")
        return original_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", _raising_import)
    monkeypatch.delitem(sys.modules, "esm.models.esmc", raising=False)

    with pytest.raises(EmbeddingDependencyError):
        EsmcEmbeddingGenerator(client=None)


def test_esmc_generate_returns_per_residue_matrix_without_pooling() -> None:
    client = _FakeClient()
    generator = EsmcEmbeddingGenerator(
        client=client,
        model_name="esmc_300m",
        from_pretrained_kwargs={},
    )
    # Inject fake SDK classes to avoid importing esm in tests.
    generator.tokenizer.protein_cls = _FakeESMProtein
    generator.model.logits_config_cls = _FakeLogitsConfig

    result = generator.generate([GenerationInput(id="Q1", sequence="ACDE")], layer_index=[0, 1], fail_fast=True)
    assert result.errors == []
    assert [record.layer_index for record in result.records] == [0, 1]
    assert result.records[0].shape == (4, 2)
    assert isinstance(result.records[0].embedding[0], list)


def test_esmc_available_layers_uses_family_hint() -> None:
    generator = EsmcEmbeddingGenerator(client=_FakeClient(), model_name="esmc_300m")
    assert generator.num_layers() == 30
    assert generator.available_layers()[:3] == [0, 1, 2]
