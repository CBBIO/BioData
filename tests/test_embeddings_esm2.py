from __future__ import annotations

import builtins
import sys
import types
from typing import Any

import pytest

from CBBIO.embeddings import EmbeddingDependencyError, GenerationInput
from CBBIO.embeddings_esm2 import Esm2EmbeddingGenerator, Esm2Preprocessor, Esm2TokenizerAdapter


class _FakeScalar:
    def __init__(self, value: float) -> None:
        self.value = value

    def item(self) -> float:
        return self.value


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

    def __ne__(self, other: Any) -> "_FakeTensor":
        _ = other
        converted = []
        for row in self.data:
            converted.append([1 if val != 0 else 0 for val in row])
        return _FakeTensor(converted, device=self.device)

    def sum(self, dim: int) -> "_FakeTensor":
        if dim != 1:
            raise ValueError("fake tensor supports dim=1 only")
        return _FakeTensor([sum(row) for row in self.data], device=self.device)

    def to(self, device: str) -> "_FakeTensor":
        return _FakeTensor(self.data, device=str(device))

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

    def __init__(self) -> None:
        self.to_args: tuple[Any, ...] | None = None
        self.to_kwargs: dict[str, Any] | None = None

    def to(self, *args: Any, **kwargs: Any) -> "_FakeModel":
        self.to_args = args
        self.to_kwargs = kwargs
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


def test_esm2_native_tokenizer_moves_tokens_and_lens_to_device() -> None:
    def _convert(data: Any) -> Any:
        _ = data
        return (["query"], ["ACDE"], _FakeTensor([[1, 2, 3, 4, 5, 1, 0]]))

    tokenizer = Esm2TokenizerAdapter(
        batch_converter=_convert,
        padding_idx=0,
        device="cuda:0",
    )

    result = tokenizer.tokenize_many(["ACDE"])

    assert result["tokens"].device == "cuda:0"
    assert result["lens"].device == "cuda:0"


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


def test_esm2_generator_records_requested_dtype(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_torch = types.SimpleNamespace(
        float32="float32",
        float16="float16",
        bfloat16="bfloat16",
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    model = _FakeModel()
    generator = Esm2EmbeddingGenerator(
        model=model,
        alphabet=_FakeAlphabet(),
        device="cuda:0",
        dtype="float16",
    )

    assert generator.model_metadata.parameters is not None
    assert generator.model_metadata.parameters["torch_dtype"] == "float16"
    assert model.to_kwargs == {"device": "cuda:0", "dtype": "float16"}


def test_esm2_available_layers_and_count() -> None:
    generator = Esm2EmbeddingGenerator(
        model=_FakeModel(),
        alphabet=_FakeAlphabet(),
    )
    assert generator.available_layers() == [0, 1, 2, 3]
    assert generator.num_layers() == 4


def test_esm2_generator_falls_back_to_transformers_when_pretrained_loader_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _FakeAutoModel:
        observed_name: str | None = None

        @staticmethod
        def from_pretrained(name: str) -> _FakeModel:
            _FakeAutoModel.observed_name = name
            return _FakeModel()

    class _FakeAutoTokenizer:
        observed_name: str | None = None

        @staticmethod
        def from_pretrained(name: str) -> _FakeAlphabet:
            _FakeAutoTokenizer.observed_name = name
            return _FakeAlphabet()

    fake_transformers = types.SimpleNamespace(
        AutoModel=_FakeAutoModel,
        AutoTokenizer=_FakeAutoTokenizer,
    )
    fake_esm = types.SimpleNamespace()

    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)
    monkeypatch.setitem(sys.modules, "esm", fake_esm)
    monkeypatch.setitem(sys.modules, "esm.pretrained", types.SimpleNamespace())

    generator = Esm2EmbeddingGenerator(model_name="esm2_t33_650M_UR50D", device="cpu")

    assert generator.model_metadata.model_name == "esm2_t33_650M_UR50D"
    assert _FakeAutoModel.observed_name == "facebook/esm2_t33_650M_UR50D"
    assert _FakeAutoTokenizer.observed_name == "facebook/esm2_t33_650M_UR50D"
