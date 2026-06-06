from __future__ import annotations

import builtins
import sys
import types
from typing import Any

import pytest

from CBBIO.embeddings import EmbeddingDependencyError, GenerationInput
from CBBIO.embeddings_esm1b import Esm1bEmbeddingGenerator, Esm1bModelAdapter, Esm1bPreprocessor


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

    def to(self, device: Any) -> "_FakeTensor":
        return self

    def tolist(self) -> Any:
        return self.data


class _FakeAlphabet:
    padding_idx = 0

    def get_batch_converter(self) -> Any:
        def _convert(data: Any) -> Any:
            labels = [str(label) for label, _sequence in data]
            sequences = [str(sequence) for _label, sequence in data]
            max_len = max(len(sequence) for sequence in sequences)
            tokens = []
            for sequence in sequences:
                residue_tokens = list(range(2, 2 + len(sequence)))
                row = [1] + residue_tokens + [1]
                row.extend([self.padding_idx] * (max_len + 2 - len(row)))
                tokens.append(row)
            return (labels, sequences, _FakeTensor(tokens))

        return _convert


class _FakeModel:
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
        token_rows = tokens.tolist()
        seq_len = len(token_rows[0])
        reps = {}
        for layer in repr_layers:
            reps[layer] = _FakeTensor(
                [
                    [[float(layer), float(row_index * 100 + pos)] for pos in range(seq_len)]
                    for row_index, _row in enumerate(token_rows)
                ]
            )
        return {"representations": reps}


class _FakeHfOutput:
    def __init__(self, hidden_states: Any) -> None:
        self.hidden_states = hidden_states


class _FakeLayerNorm:
    def __call__(self, value: Any) -> Any:
        return value + 100.0


class _FakeHfEncoder:
    emb_layer_norm_after = _FakeLayerNorm()


class _FakeHfModel:
    encoder = _FakeHfEncoder()

    def __call__(self, **kwargs: Any) -> _FakeHfOutput:
        import torch

        _ = kwargs
        hidden_states = [torch.full((1, 4, 1), float(index)) for index in range(34)]
        return _FakeHfOutput(hidden_states)


def test_esm1b_preprocessor_normalizes_sequence() -> None:
    pre = Esm1bPreprocessor()
    assert pre.preprocess("acduzob") == "ACDXXXX"


def test_esm1b_generator_raises_dependency_error_when_esm_missing(
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
        Esm1bEmbeddingGenerator()


def test_esm1b_generate_returns_per_residue_matrices_without_pooling() -> None:
    generator = Esm1bEmbeddingGenerator(
        model=_FakeModel(),
        alphabet=_FakeAlphabet(),
    )
    result = generator.generate([GenerationInput(id="P1", sequence="ACDE")], layer_index=[33], fail_fast=True)

    assert result.errors == []
    assert [record.layer_index for record in result.records] == [33]
    assert result.records[0].shape == (4, 2)
    assert isinstance(result.records[0].embedding[0], list)


def test_esm1b_generate_slices_variable_length_batches_to_residues() -> None:
    generator = Esm1bEmbeddingGenerator(
        model=_FakeModel(),
        alphabet=_FakeAlphabet(),
    )
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


def test_esm1b_cls_pooler_returns_leading_special_token() -> None:
    generator = Esm1bEmbeddingGenerator(
        model=_FakeModel(),
        alphabet=_FakeAlphabet(),
    )
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


def test_esm1b_hf_adapter_normalizes_intermediate_hidden_states_only() -> None:
    import torch

    adapter = Esm1bModelAdapter(_FakeHfModel(), normalize_hf_hidden_states=True)
    result = adapter.infer(
        {
            "input_ids": torch.tensor([[0, 5, 23, 2]]),
            "attention_mask": torch.tensor([[1, 1, 1, 1]]),
        },
        layer_index=[0, 1, 33],
    )

    assert result["sample_spans"] == [(1, 3)]
    assert float(result["layers"][0][0, 0, 0]) == 100.0
    assert float(result["layers"][1][0, 0, 0]) == 101.0
    assert float(result["layers"][33][0, 0, 0]) == 33.0


def test_esm1b_hf_adapter_can_return_raw_intermediate_hidden_states() -> None:
    import torch

    adapter = Esm1bModelAdapter(_FakeHfModel(), normalize_hf_hidden_states=False)
    result = adapter.infer(
        {
            "input_ids": torch.tensor([[0, 5, 23, 2]]),
            "attention_mask": torch.tensor([[1, 1, 1, 1]]),
        },
        layer_index=[0, 1, 33],
    )

    assert float(result["layers"][0][0, 0, 0]) == 0.0
    assert float(result["layers"][1][0, 0, 0]) == 1.0
    assert float(result["layers"][33][0, 0, 0]) == 33.0


def test_esm1b_generator_records_requested_dtype(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_torch = types.SimpleNamespace(
        float32="float32",
        float16="float16",
        bfloat16="bfloat16",
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    model = _FakeModel()
    generator = Esm1bEmbeddingGenerator(
        model=model,
        alphabet=_FakeAlphabet(),
        device="cuda:0",
        dtype="float16",
    )

    assert generator.model_metadata.parameters is not None
    assert generator.model_metadata.parameters["torch_dtype"] == "float16"
    assert model.to_kwargs == {"device": "cuda:0", "dtype": "float16"}


def test_esm1b_available_layers_and_count() -> None:
    generator = Esm1bEmbeddingGenerator(
        model=_FakeModel(),
        alphabet=_FakeAlphabet(),
    )
    assert generator.available_layers() == list(range(34))
    assert generator.num_layers() == 34


def test_esm1b_generator_falls_back_to_transformers_when_pretrained_loader_missing(
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

    generator = Esm1bEmbeddingGenerator(model_name="esm1b_t33_650M_UR50S", device="cpu")

    assert generator.model_metadata.model_name == "esm1b_t33_650M_UR50S"
    assert _FakeAutoModel.observed_name == "facebook/esm-1b"
    assert _FakeAutoTokenizer.observed_name == "facebook/esm-1b"
