from __future__ import annotations

import builtins
import sys
import types
from typing import Any

import pytest

from CBBIO.embeddings import EmbeddingDependencyError, GenerationInput
from CBBIO.embeddings_esm2 import ESM2_HF_MODEL_NAMES, Esm2EmbeddingGenerator, Esm2Preprocessor, Esm2TokenizerAdapter


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
    num_hidden_layers = 3


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
        for layer in range(4):
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


def test_esm2_preprocessor_normalizes_sequence() -> None:
    pre = Esm2Preprocessor()
    assert pre.preprocess("acduzob") == "ACDXXXX"


def test_esm2_tokenizer_moves_inputs_to_device() -> None:
    tokenizer = Esm2TokenizerAdapter(_FakeTokenizer(), device="cuda:0")

    result = tokenizer.tokenize_many(["ACDE"])

    assert result["input_ids"].device == "cuda:0"
    assert result["attention_mask"].device == "cuda:0"


def test_esm2_generator_raises_dependency_error_when_transformers_missing(
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
        Esm2EmbeddingGenerator()


def test_esm2_generate_returns_per_residue_matrices_without_pooling() -> None:
    generator = Esm2EmbeddingGenerator(model_name="facebook/test-esm2", model=_FakeHfModel(), tokenizer=_FakeTokenizer())
    result = generator.generate([GenerationInput(id="P1", sequence="ACDE")], layer_index=[3], fail_fast=True)

    assert result.errors == []
    assert [record.layer_index for record in result.records] == [3]
    assert result.records[0].shape == (4, 2)
    assert result.records[0].embedding == [[3.0, 1.0], [3.0, 2.0], [3.0, 3.0], [3.0, 4.0]]


def test_esm2_generate_slices_variable_length_batches_to_residues() -> None:
    generator = Esm2EmbeddingGenerator(model_name="facebook/test-esm2", model=_FakeHfModel(), tokenizer=_FakeTokenizer())
    result = generator.generate(
        [
            GenerationInput(id="P1", sequence="ACDE"),
            GenerationInput(id="P2", sequence="AC"),
        ],
        layer_index=[3],
        fail_fast=True,
    )

    assert [record.id for record in result.records] == ["P1", "P2"]
    assert [record.shape for record in result.records] == [(4, 2), (2, 2)]
    assert result.records[1].embedding == [[3.0, 101.0], [3.0, 102.0]]


def test_esm2_mean_and_cls_poolers_use_expected_tokens() -> None:
    generator = Esm2EmbeddingGenerator(model_name="facebook/test-esm2", model=_FakeHfModel(), tokenizer=_FakeTokenizer())

    mean = generator.generate([GenerationInput(id="P1", sequence="ACDE")], layer_index=[3], pooler="mean", fail_fast=True)
    cls = generator.generate([GenerationInput(id="P1", sequence="ACDE")], layer_index=[3], pooler="cls", fail_fast=True)

    assert mean.records[0].embedding == [3.0, 2.5]
    assert cls.records[0].embedding == [3.0, 0.0]


def test_esm2_layer_none_and_negative_indices() -> None:
    generator = Esm2EmbeddingGenerator(model_name="facebook/test-esm2", model=_FakeHfModel(), tokenizer=_FakeTokenizer())

    all_layers = generator.generate([GenerationInput(id="P1", sequence="AC")], layer_index=None, pooler="cls", fail_fast=True)
    last = generator.generate([GenerationInput(id="P1", sequence="AC")], layer_index=[-1], pooler="cls", fail_fast=True)

    assert [record.layer_index for record in all_layers.records] == [0, 1, 2, 3]
    assert [record.layer_index for record in last.records] == [3]


def test_esm2_generator_records_requested_dtype(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_torch = types.SimpleNamespace(
        float32="float32",
        float16="float16",
        bfloat16="bfloat16",
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    model = _FakeHfModel()
    generator = Esm2EmbeddingGenerator(
        model=model,
        tokenizer=_FakeTokenizer(),
        device="cuda:0",
        dtype="float16",
    )

    assert generator.model_metadata.parameters is not None
    assert generator.model_metadata.parameters["torch_dtype"] == "float16"
    assert model.to_kwargs == {"device": "cuda:0", "dtype": "float16"}


def test_esm2_available_layers_and_count() -> None:
    generator = Esm2EmbeddingGenerator(model_name="facebook/test-esm2", model=_FakeHfModel(), tokenizer=_FakeTokenizer())
    assert generator.available_layers() == [0, 1, 2, 3]
    assert generator.num_layers() == 4


def test_esm2_generator_loads_hf_model_and_tokenizer_for_all_aliases(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed_models: list[str] = []
    observed_tokenizers: list[str] = []

    class _FakeAutoModel:
        @staticmethod
        def from_pretrained(name: str, **kwargs: Any) -> _FakeHfModel:
            _ = kwargs
            observed_models.append(name)
            return _FakeHfModel()

    class _FakeAutoTokenizer:
        @staticmethod
        def from_pretrained(name: str) -> _FakeTokenizer:
            observed_tokenizers.append(name)
            return _FakeTokenizer()

    fake_transformers = types.SimpleNamespace(
        AutoModel=_FakeAutoModel,
        AutoTokenizer=_FakeAutoTokenizer,
    )
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)

    for alias, reference in ESM2_HF_MODEL_NAMES.items():
        generator = Esm2EmbeddingGenerator(model_name=alias, device="cpu")
        assert generator.model_metadata.model_reference == reference

    assert observed_models == list(ESM2_HF_MODEL_NAMES.values())
    assert observed_tokenizers == list(ESM2_HF_MODEL_NAMES.values())
