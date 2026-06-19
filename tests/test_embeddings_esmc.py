from __future__ import annotations

import builtins
import sys
import types
from typing import Any, Sequence, cast

import pytest

from CBBIO.embeddings import EmbeddingDependencyError, GenerationInput
from CBBIO.embeddings_esmc import (
    ESMC_HF_MODEL_NAMES,
    EsmcEmbeddingGenerator,
    EsmcPreprocessor,
    register_hf_esmc_architecture,
)


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

    def __ne__(self, other: object) -> "_FakeTensor":
        if isinstance(self.data, list):
            return _FakeTensor([[value != other for value in row] for row in self.data], device=self.device)
        return _FakeTensor(self.data != other, device=self.device)

    def to(self, device: str) -> "_FakeTensor":
        return _FakeTensor(self.data, device=str(device))

    def tolist(self) -> Any:
        return self.data


class _FakeTokenizerState:
    pad_token_id = 1


class _FakeSdkOutput:
    def __init__(self, hidden_states: Any) -> None:
        self.hidden_states = hidden_states


class _FakeSdkModel:
    def __init__(self) -> None:
        self.tokenizer = _FakeTokenizerState()
        self.to_args: tuple[Any, ...] | None = None
        self.to_kwargs: dict[str, Any] | None = None

    def to(self, *args: Any, **kwargs: Any) -> "_FakeSdkModel":
        self.to_args = args
        self.to_kwargs = kwargs
        return self

    def eval(self) -> None:
        return None

    def _tokenize(self, sequences: list[str]) -> _FakeTensor:
        max_len = max(len(sequence) for sequence in sequences) + 2
        token_rows = []
        for sequence in sequences:
            row = [0] + list(range(5, 5 + len(sequence))) + [2]
            row.extend([1] * (max_len - len(row)))
            token_rows.append(row)
        return _FakeTensor(token_rows)

    def forward(self, **kwargs: Any) -> _FakeSdkOutput:
        token_rows = kwargs["sequence_tokens"].tolist()
        seq_len = len(token_rows[0])
        hidden_states = []
        for layer in range(30):
            hidden_states.append(
                _FakeTensor(
                    [
                        [[float(layer), float(row_index * 100 + pos)] for pos in range(seq_len)]
                        for row_index, _row in enumerate(token_rows)
                    ]
                )
            )
        return _FakeSdkOutput(hidden_states)


class _FakeHfModel:
    def to(self, *_args: Any, **_kwargs: Any) -> "_FakeHfModel":
        return self

    def eval(self) -> None:
        return None


class _FakeTokenizer:
    name_or_path = "fake-esmc-tokenizer"


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
        EsmcEmbeddingGenerator(model_name="sdk-only-test")


def test_esmc_generate_returns_per_residue_matrix_without_pooling() -> None:
    generator = EsmcEmbeddingGenerator(
        model=_FakeSdkModel(),
        model_name="esmc_300m",
    )

    result = generator.generate([GenerationInput(id="Q1", sequence="ACDE")], layer_index=[0, 1], fail_fast=True)
    assert result.errors == []
    assert [record.layer_index for record in result.records] == [0, 1]
    assert result.records[0].shape == (4, 2)
    second_embedding = cast(Sequence[Sequence[float]], result.records[1].embedding)
    assert second_embedding[0][0] == 1.0
    assert isinstance(result.records[0].embedding[0], list)


def test_esmc_mean_and_cls_poolers_use_expected_tokens() -> None:
    generator = EsmcEmbeddingGenerator(model=_FakeSdkModel(), model_name="esmc_300m")

    mean = generator.generate([GenerationInput(id="Q1", sequence="ACDE")], layer_index=[1], pooler="mean", fail_fast=True)
    cls = generator.generate([GenerationInput(id="Q1", sequence="ACDE")], layer_index=[1], pooler="cls", fail_fast=True)

    assert mean.errors == []
    assert mean.records[0].shape == (2,)
    assert mean.records[0].embedding == [1.0, 2.5]
    assert cls.records[0].shape == (2,)
    assert cls.records[0].embedding == [1.0, 0.0]


def test_esmc_available_layers_uses_family_hint() -> None:
    generator = EsmcEmbeddingGenerator(model=_FakeSdkModel(), model_name="esmc_300m")
    assert generator.num_layers() == 30
    assert generator.available_layers()[:3] == [0, 1, 2]


def test_esmc_layer_none_requests_all_known_layers() -> None:
    generator = EsmcEmbeddingGenerator(model=_FakeSdkModel(), model_name="esmc_300m")

    result = generator.generate([GenerationInput(id="Q1", sequence="AC")], layer_index=None, pooler="cls", fail_fast=True)

    assert [record.layer_index for record in result.records] == list(range(30))


def test_esmc_negative_layer_index_resolves_from_available_layers() -> None:
    generator = EsmcEmbeddingGenerator(model=_FakeSdkModel(), model_name="esmc_300m")

    result = generator.generate([GenerationInput(id="Q1", sequence="AC")], layer_index=[-1], pooler="cls", fail_fast=True)

    assert [record.layer_index for record in result.records] == [29]


def test_esmc_generator_loads_hf_model_and_tokenizer_for_aliases(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed_models: list[str] = []
    observed_kwargs: list[dict[str, Any]] = []
    observed_tokenizers: list[str] = []

    class _FakeAutoModel:
        @staticmethod
        def register(config_class: type[object], model_class: type[object], exist_ok: bool = False) -> None:
            _ = config_class, model_class, exist_ok

        @staticmethod
        def from_pretrained(name: str, **kwargs: Any) -> _FakeHfModel:
            observed_models.append(name)
            observed_kwargs.append(dict(kwargs))
            return _FakeHfModel()

    class _FakeAutoConfig:
        @staticmethod
        def register(model_type: str, config_class: type[object], exist_ok: bool = False) -> None:
            _ = model_type, config_class, exist_ok

    class _FakeEsmConfig:
        def __init__(self, **kwargs: Any) -> None:
            for key, value in kwargs.items():
                setattr(self, key, value)

    class _FakeEsmModel:
        pass

    class _FakeAutoTokenizer:
        @staticmethod
        def from_pretrained(name: str, **kwargs: Any) -> _FakeTokenizer:
            _ = kwargs
            observed_tokenizers.append(name)
            return _FakeTokenizer()

    fake_transformers = types.SimpleNamespace(
        AutoConfig=_FakeAutoConfig,
        AutoModel=_FakeAutoModel,
        AutoTokenizer=_FakeAutoTokenizer,
        EsmConfig=_FakeEsmConfig,
        EsmModel=_FakeEsmModel,
    )
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)

    for alias, reference in ESMC_HF_MODEL_NAMES.items():
        generator = EsmcEmbeddingGenerator(model_name=alias, device="cpu")
        assert generator.model_metadata.model_reference == reference

    assert observed_models == list(ESMC_HF_MODEL_NAMES.values())
    assert observed_kwargs == [{} for _ in ESMC_HF_MODEL_NAMES]
    assert observed_tokenizers == list(ESMC_HF_MODEL_NAMES.values())


def test_esmc_generator_forwards_explicit_from_pretrained_kwargs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed_kwargs: list[dict[str, Any]] = []

    class _FakeAutoModel:
        @staticmethod
        def register(config_class: type[object], model_class: type[object], exist_ok: bool = False) -> None:
            _ = config_class, model_class, exist_ok

        @staticmethod
        def from_pretrained(name: str, **kwargs: Any) -> _FakeHfModel:
            _ = name
            observed_kwargs.append(dict(kwargs))
            return _FakeHfModel()

    class _FakeAutoConfig:
        @staticmethod
        def register(model_type: str, config_class: type[object], exist_ok: bool = False) -> None:
            _ = model_type, config_class, exist_ok

    class _FakeEsmConfig:
        def __init__(self, **kwargs: Any) -> None:
            for key, value in kwargs.items():
                setattr(self, key, value)

    class _FakeEsmModel:
        pass

    class _FakeAutoTokenizer:
        @staticmethod
        def from_pretrained(name: str, **kwargs: Any) -> _FakeTokenizer:
            _ = name, kwargs
            return _FakeTokenizer()

    fake_transformers = types.SimpleNamespace(
        AutoConfig=_FakeAutoConfig,
        AutoModel=_FakeAutoModel,
        AutoTokenizer=_FakeAutoTokenizer,
        EsmConfig=_FakeEsmConfig,
        EsmModel=_FakeEsmModel,
    )
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)

    EsmcEmbeddingGenerator(
        model_name="esmc_300m",
        device="cpu",
        from_pretrained_kwargs={"low_cpu_mem_usage": True},
    )

    assert observed_kwargs == [{"low_cpu_mem_usage": True}]


def test_register_hf_esmc_architecture_uses_matching_model_config_class(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registrations: list[tuple[type[object], type[object]]] = []

    class _FakeAutoModel:
        @staticmethod
        def register(config_class: type[object], model_class: type[object], exist_ok: bool = False) -> None:
            _ = exist_ok
            if getattr(model_class, "config_class", None) is not config_class:
                raise ValueError("model config_class must match registered config_class")
            registrations.append((config_class, model_class))

    class _FakeAutoConfig:
        @staticmethod
        def register(model_type: str, config_class: type[object], exist_ok: bool = False) -> None:
            _ = model_type, config_class, exist_ok

    class _FakeEsmConfig:
        pass

    class _FakeEsmModel:
        config_class = _FakeEsmConfig

    fake_transformers = types.SimpleNamespace(
        AutoConfig=_FakeAutoConfig,
        AutoModel=_FakeAutoModel,
        EsmConfig=_FakeEsmConfig,
        EsmModel=_FakeEsmModel,
    )
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)

    register_hf_esmc_architecture()
    register_hf_esmc_architecture()

    assert len(registrations) == 2
    assert registrations[0] == registrations[1]


def test_register_hf_esmc_architecture_resolves_cached_config() -> None:
    from transformers import AutoConfig

    register_hf_esmc_architecture()
    config = AutoConfig.from_pretrained("biohub/ESMC-300M", local_files_only=True)

    assert config.model_type == "esmc"
    assert config.hidden_size == 960
    assert config.num_hidden_layers == 30
    assert config.num_attention_heads == 15
