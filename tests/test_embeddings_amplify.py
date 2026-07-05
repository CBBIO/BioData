from __future__ import annotations

import sys
import types
from typing import Any

import pytest

from CBBIO.embeddings import EmbeddingDependencyError, EmbeddingInputError, GenerationInput
from CBBIO.embeddings.factory import Generator, available_generator_classes, available_generator_models
from CBBIO.embeddings.models import amplify as amplify_module
from CBBIO.embeddings.models.amplify import AMPLIFY_HF_MODEL_NAMES
from CBBIO import AmplifyEmbeddingGenerator, AmplifyPreprocessor, AmplifyTokenizerAdapter


def _install_fake_transformer_engine(monkeypatch: pytest.MonkeyPatch) -> None:
    def _fake_import_module(name: str) -> types.ModuleType:
        if name == "transformer_engine.pytorch":
            return types.ModuleType("transformer_engine.pytorch")
        raise ModuleNotFoundError(name)

    monkeypatch.setattr(amplify_module.importlib, "import_module", _fake_import_module)


def _as_list(value: Any) -> Any:
    tolist = getattr(value, "tolist", None)
    if callable(tolist):
        return tolist()
    return value


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
        self.to_kwargs: dict[str, Any] | None = None
        self.observed_attention_mask: Any | None = None

    def to(self, *args: Any, **kwargs: Any) -> "_FakeHfModel":
        _ = args
        self.to_kwargs = kwargs
        return self

    def eval(self) -> None:
        return None

    def __call__(self, **kwargs: Any) -> _FakeHfOutput:
        input_ids = kwargs["input_ids"]
        token_rows = input_ids.tolist()
        self.observed_attention_mask = kwargs.get("attention_mask")
        seq_len = len(token_rows[0])
        hidden_states = []
        for layer in range(3):
            payload = [
                [[float(layer), float(row_index * 100 + pos)] for pos in range(seq_len)]
                for row_index, _row in enumerate(token_rows)
            ]
            if input_ids.__class__.__module__.startswith("torch"):
                import torch

                hidden_states.append(torch.tensor(payload))
            else:
                hidden_states.append(_FakeTensor(payload))
        return _FakeHfOutput(hidden_states)


class _FakeTokenizer:
    name_or_path = "fake-amplify-tokenizer"

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


class _TorchTokenizer:
    name_or_path = "fake-amplify-tokenizer"

    def __call__(
        self,
        sequences: list[str],
        *,
        add_special_tokens: bool,
        padding: bool,
        return_tensors: str,
    ) -> dict[str, Any]:
        import torch

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
        return {
            "input_ids": torch.tensor(input_ids),
            "attention_mask": torch.tensor(attention_mask),
        }


def test_amplify_preprocessor_normalizes_sequence() -> None:
    pre = AmplifyPreprocessor()
    assert pre.preprocess("acduzob") == "ACDXXXX"


def test_amplify_tokenizer_moves_inputs_to_device() -> None:
    tokenizer = AmplifyTokenizerAdapter(_FakeTokenizer(), device="cuda:0")

    result = tokenizer.tokenize_many(["ACDE"])

    assert result["input_ids"].device == "cuda:0"
    assert result["attention_mask"].device == "cuda:0"


def test_amplify_generate_returns_per_residue_matrices_without_pooling() -> None:
    model = _FakeHfModel()
    generator = AmplifyEmbeddingGenerator(
        model_name="nvidia/AMPLIFY_120M",
        model=model,
        tokenizer=_TorchTokenizer(),
    )
    result = generator.generate(
        [
            GenerationInput(id="P1", sequence="ACDE"),
            GenerationInput(id="P2", sequence="AC"),
        ],
        layer_index=[2],
        fail_fast=True,
    )

    assert result.errors == []
    assert [record.layer_index for record in result.records] == [2, 2]
    assert result.records[0].shape == (4, 2)
    assert _as_list(result.records[0].embedding) == [[2.0, 1.0], [2.0, 2.0], [2.0, 3.0], [2.0, 4.0]]
    assert result.records[1].shape == (2, 2)
    assert _as_list(result.records[1].embedding) == [[2.0, 101.0], [2.0, 102.0]]
    assert model.observed_attention_mask is not None
    assert model.observed_attention_mask.tolist() == [
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, float("-inf"), float("-inf")],
    ]


def test_amplify_unpadded_forward_handles_variable_length_batches(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _FakeHfModel()
    generator = AmplifyEmbeddingGenerator(
        model_name="amplify_350m_chandar",
        model=model,
        tokenizer=_TorchTokenizer(),
    )
    monkeypatch.setattr(amplify_module, "_should_use_unpadded_amplify_forward", lambda attention_mask: True)

    result = generator.generate(
        [
            GenerationInput(id="P1", sequence="ACDE"),
            GenerationInput(id="P2", sequence="AC"),
        ],
        layer_index=[2],
        fail_fast=True,
    )

    assert result.errors == []
    assert [record.id for record in result.records] == ["P1", "P2"]
    assert [record.shape for record in result.records] == [(4, 2), (2, 2)]
    assert _as_list(result.records[0].embedding) == [[2.0, 1.0], [2.0, 2.0], [2.0, 3.0], [2.0, 4.0]]
    assert _as_list(result.records[1].embedding) == [[2.0, 1.0], [2.0, 2.0]]
    assert model.observed_attention_mask is None


def test_amplify_generator_loads_hf_model_and_tokenizer_with_remote_code(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_transformer_engine(monkeypatch)
    observed_models: list[tuple[str, dict[str, Any]]] = []
    observed_tokenizers: list[tuple[str, dict[str, Any]]] = []

    class _FakeAutoModel:
        @staticmethod
        def from_pretrained(name: str, **kwargs: Any) -> _FakeHfModel:
            observed_models.append((name, dict(kwargs)))
            return _FakeHfModel()

    class _FakeAutoTokenizer:
        @staticmethod
        def from_pretrained(name: str, **kwargs: Any) -> _FakeTokenizer:
            observed_tokenizers.append((name, dict(kwargs)))
            return _FakeTokenizer()

    fake_transformers = types.SimpleNamespace(
        AutoModel=_FakeAutoModel,
        AutoTokenizer=_FakeAutoTokenizer,
    )
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)

    for alias, reference in AMPLIFY_HF_MODEL_NAMES.items():
        generator = AmplifyEmbeddingGenerator(model_name=alias, device="cpu")
        assert generator.model_metadata.model_reference == reference
        assert generator.model_metadata.parameters is not None
        assert generator.model_metadata.parameters["trust_remote_code"] is True
        assert generator.model_metadata.parameters["max_sequence_length"] == 2048

    assert observed_models == [
        (reference, {"trust_remote_code": True}) for reference in AMPLIFY_HF_MODEL_NAMES.values()
    ]
    assert observed_tokenizers == [
        (reference, {"trust_remote_code": True}) for reference in AMPLIFY_HF_MODEL_NAMES.values()
    ]


def test_amplify_defaults_cuda_dtype_to_bfloat16() -> None:
    model = _FakeHfModel()
    generator = AmplifyEmbeddingGenerator(
        model_name="amplify_350m_chandar",
        model=model,
        tokenizer=_FakeTokenizer(),
        device="cuda:0",
    )

    assert generator.model_metadata.parameters is not None
    assert generator.model_metadata.parameters["torch_dtype"] == "bfloat16"
    assert generator.model_metadata.parameters["precision_policy"] == "amplify_cuda_default_bfloat16"
    assert model.to_kwargs == {"device": "cuda:0", "dtype": __import__("torch").bfloat16}


def test_amplify_preserves_explicit_cuda_dtype() -> None:
    model = _FakeHfModel()
    generator = AmplifyEmbeddingGenerator(
        model_name="amplify_350m_chandar",
        model=model,
        tokenizer=_FakeTokenizer(),
        device="cuda",
        dtype="float16",
    )

    assert generator.model_metadata.parameters is not None
    assert generator.model_metadata.parameters["torch_dtype"] == "float16"
    assert "precision_policy" not in generator.model_metadata.parameters
    assert model.to_kwargs == {"device": "cuda", "dtype": __import__("torch").float16}


def test_amplify_rejects_float32_on_cuda() -> None:
    with pytest.raises(EmbeddingInputError, match="xFormers attention"):
        AmplifyEmbeddingGenerator(
            model_name="amplify_350m_chandar",
            model=_FakeHfModel(),
            tokenizer=_FakeTokenizer(),
            device="cuda",
            dtype="float32",
        )


def test_amplify_factory_registration() -> None:
    assert "amplify" in available_generator_classes()
    assert available_generator_models("amplify") == [
        "amplify_120m",
        "amplify_350m",
        "nvidia/AMPLIFY_120M",
        "nvidia/AMPLIFY_350M",
        "amplify_120m_chandar",
        "amplify_350m_chandar",
        "chandar-lab/AMPLIFY_120M",
        "chandar-lab/AMPLIFY_350M",
    ]

    generator = Generator(
        model_class="AMPLIFY",
        model=_FakeHfModel(),
        tokenizer=_FakeTokenizer(),
    )

    assert isinstance(generator, AmplifyEmbeddingGenerator)
    assert generator.model_metadata.model_reference == "nvidia/AMPLIFY_120M"


def test_amplify_wraps_missing_remote_code_dependencies(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _fake_import_module(name: str) -> types.ModuleType:
        _ = name
        raise ModuleNotFoundError("No module named 'transformer_engine'")

    monkeypatch.setattr(amplify_module.importlib, "import_module", _fake_import_module)

    with pytest.raises(EmbeddingDependencyError, match="transformer_engine.pytorch"):
        AmplifyEmbeddingGenerator(model_name="amplify_350m", device="cpu")


def test_amplify_wraps_incomplete_transformer_engine_install(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _fake_import_module(name: str) -> types.ModuleType:
        _ = name
        raise RuntimeError(
            "Found empty `transformer-engine` meta package installed. "
            "Install `transformer-engine` with framework extensions"
        )

    monkeypatch.setattr(amplify_module.importlib, "import_module", _fake_import_module)

    with pytest.raises(EmbeddingDependencyError, match="meta package"):
        AmplifyEmbeddingGenerator(model_name="amplify_350m", device="cpu")
