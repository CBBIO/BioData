from __future__ import annotations

import builtins
from pathlib import Path
import pickle
import sys
import types
from typing import Any, Sequence

import pytest

from CBBIO.embeddings import (
    available_generator_classes,
    available_generator_models,
    EmbeddingBackendError,
    EmbeddingDependencyError,
    EmbeddingRecord,
    EmbeddingGenerator,
    EmbeddingInputError,
    Generator,
    GenerationInput,
    ModelMetadata,
    ModelAdapter,
    PostprocessorAdapter,
    PreprocessorAdapter,
    TokenizerAdapter,
    generate_from_fasta,
    load_embedding_records,
    load_embedding_records_npy,
    load_embedding_records_pickle,
    load_fasta_inputs,
    save_embedding_records_npy,
    save_embedding_records_pickle,
    RunMetadata,
)
from CBBIO.embeddings_prott5 import ProtT5EmbeddingGenerator, ProtT5Preprocessor
from CBBIO.embeddings_prostt5 import ProstT5EmbeddingGenerator
from CBBIO.embeddings_ankh3 import Ankh3EmbeddingGenerator
from CBBIO.embeddings_esmc import EsmcEmbeddingGenerator
from CBBIO.embeddings_esm2 import Esm2EmbeddingGenerator
from CBBIO.embeddings_esm1b import Esm1bEmbeddingGenerator


class _Preprocessor(PreprocessorAdapter):
    def preprocess(self, raw_sequence: str) -> str:
        return raw_sequence.strip().upper()


class _Tokenizer(TokenizerAdapter):
    def tokenize(self, sequence: str) -> Any:
        return sequence


class _Model(ModelAdapter):
    def infer(self, tokens: Any, *, layer_index: int = 0) -> Any:
        if tokens == "FAIL":
            raise RuntimeError("model exploded")
        return [len(str(tokens)), layer_index]


class _Postprocessor(PostprocessorAdapter):
    def postprocess(self, model_output: Any) -> Sequence[float]:
        return model_output


class _ToListVector:
    def __init__(self, values: list[float]) -> None:
        self.values = values

    def tolist(self) -> list[float]:
        return self.values


class _ToListModel(ModelAdapter):
    def infer(self, tokens: Any, *, layer_index: int = 0) -> Any:
        return _ToListVector([1, 2.5, float(layer_index)])


def _generator(*, model: ModelAdapter | None = None) -> EmbeddingGenerator:
    return EmbeddingGenerator(
        model_reference="test/model",
        preprocessor=_Preprocessor(),
        tokenizer=_Tokenizer(),
        model=model or _Model(),
        postprocessor=_Postprocessor(),
    )


def test_generate_happy_path_runs_full_pipeline() -> None:
    generator = _generator()
    result = generator.generate(
        [
            GenerationInput(id="p1", sequence="acde"),
            GenerationInput(id="p2", sequence="VVVV", metadata={"source": "unit"}),
        ],
        layer_index=7,
    )

    assert len(result.records) == 2
    assert result.errors == []
    assert result.records[0].id == "p1"
    assert result.records[0].embedding == [4.0, 7.0]
    assert result.records[0].shape == (2,)
    assert result.records[0].model_reference == "test/model"
    assert result.records[1].metadata == {"source": "unit"}


def test_generate_collects_errors_when_fail_fast_is_false() -> None:
    generator = _generator()
    result = generator.generate(
        [
            GenerationInput(id="bad", sequence="AB1"),
            GenerationInput(id="ok", sequence="ACDE"),
            GenerationInput(id="boom", sequence="FAIL"),
        ],
        fail_fast=False,
    )

    assert [record.id for record in result.records] == ["ok"]
    assert len(result.errors) == 2
    assert result.errors[0]["id"] == "bad"
    assert result.errors[0]["error_type"] == "EmbeddingInputError"
    assert result.errors[1]["id"] == "boom"
    assert result.errors[1]["error_type"] == "EmbeddingBackendError"


def test_generate_raises_on_first_error_when_fail_fast_true() -> None:
    generator = _generator()
    with pytest.raises(EmbeddingInputError):
        generator.generate(
            [GenerationInput(id="", sequence="ACDE"), GenerationInput(id="ok", sequence="ACDE")],
            fail_fast=True,
        )


def test_generate_normalizes_tolist_vectors_to_float_list() -> None:
    generator = _generator(model=_ToListModel())
    result = generator.generate([GenerationInput(id="p1", sequence="ACDE")], layer_index=3)

    assert len(result.records) == 1
    assert result.records[0].embedding == [1.0, 2.5, 3.0]
    assert result.records[0].shape == (3,)


def test_base_generator_available_layers_raises_when_model_adapter_does_not_expose_it() -> None:
    generator = _generator()
    with pytest.raises(EmbeddingBackendError):
        generator.available_layers()
    with pytest.raises(EmbeddingBackendError):
        generator.num_layers()


def test_generator_factory_builds_prott5_from_model_class() -> None:
    obj = Generator(model_class="protT5", name="Rostlab/prot_t5_xl_uniref50", tokenizer=object(), model=object())
    assert isinstance(obj, ProtT5EmbeddingGenerator)
    assert "Rostlab/prot_t5_xl_uniref50" in obj.family_models()


def test_generator_factory_uses_default_model_name_when_omitted() -> None:
    obj = Generator(model_class="protT5", tokenizer=object(), model=object())
    assert isinstance(obj, ProtT5EmbeddingGenerator)
    assert obj.model_reference == ProtT5EmbeddingGenerator.DEFAULT_MODEL_NAME


def test_generator_factory_builds_prostt5_from_class_kwarg() -> None:
    obj = Generator(**{"class": "prostT5", "name": "Rostlab/ProstT5", "tokenizer": object(), "model": object()})
    assert isinstance(obj, ProstT5EmbeddingGenerator)


def test_generator_factory_rejects_unknown_class() -> None:
    with pytest.raises(EmbeddingInputError):
        Generator(model_class="esmx", name="facebook/esm2")


def test_factory_catalog_reports_available_classes_and_models() -> None:
    classes = available_generator_classes()
    assert classes == ["protT5", "prostT5", "ankh3", "esmc", "esm2", "esm1b"]

    catalog = available_generator_models()
    assert set(catalog.keys()) == set(classes)
    assert "esm2_t33_650m_ur50d" in [name.lower() for name in catalog["esm2"]]

    esm1b_models = available_generator_models("esm1b")
    assert isinstance(esm1b_models, list)
    assert "esm1b_t33_650M_UR50S" in esm1b_models


def test_generator_factory_builds_ankh3_with_prefix() -> None:
    obj = Generator(
        model_class="ankh3",
        name="ElnaggarLab/ankh3-large",
        prefix="[S2S]",
        tokenizer=object(),
        model=object(),
    )
    assert isinstance(obj, Ankh3EmbeddingGenerator)


def test_generator_factory_builds_esmc_with_flash_attention_option() -> None:
    obj = Generator(
        model_class="esmc",
        name="esmc_300m",
        use_flash_attention=True,
        client=object(),
    )
    assert isinstance(obj, EsmcEmbeddingGenerator)


def test_generator_factory_builds_esm2() -> None:
    class _Model:
        num_layers = 33

        def to(self, _device: str) -> "_Model":
            return self

        def eval(self) -> None:
            return None

    class _Alphabet:
        padding_idx = 0

        def get_batch_converter(self) -> Any:
            return lambda _data: ([], [], [])

    obj = Generator(
        model_class="esm2",
        name="esm2_t33_650M_UR50D",
        model=_Model(),
        alphabet=_Alphabet(),
        batch_converter=lambda _data: ([], [], []),
    )
    assert isinstance(obj, Esm2EmbeddingGenerator)


def test_generator_factory_builds_esm1b() -> None:
    class _Model:
        def to(self, _device: str) -> "_Model":
            return self

        def eval(self) -> None:
            return None

    class _Alphabet:
        padding_idx = 0

        def get_batch_converter(self) -> Any:
            return lambda _data: ([], [], [])

    obj = Generator(
        model_class="esm1b",
        name="esm1b_t33_650M_UR50S",
        model=_Model(),
        alphabet=_Alphabet(),
        batch_converter=lambda _data: ([], [], []),
    )
    assert isinstance(obj, Esm1bEmbeddingGenerator)


@pytest.mark.skipif("Bio" not in sys.modules and __import__("importlib").util.find_spec("Bio") is None, reason="Biopython not installed")
def test_load_fasta_inputs_parses_records(tmp_path: Path) -> None:
    fasta_path = tmp_path / "input.fasta"
    fasta_path.write_text(">P1 first\nACDE\n>P2 second\nVVVV\n", encoding="utf-8")

    records = load_fasta_inputs(fasta_path)
    by_desc = load_fasta_inputs(fasta_path, id_from="description")

    assert [record.id for record in records] == ["P1", "P2"]
    assert records[0].sequence == "ACDE"
    assert by_desc[0].id == "P1 first"


@pytest.mark.skipif("Bio" not in sys.modules and __import__("importlib").util.find_spec("Bio") is None, reason="Biopython not installed")
def test_generate_from_fasta_uses_generator(tmp_path: Path) -> None:
    fasta_path = tmp_path / "input.fasta"
    fasta_path.write_text(">Q1\nACDE\n", encoding="utf-8")
    generator = _generator()

    result = generate_from_fasta(fasta_path, generator, layer_index=2)

    assert len(result.records) == 1
    assert result.records[0].id == "Q1"
    assert result.records[0].embedding == [4.0, 2.0]


def test_load_fasta_inputs_raises_dependency_error_when_biopython_missing(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    fasta_path = tmp_path / "input.fasta"
    fasta_path.write_text(">X\nACDE\n", encoding="utf-8")

    original_import = builtins.__import__

    def _raising_import(name: str, globals: Any = None, locals: Any = None, fromlist: Any = (), level: int = 0) -> Any:
        if name == "Bio" or name.startswith("Bio."):
            raise ModuleNotFoundError("No module named 'Bio'")
        return original_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", _raising_import)
    monkeypatch.delitem(sys.modules, "Bio", raising=False)
    monkeypatch.delitem(sys.modules, "Bio.SeqIO", raising=False)

    with pytest.raises(EmbeddingDependencyError):
        load_fasta_inputs(fasta_path)


def test_load_embedding_records_pickle_from_dict_of_vectors(tmp_path: Path) -> None:
    payload = {"P1": [0.1, 0.2], "P2": [0.3, 0.4]}
    data_path = tmp_path / "emb.pkl"
    data_path.write_bytes(pickle.dumps(payload))

    records = load_embedding_records_pickle(data_path, model_reference="demo/model", layer_index=5)

    assert [record.id for record in records] == ["P1", "P2"]
    assert records[0].embedding == [0.1, 0.2]
    assert records[0].shape == (2,)
    assert records[0].layer_index == 5
    assert records[0].model_reference == "demo/model"


def test_load_embedding_records_pickle_from_records_list(tmp_path: Path) -> None:
    payload = {
        "records": [
            {
                "id": "Q1",
                "embedding": [1, 2, 3],
                "layer_index": 2,
                "model_reference": "hf/x",
                "metadata": {"a": 1},
            }
        ]
    }
    data_path = tmp_path / "embeddings.pickle"
    data_path.write_bytes(pickle.dumps(payload))

    records = load_embedding_records(data_path)

    assert len(records) == 1
    assert records[0].id == "Q1"
    assert records[0].embedding == [1.0, 2.0, 3.0]
    assert records[0].shape == (3,)
    assert records[0].layer_index == 2
    assert records[0].model_reference == "hf/x"
    assert records[0].metadata == {"a": 1}


def test_load_embedding_records_npy_and_npz(tmp_path: Path) -> None:
    np = pytest.importorskip("numpy")

    npy_path = tmp_path / "emb.npy"
    np.save(npy_path, np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float32))
    npy_records = load_embedding_records_npy(
        npy_path,
        model_reference="m",
        layer_index=1,
        ids=["A", "B"],
    )
    assert [record.id for record in npy_records] == ["A", "B"]
    assert npy_records[1].embedding == pytest.approx([0.3, 0.4])

    npz_path = tmp_path / "emb.npz"
    np.savez(npz_path, arr=np.array([1.0, 2.0, 3.0], dtype=np.float32))
    npz_records = load_embedding_records(npz_path, model_reference="m2", layer_index=9)
    assert len(npz_records) == 1
    assert npz_records[0].id == "row_0"
    assert npz_records[0].shape == (3,)
    assert npz_records[0].layer_index == 9


def test_load_embedding_records_npy_rejects_id_mismatch(tmp_path: Path) -> None:
    np = pytest.importorskip("numpy")
    npy_path = tmp_path / "emb.npy"
    np.save(npy_path, np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32))

    with pytest.raises(EmbeddingInputError):
        load_embedding_records_npy(npy_path, ids=["only_one"])


def test_save_embedding_records_pickle_roundtrip_records_format(tmp_path: Path) -> None:
    records = [
        EmbeddingRecord(
            id="P1",
            embedding=[0.1, 0.2, 0.3],
            layer_index=0,
            model_reference="m",
            shape=(3,),
            metadata={"x": 1},
        )
    ]
    path = tmp_path / "saved.pkl"

    out_path = save_embedding_records_pickle(path, records, payload_format="records")
    loaded = load_embedding_records_pickle(out_path)

    assert loaded[0].id == "P1"
    assert loaded[0].embedding == pytest.approx([0.1, 0.2, 0.3])
    assert loaded[0].shape == (3,)


def test_save_embedding_records_pickle_roundtrip_mapping_format(tmp_path: Path) -> None:
    records = [
        EmbeddingRecord(
            id="A",
            embedding=[1.0, 2.0],
            layer_index=1,
            model_reference="m",
            shape=(2,),
        )
    ]
    path = tmp_path / "saved_mapping.pkl"
    save_embedding_records_pickle(path, records, payload_format="mapping")

    loaded = load_embedding_records(path, model_reference="m", layer_index=1)
    assert [record.id for record in loaded] == ["A"]
    assert loaded[0].embedding == [1.0, 2.0]


def test_save_embedding_records_npy_roundtrip(tmp_path: Path) -> None:
    np = pytest.importorskip("numpy")
    records = [
        EmbeddingRecord(id="R1", embedding=[0.1, 0.2], layer_index=0, model_reference="m", shape=(2,)),
        EmbeddingRecord(id="R2", embedding=[0.3, 0.4], layer_index=0, model_reference="m", shape=(2,)),
    ]
    path = tmp_path / "saved.npy"
    save_embedding_records_npy(path, records)

    matrix = np.load(path)
    assert matrix.shape == (2, 2)
    assert matrix[1].tolist() == pytest.approx([0.3, 0.4])

    loaded = load_embedding_records_npy(path, ids=["R1", "R2"], model_reference="m")
    assert [record.id for record in loaded] == ["R1", "R2"]


def test_save_embedding_records_npy_rejects_mixed_dimensions(tmp_path: Path) -> None:
    records = [
        EmbeddingRecord(id="R1", embedding=[0.1, 0.2], layer_index=0, model_reference="m", shape=(2,)),
        EmbeddingRecord(id="R2", embedding=[0.3], layer_index=0, model_reference="m", shape=(1,)),
    ]
    with pytest.raises(EmbeddingInputError):
        save_embedding_records_npy(tmp_path / "bad.npy", records)


def test_prott5_preprocessor_applies_expected_transform() -> None:
    pre = ProtT5Preprocessor()
    processed = pre.preprocess("acduzob")
    assert processed == "A C D X X X X"


def test_prott5_generator_raises_dependency_error_when_transformers_missing(
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
        ProtT5EmbeddingGenerator()


def test_prott5_generator_uses_t5_tokenizer_for_transformers_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _FakeAutoConfig:
        observed_name: str | None = None

        @staticmethod
        def from_pretrained(name: str) -> object:
            _FakeAutoConfig.observed_name = name
            return types.SimpleNamespace()

    class _FakeT5EncoderModel:
        observed_name: str | None = None
        observed_config: object | None = None

        @staticmethod
        def from_pretrained(name: str, config: object | None = None) -> object:
            _FakeT5EncoderModel.observed_name = name
            _FakeT5EncoderModel.observed_config = config

            class _FakeLoadedModel:
                def to(self, _device: str) -> "_FakeLoadedModel":
                    return self

                def eval(self) -> None:
                    return None

            return _FakeLoadedModel()

    class _FakeT5Tokenizer:
        observed_name: str | None = None
        observed_do_lower_case: bool | None = None

        @staticmethod
        def from_pretrained(name: str, do_lower_case: bool = False) -> object:
            _FakeT5Tokenizer.observed_name = name
            _FakeT5Tokenizer.observed_do_lower_case = do_lower_case
            return object()

    fake_transformers = types.SimpleNamespace(
        AutoConfig=_FakeAutoConfig,
        T5EncoderModel=_FakeT5EncoderModel,
        T5Tokenizer=_FakeT5Tokenizer,
    )

    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)

    generator = ProtT5EmbeddingGenerator(model_name="Rostlab/prot_t5_xl_uniref50", device="cpu")

    assert generator.model_reference == "Rostlab/prot_t5_xl_uniref50"
    assert _FakeAutoConfig.observed_name == "Rostlab/prot_t5_xl_uniref50"
    assert _FakeT5EncoderModel.observed_name == "Rostlab/prot_t5_xl_uniref50"
    assert _FakeT5Tokenizer.observed_name == "Rostlab/prot_t5_xl_uniref50"
    assert _FakeT5Tokenizer.observed_do_lower_case is False


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
    def batch_encode_plus(self, sequences: list[str], add_special_tokens: bool, padding: str) -> dict[str, Any]:
        assert add_special_tokens is True
        assert padding == "longest"
        tokens = sequences[0].split()
        length = len(tokens) + 1  # include end token
        return {"input_ids": [[1] * length], "attention_mask": [[1] * length]}


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
        length = len(input_ids.tolist()[0])  # includes end token
        hidden_states = []
        for layer in range(3):
            layer_values = [[[float(layer), float(pos)] for pos in range(length)]]
            hidden_states.append(_FakeTensor(layer_values))
        return _FakeModelOutput(tuple(hidden_states))


def test_prott5_generate_returns_all_layers_when_omitted(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_torch = types.SimpleNamespace(
        tensor=lambda values: _FakeTensor(values),
        no_grad=lambda: _FakeNoGrad(),
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    generator = ProtT5EmbeddingGenerator(
        tokenizer=_FakeTokenizer(),
        model=_FakeModel(),
    )
    result = generator.generate([GenerationInput(id="P1", sequence="ACDE")], layer_index=None)

    assert result.errors == []
    assert [record.layer_index for record in result.records] == [0, 1, 2]
    assert all(record.shape == (4, 2) for record in result.records)
    assert result.records[1].embedding[0] == [1.0, 0.0]


def test_prott5_generate_returns_requested_layers(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_torch = types.SimpleNamespace(
        tensor=lambda values: _FakeTensor(values),
        no_grad=lambda: _FakeNoGrad(),
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    generator = ProtT5EmbeddingGenerator(
        tokenizer=_FakeTokenizer(),
        model=_FakeModel(),
    )
    result = generator.generate([GenerationInput(id="P2", sequence="AAAA")], layer_index=[2, 0])

    assert result.errors == []
    assert [record.layer_index for record in result.records] == [0, 2]


def test_prott5_generator_available_layers_and_count(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_torch = types.SimpleNamespace(
        tensor=lambda values: _FakeTensor(values),
        no_grad=lambda: _FakeNoGrad(),
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    generator = ProtT5EmbeddingGenerator(
        tokenizer=_FakeTokenizer(),
        model=_FakeModel(),
    )
    assert generator.available_layers() == [0, 1, 2]
    assert generator.num_layers() == 3


def test_prott5_generation_populates_model_and_run_metadata(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_torch = types.SimpleNamespace(
        tensor=lambda values: _FakeTensor(values),
        no_grad=lambda: _FakeNoGrad(),
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    generator = ProtT5EmbeddingGenerator(
        tokenizer=_FakeTokenizer(),
        model=_FakeModel(),
        model_name="Rostlab/prot_t5_xl_uniref50",
        device="cpu",
    )
    result = generator.generate([GenerationInput(id="P3", sequence="ACDE")], layer_index=[1])

    assert isinstance(generator.model_metadata, ModelMetadata)
    assert generator.model_metadata.model_name == "Rostlab/prot_t5_xl_uniref50"
    assert generator.model_metadata.parameters == {
        "representation": "per-residue",
        "pooling": "none",
        "layer_indexing": "biodata_reversed_0_is_last_hidden",
    }
    assert isinstance(result.model_metadata, ModelMetadata)
    assert isinstance(result.run_metadata, RunMetadata)
    assert result.run_metadata is not None
    assert result.run_metadata.sequence_count == 1
    assert result.run_metadata.requested_layers == [1]
    assert result.run_metadata.resolved_layers == [1]
