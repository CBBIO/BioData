from __future__ import annotations

import builtins
from pathlib import Path
import pickle
import sys
import types
from typing import Any, Dict, List, Self, Sequence, cast

import pytest

from CBBIO.embeddings import (
    batch_generation_inputs,
    EmbeddingBackendError,
    EmbeddingDependencyError,
    EmbeddingRecord,
    EmbeddingGenerator,
    EmbeddingInputError,
    GenerationInput,
    ModelMetadata,
    ModelAdapter,
    PostprocessorAdapter,
    PreprocessorAdapter,
    TokenizerAdapter,
    generate_from_fasta,
    generate_from_fasta_batches,
    iter_embedding_records_from_fasta,
    iter_fasta_inputs,
    load_fasta_inputs,
    RunMetadata,
)
from CBBIO import (
    AmplifyEmbeddingGenerator,
    Ankh3EmbeddingGenerator,
    available_generator_classes,
    available_generator_models,
    EmbeddingWriter,
    Esm1bEmbeddingGenerator,
    Esm2EmbeddingGenerator,
    EsmcEmbeddingGenerator,
    FastaBatcher,
    Generator,
    H5EmbeddingReader,
    IterableBatcher,
    generate_fasta_h5,
    generate_fasta_npy_shards,
    generate_fasta_pickle_shards,
    load_embedding_records,
    load_embedding_records_h5,
    load_embedding_records_npy,
    load_embedding_records_pickle,
    mean_pool_embedding_record,
    ProteinGlmEmbeddingGenerator,
    ProstT5EmbeddingGenerator,
    ProtT5EmbeddingGenerator,
    ProtT5Preprocessor,
    save_embedding_records_h5,
    save_embedding_records_npy,
    save_embedding_records_npy_shards,
    save_embedding_records_pickle,
    save_embedding_records_pickle_shards,
    pooler_factory,
    run_embedding_generation,
)


_MODEL_GENERATOR_TYPES: tuple[type[Any], ...] = (
    ProtT5EmbeddingGenerator,
    ProstT5EmbeddingGenerator,
    Ankh3EmbeddingGenerator,
    AmplifyEmbeddingGenerator,
    ProteinGlmEmbeddingGenerator,
    EsmcEmbeddingGenerator,
    Esm2EmbeddingGenerator,
    Esm1bEmbeddingGenerator,
)

_MODEL_ALIAS_CASES = [
    (generator_type, alias, model_reference)
    for generator_type in _MODEL_GENERATOR_TYPES
    for alias, model_reference in generator_type.MODEL_ALIASES.items()
]


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


class _ScalePooler:
    name = "scale"

    def __call__(self, residue_tensor: Any) -> list[float]:
        return [2.0 * float(cast(Any, value)) for value in cast(Sequence[Any], residue_tensor)]


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


def test_generate_applies_custom_callable_pooler() -> None:
    generator = _generator(model=_ToListModel())

    result = generator.generate(
        [GenerationInput(id="p1", sequence="ACDE")],
        layer_index=3,
        pooler=_ScalePooler(),
    )

    assert len(result.records) == 1
    assert result.records[0].embedding == [2.0, 5.0, 6.0]
    assert result.records[0].shape == (3,)


def test_generate_with_mean_pooler_keeps_vector_outputs() -> None:
    generator = _generator(model=_ToListModel())

    result = generator.generate(
        [GenerationInput(id="p1", sequence="ACDE")],
        layer_index=3,
        pooler="mean",
    )

    assert len(result.records) == 1
    assert result.records[0].embedding == [1.0, 2.5, 3.0]
    assert result.records[0].shape == (3,)


def test_pooler_factory_supports_cls_aliases() -> None:
    cls_pooler = pooler_factory("cls")
    bos_pooler = pooler_factory("bos")

    assert cls_pooler is not None
    assert bos_pooler is not None
    assert cls_pooler.name == "cls"
    assert bos_pooler.name == "cls"


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


def test_prott5_generator_records_requested_dtype(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_torch = types.SimpleNamespace(float32="float32", float16="float16", bfloat16="bfloat16")
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    obj = ProtT5EmbeddingGenerator(tokenizer=object(), model=object(), dtype="float16")

    assert obj.model_metadata.parameters is not None
    assert obj.model_metadata.parameters["torch_dtype"] == "float16"


def test_prott5_generator_rejects_unknown_dtype() -> None:
    with pytest.raises(EmbeddingInputError):
        ProtT5EmbeddingGenerator(tokenizer=object(), model=object(), dtype="int8")


def test_generator_factory_builds_prostt5_from_class_kwarg() -> None:
    obj = Generator(**{"class": "prostT5", "name": "Rostlab/ProstT5", "tokenizer": object(), "model": object()})
    assert isinstance(obj, ProstT5EmbeddingGenerator)


def test_generator_factory_rejects_unknown_class() -> None:
    with pytest.raises(EmbeddingInputError):
        Generator(model_class="esmx", name="facebook/esm2")


def test_factory_catalog_reports_available_classes_and_models() -> None:
    classes = available_generator_classes()
    assert classes == ["prott5", "prostt5", "ankh3", "amplify", "proteinglm", "esmc", "esm2", "esm1b"]

    catalog = cast(Dict[str, List[str]], available_generator_models())
    assert set(catalog.keys()) == set(classes)
    assert catalog["amplify"] == [
        "amplify_120m",
        "amplify_350m",
        "nvidia/AMPLIFY_120M",
        "nvidia/AMPLIFY_350M",
        "amplify_120m_chandar",
        "amplify_350m_chandar",
        "chandar-lab/AMPLIFY_120M",
        "chandar-lab/AMPLIFY_350M",
    ]
    assert "facebook/esm2_t33_650m_ur50d" in [name.lower() for name in catalog["esm2"]]
    assert catalog["proteinglm"][:3] == ["proteinglm_1b_mlm", "proteinglm_3b_mlm", "proteinglm_10b_mlm"]

    esm1b_models = cast(List[str], available_generator_models("esm1b"))
    assert "esm1b_t33_650M_UR50S" in esm1b_models


@pytest.mark.parametrize(
    ("model_class", "generator_type"),
    [
        ("prott5", ProtT5EmbeddingGenerator),
        ("protT5", ProtT5EmbeddingGenerator),
        ("prot_t5", ProtT5EmbeddingGenerator),
        ("Prot-T5", ProtT5EmbeddingGenerator),
        ("prostt5", ProstT5EmbeddingGenerator),
        ("prostT5", ProstT5EmbeddingGenerator),
        ("prost_t5", ProstT5EmbeddingGenerator),
        ("Prost-T5", ProstT5EmbeddingGenerator),
        ("ankh3", Ankh3EmbeddingGenerator),
        ("Ankh3-Large", Ankh3EmbeddingGenerator),
        ("ankh3_large", Ankh3EmbeddingGenerator),
        ("AMPLIFY", AmplifyEmbeddingGenerator),
        ("amplify_120m", AmplifyEmbeddingGenerator),
        ("amplify_350m", AmplifyEmbeddingGenerator),
        ("ProteinGLM", ProteinGlmEmbeddingGenerator),
        ("ProteinPGLM", ProteinGlmEmbeddingGenerator),
        ("pglm", ProteinGlmEmbeddingGenerator),
        ("proteinglm_mlm", ProteinGlmEmbeddingGenerator),
        ("esmc", EsmcEmbeddingGenerator),
        ("esmC", EsmcEmbeddingGenerator),
        ("esm-c", EsmcEmbeddingGenerator),
        ("esmc3", EsmcEmbeddingGenerator),
        ("ESM3c", EsmcEmbeddingGenerator),
        ("esm2", Esm2EmbeddingGenerator),
        ("esm-2", Esm2EmbeddingGenerator),
        ("ESM", Esm2EmbeddingGenerator),
        ("esm1b", Esm1bEmbeddingGenerator),
        ("esm-1b", Esm1bEmbeddingGenerator),
    ],
)
def test_generator_factory_accepts_canonical_names_and_aliases(
    model_class: str,
    generator_type: type[EmbeddingGenerator],
) -> None:
    model = types.SimpleNamespace(config=types.SimpleNamespace(num_hidden_layers=1))
    generator = Generator(model_class=model_class, model=model, tokenizer=object())

    assert isinstance(generator, generator_type)


@pytest.mark.parametrize(
    "generator_type",
    _MODEL_GENERATOR_TYPES,
)
def test_model_generators_share_construction_classmethods(
    generator_type: type[Any],
) -> None:
    model = types.SimpleNamespace(config=types.SimpleNamespace(num_hidden_layers=1))
    tokenizer = object()

    pretrained = generator_type.from_pretrained(
        generator_type.DEFAULT_MODEL_NAME,
        model=model,
        tokenizer=tokenizer,
    )
    wrapped = generator_type.from_model_and_tokenizer(model, tokenizer)

    assert isinstance(pretrained, generator_type)
    assert isinstance(wrapped, generator_type)


@pytest.mark.parametrize("generator_type", _MODEL_GENERATOR_TYPES)
def test_model_generators_share_catalog_interface(generator_type: type[Any]) -> None:
    assert generator_type.GENERATOR_CLASS == generator_type.GENERATOR_CLASS.lower()
    assert isinstance(generator_type.GENERATOR_ALIASES, tuple)
    assert isinstance(generator_type.MODEL_ALIASES, dict)
    assert generator_type.DEFAULT_MODEL_NAME in generator_type.FAMILY_MODELS
    assert isinstance(generator_type.SUPPORTED_POOLERS, tuple)


@pytest.mark.parametrize(
    ("generator_type", "model_alias", "expected_model_reference"),
    _MODEL_ALIAS_CASES,
)
def test_model_generators_resolve_every_model_alias(
    generator_type: type[Any],
    model_alias: str,
    expected_model_reference: str,
) -> None:
    model = types.SimpleNamespace(config=types.SimpleNamespace(num_hidden_layers=1))

    generator = generator_type.from_pretrained(
        model_alias,
        model=model,
        tokenizer=object(),
    )

    assert generator.model_reference == expected_model_reference


@pytest.mark.parametrize("generator_type", _MODEL_GENERATOR_TYPES)
def test_model_generators_accept_every_advertised_family_model(
    generator_type: type[Any],
) -> None:
    model = types.SimpleNamespace(config=types.SimpleNamespace(num_hidden_layers=1))

    for model_name in generator_type.FAMILY_MODELS:
        generator = generator_type.from_pretrained(
            model_name,
            model=model,
            tokenizer=object(),
        )

        assert generator.model_reference


def test_generator_factory_builds_ankh3_with_prefix() -> None:
    obj = Generator(
        model_class="ankh3",
        name="ElnaggarLab/ankh3-large",
        prefix="[S2S]",
        tokenizer=object(),
        model=object(),
    )
    assert isinstance(obj, Ankh3EmbeddingGenerator)


def test_generator_factory_builds_esmc() -> None:
    obj = Generator(
        model_class="esmc",
        name="esmc_300m",
        model=object(),
        tokenizer=object(),
    )
    assert isinstance(obj, EsmcEmbeddingGenerator)


def test_generator_factory_builds_esmc_from_size_alias() -> None:
    obj = Generator(
        model_class="esm-c",
        name="esmc600m",
        model=object(),
        tokenizer=object(),
    )

    assert isinstance(obj, EsmcEmbeddingGenerator)
    assert obj.model_reference == "biohub/ESMC-600M"
    assert obj.model_metadata.parameters is not None
    assert obj.model_metadata.parameters["sdk_model_name"] == "esmc_600m"


def test_generator_factory_builds_proteinglm() -> None:
    obj = Generator(
        model_class="proteinglm",
        name="proteinglm_1b_mlm",
        model=object(),
        tokenizer=object(),
    )
    assert isinstance(obj, ProteinGlmEmbeddingGenerator)


def test_generator_factory_builds_esm2() -> None:
    obj = Generator(
        model_class="esm2",
        name="esm2_t33_650M_UR50D",
        model=object(),
        tokenizer=object(),
    )
    assert isinstance(obj, Esm2EmbeddingGenerator)


def test_generator_factory_builds_esm2_from_size_alias() -> None:
    obj = Generator(
        model_class="esm",
        name="esm2_650m",
        model=object(),
        tokenizer=object(),
    )

    assert isinstance(obj, Esm2EmbeddingGenerator)
    assert obj.model_reference == "facebook/esm2_t33_650M_UR50D"


@pytest.mark.parametrize(
    ("model_class", "model_name", "expected_type", "expected_reference"),
    [
        ("amplify", "amplify_350m", AmplifyEmbeddingGenerator, "nvidia/AMPLIFY_350M"),
        ("ankh3", "ankh3_base", Ankh3EmbeddingGenerator, "ElnaggarLab/ankh-base"),
        ("esmc", "esmc_600m", EsmcEmbeddingGenerator, "biohub/ESMC-600M"),
        ("esm2", "esm2_650m", Esm2EmbeddingGenerator, "facebook/esm2_t33_650M_UR50D"),
        ("esm1b", "esm1b_650m", Esm1bEmbeddingGenerator, "esm1b_t33_650M_UR50S"),
        ("proteinglm", "proteinglm_3b", ProteinGlmEmbeddingGenerator, "biomap-research/proteinglm-3b-mlm"),
        ("prott5", "prott5_xl_uniref50", ProtT5EmbeddingGenerator, "Rostlab/prot_t5_xl_uniref50"),
    ],
)
def test_generator_factory_accepts_family_size_model_aliases(
    model_class: str,
    model_name: str,
    expected_type: type[Any],
    expected_reference: str,
) -> None:
    model = types.SimpleNamespace(config=types.SimpleNamespace(num_hidden_layers=1))
    obj = Generator(
        model_class=model_class,
        name=model_name,
        model=model,
        tokenizer=object(),
    )

    assert isinstance(obj, expected_type)
    assert obj.model_reference == expected_reference


def test_generator_factory_builds_esm1b() -> None:
    obj = Generator(
        model_class="esm1b",
        name="esm1b_t33_650M_UR50S",
        model=object(),
        tokenizer=object(),
    )
    assert isinstance(obj, Esm1bEmbeddingGenerator)


def test_batch_generation_inputs_groups_iterable_without_materializing_all_inputs() -> None:
    records = (GenerationInput(id=f"P{idx}", sequence="ACDE") for idx in range(5))

    batches = list(batch_generation_inputs(records, batch_size=2))

    assert [[record.id for record in batch] for batch in batches] == [
        ["P0", "P1"],
        ["P2", "P3"],
        ["P4"],
    ]


def test_batch_generation_inputs_respects_padded_token_budget() -> None:
    records = [
        GenerationInput(id="P0", sequence="A" * 9),
        GenerationInput(id="P1", sequence="A" * 8),
        GenerationInput(id="P2", sequence="A" * 3),
        GenerationInput(id="P3", sequence="A" * 2),
    ]

    batches = list(batch_generation_inputs(records, batch_size=10, max_batch_tokens=20))

    assert [[record.id for record in batch] for batch in batches] == [["P0", "P1"], ["P2", "P3"]]


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
def test_iter_fasta_inputs_yields_records_lazily(tmp_path: Path) -> None:
    fasta_path = tmp_path / "input.fasta"
    fasta_path.write_text(">P1 first\nACDE\n>P2 second\nVVVV\n", encoding="utf-8")

    records = iter_fasta_inputs(fasta_path)

    first = next(records)
    second = next(records)

    assert first.id == "P1"
    assert first.sequence == "ACDE"
    assert second.id == "P2"


@pytest.mark.skipif("Bio" not in sys.modules and __import__("importlib").util.find_spec("Bio") is None, reason="Biopython not installed")
def test_generate_from_fasta_uses_generator(tmp_path: Path) -> None:
    fasta_path = tmp_path / "input.fasta"
    fasta_path.write_text(">Q1\nACDE\n", encoding="utf-8")
    generator = _generator()

    with pytest.warns(DeprecationWarning):
        result = generate_from_fasta(fasta_path, generator, layer_index=2)

    assert len(result.records) == 1
    assert result.records[0].id == "Q1"
    assert result.records[0].embedding == [4.0, 2.0]


@pytest.mark.skipif("Bio" not in sys.modules and __import__("importlib").util.find_spec("Bio") is None, reason="Biopython not installed")
def test_generate_from_fasta_batches_yields_one_result_per_batch(tmp_path: Path) -> None:
    fasta_path = tmp_path / "input.fasta"
    fasta_path.write_text(">Q1\nACDE\n>Q2\nAAAA\n>Q3\nVVVV\n", encoding="utf-8")
    generator = _generator()

    with pytest.warns(DeprecationWarning):
        results = list(generate_from_fasta_batches(fasta_path, generator, batch_size=2, layer_index=5))

    assert [len(result.records) for result in results] == [2, 1]
    assert [[record.id for record in result.records] for result in results] == [["Q1", "Q2"], ["Q3"]]
    assert results[0].records[0].embedding == [4.0, 5.0]


@pytest.mark.skipif("Bio" not in sys.modules and __import__("importlib").util.find_spec("Bio") is None, reason="Biopython not installed")
def test_generate_from_fasta_batch_size_aggregates_results(tmp_path: Path) -> None:
    fasta_path = tmp_path / "input.fasta"
    fasta_path.write_text(">Q1\nACDE\n>Q2\nAAAA\n>Q3\nVVVV\n", encoding="utf-8")
    generator = _generator()

    with pytest.warns(DeprecationWarning):
        result = generate_from_fasta(fasta_path, generator, layer_index=6, batch_size=2)

    assert [record.id for record in result.records] == ["Q1", "Q2", "Q3"]
    assert result.run_metadata is not None
    assert result.run_metadata.sequence_count == 3
    assert result.run_metadata.failure_count == 0


@pytest.mark.skipif("Bio" not in sys.modules and __import__("importlib").util.find_spec("Bio") is None, reason="Biopython not installed")
def test_iter_embedding_records_from_fasta_streams_records(tmp_path: Path) -> None:
    fasta_path = tmp_path / "input.fasta"
    fasta_path.write_text(">Q1\nACDE\n>Q2\nAAAA\n>Q3\nVVVV\n", encoding="utf-8")
    generator = _generator()

    with pytest.warns(DeprecationWarning):
        records = list(iter_embedding_records_from_fasta(fasta_path, generator, batch_size=2, layer_index=4))

    assert [record.id for record in records] == ["Q1", "Q2", "Q3"]
    assert records[1].embedding == [4.0, 4.0]


def test_batch_generation_inputs_rejects_non_positive_batch_size() -> None:
    with pytest.raises(EmbeddingInputError):
        list(batch_generation_inputs([], batch_size=0))


def test_iterable_batcher_defaults_to_singletons() -> None:
    records = [GenerationInput(id=f"P{idx}", sequence="A" * (idx + 1)) for idx in range(3)]

    batches = list(IterableBatcher(records))

    assert [[record.id for record in batch] for batch in batches] == [["P0"], ["P1"], ["P2"]]


def test_iterable_batcher_batch_size_only() -> None:
    records = [GenerationInput(id=f"P{idx}", sequence="ACDE") for idx in range(5)]

    batches = list(IterableBatcher(records, batch_size=2))

    assert [[record.id for record in batch] for batch in batches] == [["P0", "P1"], ["P2", "P3"], ["P4"]]


def test_iterable_batcher_token_budget_only() -> None:
    records = [
        GenerationInput(id="P0", sequence="A" * 9),
        GenerationInput(id="P1", sequence="A" * 8),
        GenerationInput(id="P2", sequence="A" * 3),
    ]

    batches = list(IterableBatcher(records, max_batch_tokens=20))

    assert [[record.id for record in batch] for batch in batches] == [["P0", "P1"], ["P2"]]


def test_iterable_batcher_supports_size_and_token_caps() -> None:
    records = [
        GenerationInput(id="P0", sequence="A" * 9),
        GenerationInput(id="P1", sequence="A" * 8),
        GenerationInput(id="P2", sequence="A" * 3),
        GenerationInput(id="P3", sequence="A" * 2),
    ]

    batches = list(IterableBatcher(records, batch_size=3, max_batch_tokens=20))

    assert [[record.id for record in batch] for batch in batches] == [["P0", "P1"], ["P2", "P3"]]


def test_iterable_batcher_limit_applies_first_accepted_records() -> None:
    records = [GenerationInput(id=f"P{idx}", sequence="ACDE") for idx in range(5)]

    batches = list(IterableBatcher(records, batch_size=10, limit=3))

    assert [[record.id for record in batch] for batch in batches] == [["P0", "P1", "P2"]]


def test_iterable_batcher_can_sort_by_length_window() -> None:
    records = [
        GenerationInput(id="P0", sequence="A"),
        GenerationInput(id="P1", sequence="AAAAA"),
        GenerationInput(id="P2", sequence="AAA"),
        GenerationInput(id="P3", sequence="AA"),
    ]

    batches = list(IterableBatcher(records, batch_size=10, length_sort_window=3))

    assert [[record.id for record in batch] for batch in batches] == [["P1", "P2", "P0", "P3"]]


@pytest.mark.skipif("Bio" not in sys.modules and __import__("importlib").util.find_spec("Bio") is None, reason="Biopython not installed")
def test_fasta_batcher_defaults_to_singletons(tmp_path: Path) -> None:
    fasta_path = tmp_path / "input.fasta"
    fasta_path.write_text(">Q1\nA\n>Q2\nAA\n", encoding="utf-8")

    batches = list(FastaBatcher(fasta_path))

    assert [[record.id for record in batch] for batch in batches] == [["Q1"], ["Q2"]]


@pytest.mark.skipif("Bio" not in sys.modules and __import__("importlib").util.find_spec("Bio") is None, reason="Biopython not installed")
def test_fasta_batcher_batch_size_only(tmp_path: Path) -> None:
    fasta_path = tmp_path / "input.fasta"
    fasta_path.write_text(">Q1\nA\n>Q2\nAA\n>Q3\nAAA\n", encoding="utf-8")

    batches = list(FastaBatcher(fasta_path, batch_size=2))

    assert [[record.id for record in batch] for batch in batches] == [["Q1", "Q2"], ["Q3"]]


@pytest.mark.skipif("Bio" not in sys.modules and __import__("importlib").util.find_spec("Bio") is None, reason="Biopython not installed")
def test_fasta_batcher_token_budget_only(tmp_path: Path) -> None:
    fasta_path = tmp_path / "input.fasta"
    fasta_path.write_text(">Q1\nAAAAAAAAA\n>Q2\nAAAAAAAA\n>Q3\nAAA\n", encoding="utf-8")

    batches = list(FastaBatcher(fasta_path, max_batch_tokens=20))

    assert [[record.id for record in batch] for batch in batches] == [["Q1", "Q2"], ["Q3"]]


@pytest.mark.skipif("Bio" not in sys.modules and __import__("importlib").util.find_spec("Bio") is None, reason="Biopython not installed")
def test_fasta_batcher_semantics_and_skipped_tsv(tmp_path: Path) -> None:
    fasta_path = tmp_path / "input.fasta"
    skipped_path = tmp_path / "skipped.tsv"
    fasta_path.write_text(">Q1\nA\n>Q2\nAAAAA\n>Q3\nAAA\n>Q4\nAA\n", encoding="utf-8")

    batcher = FastaBatcher(
        fasta_path,
        batch_size=2,
        max_batch_tokens=12,
        limit=3,
        length_sort_window=3,
        max_sequence_length=4,
        skipped_path=skipped_path,
    )
    batches = list(batcher)

    assert [[record.id for record in batch] for batch in batches] == [["Q3", "Q4"], ["Q1"]]
    assert batcher.skipped_count == 1
    assert batcher.skipped == [{"id": "Q2", "length": 5, "reason": "length>4"}]
    assert skipped_path.read_text(encoding="utf-8").splitlines() == [
        "id\tlength\treason",
        "Q2\t5\tlength>4",
    ]


def test_embedding_writer_memory_keeps_records() -> None:
    writer = EmbeddingWriter(format="memory")
    writer.write([EmbeddingRecord(id="P1", embedding=[1.0, 2.0], layer_index=0, model_reference="m", shape=(2,))])
    writer.close()

    assert writer.record_count == 1
    assert writer.records[0].id == "P1"


def test_embedding_writer_rejects_invalid_parameter_combinations(tmp_path: Path) -> None:
    with pytest.raises(EmbeddingInputError):
        EmbeddingWriter(format="memory", path=tmp_path / "x.pkl")
    with pytest.raises(EmbeddingInputError):
        EmbeddingWriter(format="pkl")
    with pytest.raises(EmbeddingInputError):
        EmbeddingWriter(format="bad")


def test_embedding_writer_pkl_and_npy_shards(tmp_path: Path) -> None:
    np = pytest.importorskip("numpy")
    records = [
        EmbeddingRecord(id=f"P{idx}", embedding=[float(idx), 1.0], layer_index=0, model_reference="m", shape=(2,))
        for idx in range(3)
    ]

    pkl_writer = EmbeddingWriter(format="pkl", path=tmp_path / "out.pkl", records_per_shard=2)
    pkl_writer.write(records)
    pkl_writer.close()
    assert [path.name for path in pkl_writer.paths] == ["out.shard_000001.pkl", "out.shard_000002.pkl"]
    assert load_embedding_records_pickle(pkl_writer.paths[0])[0].id == "P0"

    npy_writer = EmbeddingWriter(format="npy", path=tmp_path / "out.npy", records_per_shard=2)
    npy_writer.write(records)
    npy_writer.close()
    assert [path.name for path in npy_writer.paths] == ["out.shard_000001.npy", "out.shard_000002.npy"]
    assert np.load(npy_writer.paths[0]).shape == (2, 2)
    assert npy_writer.id_paths[0].read_text(encoding="utf-8").splitlines() == ["P0", "P1"]


def test_embedding_writer_h5(tmp_path: Path) -> None:
    h5py = pytest.importorskip("h5py")
    writer = EmbeddingWriter(format="h5", path=tmp_path / "out.h5", write_batch_size=1)
    writer.write(
        [
            EmbeddingRecord(id="P1", embedding=[1.0, 2.0], layer_index=0, model_reference="m", shape=(2,)),
            EmbeddingRecord(id="P2", embedding=[3.0, 4.0], layer_index=0, model_reference="m", shape=(2,)),
        ]
    )
    writer.close()

    assert writer.record_count == 2
    with h5py.File(tmp_path / "out.h5", "r") as handle:
        assert handle["embeddings"].shape == (2, 2)
        assert handle["ids"].asstr()[:].tolist() == ["P1", "P2"]


class _BatchSensitiveGenerator:
    model_reference = "fake/model"

    def __init__(self, *, fail_ids: set[str] | None = None, oom_ids: set[str] | None = None) -> None:
        self.fail_ids = fail_ids or set()
        self.oom_ids = oom_ids or set()
        self.calls: List[List[str]] = []

    def generate(
        self,
        records: Sequence[GenerationInput],
        *,
        layer_index: int | Sequence[int] | None = 0,
        pooler: Any = None,
        fail_fast: bool = False,
    ) -> Any:
        self.calls.append([record.id for record in records])
        if any(record.id in self.oom_ids for record in records):
            raise RuntimeError("CUDA out of memory")
        if any(record.id in self.fail_ids for record in records):
            raise RuntimeError("bad protein")
        return types.SimpleNamespace(
            records=[
                EmbeddingRecord(
                    id=record.id,
                    embedding=[float(len(record.sequence))],
                    layer_index=0,
                    model_reference=self.model_reference,
                    shape=(1,),
                )
                for record in records
            ],
            errors=[],
        )


def test_run_embedding_generation_success_and_progress() -> None:
    events: List[Dict[str, Any]] = []
    writer = EmbeddingWriter(format="memory")
    batcher = IterableBatcher([GenerationInput(id="P1", sequence="ACDE")], batch_size=1, max_batch_tokens=10)

    with pytest.warns(RuntimeWarning):
        result = run_embedding_generation(
            cast(Any, _BatchSensitiveGenerator()),
            batcher,
            writer,
            pooler=pooler_factory("mean"),
            progress_callback=events.append,
        )

    assert result.record_count == 1
    assert writer.records[0].embedding == [4.0]
    assert any(event["event"] == "batch_completed" for event in events)


def test_run_embedding_generation_collects_errors_and_bisects_oom() -> None:
    records = [
        GenerationInput(id="ok1", sequence="AA"),
        GenerationInput(id="oom", sequence="AA"),
        GenerationInput(id="ok2", sequence="AA"),
    ]
    generator = _BatchSensitiveGenerator(oom_ids={"oom"})
    writer = EmbeddingWriter(format="memory")

    with pytest.warns(RuntimeWarning):
        result = run_embedding_generation(
            cast(Any, generator),
            IterableBatcher(records, batch_size=3, max_batch_tokens=20),
            writer,
            fail_fast=False,
        )

    assert [record.id for record in writer.records] == ["ok1", "ok2"]
    assert result.error_count == 1
    assert result.errors[0]["id"] == "oom"
    assert result.errors[0]["reason"] == "cuda_oom"
    assert ["ok1", "oom", "ok2"] in generator.calls
    assert ["oom"] in generator.calls


def test_run_embedding_generation_fail_fast_raises_singleton_oom() -> None:
    writer = EmbeddingWriter(format="memory")
    with pytest.warns(RuntimeWarning):
        with pytest.raises(RuntimeError):
            run_embedding_generation(
                cast(Any, _BatchSensitiveGenerator(oom_ids={"oom"})),
                IterableBatcher([GenerationInput(id="oom", sequence="AA")], batch_size=1),
                writer,
                fail_fast=True,
            )


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


def test_save_embedding_records_pickle_creates_parent_directories(tmp_path: Path) -> None:
    records = [
        EmbeddingRecord(
            id="A",
            embedding=[1.0, 2.0],
            layer_index=1,
            model_reference="m",
            shape=(2,),
        )
    ]
    path = tmp_path / "nested" / "pickle" / "saved_mapping.pkl"

    save_embedding_records_pickle(path, records, payload_format="mapping")

    assert path.exists()
    loaded = load_embedding_records(path, model_reference="m", layer_index=1)
    assert [record.id for record in loaded] == ["A"]


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


def test_save_embedding_records_npy_creates_parent_directories(tmp_path: Path) -> None:
    np = pytest.importorskip("numpy")
    records = [
        EmbeddingRecord(id="R1", embedding=[0.1, 0.2], layer_index=0, model_reference="m", shape=(2,)),
        EmbeddingRecord(id="R2", embedding=[0.3, 0.4], layer_index=0, model_reference="m", shape=(2,)),
    ]
    path = tmp_path / "nested" / "npy" / "saved.npy"

    save_embedding_records_npy(path, records)

    assert path.exists()
    matrix = np.load(path)
    assert matrix.shape == (2, 2)


def test_save_embedding_records_npy_rejects_mixed_dimensions(tmp_path: Path) -> None:
    records = [
        EmbeddingRecord(id="R1", embedding=[0.1, 0.2], layer_index=0, model_reference="m", shape=(2,)),
        EmbeddingRecord(id="R2", embedding=[0.3], layer_index=0, model_reference="m", shape=(1,)),
    ]
    with pytest.raises(EmbeddingInputError):
        save_embedding_records_npy(tmp_path / "bad.npy", records)


def test_mean_pool_embedding_record_pools_matrix() -> None:
    record = EmbeddingRecord(
        id="M1",
        embedding=[[1.0, 3.0], [5.0, 7.0]],
        layer_index=0,
        model_reference="m",
        shape=(2, 2),
    )

    pooled = mean_pool_embedding_record(record)

    assert pooled.id == "M1"
    assert pooled.embedding == [3.0, 5.0]
    assert pooled.shape == (2,)


def test_save_embedding_records_pickle_shards_streams_records(tmp_path: Path) -> None:
    records = [
        EmbeddingRecord(id=f"R{idx}", embedding=[float(idx)], layer_index=0, model_reference="m", shape=(1,))
        for idx in range(5)
    ]

    result = save_embedding_records_pickle_shards(
        tmp_path / "embeddings.pkl",
        records,
        records_per_shard=2,
    )

    assert result.record_count == 5
    assert [path.name for path in result.paths] == [
        "embeddings.shard_000001.pkl",
        "embeddings.shard_000002.pkl",
        "embeddings.shard_000003.pkl",
    ]
    loaded = load_embedding_records_pickle(result.paths[1])
    assert [record.id for record in loaded] == ["R2", "R3"]


def test_save_embedding_records_npy_shards_streams_records(tmp_path: Path) -> None:
    np = pytest.importorskip("numpy")
    records = [
        EmbeddingRecord(id=f"R{idx}", embedding=[float(idx), float(idx + 1)], layer_index=0, model_reference="m", shape=(2,))
        for idx in range(5)
    ]

    result = save_embedding_records_npy_shards(
        tmp_path / "embeddings.npy",
        records,
        records_per_shard=2,
    )

    assert result.record_count == 5
    assert [path.name for path in result.paths] == [
        "embeddings.shard_000001.npy",
        "embeddings.shard_000002.npy",
        "embeddings.shard_000003.npy",
    ]
    assert [path.name for path in result.id_paths] == [
        "embeddings.shard_000001.ids.txt",
        "embeddings.shard_000002.ids.txt",
        "embeddings.shard_000003.ids.txt",
    ]
    matrix = np.load(result.paths[1])
    assert matrix.shape == (2, 2)
    assert matrix[0].tolist() == pytest.approx([2.0, 3.0])
    assert matrix[1].tolist() == pytest.approx([3.0, 4.0])
    assert result.id_paths[1].read_text(encoding="utf-8").splitlines() == ["R2", "R3"]


def test_save_embedding_records_h5_appends_records(tmp_path: Path) -> None:
    h5py = pytest.importorskip("h5py")
    records_1 = [
        EmbeddingRecord(id="R1", embedding=[1.0, 2.0], layer_index=0, model_reference="m", shape=(2,)),
    ]
    records_2 = [
        EmbeddingRecord(id="R2", embedding=[3.0, 4.0], layer_index=0, model_reference="m", shape=(2,)),
    ]
    path = tmp_path / "embeddings.h5"

    first = save_embedding_records_h5(path, records_1)
    second = save_embedding_records_h5(path, records_2, append=True)

    assert first.record_count == 1
    assert second.record_count == 2
    with h5py.File(path, "r") as handle:
        assert handle["embeddings"].shape == (2, 2)
        assert handle["embeddings"][1].tolist() == pytest.approx([3.0, 4.0])
        assert handle["ids"].asstr()[:].tolist() == ["R1", "R2"]


def test_save_and_load_embedding_records_h5_vector_round_trip(tmp_path: Path) -> None:
    h5py = pytest.importorskip("h5py")
    records = [
        EmbeddingRecord(id="R1", embedding=[1.0, 2.0], layer_index=0, model_reference="m", shape=(2,)),
        EmbeddingRecord(id="R2", embedding=[3.0, 4.0], layer_index=1, model_reference="m", shape=(2,)),
    ]
    path = tmp_path / "vector.h5"

    save_embedding_records_h5(path, records)
    loaded = load_embedding_records_h5(path)

    assert [record.id for record in loaded] == ["R1", "R2"]
    assert [record.embedding for record in loaded] == [[1.0, 2.0], [3.0, 4.0]]
    assert [record.shape for record in loaded] == [(2,), (2,)]
    with h5py.File(path, "r") as handle:
        assert handle.attrs["payload_kind"] == "vector"
        assert handle["embeddings"].shape == (2, 2)


def test_save_and_load_embedding_records_h5_matrix_round_trip(tmp_path: Path) -> None:
    h5py = pytest.importorskip("h5py")
    records = [
        EmbeddingRecord(
            id="M1",
            embedding=[[1.0, 2.0], [3.0, 4.0]],
            layer_index=0,
            model_reference="m",
            shape=(2, 2),
        ),
        EmbeddingRecord(
            id="M2",
            embedding=[[5.0, 6.0]],
            layer_index=0,
            model_reference="m",
            shape=(1, 2),
        ),
    ]
    path = tmp_path / "matrix.h5"

    save_embedding_records_h5(path, records)
    loaded = load_embedding_records_h5(path)

    assert [record.id for record in loaded] == ["M1", "M2"]
    assert cast(Any, loaded[0].embedding).tolist() == [[1.0, 2.0], [3.0, 4.0]]
    assert cast(Any, loaded[1].embedding).tolist() == [[5.0, 6.0]]
    assert [record.shape for record in loaded] == [(2, 2), (1, 2)]
    with h5py.File(path, "r") as handle:
        assert handle.attrs["payload_kind"] == "matrix"
        assert handle["matrix_offsets"][:].tolist() == [0, 4, 6]


def test_save_and_load_embedding_records_h5_mixed_round_trip(tmp_path: Path) -> None:
    h5py = pytest.importorskip("h5py")
    records = [
        EmbeddingRecord(id="V1", embedding=[1.0, 2.0], layer_index=0, model_reference="m", shape=(2,)),
        EmbeddingRecord(
            id="M1",
            embedding=[[7.0, 8.0], [9.0, 10.0]],
            layer_index=0,
            model_reference="m",
            shape=(2, 2),
        ),
        EmbeddingRecord(id="V2", embedding=[3.0, 4.0], layer_index=1, model_reference="m", shape=(2,)),
    ]
    path = tmp_path / "mixed.h5"

    save_embedding_records_h5(path, records)
    loaded = load_embedding_records_h5(path)

    assert [record.id for record in loaded] == ["V1", "M1", "V2"]
    assert loaded[0].embedding == [1.0, 2.0]
    assert cast(Any, loaded[1].embedding).tolist() == [[7.0, 8.0], [9.0, 10.0]]
    assert loaded[2].embedding == [3.0, 4.0]
    with h5py.File(path, "r") as handle:
        assert handle.attrs["payload_kind"] == "mixed"
        assert handle["payload_kind"].asstr()[:].tolist() == ["vector", "matrix", "vector"]
        assert handle["vector_index"][:].tolist() == [0, -1, 1]
        assert handle["matrix_index"][:].tolist() == [-1, 0, -1]
        assert handle["embeddings"].shape == (2, 2)
        assert handle["matrix_offsets"][:].tolist() == [0, 4]


def test_load_embedding_records_h5_filters_and_slices_matrix_payloads(tmp_path: Path) -> None:
    pytest.importorskip("h5py")
    records = [
        EmbeddingRecord(
            id="P1",
            embedding=[[1.0, 10.0], [2.0, 20.0], [3.0, 30.0], [4.0, 40.0]],
            layer_index=0,
            model_reference="m",
            shape=(4, 2),
        ),
        EmbeddingRecord(
            id="P1",
            embedding=[[101.0, 1001.0], [102.0, 1002.0]],
            layer_index=1,
            model_reference="m",
            shape=(2, 2),
        ),
        EmbeddingRecord(
            id="P2",
            embedding=[[5.0, 50.0]],
            layer_index=0,
            model_reference="m",
            shape=(1, 2),
        ),
    ]
    path = tmp_path / "matrix_segments.h5"

    save_embedding_records_h5(path, records)
    loaded = load_embedding_records_h5(path, ids="P1", layer_index=0, residue_start=1, residue_end=3)

    assert len(loaded) == 1
    assert loaded[0].id == "P1"
    assert loaded[0].layer_index == 0
    assert loaded[0].shape == (2, 2)
    assert cast(Any, loaded[0].embedding).tolist() == [[2.0, 20.0], [3.0, 30.0]]


def test_h5_embedding_reader_selects_pool_method_for_same_id_and_layer(tmp_path: Path) -> None:
    h5py = pytest.importorskip("h5py")
    records = [
        EmbeddingRecord(
            id="P1",
            embedding=[[1.0, 2.0], [3.0, 4.0]],
            layer_index=0,
            model_reference="m",
            shape=(2, 2),
            metadata={"pooling": "none"},
        ),
        EmbeddingRecord(
            id="P1",
            embedding=[2.0, 3.0],
            layer_index=0,
            model_reference="m",
            shape=(2,),
            metadata={"pooling": "mean"},
        ),
        EmbeddingRecord(
            id="P1",
            embedding=[1.0, 2.0],
            layer_index=0,
            model_reference="m",
            shape=(2,),
            metadata={"pooling": "cls"},
        ),
    ]
    path = tmp_path / "pooled_variants.h5"

    save_embedding_records_h5(path, records)
    reader = H5EmbeddingReader(path)
    mean_record = reader.read("P1", layer_index=0, pool_method="mean")
    raw_record = reader.read("P1", layer_index=0, pool_method="none", residue_slice=(0, 1))

    assert mean_record.embedding == [2.0, 3.0]
    assert mean_record.metadata == {"pooling": "mean"}
    assert raw_record.shape == (1, 2)
    assert cast(Any, raw_record.embedding).tolist() == [[1.0, 2.0]]
    with pytest.raises(EmbeddingInputError):
        reader.read("P1", layer_index=0)
    with h5py.File(path, "r") as handle:
        assert handle["pool_method"].asstr()[:].tolist() == ["none", "mean", "cls"]


@pytest.mark.skipif("Bio" not in sys.modules and __import__("importlib").util.find_spec("Bio") is None, reason="Biopython not installed")
def test_generate_fasta_pickle_shards_streams_generation_results(tmp_path: Path) -> None:
    fasta_path = tmp_path / "input.fasta"
    fasta_path.write_text(">Q1\nACDE\n>Q2\nAAAA\n>Q3\nVVVV\n", encoding="utf-8")
    events: List[Dict[str, Any]] = []

    with pytest.warns(DeprecationWarning):
        result = generate_fasta_pickle_shards(
            fasta_path,
            _generator(),
            tmp_path / "out.pkl",
            batch_size=2,
            records_per_shard=2,
            layer_index=4,
            progress_callback=events.append,
        )

    assert result.record_count == 3
    assert result.error_count == 0
    assert [path.name for path in result.paths] == ["out.shard_000001.pkl", "out.shard_000002.pkl"]
    assert any(event["event"] == "batch_completed" for event in events)
    loaded = load_embedding_records_pickle(result.paths[0])
    assert [record.id for record in loaded] == ["Q1", "Q2"]
    assert loaded[0].embedding == [4.0, 4.0]


@pytest.mark.skipif("Bio" not in sys.modules and __import__("importlib").util.find_spec("Bio") is None, reason="Biopython not installed")
def test_generate_fasta_pickle_shards_can_sort_by_length_window(tmp_path: Path) -> None:
    fasta_path = tmp_path / "input.fasta"
    fasta_path.write_text(">Q1\nA\n>Q2\nAAAAA\n>Q3\nAAA\n>Q4\nAA\n", encoding="utf-8")

    with pytest.warns(DeprecationWarning):
        result = generate_fasta_pickle_shards(
            fasta_path,
            _generator(),
            tmp_path / "out.pkl",
            batch_size=2,
            records_per_shard=10,
            length_sort_window=3,
        )

    assert result.record_count == 4
    loaded = load_embedding_records_pickle(result.paths[0])
    assert [record.id for record in loaded] == ["Q2", "Q3", "Q1", "Q4"]


@pytest.mark.skipif("Bio" not in sys.modules and __import__("importlib").util.find_spec("Bio") is None, reason="Biopython not installed")
def test_generate_fasta_pickle_shards_writes_all_length_skips(tmp_path: Path) -> None:
    fasta_path = tmp_path / "input.fasta"
    skipped_path = tmp_path / "skipped.tsv"
    fasta_path.write_text(">Q1\nAAAAA\n>Q2\nAA\n>Q3\nAAAAAA\n", encoding="utf-8")

    with pytest.warns(DeprecationWarning):
        result = generate_fasta_pickle_shards(
            fasta_path,
            _generator(),
            tmp_path / "out.pkl",
            batch_size=2,
            records_per_shard=10,
            max_sequence_length=3,
            skipped_path=skipped_path,
        )

    assert result.record_count == 1
    assert result.skipped_count == 2
    assert skipped_path.read_text(encoding="utf-8").splitlines() == [
        "id\tlength\treason",
        "Q1\t5\tlength>3",
        "Q3\t6\tlength>3",
    ]


@pytest.mark.skipif("Bio" not in sys.modules and __import__("importlib").util.find_spec("Bio") is None, reason="Biopython not installed")
def test_generate_fasta_npy_shards_streams_generation_results(tmp_path: Path) -> None:
    np = pytest.importorskip("numpy")
    fasta_path = tmp_path / "input.fasta"
    fasta_path.write_text(">Q1\nACDE\n>Q2\nAAAA\n>Q3\nVVVV\n", encoding="utf-8")
    events: List[Dict[str, Any]] = []

    with pytest.warns(DeprecationWarning):
        result = generate_fasta_npy_shards(
            fasta_path,
            _generator(),
            tmp_path / "out.npy",
            batch_size=2,
            records_per_shard=2,
            layer_index=4,
            progress_callback=events.append,
        )

    assert result.record_count == 3
    assert result.error_count == 0
    assert [path.name for path in result.paths] == ["out.shard_000001.npy", "out.shard_000002.npy"]
    assert [path.name for path in result.id_paths] == ["out.shard_000001.ids.txt", "out.shard_000002.ids.txt"]
    assert any(event["event"] == "batch_completed" for event in events)
    matrix = np.load(result.paths[0])
    assert matrix.shape == (2, 2)
    assert matrix[0].tolist() == pytest.approx([4.0, 4.0])
    assert matrix[1].tolist() == pytest.approx([4.0, 4.0])
    assert result.id_paths[0].read_text(encoding="utf-8").splitlines() == ["Q1", "Q2"]


@pytest.mark.skipif("Bio" not in sys.modules and __import__("importlib").util.find_spec("Bio") is None, reason="Biopython not installed")
def test_generate_fasta_h5_streams_generation_results(tmp_path: Path) -> None:
    h5py = pytest.importorskip("h5py")
    fasta_path = tmp_path / "input.fasta"
    fasta_path.write_text(">Q1\nACDE\n>Q2\nAAAA\n>Q3\nVVVV\n", encoding="utf-8")
    events: List[Dict[str, Any]] = []

    with pytest.warns(DeprecationWarning):
        result = generate_fasta_h5(
            fasta_path,
            _generator(),
            tmp_path / "out.h5",
            batch_size=2,
            layer_index=4,
            progress_callback=events.append,
        )

    assert result.record_count == 3
    assert result.error_count == 0
    assert any(event["event"] == "batch_completed" for event in events)
    with h5py.File(result.path, "r") as handle:
        assert handle["embeddings"].shape == (3, 2)
        assert handle["embeddings"][0].tolist() == pytest.approx([4.0, 4.0])
        assert handle["ids"].asstr()[:].tolist() == ["Q1", "Q2", "Q3"]


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

    with pytest.raises(EmbeddingDependencyError, match="transformers"):
        ProtT5EmbeddingGenerator()


@pytest.mark.parametrize(
    ("model_name", "expected_model_reference"),
    [
        ("prott5_xxl_bfd", "Rostlab/prot_t5_xxl_bfd"),
        ("prot_t5_xl_bfd", "Rostlab/prot_t5_xl_bfd"),
        ("prot-t5-xxl-uniref50", "Rostlab/prot_t5_xxl_uniref50"),
        ("prott5_xl_half_uniref50_enc", "Rostlab/prot_t5_xl_half_uniref50-enc"),
    ],
)
def test_prott5_generator_uses_t5_tokenizer_for_transformers_fallback(
    monkeypatch: pytest.MonkeyPatch,
    model_name: str,
    expected_model_reference: str,
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

    generator = ProtT5EmbeddingGenerator(model_name=model_name, device="cpu")

    assert generator.model_reference == expected_model_reference
    assert _FakeAutoConfig.observed_name == expected_model_reference
    assert _FakeT5EncoderModel.observed_name == expected_model_reference
    assert _FakeT5Tokenizer.observed_name == expected_model_reference
    assert _FakeT5Tokenizer.observed_do_lower_case is False


class _FakeScalar:
    def __init__(self, value: float) -> None:
        self.value = value

    def item(self) -> float:
        return self.value


class _FakeTensor:
    def __init__(self, data: Any) -> None:
        self.data = data

    @property
    def shape(self) -> tuple[int, ...]:
        values = self.data
        dims: list[int] = []
        while isinstance(values, list):
            dims.append(len(values))
            values = values[0] if values else None
        return tuple(dims)

    def to(self, _device: str) -> "_FakeTensor":
        return self

    def float(self) -> "_FakeTensor":
        return self

    def detach(self) -> "_FakeTensor":
        return self

    def cpu(self) -> "_FakeTensor":
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

    def mean(self, *, dim: int) -> "_FakeTensor":
        if dim != 0:
            raise ValueError("fake tensor supports dim=0 only")
        rows = self.data
        if not rows:
            return _FakeTensor([])
        width = len(rows[0])
        return _FakeTensor([sum(float(row[index]) for row in rows) / float(len(rows)) for index in range(width)])

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
        lengths = [len(sequence.split()) + 1 for sequence in sequences]  # include end token
        max_len = max(lengths)
        return {
            "input_ids": [[1] * length + [0] * (max_len - length) for length in lengths],
            "attention_mask": [[1] * length + [0] * (max_len - length) for length in lengths],
        }


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
        input_rows = input_ids.tolist()
        batch_size = len(input_rows)
        length = len(input_rows[0])  # includes end token and padding
        hidden_states = []
        for layer in range(3):
            layer_values = [
                [[float(layer), float(row_index), float(pos)] for pos in range(length)]
                for row_index in range(batch_size)
            ]
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
    assert all(record.shape == (4, 3) for record in result.records)
    assert result.records[1].embedding[0] == [1.0, 0.0, 0.0]


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
    assert result.records[0].embedding[0] == [0.0, 0.0, 0.0]
    assert result.records[1].embedding[0] == [2.0, 0.0, 0.0]


def test_prott5_generate_batches_records_in_one_model_call(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_torch = types.SimpleNamespace(
        tensor=lambda values: _FakeTensor(values),
        no_grad=lambda: _FakeNoGrad(),
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    model = _FakeModel()
    generator = ProtT5EmbeddingGenerator(
        tokenizer=_FakeTokenizer(),
        model=model,
    )
    result = generator.generate(
        [GenerationInput(id="P1", sequence="ACDE"), GenerationInput(id="P2", sequence="AA")],
        layer_index=0,
    )

    assert result.errors == []
    assert [record.id for record in result.records] == ["P1", "P2"]
    assert result.records[0].shape == (4, 3)
    assert result.records[1].shape == (2, 3)
    assert result.records[1].embedding[0] == [0.0, 1.0, 0.0]


def test_prott5_generate_with_mean_pooler_returns_vectors(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_torch = types.SimpleNamespace(
        tensor=lambda values: _FakeTensor(values),
        no_grad=lambda: _FakeNoGrad(),
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    generator = ProtT5EmbeddingGenerator(
        tokenizer=_FakeTokenizer(),
        model=_FakeModel(),
    )
    result = generator.generate(
        [GenerationInput(id="P1", sequence="ACDE"), GenerationInput(id="P2", sequence="AA")],
        layer_index=0,
        pooler="mean",
    )

    assert result.errors == []
    assert [record.shape for record in result.records] == [(3,), (3,)]
    assert result.records[0].embedding == [0.0, 0.0, 1.5]
    assert result.records[1].embedding == [0.0, 1.0, 0.5]


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
        "layer_indexing": "hf_native_0_is_first_hidden",
    }
    assert isinstance(result.model_metadata, ModelMetadata)
    assert isinstance(result.run_metadata, RunMetadata)
    assert result.run_metadata is not None
    assert result.run_metadata.sequence_count == 1
    assert result.run_metadata.requested_layers == [1]
    assert result.run_metadata.resolved_layers == [1]
