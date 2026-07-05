from __future__ import annotations

import logging
import sys
import types

import pytest

from CBBIO.embeddings.utils.transformers import load_esm_tokenizer, suppress_esm_tokenizer_class_warning


def test_suppress_esm_tokenizer_class_warning_filters_only_esm_name_mismatch() -> None:
    logger = logging.getLogger("transformers.tokenization_utils_base")
    esm_record = logger.makeRecord(
        logger.name,
        logging.WARNING,
        __file__,
        1,
        (
            "The tokenizer class you load from this checkpoint is not the same type as the class this function is "
            "called from. The tokenizer class you load from this checkpoint is 'ESMTokenizer'. The class this "
            "function is called from is 'EsmTokenizer'."
        ),
        (),
        None,
    )
    other_record = logger.makeRecord(logger.name, logging.WARNING, __file__, 1, "keep this warning", (), None)
    esmc_record = logger.makeRecord(
        logger.name,
        logging.WARNING,
        __file__,
        1,
        (
            "The tokenizer class you load from this checkpoint is not the same type as the class this function is "
            "called from. The tokenizer class you load from this checkpoint is 'ESMCTokenizer'. The class this "
            "function is called from is 'EsmTokenizer'."
        ),
        (),
        None,
    )

    existing_filter_count = len(logger.filters)
    with suppress_esm_tokenizer_class_warning():
        added_filters = logger.filters[existing_filter_count:]
        assert len(added_filters) == 1
        assert added_filters[0].filter(esm_record) is False
        assert added_filters[0].filter(esmc_record) is False
        assert added_filters[0].filter(other_record) is True


def test_load_esm_tokenizer_falls_back_to_esmc_sequence_tokenizer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _FakeAutoTokenizer:
        @staticmethod
        def from_pretrained(model_name: str) -> object:
            _ = model_name
            raise ValueError("Tokenizer class ESMCTokenizer does not exist or is not currently imported.")

    class _FakeEsmSequenceTokenizer:
        pass

    fake_transformers = types.SimpleNamespace(AutoTokenizer=_FakeAutoTokenizer)
    fake_sequence_tokenizer = types.SimpleNamespace(EsmSequenceTokenizer=_FakeEsmSequenceTokenizer)
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)
    monkeypatch.setitem(sys.modules, "esm.tokenization.sequence_tokenizer", fake_sequence_tokenizer)

    tokenizer = load_esm_tokenizer("biohub/ESMC-300M")

    assert isinstance(tokenizer, _FakeEsmSequenceTokenizer)
