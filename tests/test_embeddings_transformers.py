from __future__ import annotations

import logging

from CBBIO.embeddings_transformers import suppress_esm_tokenizer_class_warning


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

    existing_filter_count = len(logger.filters)
    with suppress_esm_tokenizer_class_warning():
        added_filters = logger.filters[existing_filter_count:]
        assert len(added_filters) == 1
        assert added_filters[0].filter(esm_record) is False
        assert added_filters[0].filter(other_record) is True
