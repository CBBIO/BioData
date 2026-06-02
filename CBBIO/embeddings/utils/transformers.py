"""Shared Hugging Face Transformers helpers for embedding generators."""

from __future__ import annotations

from contextlib import contextmanager
import logging
from typing import Any, Iterator, cast


_ESM_TOKENIZER_CLASS_WARNING = "The tokenizer class you load from this checkpoint is not the same type"
_ESM_TOKENIZER_CLASS_NAMES = ("'ESMTokenizer'", "'EsmTokenizer'")


class _EsmTokenizerClassWarningFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        message = record.getMessage()
        if _ESM_TOKENIZER_CLASS_WARNING not in message:
            return True
        return not all(name in message for name in _ESM_TOKENIZER_CLASS_NAMES)


@contextmanager
def suppress_esm_tokenizer_class_warning() -> Iterator[None]:
    """Suppress the harmless ESMTokenizer/EsmTokenizer checkpoint metadata warning."""

    logger = logging.getLogger("transformers.tokenization_utils_base")
    filter_obj = _EsmTokenizerClassWarningFilter()
    logger.addFilter(filter_obj)
    try:
        yield
    finally:
        logger.removeFilter(filter_obj)


def load_esm_tokenizer(model_name: str) -> Any:
    """Load an ESM tokenizer across transformers versions/checkpoint metadata variants."""

    try:
        from transformers import AutoTokenizer  # type: ignore
    except ModuleNotFoundError as exc:
        raise RuntimeError("transformers is required to load ESM tokenizers") from exc

    with suppress_esm_tokenizer_class_warning():
        try:
            return cast(Any, AutoTokenizer).from_pretrained(model_name)
        except ValueError as exc:
            message = str(exc)
            if "Tokenizer class ESMTokenizer" not in message:
                raise

    # Some checkpoints reference legacy `ESMTokenizer`; retry with concrete class.
    try:
        from transformers import EsmTokenizer  # type: ignore
    except Exception as exc:
        raise RuntimeError(
            "Failed to load tokenizer via AutoTokenizer and EsmTokenizer fallback is unavailable"
        ) from exc

    return cast(Any, EsmTokenizer).from_pretrained(model_name)


__all__ = ["load_esm_tokenizer", "suppress_esm_tokenizer_class_warning"]
