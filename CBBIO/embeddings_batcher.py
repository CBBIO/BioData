"""Input batchers for embedding generation jobs."""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any, Callable, Dict, List, Literal, TextIO

from .embeddings import (
    EmbeddingInputError,
    GenerationInput,
    iter_fasta_inputs,
)


def _validate_optional_positive(name: str, value: int | None) -> int | None:
    if value is None:
        return None
    resolved = int(value)
    if resolved < 1:
        raise EmbeddingInputError(f"{name} must be >= 1 when provided.")
    return resolved


def _estimated_tokens(record: GenerationInput) -> int:
    return len(record.sequence) + 1


def _batch_records(
    records: Iterable[GenerationInput],
    *,
    batch_size: int | None,
    max_batch_tokens: int | None,
    token_estimator: Callable[[GenerationInput], int] | None = None,
) -> Iterator[List[GenerationInput]]:
    resolved_batch_size = _validate_optional_positive("batch_size", batch_size)
    resolved_max_tokens = _validate_optional_positive("max_batch_tokens", max_batch_tokens)
    if resolved_batch_size is None and resolved_max_tokens is None:
        resolved_batch_size = 1

    estimate = token_estimator if token_estimator is not None else _estimated_tokens
    batch: List[GenerationInput] = []
    batch_max_tokens = 0

    for record in records:
        record_tokens = estimate(record)
        candidate_max_tokens = max(batch_max_tokens, record_tokens)
        exceeds_size = resolved_batch_size is not None and len(batch) >= resolved_batch_size
        exceeds_tokens = (
            resolved_max_tokens is not None
            and batch
            and (len(batch) + 1) * candidate_max_tokens > resolved_max_tokens
        )
        if batch and (exceeds_size or exceeds_tokens):
            yield batch
            batch = []
            batch_max_tokens = 0

        batch.append(record)
        batch_max_tokens = max(batch_max_tokens, record_tokens)

    if batch:
        yield batch


def _limit_records(records: Iterable[GenerationInput], limit: int | None) -> Iterator[GenerationInput]:
    resolved_limit = _validate_optional_positive("limit", limit)
    if resolved_limit is None:
        yield from records
        return

    for index, record in enumerate(records):
        if index >= resolved_limit:
            break
        yield record


def _iter_length_sorted_windows(
    records: Iterable[GenerationInput],
    *,
    window_size: int | None,
) -> Iterator[GenerationInput]:
    resolved_window_size = _validate_optional_positive("length_sort_window", window_size)
    if resolved_window_size is None:
        yield from records
        return

    pending: List[GenerationInput] = []
    for record in records:
        pending.append(record)
        if len(pending) >= resolved_window_size:
            pending.sort(key=lambda item: len(item.sequence), reverse=True)
            yield from pending
            pending = []

    if pending:
        pending.sort(key=lambda item: len(item.sequence), reverse=True)
        yield from pending


class _SkippedTsvWriter:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._handle: TextIO = self.path.open("w", encoding="utf-8", buffering=1)
        self._handle.write("id\tlength\treason\n")

    def write(self, event: Dict[str, Any]) -> None:
        self._handle.write(
            f"{_tsv_cell(event.get('id', ''))}\t{int(event.get('length', 0))}\t{_tsv_cell(event.get('reason', ''))}\n"
        )

    def close(self) -> None:
        self._handle.close()


def _tsv_cell(value: object) -> str:
    return str(value).replace("\t", " ").replace("\r", " ").replace("\n", " ")


class FastaBatcher:
    """Lazily read FASTA records and yield generation batches."""

    def __init__(
        self,
        path: str | Path,
        *,
        batch_size: int | None = None,
        max_batch_tokens: int | None = None,
        limit: int | None = None,
        length_sort_window: int | None = None,
        max_sequence_length: int | None = None,
        skipped_path: str | Path | None = None,
        id_from: Literal["record_id", "description"] = "record_id",
        token_estimator: Callable[[GenerationInput], int] | None = None,
    ) -> None:
        self.path = Path(path)
        self.batch_size = _validate_optional_positive("batch_size", batch_size)
        self.max_batch_tokens = _validate_optional_positive("max_batch_tokens", max_batch_tokens)
        self.limit = _validate_optional_positive("limit", limit)
        self.length_sort_window = _validate_optional_positive("length_sort_window", length_sort_window)
        self.max_sequence_length = _validate_optional_positive("max_sequence_length", max_sequence_length)
        self.skipped_path = Path(skipped_path) if skipped_path is not None else None
        if id_from not in {"record_id", "description"}:
            raise EmbeddingInputError("id_from must be one of: 'record_id', 'description'.")
        self.id_from: Literal["record_id", "description"] = id_from
        self.token_estimator = token_estimator
        self.skipped: List[Dict[str, Any]] = []
        self.skipped_count = 0

    def __iter__(self) -> Iterator[List[GenerationInput]]:
        self.skipped = []
        self.skipped_count = 0
        writer = _SkippedTsvWriter(self.skipped_path) if self.skipped_path is not None else None
        try:
            records = self._iter_accepted_records(writer.write if writer is not None else None)
            records = _iter_length_sorted_windows(records, window_size=self.length_sort_window)
            records = _limit_records(records, self.limit)
            yield from _batch_records(
                records,
                batch_size=self.batch_size,
                max_batch_tokens=self.max_batch_tokens,
                token_estimator=self.token_estimator,
            )
        finally:
            if writer is not None:
                writer.close()

    def _iter_accepted_records(
        self,
        skipped_callback: Callable[[Dict[str, Any]], None] | None,
    ) -> Iterator[GenerationInput]:
        for record in iter_fasta_inputs(self.path, id_from=self.id_from):
            sequence_length = len(record.sequence)
            if self.max_sequence_length is not None and sequence_length > self.max_sequence_length:
                event = {
                    "id": record.id,
                    "length": sequence_length,
                    "reason": f"length>{self.max_sequence_length}",
                }
                self.skipped_count += 1
                if len(self.skipped) < 5:
                    self.skipped.append(event)
                if skipped_callback is not None:
                    skipped_callback(event)
                continue
            yield record


class IterableBatcher:
    """Batch an in-memory or streaming iterable of ``GenerationInput`` records."""

    def __init__(
        self,
        records: Iterable[GenerationInput],
        *,
        batch_size: int | None = None,
        max_batch_tokens: int | None = None,
        limit: int | None = None,
        max_sequence_length: int | None = None,
        skipped_path: str | Path | None = None,
        token_estimator: Callable[[GenerationInput], int] | None = None,
    ) -> None:
        self.records = records
        self.batch_size = _validate_optional_positive("batch_size", batch_size)
        self.max_batch_tokens = _validate_optional_positive("max_batch_tokens", max_batch_tokens)
        self.limit = _validate_optional_positive("limit", limit)
        self.max_sequence_length = _validate_optional_positive("max_sequence_length", max_sequence_length)
        self.skipped_path = Path(skipped_path) if skipped_path is not None else None
        self.token_estimator = token_estimator
        self.skipped: List[Dict[str, Any]] = []
        self.skipped_count = 0

    def __iter__(self) -> Iterator[List[GenerationInput]]:
        self.skipped = []
        self.skipped_count = 0
        writer = _SkippedTsvWriter(self.skipped_path) if self.skipped_path is not None else None
        try:
            records = self._iter_accepted_records(writer.write if writer is not None else None)
            records = _limit_records(records, self.limit)
            yield from _batch_records(
                records,
                batch_size=self.batch_size,
                max_batch_tokens=self.max_batch_tokens,
                token_estimator=self.token_estimator,
            )
        finally:
            if writer is not None:
                writer.close()

    def _iter_accepted_records(
        self,
        skipped_callback: Callable[[Dict[str, Any]], None] | None,
    ) -> Iterator[GenerationInput]:
        for record in self.records:
            sequence_length = len(record.sequence)
            if self.max_sequence_length is not None and sequence_length > self.max_sequence_length:
                event = {
                    "id": record.id,
                    "length": sequence_length,
                    "reason": f"length>{self.max_sequence_length}",
                }
                self.skipped_count += 1
                if len(self.skipped) < 5:
                    self.skipped.append(event)
                if skipped_callback is not None:
                    skipped_callback(event)
                continue
            yield record


__all__ = ["FastaBatcher", "IterableBatcher"]
