from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


_SCRIPT_PATH = Path(__file__).resolve().parent.parent / "scripts" / "benchmark_fasta_embedding_search.py"
_SPEC = importlib.util.spec_from_file_location("benchmark_fasta_embedding_search", _SCRIPT_PATH)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)


def test_pool_embedding_leaves_flat_vector_unchanged() -> None:
    pooled = _MODULE._pool_embedding([1, 2.5, 3], pooling="mean")
    assert pooled == [1.0, 2.5, 3.0]


def test_pool_embedding_mean_reduces_per_residue_matrix() -> None:
    pooled = _MODULE._pool_embedding([[1.0, 2.0], [3.0, 6.0], [5.0, 10.0]], pooling="mean")
    assert pooled == pytest.approx([3.0, 6.0])


def test_pool_embedding_first_uses_first_residue_vector() -> None:
    pooled = _MODULE._pool_embedding([[7.0, 8.0], [9.0, 10.0]], pooling="first")
    assert pooled == pytest.approx([7.0, 8.0])


def test_chunk_records_splits_inputs() -> None:
    records = [
        _MODULE.GenerationInput(id="A", sequence="AAAA"),
        _MODULE.GenerationInput(id="B", sequence="BBBB"),
        _MODULE.GenerationInput(id="C", sequence="CCCC"),
    ]
    chunks = _MODULE._chunk_records(records, 2)
    assert [[record.id for record in chunk] for chunk in chunks] == [["A", "B"], ["C"]]


def test_sort_records_by_length_orders_shorter_first() -> None:
    records = [
        _MODULE.GenerationInput(id="B", sequence="AAAA"),
        _MODULE.GenerationInput(id="A", sequence="AA"),
        _MODULE.GenerationInput(id="C", sequence="AA"),
    ]
    ordered = _MODULE._sort_records_by_length(records)
    assert [record.id for record in ordered] == ["A", "C", "B"]


def test_chunk_records_by_token_budget_respects_budget() -> None:
    records = [
        _MODULE.GenerationInput(id="A", sequence="AA"),
        _MODULE.GenerationInput(id="B", sequence="AAA"),
        _MODULE.GenerationInput(id="C", sequence="AAAA"),
        _MODULE.GenerationInput(id="D", sequence="AAAAA"),
    ]
    chunks = _MODULE._chunk_records_by_token_budget(records, 8)
    assert [[record.id for record in chunk] for chunk in chunks] == [["A", "B"], ["C"], ["D"]]


def test_schedule_embedding_chunks_can_sort_then_budget() -> None:
    records = [
        _MODULE.GenerationInput(id="D", sequence="AAAAA"),
        _MODULE.GenerationInput(id="A", sequence="AA"),
        _MODULE.GenerationInput(id="C", sequence="AAAA"),
        _MODULE.GenerationInput(id="B", sequence="AAA"),
    ]
    chunks = _MODULE._schedule_embedding_chunks(
        records,
        chunk_size=1,
        max_tokens_per_batch=8,
        sort_by_length=True,
    )
    assert [[record.id for record in chunk] for chunk in chunks] == [["A", "B"], ["C"], ["D"]]
