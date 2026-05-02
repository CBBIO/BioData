"""Tests for CBBIO.sequence_similarity (parasail-based pairwise similarity)."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Optional
from unittest.mock import MagicMock, patch

import pytest

from CBBIO.sequence_similarity import (
    DEFAULT_ALGORITHM,
    DEFAULT_GAP_EXTEND,
    DEFAULT_GAP_OPEN,
    DEFAULT_MATRIX,
    SequenceSimilarityDependencyError,
    SequenceSimilarityInputError,
    SimilarityResult,
    _extract_id,
    _extract_sequence,
    _normalise,
    compute_similarity,
    pairwise_similarity_matrix,
    similarity_to_dict,
)


# ---------------------------------------------------------------------------
# Helpers / minimal GenerationInput-like object
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class _FakeInput:
    """Minimal stand-in for GenerationInput."""

    id: str
    sequence: str


# ---------------------------------------------------------------------------
# Unit tests for internal helpers
# ---------------------------------------------------------------------------

def test_normalise_identical() -> None:
    assert _normalise(10, 10, 10) == pytest.approx(1.0)


def test_normalise_zero_score() -> None:
    assert _normalise(0, 10, 10) == pytest.approx(0.0)


def test_normalise_negative_clamped() -> None:
    assert _normalise(-5, 10, 10) == pytest.approx(0.0)


def test_normalise_zero_self_score() -> None:
    assert _normalise(5, 0, 10) == pytest.approx(0.0)
    assert _normalise(5, 10, 0) == pytest.approx(0.0)


def test_normalise_partial() -> None:
    result = _normalise(6, 9, 4)
    assert result == pytest.approx(6.0 / math.sqrt(9 * 4))


def test_extract_sequence_from_string() -> None:
    assert _extract_sequence("ACGT") == "ACGT"


def test_extract_sequence_from_generation_input() -> None:
    obj = _FakeInput(id="prot1", sequence="MKTLL")
    assert _extract_sequence(obj) == "MKTLL"


def test_extract_sequence_invalid() -> None:
    with pytest.raises(SequenceSimilarityInputError, match="Cannot extract"):
        _extract_sequence(42)  # type: ignore[arg-type]


def test_extract_id_from_string() -> None:
    assert _extract_id("ACGT") is None


def test_extract_id_from_generation_input() -> None:
    obj = _FakeInput(id="prot1", sequence="MKTLL")
    assert _extract_id(obj) == "prot1"


def test_extract_id_missing() -> None:
    class _NoID:
        pass

    assert _extract_id(_NoID()) is None


# ---------------------------------------------------------------------------
# compute_similarity – happy path (real parasail)
# ---------------------------------------------------------------------------

def test_compute_similarity_identical_strings() -> None:
    result = compute_similarity("ACGT", "ACGT")
    assert isinstance(result, SimilarityResult)
    assert result.similarity == pytest.approx(1.0)
    assert result.id_a is None
    assert result.id_b is None
    assert result.algorithm == "local"
    assert result.matrix == "blosum62"


def test_compute_similarity_identical_generation_inputs() -> None:
    a = _FakeInput(id="p1", sequence="MKTLL")
    b = _FakeInput(id="p2", sequence="MKTLL")
    result = compute_similarity(a, b)
    assert result.similarity == pytest.approx(1.0)
    assert result.id_a == "p1"
    assert result.id_b == "p2"


def test_compute_similarity_different_sequences() -> None:
    result = compute_similarity("AAAA", "CCCC")
    assert 0.0 <= result.similarity <= 1.0


def test_compute_similarity_global_algorithm() -> None:
    result = compute_similarity("ACGT", "ACGT", algorithm="global")
    assert result.similarity == pytest.approx(1.0)
    assert result.algorithm == "global"


def test_compute_similarity_semiglobal_algorithm() -> None:
    result = compute_similarity("ACGT", "ACGT", algorithm="semiglobal")
    assert result.similarity == pytest.approx(1.0)
    assert result.algorithm == "semiglobal"


def test_compute_similarity_pam250_matrix() -> None:
    result = compute_similarity("MKTLL", "MKTLL", matrix="pam250")
    assert result.similarity == pytest.approx(1.0)
    assert result.matrix == "pam250"


def test_compute_similarity_mixed_input_types() -> None:
    obj = _FakeInput(id="prot1", sequence="ACGT")
    result = compute_similarity("ACGT", obj)
    assert result.similarity == pytest.approx(1.0)
    assert result.id_a is None
    assert result.id_b == "prot1"


# ---------------------------------------------------------------------------
# compute_similarity – error paths
# ---------------------------------------------------------------------------

def test_compute_similarity_empty_seq_a() -> None:
    with pytest.raises(SequenceSimilarityInputError, match="seq_a"):
        compute_similarity("", "ACGT")


def test_compute_similarity_empty_seq_b() -> None:
    with pytest.raises(SequenceSimilarityInputError, match="seq_b"):
        compute_similarity("ACGT", "")


def test_compute_similarity_unknown_algorithm() -> None:
    with pytest.raises(SequenceSimilarityInputError, match="algorithm"):
        compute_similarity("ACGT", "ACGT", algorithm="unknown")  # type: ignore[arg-type]


def test_compute_similarity_unknown_matrix() -> None:
    with pytest.raises(SequenceSimilarityInputError, match="matrix"):
        compute_similarity("ACGT", "ACGT", matrix="notamatrix")


def test_compute_similarity_negative_gap_open() -> None:
    with pytest.raises(SequenceSimilarityInputError, match="gap_open"):
        compute_similarity("ACGT", "ACGT", gap_open=-1)


def test_compute_similarity_negative_gap_extend() -> None:
    with pytest.raises(SequenceSimilarityInputError, match="gap_extend"):
        compute_similarity("ACGT", "ACGT", gap_extend=-1)


def test_compute_similarity_missing_parasail(monkeypatch: pytest.MonkeyPatch) -> None:
    import builtins
    original_import = builtins.__import__

    def mock_import(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == "parasail":
            raise ModuleNotFoundError("No module named 'parasail'")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", mock_import)
    with pytest.raises(SequenceSimilarityDependencyError, match="parasail"):
        compute_similarity("ACGT", "ACGT")


# ---------------------------------------------------------------------------
# pairwise_similarity_matrix – happy path
# ---------------------------------------------------------------------------

def test_pairwise_matrix_single_sequence() -> None:
    matrix = pairwise_similarity_matrix(["ACGT"])
    assert len(matrix) == 1
    assert len(matrix[0]) == 1
    assert matrix[0][0].similarity == pytest.approx(1.0)


def test_pairwise_matrix_symmetry() -> None:
    seqs = ["MKTLL", "ACDEF", "GHIKL"]
    matrix = pairwise_similarity_matrix(seqs)
    n = len(seqs)
    for i in range(n):
        for j in range(n):
            assert matrix[i][j].similarity == pytest.approx(matrix[j][i].similarity, abs=1e-9)


def test_pairwise_matrix_diagonal_is_one() -> None:
    seqs = ["MKTLL", "ACDEF"]
    matrix = pairwise_similarity_matrix(seqs)
    for i in range(len(seqs)):
        assert matrix[i][i].similarity == pytest.approx(1.0)


def test_pairwise_matrix_with_generation_inputs() -> None:
    inputs = [
        _FakeInput(id="a", sequence="MKTLL"),
        _FakeInput(id="b", sequence="MKTLL"),
    ]
    matrix = pairwise_similarity_matrix(inputs)
    assert matrix[0][1].id_a == "a"
    assert matrix[0][1].id_b == "b"
    assert matrix[0][1].similarity == pytest.approx(1.0)


def test_pairwise_matrix_values_in_range() -> None:
    seqs = ["ACGT", "AGGT", "TTTT"]
    matrix = pairwise_similarity_matrix(seqs)
    for row in matrix:
        for cell in row:
            assert 0.0 <= cell.similarity <= 1.0 + 1e-9


def test_pairwise_matrix_empty_raises() -> None:
    with pytest.raises(SequenceSimilarityInputError, match="non-empty"):
        pairwise_similarity_matrix([])


def test_pairwise_matrix_empty_sequence_in_list_raises() -> None:
    with pytest.raises(SequenceSimilarityInputError):
        pairwise_similarity_matrix(["ACGT", ""])


# ---------------------------------------------------------------------------
# similarity_to_dict
# ---------------------------------------------------------------------------

def test_similarity_to_dict_with_ids() -> None:
    inputs = [
        _FakeInput(id="p1", sequence="MKTLL"),
        _FakeInput(id="p2", sequence="ACDEF"),
    ]
    matrix = pairwise_similarity_matrix(inputs)
    d = similarity_to_dict(matrix)
    assert ("p1", "p1") in d
    assert ("p1", "p2") in d
    assert ("p2", "p1") in d
    assert d[("p1", "p1")] == pytest.approx(1.0)
    assert d[("p1", "p2")] == pytest.approx(d[("p2", "p1")], abs=1e-9)


def test_similarity_to_dict_without_ids() -> None:
    matrix = pairwise_similarity_matrix(["ACGT", "AGGT"])
    d = similarity_to_dict(matrix)
    assert (None, None) in d


# ---------------------------------------------------------------------------
# Default constant values
# ---------------------------------------------------------------------------

def test_default_constants() -> None:
    assert DEFAULT_ALGORITHM == "local"
    assert DEFAULT_MATRIX == "blosum62"
    assert DEFAULT_GAP_OPEN == 11
    assert DEFAULT_GAP_EXTEND == 1
