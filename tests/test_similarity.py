from __future__ import annotations

import pytest

from CBBIO.similarity import (
    AlignmentResult,
    InvalidSequenceError,
    SequenceSimilarityError,
    SimilarityDependencyError,
    UnknownMatrixError,
    align_sequences,
)


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------

_SEQ_SHORT = "MKTAYIAK"
_SEQ_IDENTICAL = "ACDEFGHIKLMNPQRSTVWY"
_SEQ_ONE_MISMATCH = "ACDEFXHIKLMNPQRSTVWY"   # G -> X at position 5
_SEQ_ONE_GAP = "ACDEFHIKLMNPQRSTVWY"          # G deleted at position 5


# ---------------------------------------------------------------------------
# Basic return-type and structure
# ---------------------------------------------------------------------------


def test_align_returns_alignment_result() -> None:
    result = align_sequences(_SEQ_IDENTICAL, _SEQ_IDENTICAL)
    assert isinstance(result, AlignmentResult)


def test_align_identical_sequences_perfect_identity() -> None:
    result = align_sequences(_SEQ_IDENTICAL, _SEQ_IDENTICAL)
    assert result.matches == len(_SEQ_IDENTICAL)
    assert result.mismatches == 0
    assert result.gaps == 0
    assert result.identity == pytest.approx(100.0)
    assert result.positives == pytest.approx(100.0)
    assert result.alignment_length == len(_SEQ_IDENTICAL)


def test_align_identity_with_one_mismatch() -> None:
    result = align_sequences(_SEQ_IDENTICAL, _SEQ_ONE_MISMATCH)
    assert result.mismatches == 1
    assert result.gaps == 0
    assert result.alignment_length == len(_SEQ_IDENTICAL)
    assert result.identity < 100.0
    assert result.identity == pytest.approx(95.0)


def test_align_identity_with_one_gap() -> None:
    result = align_sequences(_SEQ_IDENTICAL, _SEQ_ONE_GAP)
    # SW local alignment: query has 1 extra character (G) → gap in reference
    assert result.gaps == 1
    assert result.mismatches == 0
    assert result.matches == len(_SEQ_ONE_GAP)
    assert result.identity < 100.0


def test_align_score_is_positive_for_similar_sequences() -> None:
    result = align_sequences(_SEQ_IDENTICAL, _SEQ_IDENTICAL)
    assert result.score > 0


# ---------------------------------------------------------------------------
# Aligned string content
# ---------------------------------------------------------------------------


def test_aligned_strings_same_length_as_midline() -> None:
    result = align_sequences(_SEQ_IDENTICAL, _SEQ_ONE_GAP)
    assert len(result.query_aligned) == len(result.ref_aligned) == len(result.midline)


def test_aligned_strings_identical_sequences() -> None:
    result = align_sequences(_SEQ_IDENTICAL, _SEQ_IDENTICAL)
    assert result.query_aligned == _SEQ_IDENTICAL
    assert result.ref_aligned == _SEQ_IDENTICAL
    assert result.midline == "|" * len(_SEQ_IDENTICAL)


def test_midline_contains_dot_for_mismatch() -> None:
    result = align_sequences(_SEQ_IDENTICAL, _SEQ_ONE_MISMATCH)
    assert "." in result.midline


def test_midline_contains_space_for_gap() -> None:
    result = align_sequences(_SEQ_IDENTICAL, _SEQ_ONE_GAP)
    # gap appears as a space in the midline
    assert " " in result.midline


# ---------------------------------------------------------------------------
# Local vs global mode
# ---------------------------------------------------------------------------


def test_mode_local_is_default() -> None:
    result = align_sequences(_SEQ_IDENTICAL, _SEQ_ONE_MISMATCH)
    assert result.mode == "local"


def test_mode_global_reported_correctly() -> None:
    result = align_sequences(_SEQ_IDENTICAL, _SEQ_ONE_MISMATCH, mode="global")
    assert result.mode == "global"


def test_local_and_global_same_result_for_identical_sequences() -> None:
    local = align_sequences(_SEQ_IDENTICAL, _SEQ_IDENTICAL, mode="local")
    global_ = align_sequences(_SEQ_IDENTICAL, _SEQ_IDENTICAL, mode="global")
    assert local.score == global_.score
    assert local.matches == global_.matches
    assert local.identity == pytest.approx(global_.identity)


def test_local_score_ge_zero_for_unrelated_sequences() -> None:
    # SW local score is always >= 0
    result = align_sequences("AAAAAAAAAA", "CCCCCCCCCC", mode="local")
    assert result.score >= 0


# ---------------------------------------------------------------------------
# Custom gap penalties and substitution matrix
# ---------------------------------------------------------------------------


def test_custom_gap_penalties_accepted() -> None:
    result = align_sequences(_SEQ_IDENTICAL, _SEQ_ONE_GAP, gap_open=5, gap_extend=2)
    assert isinstance(result, AlignmentResult)


def test_custom_matrix_blosum50() -> None:
    result = align_sequences(_SEQ_IDENTICAL, _SEQ_IDENTICAL, matrix="blosum50")
    assert result.identity == pytest.approx(100.0)


def test_custom_matrix_pam250() -> None:
    result = align_sequences(_SEQ_SHORT, _SEQ_SHORT, matrix="pam250")
    assert result.identity == pytest.approx(100.0)


# ---------------------------------------------------------------------------
# Statistics consistency
# ---------------------------------------------------------------------------


def test_alignment_length_equals_sum_of_components() -> None:
    for seq2 in [_SEQ_IDENTICAL, _SEQ_ONE_MISMATCH, _SEQ_ONE_GAP]:
        result = align_sequences(_SEQ_IDENTICAL, seq2)
        assert result.matches + result.mismatches + result.gaps == result.alignment_length


def test_identity_range() -> None:
    for seq2 in [_SEQ_IDENTICAL, _SEQ_ONE_MISMATCH, _SEQ_ONE_GAP]:
        result = align_sequences(_SEQ_IDENTICAL, seq2)
        assert 0.0 <= result.identity <= 100.0


def test_positives_ge_identity() -> None:
    # positives (similar) is always >= identity (matches)
    result = align_sequences(_SEQ_IDENTICAL, _SEQ_ONE_MISMATCH)
    assert result.positives >= result.identity


def test_gaps_match_gap_chars_in_aligned_strings() -> None:
    result = align_sequences(_SEQ_IDENTICAL, _SEQ_ONE_GAP)
    expected_gaps = result.query_aligned.count("-") + result.ref_aligned.count("-")
    assert result.gaps == expected_gaps


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------


def test_empty_seq1_raises_invalid_sequence_error() -> None:
    with pytest.raises(InvalidSequenceError):
        align_sequences("", _SEQ_IDENTICAL)


def test_empty_seq2_raises_invalid_sequence_error() -> None:
    with pytest.raises(InvalidSequenceError):
        align_sequences(_SEQ_IDENTICAL, "")


def test_non_string_seq1_raises_invalid_sequence_error() -> None:
    with pytest.raises(InvalidSequenceError):
        align_sequences(None, _SEQ_IDENTICAL)  # type: ignore[arg-type]


def test_non_string_seq2_raises_invalid_sequence_error() -> None:
    with pytest.raises(InvalidSequenceError):
        align_sequences(_SEQ_IDENTICAL, 123)  # type: ignore[arg-type]


def test_unknown_matrix_raises_error() -> None:
    with pytest.raises(UnknownMatrixError):
        align_sequences(_SEQ_IDENTICAL, _SEQ_IDENTICAL, matrix="nonexistent_matrix")


def test_error_hierarchy() -> None:
    assert issubclass(SimilarityDependencyError, SequenceSimilarityError)
    assert issubclass(InvalidSequenceError, SequenceSimilarityError)
    assert issubclass(UnknownMatrixError, SequenceSimilarityError)
