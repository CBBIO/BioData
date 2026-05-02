"""Pairwise sequence similarity using parasail.

This module is intentionally independent from database access code.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


AlignmentMode = Literal["local", "global"]

DEFAULT_GAP_OPEN: int = 10
DEFAULT_GAP_EXTEND: int = 1
DEFAULT_MATRIX: str = "blosum62"
DEFAULT_MODE: AlignmentMode = "local"


class SequenceSimilarityError(Exception):
    """Base exception for ``CBBIO.similarity`` errors."""


class SimilarityDependencyError(SequenceSimilarityError):
    """Raised when the ``parasail`` dependency is not installed."""


class InvalidSequenceError(SequenceSimilarityError):
    """Raised when one or both input sequences are invalid."""


class UnknownMatrixError(SequenceSimilarityError):
    """Raised when the requested substitution matrix does not exist in parasail."""


@dataclass(frozen=True)
class AlignmentResult:
    """Statistics and aligned strings for a pairwise sequence alignment.

    Attributes
    ----------
    score:
        Raw alignment score returned by parasail.
    alignment_length:
        Total number of columns in the alignment (matches + mismatches + gap
        columns).
    matches:
        Number of identical aligned positions.
    mismatches:
        Number of substituted (non-identical, non-gap) aligned positions.
    gaps:
        Total number of gap characters across both aligned sequences.
    identity:
        Percentage of identical positions over the alignment length
        (``matches / alignment_length * 100``).  ``0.0`` when
        ``alignment_length`` is zero.
    positives:
        Percentage of positions with a positive substitution score
        (matches + conservative substitutions) over the alignment length.
        ``0.0`` when ``alignment_length`` is zero.
    mode:
        Alignment mode used: ``"local"`` (Smith-Waterman) or ``"global"``
        (Needleman-Wunsch).
    query_aligned:
        The first (query) sequence in the alignment, with ``"-"`` for gaps.
    ref_aligned:
        The second (reference) sequence in the alignment, with ``"-"`` for
        gaps.
    midline:
        Per-column comparison string where ``"|"`` denotes an identical pair,
        ``"."`` a substitution, and ``" "`` a gap column.
    """

    score: int
    alignment_length: int
    matches: int
    mismatches: int
    gaps: int
    identity: float
    positives: float
    mode: AlignmentMode
    query_aligned: str
    ref_aligned: str
    midline: str


def align_sequences(
    seq1: str,
    seq2: str,
    *,
    mode: AlignmentMode = DEFAULT_MODE,
    gap_open: int = DEFAULT_GAP_OPEN,
    gap_extend: int = DEFAULT_GAP_EXTEND,
    matrix: str = DEFAULT_MATRIX,
) -> AlignmentResult:
    """Align two sequences and return pairwise similarity statistics.

    Uses the `parasail <https://github.com/jeffdaily/parasail-python>`_
    library with SIMD-accelerated Smith-Waterman (local) or
    Needleman-Wunsch (global) alignment.

    Parameters
    ----------
    seq1:
        Query sequence string (e.g. an amino-acid or nucleotide sequence).
        Must be a non-empty string.
    seq2:
        Reference sequence string.  Must be a non-empty string.
    mode:
        ``"local"`` (default) for Smith-Waterman local alignment; ``"global"``
        for Needleman-Wunsch global alignment.
    gap_open:
        Gap-opening penalty (positive integer, default ``10``).
    gap_extend:
        Gap-extension penalty (positive integer, default ``1``).
    matrix:
        Name of the substitution matrix available in parasail (default
        ``"blosum62"``).  Examples: ``"blosum50"``, ``"pam250"``,
        ``"dnafull"``.

    Returns
    -------
    AlignmentResult
        Dataclass containing the alignment score, length, identity,
        positives percentage, gap/mismatch counts, and the aligned strings.

    Raises
    ------
    SimilarityDependencyError
        If the ``parasail`` package is not installed.
    InvalidSequenceError
        If either sequence is empty or not a string.
    UnknownMatrixError
        If *matrix* is not available in parasail.
    SequenceSimilarityError
        For other alignment errors.
    """
    try:
        import parasail as _parasail  # type: ignore[import-untyped]
    except ModuleNotFoundError as exc:
        raise SimilarityDependencyError(
            "Missing dependency 'parasail'. Install with: pip install parasail"
        ) from exc

    if not isinstance(seq1, str) or not seq1:
        raise InvalidSequenceError("seq1 must be a non-empty string.")
    if not isinstance(seq2, str) or not seq2:
        raise InvalidSequenceError("seq2 must be a non-empty string.")

    subst_matrix = getattr(_parasail, matrix, None)
    if subst_matrix is None or not isinstance(subst_matrix, _parasail.Matrix):
        raise UnknownMatrixError(
            f"Matrix {matrix!r} is not available in parasail. "
            "See parasail documentation for valid matrix names (e.g. 'blosum62', 'pam250')."
        )

    if mode == "local":
        trace_fn = _parasail.sw_trace_striped_sat
        stats_fn = _parasail.sw_stats_striped_sat
    else:
        trace_fn = _parasail.nw_trace_striped_sat
        stats_fn = _parasail.nw_stats_striped_sat

    try:
        trace_result = trace_fn(seq1, seq2, gap_open, gap_extend, subst_matrix)
        stats_result = stats_fn(seq1, seq2, gap_open, gap_extend, subst_matrix)
    except Exception as exc:
        raise SequenceSimilarityError(f"Alignment failed: {exc}") from exc

    tb = trace_result.traceback
    query_aligned: str = tb.query
    ref_aligned: str = tb.ref
    midline: str = tb.comp

    score: int = int(stats_result.score)
    alignment_length: int = int(stats_result.length)
    matches: int = int(stats_result.matches)
    positives_count: int = int(stats_result.similar)

    gaps: int = query_aligned.count("-") + ref_aligned.count("-")
    mismatches: int = max(0, alignment_length - matches - gaps)

    if alignment_length > 0:
        identity = matches / alignment_length * 100.0
        positives = positives_count / alignment_length * 100.0
    else:
        identity = 0.0
        positives = 0.0

    return AlignmentResult(
        score=score,
        alignment_length=alignment_length,
        matches=matches,
        mismatches=mismatches,
        gaps=gaps,
        identity=identity,
        positives=positives,
        mode=mode,
        query_aligned=query_aligned,
        ref_aligned=ref_aligned,
        midline=midline,
    )


__all__ = [
    "AlignmentMode",
    "AlignmentResult",
    "DEFAULT_GAP_EXTEND",
    "DEFAULT_GAP_OPEN",
    "DEFAULT_MATRIX",
    "DEFAULT_MODE",
    "InvalidSequenceError",
    "SequenceSimilarityError",
    "SimilarityDependencyError",
    "UnknownMatrixError",
    "align_sequences",
]
