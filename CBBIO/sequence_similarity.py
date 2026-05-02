"""Pairwise sequence similarity using parasail SIMD-accelerated alignments.

This module is intentionally independent from database access code and works
directly with plain amino-acid / nucleotide strings as well as with
:class:`~CBBIO.embeddings.GenerationInput` objects.

Supported algorithms
--------------------
* ``"local"``    – Smith-Waterman local alignment (default).
* ``"global"``   – Needleman-Wunsch global alignment.
* ``"semiglobal"`` – Semi-global (end-gap free) alignment.

Similarity normalisation
------------------------
The raw alignment score is normalised by the geometric mean of each
sequence's self-alignment score, yielding a value in **[0, 1]**:

    similarity = score(a, b) / sqrt(score(a, a) * score(b, b))

Negative raw scores (possible with the global algorithm and mismatched
sequences) are clamped to 0 before normalisation so the result is always
a non-negative float.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .types import AlignmentAlgorithm, ScoringMatrix


class SequenceSimilarityError(Exception):
    """Base exception for ``CBBIO.sequence_similarity``."""


class SequenceSimilarityDependencyError(SequenceSimilarityError):
    """Raised when the ``parasail`` library is not installed."""


class SequenceSimilarityInputError(SequenceSimilarityError):
    """Raised when input sequences or parameters are invalid."""


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _require_parasail() -> Any:
    """Import and return the :mod:`parasail` module or raise a descriptive error."""
    try:
        import parasail  # type: ignore
        return parasail
    except ModuleNotFoundError as exc:
        raise SequenceSimilarityDependencyError(
            "Missing dependency 'parasail'. Install with: pip install parasail"
        ) from exc


def _resolve_matrix(parasail: Any, matrix_name: str) -> Any:
    """Return the parasail matrix object for *matrix_name*."""
    name = matrix_name.strip().lower()
    matrix = getattr(parasail, name, None)
    if matrix is None:
        raise SequenceSimilarityInputError(
            f"Unknown scoring matrix '{matrix_name}'. "
            "Examples: 'blosum62', 'pam250', 'dnafull'."
        )
    return matrix


def _align(
    parasail: Any,
    seq_a: str,
    seq_b: str,
    *,
    algorithm: AlignmentAlgorithm,
    matrix: Any,
    gap_open: int,
    gap_extend: int,
) -> int:
    """Run the requested alignment and return the raw integer score."""
    alg = algorithm.strip().lower()
    if alg == "local":
        result = parasail.sw_striped_32(seq_a, seq_b, gap_open, gap_extend, matrix)
    elif alg == "global":
        result = parasail.nw_striped_32(seq_a, seq_b, gap_open, gap_extend, matrix)
    elif alg == "semiglobal":
        result = parasail.sg_striped_32(seq_a, seq_b, gap_open, gap_extend, matrix)
    else:
        raise SequenceSimilarityInputError(
            f"Unknown alignment algorithm '{algorithm}'. "
            "Use: 'local', 'global', or 'semiglobal'."
        )
    return int(result.score)


def _self_score(
    parasail: Any,
    seq: str,
    *,
    algorithm: AlignmentAlgorithm,
    matrix: Any,
    gap_open: int,
    gap_extend: int,
) -> int:
    """Return the self-alignment score for a single sequence."""
    return _align(
        parasail,
        seq,
        seq,
        algorithm=algorithm,
        matrix=matrix,
        gap_open=gap_open,
        gap_extend=gap_extend,
    )


def _normalise(score: int, self_a: int, self_b: int) -> float:
    """Return normalised similarity in [0, 1]."""
    if self_a <= 0 or self_b <= 0:
        return 0.0
    raw = max(score, 0)
    return raw / math.sqrt(float(self_a) * float(self_b))


def _extract_sequence(seq_or_input: object) -> str:
    """Return the sequence string from either a plain string or a GenerationInput."""
    if isinstance(seq_or_input, str):
        return seq_or_input
    seq_attr = getattr(seq_or_input, "sequence", None)
    if isinstance(seq_attr, str):
        return seq_attr
    raise SequenceSimilarityInputError(
        f"Cannot extract a sequence string from {type(seq_or_input).__name__}. "
        "Provide a str or a GenerationInput-compatible object with a 'sequence' attribute."
    )


# ---------------------------------------------------------------------------
# Public dataclass for individual results
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SimilarityResult:
    """Pairwise similarity result between two sequences.

    Attributes
    ----------
    id_a:
        Identifier of the first sequence (``None`` when not provided).
    id_b:
        Identifier of the second sequence (``None`` when not provided).
    score:
        Raw alignment score.
    similarity:
        Normalised similarity in **[0, 1]**.
    algorithm:
        Alignment algorithm used.
    matrix:
        Name of the scoring matrix used.
    """

    id_a: Optional[str]
    id_b: Optional[str]
    score: int
    similarity: float
    algorithm: AlignmentAlgorithm
    matrix: ScoringMatrix


# ---------------------------------------------------------------------------
# Core public functions
# ---------------------------------------------------------------------------

DEFAULT_ALGORITHM: AlignmentAlgorithm = "local"
DEFAULT_MATRIX: ScoringMatrix = "blosum62"
DEFAULT_GAP_OPEN: int = 11
DEFAULT_GAP_EXTEND: int = 1


def compute_similarity(
    seq_a: object,
    seq_b: object,
    *,
    algorithm: AlignmentAlgorithm = DEFAULT_ALGORITHM,
    matrix: ScoringMatrix = DEFAULT_MATRIX,
    gap_open: int = DEFAULT_GAP_OPEN,
    gap_extend: int = DEFAULT_GAP_EXTEND,
) -> SimilarityResult:
    """Compute the pairwise similarity between two sequences.

    Parameters
    ----------
    seq_a, seq_b:
        Either plain amino-acid / nucleotide strings or objects that expose a
        ``sequence`` attribute (e.g. :class:`~CBBIO.embeddings.GenerationInput`).
    algorithm:
        Alignment algorithm to use: ``"local"`` (Smith-Waterman, default),
        ``"global"`` (Needleman-Wunsch), or ``"semiglobal"``.
    matrix:
        Name of the parasail scoring matrix, e.g. ``"blosum62"`` (default)
        or ``"pam250"``.
    gap_open:
        Gap-open penalty (positive integer, default 11).
    gap_extend:
        Gap-extension penalty (positive integer, default 1).

    Returns
    -------
    SimilarityResult
        Dataclass containing the raw score and the normalised similarity.

    Raises
    ------
    SequenceSimilarityDependencyError
        If :mod:`parasail` is not installed.
    SequenceSimilarityInputError
        If input sequences or parameters are invalid.
    """
    parasail_mod = _require_parasail()

    str_a = _extract_sequence(seq_a)
    str_b = _extract_sequence(seq_b)

    if not str_a:
        raise SequenceSimilarityInputError("seq_a must be a non-empty sequence string.")
    if not str_b:
        raise SequenceSimilarityInputError("seq_b must be a non-empty sequence string.")

    _validate_penalties(gap_open, gap_extend)

    matrix_obj = _resolve_matrix(parasail_mod, matrix)

    raw_score = _align(
        parasail_mod,
        str_a,
        str_b,
        algorithm=algorithm,
        matrix=matrix_obj,
        gap_open=gap_open,
        gap_extend=gap_extend,
    )

    self_a = _self_score(
        parasail_mod,
        str_a,
        algorithm=algorithm,
        matrix=matrix_obj,
        gap_open=gap_open,
        gap_extend=gap_extend,
    )
    self_b = _self_score(
        parasail_mod,
        str_b,
        algorithm=algorithm,
        matrix=matrix_obj,
        gap_open=gap_open,
        gap_extend=gap_extend,
    )

    sim = _normalise(raw_score, self_a, self_b)

    id_a: Optional[str] = _extract_id(seq_a)
    id_b: Optional[str] = _extract_id(seq_b)

    return SimilarityResult(
        id_a=id_a,
        id_b=id_b,
        score=raw_score,
        similarity=sim,
        algorithm=algorithm,
        matrix=matrix,
    )


def pairwise_similarity_matrix(
    sequences: Sequence[object],
    *,
    algorithm: AlignmentAlgorithm = DEFAULT_ALGORITHM,
    matrix: ScoringMatrix = DEFAULT_MATRIX,
    gap_open: int = DEFAULT_GAP_OPEN,
    gap_extend: int = DEFAULT_GAP_EXTEND,
) -> List[List[SimilarityResult]]:
    """Compute all pairwise similarities for a collection of sequences.

    Parameters
    ----------
    sequences:
        A list of plain strings or :class:`~CBBIO.embeddings.GenerationInput`-
        compatible objects.  Must contain at least one entry.
    algorithm, matrix, gap_open, gap_extend:
        Same as :func:`compute_similarity`.

    Returns
    -------
    list[list[SimilarityResult]]
        A symmetric *N × N* matrix of :class:`SimilarityResult` objects.
        ``result[i][j]`` is the similarity between ``sequences[i]`` and
        ``sequences[j]``.  The diagonal entries have ``similarity == 1.0``.

    Raises
    ------
    SequenceSimilarityInputError
        If *sequences* is empty or any entry is invalid.
    SequenceSimilarityDependencyError
        If :mod:`parasail` is not installed.
    """
    if not sequences:
        raise SequenceSimilarityInputError("sequences must be a non-empty list.")

    parasail_mod = _require_parasail()
    _validate_penalties(gap_open, gap_extend)
    matrix_obj = _resolve_matrix(parasail_mod, matrix)

    # Pre-extract strings and IDs once
    str_seqs: List[str] = []
    ids: List[Optional[str]] = []
    for i, seq in enumerate(sequences):
        s = _extract_sequence(seq)
        if not s:
            raise SequenceSimilarityInputError(f"sequences[{i}] is an empty string.")
        str_seqs.append(s)
        ids.append(_extract_id(seq))

    n = len(str_seqs)

    # Pre-compute self-scores to avoid duplicating work on the diagonal
    self_scores: List[int] = [
        _self_score(
            parasail_mod,
            s,
            algorithm=algorithm,
            matrix=matrix_obj,
            gap_open=gap_open,
            gap_extend=gap_extend,
        )
        for s in str_seqs
    ]

    # Build upper-triangle, then mirror
    raw_scores: List[List[int]] = [[0] * n for _ in range(n)]
    for i in range(n):
        raw_scores[i][i] = self_scores[i]
        for j in range(i + 1, n):
            s = _align(
                parasail_mod,
                str_seqs[i],
                str_seqs[j],
                algorithm=algorithm,
                matrix=matrix_obj,
                gap_open=gap_open,
                gap_extend=gap_extend,
            )
            raw_scores[i][j] = s
            raw_scores[j][i] = s

    result: List[List[SimilarityResult]] = []
    for i in range(n):
        row: List[SimilarityResult] = []
        for j in range(n):
            sim = _normalise(raw_scores[i][j], self_scores[i], self_scores[j])
            row.append(
                SimilarityResult(
                    id_a=ids[i],
                    id_b=ids[j],
                    score=raw_scores[i][j],
                    similarity=sim,
                    algorithm=algorithm,
                    matrix=matrix,
                )
            )
        result.append(row)

    return result


def similarity_to_dict(
    result_matrix: List[List[SimilarityResult]],
) -> Dict[Tuple[Optional[str], Optional[str]], float]:
    """Flatten a similarity matrix into a ``{(id_a, id_b): similarity}`` mapping.

    Useful when the sequences have meaningful string IDs (e.g. protein
    accessions coming from :class:`~CBBIO.embeddings.GenerationInput` objects).
    Pairs where either ID is ``None`` are still included.

    Parameters
    ----------
    result_matrix:
        Output of :func:`pairwise_similarity_matrix`.

    Returns
    -------
    dict
        Mapping of ``(id_a, id_b)`` tuples to normalised similarity floats.
    """
    out: Dict[Tuple[Optional[str], Optional[str]], float] = {}
    for row in result_matrix:
        for cell in row:
            out[(cell.id_a, cell.id_b)] = cell.similarity
    return out


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _extract_id(seq_or_input: object) -> Optional[str]:
    """Return the ID attribute if present, otherwise ``None``."""
    if isinstance(seq_or_input, str):
        return None
    id_attr = getattr(seq_or_input, "id", None)
    if id_attr is None:
        return None
    return str(id_attr)


def _validate_penalties(gap_open: int, gap_extend: int) -> None:
    if gap_open < 0:
        raise SequenceSimilarityInputError(
            f"gap_open must be a non-negative integer, got {gap_open}."
        )
    if gap_extend < 0:
        raise SequenceSimilarityInputError(
            f"gap_extend must be a non-negative integer, got {gap_extend}."
        )


__all__ = [
    "DEFAULT_ALGORITHM",
    "DEFAULT_GAP_EXTEND",
    "DEFAULT_GAP_OPEN",
    "DEFAULT_MATRIX",
    "SequenceSimilarityDependencyError",
    "SequenceSimilarityError",
    "SequenceSimilarityInputError",
    "SimilarityResult",
    "compute_similarity",
    "pairwise_similarity_matrix",
    "similarity_to_dict",
]
