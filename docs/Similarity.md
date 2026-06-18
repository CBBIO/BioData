# CBBIO/similarity.py Documentation

## General Description
`CBBIO.similarity` provides pairwise sequence alignment and similarity
statistics powered by [parasail](https://github.com/jeffdaily/parasail-python).
It is intentionally independent from database access code and works directly
with plain string sequences, including those returned by `BioDataClient`
methods such as `get_protein_sequence` and `get_protein_sequences`.

Both local (Smith-Waterman) and global (Needleman-Wunsch) alignment modes are
supported.  The module returns a rich result dataclass covering score,
alignment length, match/mismatch/gap counts, percentage identity, and
percentage positives.

Main dependencies:
- `parasail` for SIMD-accelerated sequence alignment.

## Constants

| Name | Default | Description |
|------|---------|-------------|
| `DEFAULT_MODE` | `"local"` | Default alignment mode (Smith-Waterman). |
| `DEFAULT_GAP_OPEN` | `10` | Default gap-opening penalty. |
| `DEFAULT_GAP_EXTEND` | `1` | Default gap-extension penalty. |
| `DEFAULT_MATRIX` | `"blosum62"` | Default substitution matrix name. |

## Type Alias

### `AlignmentMode`
`Literal["local", "global"]` — selects Smith-Waterman (`"local"`) or
Needleman-Wunsch (`"global"`) alignment.

## Exceptions

- `SequenceSimilarityError`: base module-level exception.
- `SimilarityDependencyError`: raised when `parasail` is not installed.
- `InvalidSequenceError`: raised when either input sequence is empty or not a
  string.
- `UnknownMatrixError`: raised when the requested substitution matrix is not
  available in parasail.

## Dataclass: `AlignmentResult`

Frozen dataclass returned by `align_sequences`.

| Field | Type | Description |
|-------|------|-------------|
| `score` | `int` | Raw alignment score from parasail. |
| `alignment_length` | `int` | Total alignment columns (matches + mismatches + gap columns). |
| `matches` | `int` | Number of identical aligned positions. |
| `mismatches` | `int` | Number of substituted (non-identical, non-gap) positions. |
| `gaps` | `int` | Total gap characters across both aligned sequences. |
| `identity` | `float` | Percentage identity: `matches / alignment_length × 100`. `0.0` when `alignment_length` is zero. |
| `positives` | `float` | Percentage of positions with positive substitution score (matches + conservative substitutions) over `alignment_length`. `0.0` when `alignment_length` is zero. |
| `mode` | `AlignmentMode` | Alignment mode used (`"local"` or `"global"`). |
| `query_aligned` | `str` | First (query) sequence in the alignment; `"-"` marks gap positions. |
| `ref_aligned` | `str` | Second (reference) sequence in the alignment; `"-"` marks gap positions. |
| `midline` | `str` | Per-column comparison: `"\|"` identical, `"."` substitution, `" "` gap. |

Invariant: `matches + mismatches + gaps == alignment_length`.

## Function: `align_sequences`

```python
align_sequences(
    seq1: str,
    seq2: str,
    *,
    mode: AlignmentMode = "local",
    gap_open: int = 10,
    gap_extend: int = 1,
    matrix: str = "blosum62",
) -> AlignmentResult
```

Aligns two sequences and returns pairwise similarity statistics.

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `seq1` | `str` | — | Query sequence (amino-acid or nucleotide). Must be non-empty. |
| `seq2` | `str` | — | Reference sequence. Must be non-empty. |
| `mode` | `AlignmentMode` | `"local"` | `"local"` for Smith-Waterman; `"global"` for Needleman-Wunsch. |
| `gap_open` | `int` | `10` | Gap-opening penalty (positive integer). |
| `gap_extend` | `int` | `1` | Gap-extension penalty (positive integer). |
| `matrix` | `str` | `"blosum62"` | Substitution matrix name available in parasail (e.g. `"blosum50"`, `"pam250"`, `"dnafull"`). |

### Returns
`AlignmentResult` — see dataclass fields above.

### Raises
- `SimilarityDependencyError` if `parasail` is not installed.
- `InvalidSequenceError` if either sequence is empty or not a string.
- `UnknownMatrixError` if *matrix* is not available in parasail.
- `SequenceSimilarityError` for other alignment errors.

## Usage Examples

### Local alignment (default)

```python
from CBBIO import align_sequences

result = align_sequences("ACDEFGHIKLMNPQRSTVWY", "ACDEFHIKLMNPQRSTVWY")
print(f"Score:            {result.score}")
print(f"Alignment length: {result.alignment_length}")
print(f"Matches:          {result.matches}")
print(f"Mismatches:       {result.mismatches}")
print(f"Gaps:             {result.gaps}")
print(f"% Identity:       {result.identity:.1f}%")
print(f"% Positives:      {result.positives:.1f}%")
print(result.query_aligned)
print(result.midline)
print(result.ref_aligned)
```

### Global alignment

```python
from CBBIO import align_sequences

result = align_sequences(seq1, seq2, mode="global")
```

### Custom gap penalties and matrix

```python
result = align_sequences(seq1, seq2, gap_open=5, gap_extend=2, matrix="blosum50")
```

### Reading the alignment strings

`query_aligned` and `ref_aligned` are the two aligned sequences (with `-` for gaps).
`midline` shows `|` for identical positions, `.` for conservative substitutions, and ` ` for gaps:

```
ACDEFGHIKLMNPQRSTVWY
|||||-||||||||||||||
ACDEF-HIKLMNPQRSTVWY
```

### Using with BioDataClient sequences

```python
from CBBIO import connect, align_sequences

with connect() as client:
    seq1 = client.get_protein_sequence("P12345")
    seq2 = client.get_protein_sequence("Q67890")

if seq1 and seq2:
    result = align_sequences(seq1, seq2)
    print(f"Identity: {result.identity:.2f}%")
    print(f"Positives: {result.positives:.2f}%")
    print(f"Score: {result.score}")
```

### Screening a set of proteins for similarity

```python
from CBBIO import connect, align_sequences

with connect() as client:
    sequences = client.get_protein_sequences(["P12345", "Q67890", "A11111"])

query_seq = sequences["P12345"]
for pid, seq in sequences.items():
    if pid == "P12345":
        continue
    result = align_sequences(query_seq, seq)
    print(f"{pid}  identity={result.identity:.1f}%  score={result.score}")
```
