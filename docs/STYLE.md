# CBBIO Coding Style Guide

This document is the canonical style reference for all code in the CBBIO package. Follow it for new code and when editing existing code. The goal is a codebase that reads consistently regardless of who wrote it.

---

## Naming

| Thing | Convention | Example |
|---|---|---|
| Function, method, variable | `snake_case` | `load_fasta_inputs`, `layer_index` |
| Class, type alias | `PascalCase` | `EmbeddingGenerator`, `EmbeddingPayload` |
| Module-level constant | `UPPER_SNAKE_CASE` | `DEFAULT_BATCH_SIZE`, `RESIDUE_SOURCE_SPECS` |
| Private / internal | `_leading_underscore` | `_validate_sequence`, `_pending` |
| Callable factory returning an object | `PascalCase` | `Generator(model_class="esm2")`, `Task(name=...)` |

- Never abbreviate unless the abbreviation is universally understood in the domain (`seq`, `id`, `db`, `ic` are fine; `gen`, `emb`, `spec` are not).
- Boolean parameters and properties read as questions: `include_self`, `fail_fast`, `load_obsolete`.
- Collection parameters that hold multiple items are plural: `layer_indices`, `protein_ids`, `records`.

---

## Type Hints

All public functions and methods must be fully annotated (parameters and return type).

```python
# Good
def load_fasta_inputs(path: str | Path, *, id_from: str = "record_id") -> list[GenerationInput]:

# Bad — missing annotations
def load_fasta_inputs(path, id_from="record_id"):
```

- Use PEP 604 `X | Y` union syntax, not `Union[X, Y]`.
- Use `Sequence[T]` (covariant) for input parameters that only need iteration. Reserve `list[T]` for return values that callers may mutate.
- Use `Mapping[K, V]` for input dicts. Use `dict[K, V]` only when the caller needs a mutable dict.
- Use `TypeAlias` for domain-specific string types so they are self-documenting:

```python
from typing import TypeAlias

ProteinID: TypeAlias = str
SplitName: TypeAlias = Literal["train", "val", "test"]
```

- Annotate `None` returns explicitly: `-> None`.
- For factory classmethods, use `Self` (from `typing`):

```python
from typing import Self

@classmethod
def from_pretrained(cls, model_name: str, *, device: str = "cpu") -> Self:
```

---

## Docstrings

Every public class, function, and method needs at minimum a one-line summary docstring. For anything with non-obvious parameters, add an `Args` / `Returns` / `Raises` block (Google style).

```python
def align_sequences(
    seq1: str,
    seq2: str,
    *,
    mode: AlignmentMode = "local",
    gap_open: int = 10,
    gap_extend: int = 1,
    matrix: str = "blosum62",
) -> AlignmentResult:
    """Align two protein sequences and return the alignment result.

    Args:
        seq1: Query sequence (uppercase single-letter amino acid codes).
        seq2: Reference sequence.
        mode: ``"local"`` (Smith-Waterman) or ``"global"`` (Needleman-Wunsch).
        gap_open: Gap-open penalty (positive integer).
        gap_extend: Gap-extend penalty (positive integer).
        matrix: Substitution matrix name (e.g. ``"blosum62"``).

    Returns:
        AlignmentResult with score, identity, positives, and aligned strings.

    Raises:
        SimilarityDependencyError: If the ``parasail`` library is not installed.
        InvalidSequenceError: If either sequence is empty or contains invalid characters.
    """
```

Rules:
- One-line summary ends with a period.
- Do not repeat the function name in the summary ("Returns the alignment…" not "align_sequences returns…").
- Skip `Args` / `Returns` / `Raises` when the signature is already self-documenting (single-param helpers, trivial getters).
- No multi-paragraph docstrings explaining design decisions — those belong in comments or commit messages.

---

## Comments

Add a comment only when the **why** is non-obvious. Never explain what the code does (the names do that).

```python
# Good — explains a non-obvious constraint
# parasail requires upper-case AA letters; lower-case passes silently but gives wrong scores
sequence = sequence.upper()

# Bad — restates the code
# Convert sequence to upper case
sequence = sequence.upper()
```

A comment indicating a workaround should include a reference:

```python
# ESM-1b weights were trained with post-norm but the HF model uses pre-norm,
# causing ~10% accuracy drop. Use torch.hub instead of HF for this model.
```

---

## Imports

Three groups, each separated by a blank line:

1. Standard library
2. Third-party packages
3. Internal CBBIO imports

```python
import os
from pathlib import Path
from typing import Sequence

import numpy as np
import torch

from CBBIO.embeddings import EmbeddingGenerator, EmbeddingRecord
from CBBIO.types import ProteinID
```

- No wildcard imports (`from x import *`).
- Every public module must define `__all__` listing its exported names.
- Lazy imports (inside functions) are acceptable for optional heavy dependencies:

```python
def _import_faiss(allow_missing: bool = False):
    try:
        import faiss
        return faiss
    except ImportError:
        if allow_missing:
            return None
        raise EmbeddingDependencyError("faiss is required. Install with: pip install faiss-gpu")
```

---

## Dataclasses

- `@dataclass(frozen=True)` for immutable value objects (results, specs, metadata).
- Plain `@dataclass` for containers that accumulate state during a run.
- Never use a bare mutable default — always use `field(default_factory=...)`:

```python
from dataclasses import dataclass, field

# Good
@dataclass
class GenerationResult:
    records: list[EmbeddingRecord] = field(default_factory=list)
    errors: list[dict] = field(default_factory=list)

# Bad — mutable default shared across instances
@dataclass
class GenerationResult:
    records: list[EmbeddingRecord] = []
```

- Use `__post_init__` for validation, not for computation:

```python
@dataclass(frozen=True)
class PredictionSpec:
    objective: ObjectiveName
    level: TaskLevel = "protein"

    def __post_init__(self) -> None:
        valid = {"regression", "binary", "multiclass", "multilabel"}
        if self.objective not in valid:
            raise ValueError(f"objective must be one of {valid}, got {self.objective!r}")
```

---

## Exceptions

Each subsystem defines a base exception class. Raise the most specific subclass; never raise the base class directly.

```python
# Base per subsystem
class EmbeddingGenerationError(Exception): ...
class EmbeddingInputError(EmbeddingGenerationError): ...
class EmbeddingDependencyError(EmbeddingGenerationError): ...

# Usage — raise specific, not base
raise EmbeddingInputError(f"Sequence for {record.id!r} contains invalid characters: {bad!r}")
```

Exception messages must be actionable: say what went wrong, and when possible, how to fix it.

```python
# Good
raise EmbeddingDependencyError(
    "torch is required for GPU inference. Install with: pip install torch"
)

# Bad
raise EmbeddingDependencyError("Missing dependency")
```

---

## EmbeddingGenerator Subclasses

Every model adapter must define these class attributes:

| Attribute | Type | Required | Description |
|---|---|---|---|
| `GENERATOR_CLASS` | `str` | Yes | Canonical short name used with `Generator(model_class=...)` |
| `DEFAULT_MODEL_NAME` | `str` | Yes | HuggingFace repo or identifier used when no `name` is given |
| `FAMILY_MODELS` | `list[str]` | Yes | All model names this generator supports |
| `SUPPORTED_POOLERS` | `list[str]` | Yes | Pool method names accepted; empty list means residue-level only |
| `GENERATOR_ALIASES` | `tuple[str, ...]` | No | Alternative names registered in the factory |

Every model adapter must define these factory classmethods:

```python
@classmethod
def from_pretrained(
    cls,
    model_name: str,
    *,
    device: str = "cpu",
    cache_dir: str | Path | None = None,
    **kwargs,
) -> Self:
    """Load model and tokenizer from HuggingFace Hub."""

# Optional — when users may already have a loaded model
@classmethod
def from_model_and_tokenizer(
    cls,
    model: Any,
    tokenizer: Any,
    *,
    device: str = "cpu",
) -> Self:
    """Wrap an already-loaded model and tokenizer."""
```

---

## Dataset Collection Pattern

Every probing dataset must belong to a `DatasetCollection`. Collections expose discovery,
download, and loading through one interface:

```python
from collections.abc import Mapping, Sequence
from pathlib import Path

from CBBIO import (
    DatasetCollection,
    DatasetMetadata,
    DatasetSplitter,
    ProteinDataset,
    ResidueDataset,
)


class ExampleCollection(DatasetCollection):
    id = "example"
    display_name = "Example datasets"

    def list_datasets(self) -> list[DatasetMetadata]:
        """Return datasets published by this collection."""

    def download(
        self,
        root: str | Path,
        *,
        name: str,
        force: bool = False,
    ) -> list[Path]:
        """Download one dataset and return its local paths."""

    def load(
        self,
        root: str | Path,
        *,
        name: str,
        split: str | Sequence[str] | None = None,
        target: str | None = None,
        download: bool = False,
        max_examples_per_split: Mapping[str, int | None] | None = None,
        splitter: DatasetSplitter | None = None,
    ) -> ProteinDataset | ResidueDataset:
        """Load one dataset into the canonical representation."""
```

The collection owns each `DatasetMetadata`. The catalog indexes that metadata and delegates
`load_dataset()` and `download_dataset()` back to the owning collection. Do not duplicate dataset
metadata in the catalog.

New datasets must be reachable through their collection. Add source-specific functions only when
the provider needs options that do not fit `load_dataset()` or `download_dataset()`.

---

## Private vs. Public

- Functions and methods meant only for internal use get a `_` prefix. Do not export them from `__all__`.
- A function that is public API but heavy / slow should be clearly documented rather than hidden.
- If a helper is used only within one file, define it at module level with `_` prefix. If it is shared across files in a package, put it in the package's `utils.py` or `_utils.py`.

---

## General

- Maximum line length: 100 characters.
- Use `f-strings` for string interpolation (not `%` or `.format()`).
- Prefer `pathlib.Path` over string paths for file system operations.
- Prefer keyword-only arguments (`*` separator) for functions with three or more parameters, especially when several have the same type.
- Do not use `print()` in library code. Use `logging` or pass output back to the caller. Silent is better than noisy; noisy is better than wrong.
- Never silence exceptions with a bare `except: pass`. Catch the specific exception and either re-raise, log, or return a documented sentinel.
