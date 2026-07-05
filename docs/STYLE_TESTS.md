# CBBIO Test Style Guide

This document describes how to write tests for CBBIO. Follow it for new tests and when editing existing ones. The test suite uses `pytest`.

---

## Principles

**Tests are the first documentation of edge cases.** A test name like `test_generate_collects_errors_when_fail_fast_is_false` tells a reader more than any comment ever could. Invest in names.

**One behavior per test.** Each test verifies one thing. If a test needs to verify a precondition before reaching its real assertion, split it into two tests or use a fixture.

**Tests must not depend on order.** Every test must be able to run in isolation. No shared mutable state between tests.

**No real network or filesystem calls in unit tests.** Use `tmp_path`, in-memory data, or stub fixtures. Reserve actual I/O for integration tests.

---

## Test Names

Names follow the pattern `test_<subject>_<scenario>` or `test_<subject>_<expected_result>`. The name must read as a plain English sentence if you replace `_` with spaces.

```python
# Good — reads as a sentence
def test_generate_collects_errors_when_fail_fast_is_false(): ...
def test_align_identical_sequences_perfect_identity(): ...
def test_protein_dataset_raises_on_duplicate_id(): ...

# Bad — vague, no scenario
def test_generate(): ...
def test_error(): ...
def test_dataset_1(): ...
```

For tests that verify a specific exception is raised, name them `test_<subject>_raises_<exception_type>_when_<condition>`:

```python
def test_go_term_raises_not_found_when_id_is_missing(): ...
def test_probe_spec_raises_on_negative_epochs(): ...
```

---

## File Layout

One test file per source module:

| Source module | Test file |
|---|---|
| `CBBIO/embeddings/` | `tests/test_embeddings.py` |
| `CBBIO/probing/` | `tests/test_probing.py` |
| `CBBIO/GO.py` | `tests/test_go.py` |
| `CBBIO/Taxonomy.py` | `tests/test_taxonomy.py` |
| `CBBIO/similarity.py` | `tests/test_similarity.py` |
| `CBBIO/BioData.py` (unit) | `tests/test_biodata.py` |
| `CBBIO/BioData.py` (integration) | `tests/test_biodata_integration.py` |
| Model-specific generators | `tests/test_embeddings_<model>.py` |

Within a file, group related tests with a section comment:

```python
# ---------------------------------------------------------------------------
# Fixtures and helpers
# ---------------------------------------------------------------------------

# ... fixture definitions ...

# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------

def test_align_identical_sequences_perfect_identity(): ...

# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------

def test_align_raises_on_empty_sequence(): ...
```

---

## Fixtures

Use `pytest.fixture` for test data that is reused across several tests. Define fixtures at the top of the file (or in `conftest.py` if shared across files).

```python
import pytest
from pathlib import Path

OBO_CONTENT = """format-version: 1.2
[Term]
id: GO:0000001
name: root process
namespace: biological_process
"""

@pytest.fixture()
def go_obo_path(tmp_path: Path) -> Path:
    path = tmp_path / "mini-go.obo"
    path.write_text(OBO_CONTENT, encoding="utf-8")
    return path
```

Rules for fixtures:

- **Prefer the smallest fixture that makes the test pass.** Do not load a 1000-term OBO file if 4 terms are enough.
- **Use `tmp_path`** (pytest's built-in) for all temporary files. Never write to a hardcoded path.
- **Prefer function scope** (the default) unless the fixture is expensive and truly safe to share — then use `scope="module"`.
- **Name fixtures after what they provide**, not what they do: `go_obo_path`, not `create_go_file`.
- **Module-level constants for inline data** (strings, small dicts) are fine and readable. Use `UPPER_SNAKE_CASE` and prefix them with `_` to signal test-only:

```python
_SEQ_IDENTICAL    = "ACDEFGHIKLMNPQRSTVWY"
_SEQ_ONE_MISMATCH = "ACDEFXHIKLMNPQRSTVWY"  # G → X at position 5
_SEQ_ONE_GAP      = "ACDEFHIKLMNPQRSTVWY"   # G deleted at position 5
```

---

## Test Helpers (Private Stub Classes)

When the code under test requires an adapter or collaborator, define a minimal stub. Name stubs with a leading underscore and a `_` prefix to signal they are test-internal.

```python
class _Preprocessor(PreprocessorAdapter):
    def preprocess(self, raw_sequence: str) -> str:
        return raw_sequence.strip().upper()


class _Tokenizer(TokenizerAdapter):
    def tokenize(self, sequence: str) -> Any:
        return sequence


class _Model(ModelAdapter):
    def infer(self, tokens: Any, *, layer_index: int = 0) -> Any:
        if tokens == "FAIL":
            raise RuntimeError("model exploded")
        return [len(str(tokens)), layer_index]
```

Use a private factory function when creating a standard object requires several lines:

```python
def _generator(*, model: ModelAdapter | None = None) -> EmbeddingGenerator:
    return EmbeddingGenerator(
        model_reference="test/model",
        preprocessor=_Preprocessor(),
        tokenizer=_Tokenizer(),
        model=model or _Model(),
        postprocessor=_Postprocessor(),
    )
```

---

## Assertions

**Assert the minimum needed to prove the behavior.** Do not assert every field of a result when only one matters.

```python
# Good — proves the failing record was collected, not every field
assert result.errors[0]["id"] == "bad"
assert result.errors[0]["error_type"] == "EmbeddingInputError"

# Bad — overly broad, will break on unrelated changes
assert result == expected_result
```

Use `pytest.approx` for float comparisons:

```python
assert result.identity == pytest.approx(95.0)
assert result.identity == pytest.approx(0.743, abs=1e-3)
```

Use `pytest.raises` as a context manager for exception tests. Always assert something about the exception — at minimum the type, ideally a substring of the message:

```python
def test_probe_spec_raises_on_negative_epochs() -> None:
    with pytest.raises(EmbeddingInputError, match="epochs"):
        ProbeSpec(epochs=-1)
```

Do not use `assert ... is True` or `assert ... is False` — use bare `assert` and `assert not`:

```python
assert go.has_term("GO:0000001")         # good
assert not go.has_term("GO:9999999")     # good
assert go.has_term("GO:0000001") is True # bad
```

---

## Optional Dependencies

Use `pytest.importorskip` at module scope to skip an entire test file when a heavy optional dependency (torch, transformers, etc.) is not installed:

```python
torch = pytest.importorskip("torch")
```

For tests that require an optional dependency only within the test, use `pytest.importorskip` inside the test function:

```python
def test_something_with_faiss() -> None:
    faiss = pytest.importorskip("faiss")
    ...
```

Do not use `try/except ImportError` in tests — `importorskip` produces a cleaner skip message.

---

## Integration Tests

Integration tests require an external resource (database, network, GPU). They live in a dedicated file (`test_biodata_integration.py`) and must be skippable with a clear human-readable message.

```python
_RUN_HINT = (
    "Integration tests are skipped. To run them:\n"
    "1) Copy tests/assets/config_test.yaml.example to config_test.yaml and edit credentials.\n"
    "2) Ensure PostgreSQL is running and reachable.\n"
    "3) Run: poetry run pytest -q tests/test_biodata_integration.py"
)

@pytest.fixture(scope="module")
def integration_client():
    config_path = Path(os.getenv("BIODATA_TEST_CONFIG", "config_test.yaml"))
    if not config_path.exists():
        pytest.skip(f"{_RUN_HINT}\nMissing config file: {config_path}")
    ...
```

Rules for integration tests:

- Always use `pytest.skip` (not `pytest.xfail`) when the environment is unavailable.
- Use `scope="module"` for expensive setup fixtures (database connections).
- Call `health_check()` at fixture setup; skip if required tables are missing.
- Do not add integration assertions to unit test files — keep the two completely separate.

---

## What Not to Test

- **Private functions and methods** (`_` prefix). Test them indirectly through public APIs.
- **Third-party library behavior.** If goatools parses OBO correctly, that is goatools' test, not ours. Test how we use it.
- **Every permutation of arguments.** Cover the meaningful boundaries: happy path, missing input, invalid input, edge case (empty list, single item).
- **Implementation details.** If a refactor changes how a result is computed but not what it produces, no test should break.

---

## Running Tests

```bash
# All unit tests
poetry run pytest -q

# One file
poetry run pytest -q tests/test_go.py

# One test by name pattern
poetry run pytest -q -k "test_go_ancestors"

# Integration tests (requires config_test.yaml)
poetry run pytest -q tests/test_biodata_integration.py

# With coverage
poetry run pytest --cov=CBBIO --cov-report=term-missing -q
```

---

## Checklist Before Committing Tests

- [ ] Every test name reads as a plain English sentence with `_` replaced by spaces.
- [ ] No test depends on another test's side effects.
- [ ] No hardcoded file paths — use `tmp_path`.
- [ ] Float comparisons use `pytest.approx`.
- [ ] Exception tests use `pytest.raises` with a `match=` assertion.
- [ ] Optional dependencies use `pytest.importorskip`, not try/except.
- [ ] Integration tests are in a separate file and skip gracefully without a DB.
- [ ] No private methods or internal behavior is tested directly.
