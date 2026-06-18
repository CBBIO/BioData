# CBBIO — Agent Instructions

## What this project is

CBBIO is a Python library for protein language model embeddings, supervised probing tasks, nearest-neighbor search (pgvector + GPU backends), and GO/taxonomy analysis. The package is `CBBIO`; the installable name is `biodata`.

## Style guides

All style rules live in `docs/`. Read the relevant guide before making changes.

| Topic | File |
|---|---|
| Python code | `docs/STYLE.md` |
| Documentation | `docs/STYLE_DOCS.md` |
| Tests | `docs/STYLE_TESTS.md` |
| Versioning and branching | `docs/STYLE_VERSIONING.md` |

---

## Non-negotiable rules (read these even if you skip the guides)

### Code

- **All public functions and methods must have full type annotations** — parameters and return type. No exceptions.
- **Use `X | Y` union syntax**, never `Union[X, Y]`.
- **Mutable dataclass fields always use `field(default_factory=...)`**, never a bare `[]` or `{}` default.
- **No `print()` in library code.** Use `logging` or return values.
- **Optional heavy dependencies (torch, transformers, faiss) must be imported lazily**, inside the function that needs them, with a clear `EmbeddingDependencyError` if missing.
- **Private names get a `_` prefix** and must not appear in `__all__`.

### Imports in examples, tests, and docs

Always import from the public `CBBIO` namespace:

```python
# correct
from CBBIO import Generator, FastaBatcher, EmbeddingWriter, run_embedding_generation

# wrong — leaks internal paths
from CBBIO.embeddings.models.esm2 import Esm2EmbeddingGenerator
from CBBIO.probing.runner import run_task_on_layer
```

The two known exceptions (not re-exported from top-level `CBBIO`):
- `from CBBIO.Taxonomy import normalize_taxonomy_id, compute_taxon_ic_and_lin_maps`
- Internal adapter base classes when writing a new model adapter

### Adding a new embedding model

Every `EmbeddingGenerator` subclass must define all four class attributes:

```python
GENERATOR_CLASS   = "mymodel"                        # used with Generator(model_class=...)
DEFAULT_MODEL_NAME = "org/mymodel-base"              # used when no name= is given
FAMILY_MODELS     = ["org/mymodel-base", "org/mymodel-large"]
SUPPORTED_POOLERS = ("none", "mean")                 # empty tuple = residue-only
```

And must implement:

```python
@classmethod
def from_pretrained(cls, model_name: str, *, device: str = "cpu", **kwargs) -> Self: ...
```

### Tests

- **Test names are sentences**: `test_generate_collects_errors_when_fail_fast_is_false`, not `test_error`.
- **Float comparisons always use `pytest.approx`**.
- **Exception tests always include `match=`**: `pytest.raises(SomeError, match="keyword")`.
- **Optional deps use `pytest.importorskip`**, not try/except.
- **Integration tests live in `tests/test_biodata_integration.py`** and skip gracefully with `pytest.skip` when the DB is unavailable.

### Documentation

- **Never document private methods** (names starting with `_`).
- **Lead every doc section with a working code example**, not a description.
- **Exceptions are always documented as a table at the end** of the doc file.

### Commits and versioning

- **Commit message format**: `feat:`, `fix:`, `chore:`, `docs:`, `test:`, `perf:` prefix, imperative mood, no trailing period.
- **Version lives only in `pyproject.toml`**. Do not hardcode it elsewhere.
- **Tags are annotated**: `git tag -a 0.2.0 -m "Release 0.2.0"`. Never lightweight tags.

---

## Known gotchas

- **ESM-1b**: do not load from HuggingFace `facebook/esm-1b` directly — the HF code uses pre-norm but the weights are post-norm. The `Esm1bEmbeddingGenerator` loads via `torch.hub` to avoid the mismatch.
- **AMPLIFY on CUDA**: defaults to `bfloat16` because xFormers attention does not support `float32`. Do not change this default.
- **ESM-C batching**: some SDK versions reject multi-sequence batched logit calls. Default `batch_size=1` in examples unless you know the installed SDK supports it.
- **`filter_redundant_to_test_mmseqs`**: requires `mmseqs` binary on `PATH`. Do not call it in unit tests.
- **`0.x` versioning**: the project is pre-1.0. MINOR bumps (`0.2.0`, `0.3.0`) may contain breaking changes.

---

## Running checks

```bash
poetry run pytest -q                                      # unit tests
poetry run pytest -q tests/test_biodata_integration.py   # integration tests (needs config_test.yaml)
poetry run pyright                                        # type checking
```
