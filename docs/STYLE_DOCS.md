# CBBIO Documentation Style Guide

This document describes how to write and maintain documentation for CBBIO. Follow it when writing new docs or expanding existing ones.

---

## Principles

**Be practical first.** Every document should answer "how do I use this?" before it answers "what does this do?". Lead with a working example, not a description.

**One concept per sentence.** If you need a comma to separate two ideas in a sentence, consider splitting it into two sentences or a list.

**Write for someone reading in a hurry.** They will scan headers first, read examples second, and read prose last. Structure your writing accordingly.

**Don't document the obvious.** If a parameter is named `path` and has type `str | Path`, you do not need to explain it. Only document what is non-obvious.

---

## Document Structure

Every module or subsystem doc follows this order:

1. **One-line description** — what this module does, in one sentence.
2. **Minimal working example** — the simplest possible code that demonstrates the module. No setup boilerplate beyond necessary imports.
3. **Concept section** (when needed) — brief explanation of any non-obvious domain concepts, as a table or short prose. Skip this if the API is self-explanatory.
4. **How-to sections** — one section per major workflow, named after what the user wants to accomplish ("Loading the Ontology", "Nearest-Neighbor Search", "Running a Probing Task").
5. **Reference table(s)** — compact parameter or field tables, linked from how-to sections.
6. **Exceptions** — always last, as a table.

Do not add an "Overview" section that restates the one-line description. Do not add a "See also" section unless it contains a specific, non-obvious link.

---

## Headings

Use ATX-style headings (`#`, `##`, `###`). Never skip a level.

- `#` — document title (one per file)
- `##` — major section (workflow, concept group)
- `###` — sub-section within a workflow

Keep heading text short. It is a label, not a sentence. No trailing punctuation.

```markdown
# CBBIO GO                         ← good
## Information Content             ← good
### Full Workflow Example           ← good

# CBBIO GO Ontology Module         ← redundant ("module")
## How to Compute Information Content  ← too long
```

---

## Code Examples

Every code block must be runnable. Use imports that resolve from the public `CBBIO` namespace:

```python
# Good — uses public API
from CBBIO import load_go, align_sequences

# Bad — leaks internal paths
from CBBIO.GO import GOOntology
from CBBIO.similarity import align_sequences
```

Exceptions: when a name is not re-exported from the top-level `CBBIO` (check `CBBIO/__init__.py`), import from the submodule and note it:

```python
# normalize_taxonomy_id is not in the top-level namespace
from CBBIO.Taxonomy import normalize_taxonomy_id
```

### Code block rules

- Always specify the language: ` ```python `, ` ```bash `, ` ```yaml `.
- Show realistic identifiers and values, not placeholder names like `foo` or `bar`.
- Show the output of a call as a comment when it adds meaning:

```python
node = tax.taxon("9606")
print(node["name"])   # "Homo sapiens"
print(node["rank"])   # "species"
```

- Keep examples as short as possible. Stop after demonstrating the point — do not add error handling, logging, or validation unless that is the point being demonstrated.
- One example per concept. Do not chain multiple unrelated calls into one block.

### When to show output

Show output when:
- The return type is not obvious from the signature
- The shape or structure of the return value matters
- A numeric result helps calibrate expectations

```python
sim = onto.semantic_similarity("GO:0006355", "GO:0006351", method="lin")
print(f"Similarity: {sim:.3f}")   # 0.743
```

Do not show output when it is obvious (a list of strings, a bool, `None`).

---

## Tables

Use a table when you have three or more items with two or more attributes. Use a bullet list when items have only one attribute. Do not use a table for two items.

```markdown
# Good — table for parameters with multiple attributes
| Parameter | Default | Description |
|---|---|---|
| `kind` | `"linear"` | `"linear"` or `"mlp"` |
| `epochs` | `100` | Training epochs |

# Good — list for a simple enumeration
Supported methods:
- `"resnik"` — IC of the most informative common ancestor
- `"lin"` — normalized Resnik

# Bad — table with one meaningful column
| Method |
|---|
| resnik |
| lin |
```

Column headers: capitalize the first word only, no trailing punctuation.

Always include a header row. Always use the compact `|---|` separator (no extra dashes).

---

## Prose Style

**Active voice.** "Returns a list" not "A list is returned".

**Present tense.** "The method raises `NotFoundError`" not "The method will raise".

**Second person.** "You need annotation data" not "The user needs annotation data".

**No filler phrases.** Avoid "Note that", "Keep in mind", "It is worth mentioning", "As mentioned above". Say the thing directly.

**No apologetic or tentative language.** Avoid "simply", "just", "easily", "it is easy to", "straightforward".

```markdown
# Good
Pass `use_ann=True` to enable approximate nearest neighbor indexing.

# Bad
It is worth noting that you can simply pass `use_ann=True` to easily enable approximate nearest neighbor indexing.
```

---

## Inline Code

Use backtick code formatting for:
- Function and method names: `align_sequences()`
- Parameter names: `layer_index`
- Class names: `EmbeddingRecord`
- Literal values: `"cosine"`, `True`, `None`
- File paths: `config.yaml`
- Exception names: `NotFoundError`

Do not use code formatting for general concepts or product names ("the BioData database", "HuggingFace", "pgvector").

---

## Parameters and Fields

Document parameters as a table when there are three or more. Include the default value column only when defaults exist and vary meaningfully.

```markdown
| Field | Type | Description |
|---|---|---|
| `id` | `str` | Unique protein identifier |
| `embedding` | `Sequence[float]` or `Sequence[Sequence[float]]` | A 1-D vector (pooled) or 2-D matrix (per-residue) |
| `layer_index` | `int` | Which model layer this came from |
```

Do not document every field of every dataclass. Focus on the ones users will access. If all fields are self-explanatory, a one-line summary suffices.

---

## Exceptions Section

Always document exceptions as a table at the end of the document:

```markdown
## Exceptions

| Exception | When raised |
|---|---|
| `GOError` | Base exception for all GO errors |
| `GOTermNotFoundError` | Requested GO term is not in the loaded DAG |
| `GOCountsNotPreparedError` | IC/similarity called before `prepare_term_counts()` |
```

"When raised" should be one short phrase, not a sentence. Do not include internal-only exceptions.

---

## What Not to Document

- **Private methods and functions** (names starting with `_`). These are implementation details.
- **Internal module structure** (`_merge_dicts`, `_cursor`, `_as_optional_str` etc.). Document the behavior, not the mechanism.
- **Behavior the code already expresses clearly.** If a function is called `list_residue_sources()` and returns a list of residue source names, you do not need to explain what it does.
- **Implementation notes** ("This delegates to `search/service.py`"). Mention architecture only when it affects how users call the code.
- **Future plans or TODOs.** If it is not shipped, it is not documented.

---

## File Naming and Location

All documentation lives under `docs/`. One file per module or subsystem.

| File | Covers |
|---|---|
| `AgentQuickStart.md` | Fast orientation for coding agents |
| `AgentWorkflows.md` | Common coding-agent task recipes |
| `AgentApiMap.md` | Public API map for coding agents |
| `AgentExamplePrompts.md` | Example prompts for coding agents |
| `examples/` | Code examples paired with agent prompts |
| `GettingStarted.md` | First-use tutorial for all modules |
| `Embeddings.md` | Embedding generation |
| `Probing.md` | Probing tasks and benchmarks |
| `BioData.md` | Database client and search |
| `GO.md` | GO ontology |
| `Taxonomy.md` | Taxonomy |
| `Similarity.md` | Sequence alignment |
| `STYLE.md` | Python coding style |
| `STYLE_DOCS.md` | Documentation style (this file) |

Do not create sub-directories under `docs/` unless the module itself is large enough to warrant a multi-file guide (e.g. a dedicated `embeddings/` folder with separate files for models, I/O, and pooling).

---

## Checklist Before Committing a Doc Change

- [ ] Every code example has explicit imports.
- [ ] All imports resolve from the public `CBBIO` namespace (or the submodule path is noted).
- [ ] No private methods are documented.
- [ ] Every section header is a label, not a sentence.
- [ ] Tables have a header row and compact `|---|` separators.
- [ ] No filler phrases ("Note that", "simply", "just").
- [ ] Exceptions are documented as a table at the end.
