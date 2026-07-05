# CBBIO GO

`CBBIO.GO` provides GO ontology utilities built on top of `goatools`. It works independently from the database and operates directly on local OBO files and annotation mappings.

```python
from CBBIO import load_go, read_annotations_tsv

onto = load_go("notebooks/data/go-basic.obo")
```

Download a current OBO file from the Gene Ontology website or use the one in this repo (`notebooks/data/go-basic.obo`).

---

## Loading the Ontology

```python
from CBBIO import load_go

onto = load_go(
    "notebooks/data/go-basic.obo",
    load_obsolete=False,   # set True to include deprecated terms
    quiet=True,            # suppress goatools parser output
)

# Total number of terms
print(len(onto.go_ids))

# Check if a term exists
print(onto.has_term("GO:0006355"))   # True

# Fetch term metadata
term = onto.term("GO:0006355")
print(term["name"])       # "regulation of DNA-templated transcription"
print(term["namespace"])  # "biological_process"
print(term["depth"])      # integer depth in DAG
```

---

## Hierarchy Navigation

```python
# All ancestors (root included)
ancestors = onto.ancestors("GO:0006355")

# Direct parents only
parents = onto.direct_parents("GO:0006355")

# All descendants
descendants = onto.descendants("GO:0006355")

# Direct children
children = onto.direct_children("GO:0006355")

# Shared ancestors of two terms
shared = onto.common_ancestors("GO:0006355", "GO:0006351")
```

### Relationship queries

```python
rel = onto.find_relation("GO:0006355", "GO:0003700")
# → "descendant" | "ancestor" | "parent" | "child" | "same" | "non-related"

# Check ancestry
print(onto.is_ancestor("GO:0008150", "GO:0006355"))   # True
print(onto.is_descendant("GO:0006355", "GO:0008150")) # True

# Are they on the same path?
print(onto.are_in_the_same_path("GO:0006355", "GO:0008150"))  # True
```

---

## Information Content and Semantic Similarity

You need annotation data to compute IC. Load annotations from a TSV file (one `entity_id<TAB>go_id` per line):

```python
from CBBIO import load_go, read_annotations_tsv

onto = load_go("notebooks/data/go-basic.obo")
annotations = read_annotations_tsv("annotations.tsv")
# → {"P12345": {"GO:0006355", "GO:0003700"}, ...}

onto.prepare_term_counts(annotations)
```

Now compute IC and similarity:

```python
# Information content (higher = more specific term)
ic = onto.information_content("GO:0006355")
print(f"IC: {ic:.3f}")

# Pairwise semantic similarity
sim = onto.semantic_similarity("GO:0006355", "GO:0006351", method="lin")
print(f"Similarity: {sim:.3f}")
```

Supported similarity methods:

| Method | Description | Requires IC? |
|---|---|---|
| `"resnik"` | IC of the most informative common ancestor | Yes |
| `"lin"` | Normalized Resnik (divides by IC of both terms) | Yes |
| `"schlicker"` | Lin with relevance weighting | Yes |
| `"wang"` | Graph-based, no corpus IC needed | No |

### Group similarity (protein vs. protein)

Compare two GO annotation sets (e.g. the annotations of two proteins):

```python
go_a = {"GO:0006355", "GO:0003700", "GO:0045893"}
go_b = {"GO:0006351", "GO:0008168"}

sim = onto.group_similarity(go_a, go_b, method="lin", aggregate="bma")
print(f"Group similarity: {sim:.3f}")
```

`aggregate="bma"` (Best Match Average) scores each term in one set against the best-matching term in the other set, then averages both directions.

---

## Term Utilities

```python
# Get display names for a set of GO IDs
names = onto.term_names({"GO:0006355", "GO:0003700"})
# → ["regulation of DNA-templated transcription", "DNA binding"]

# Formatted string
label = onto.format_term_names({"GO:0006355", "GO:0003700"})
# → "DNA binding; regulation of DNA-templated transcription"

# Map namespace to category code ("mf" | "bp" | "cc")
cat = onto.category_for_term("GO:0006355")   # "bp"

# Split annotation dict by category
by_cat = onto.split_annotations_by_category(annotations)
# → {"mf": {...}, "bp": {...}, "cc": {...}}

# Remove invalid GO IDs from a list
valid = onto.filter_valid_terms(["GO:9999999", "GO:0006355"])
# → ["GO:0006355"]

# Branch distance between two terms
dist = onto.minimal_branch_length("GO:0006355", "GO:0003700")
```

---

## Full Workflow Example

Compute functional similarity between two proteins based on their GO annotations from the BioData database:

```python
from CBBIO import connect, load_go, read_annotations_tsv

client = connect()
onto   = load_go("notebooks/data/go-basic.obo")

# Fetch all GO IDs in the database (use as corpus for IC)
all_go_ids = client.fetch_protein_go_ids()    # {protein_id: {go_id, ...}}
onto.prepare_term_counts(all_go_ids)

# Fetch annotations for two specific proteins
ann = client.fetch_go_annotations(["P12345", "Q67890"])
go_a = {t.go_id for t in ann.get("P12345", [])}
go_b = {t.go_id for t in ann.get("Q67890", [])}

sim = onto.group_similarity(go_a, go_b, method="lin")
print(f"Functional similarity: {sim:.3f}")
```

---

## Exceptions

| Exception | When raised |
|---|---|
| `GOError` | Base exception for all GO errors |
| `GOTermNotFoundError` | Requested GO term is not in the loaded DAG |
| `GOCountsNotPreparedError` | IC/similarity called before `prepare_term_counts()` |
