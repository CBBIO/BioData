# CBBIO Taxonomy

`CBBIO.Taxonomy` provides taxonomy utilities for NCBI taxdump files. It works independently from the database.

Download the taxdump from NCBI: https://ftp.ncbi.nlm.nih.gov/pub/taxonomy/taxdump.tar.gz

```python
from CBBIO import load_taxonomy

tax = load_taxonomy("/path/to/taxdump/")
```

The directory must contain `nodes.dmp` and `names.dmp` (the standard NCBI taxdump files).

---

## Basic Term Lookup

```python
# Check if a taxon exists
print(tax.has_taxon("9606"))   # True

# Fetch metadata
node = tax.taxon("9606")
print(node["name"])      # "Homo sapiens"
print(node["rank"])      # "species"
print(node["depth"])     # integer depth in the tree
print(node["parent_id"]) # "9605"

# Total taxon count
print(len(tax.taxon_ids))
```

---

## Lineage Navigation

```python
# Full lineage from root to the taxon (ordered root → leaf)
lineage = tax.lineage("9606", include_self=True, include_root=True)
# → ["1", "131567", "2759", "33154", ..., "9606"]

# Direct ancestors (all nodes on the path to root)
ancestors = tax.ancestors("9606")

# Direct children
children = tax.children("9606")

# All descendants (deep)
descendants = tax.descendants("9606")

# Parent
parent_id = tax.parent("9606")   # "9605"

# Siblings
siblings = tax.siblings("9606")  # other species in the same genus
```

---

## Lowest Common Ancestor

```python
# LCA taxon ID
lca_id = tax.lowest_common_ancestor("9606", "10090")  # human vs mouse
# → "314146"  (Euarchontoglires)

# LCA rank
rank = tax.lowest_common_ancestor_rank("9606", "10090")
# → "superorder"

# Full LCA metadata
clade = tax.lowest_common_ancestor_clade("9606", "10090")
print(clade["name"], clade["rank"], clade["depth"])

# Check if two taxa share a clade at a given rank
print(tax.shares_clade_at_rank("9606", "10090", "order"))  # True (both Primates + Rodentia → no)
print(tax.shares_clade_at_rank("9606", "9598",  "family")) # True (both Hominidae)
```

---

## Similarity Measures

### Wu-Palmer similarity (depth-based, no corpus needed)

```python
sim = tax.wu_palmer_similarity("9606", "10090")
# 2 * depth(LCA) / (depth(a) + depth(b))
print(f"Wu-Palmer: {sim:.3f}")

# Normalized LCA depth (similar, alternative formulation)
score = tax.normalized_lca_depth("9606", "10090")
```

### Lin similarity (requires annotation corpus)

Lin similarity uses information content: `2 * IC(LCA) / (IC(a) + IC(b))`.

```python
from CBBIO import load_taxonomy, read_taxonomy_annotations_tsv

tax = load_taxonomy("/path/to/taxdump/")

# Load protein→taxonomy annotations
annotations = read_taxonomy_annotations_tsv("protein_taxa.tsv")
# → {"P12345": {"9606"}, "Q67890": {"10090"}, ...}

tax.prepare_taxon_counts(annotations, mode="observed")

sim = tax.lin_similarity("9606", "10090", mode="observed")
print(f"Lin similarity: {sim:.3f}")
```

IC modes:

| Mode | Description |
|---|---|
| `"observed"` | Based on annotation frequencies in the provided corpus |
| `"subtree"` | Based on subtree size (no corpus needed) |
| `"whole_db"` | Reads taxonomy assignments from the BioData database |

### Batch similarity computation

```python
from CBBIO.Taxonomy import compute_taxon_ic_and_lin_maps

ic_map, lin_map = compute_taxon_ic_and_lin_maps(
    tax,
    query_tax_ids={"9606", "10090"},
    subject_tax_ids={"9606", "9598", "10090"},
)

for (q, s), score in lin_map.items():
    print(f"{q} vs {s}: Lin = {score:.3f}")
```

---

## ID Normalization

Taxonomy IDs sometimes arrive as floats from CSV or database reads:

```python
from CBBIO.Taxonomy import normalize_taxonomy_id

normalize_taxonomy_id("9606.0")  # → "9606"
normalize_taxonomy_id(9606.0)    # → "9606"
normalize_taxonomy_id("nan")     # → ""
normalize_taxonomy_id(None)      # → ""
```

---

## Loading Annotation Files

```python
from CBBIO import read_taxonomy_annotations_tsv

annotations = read_taxonomy_annotations_tsv(
    "protein_taxa.tsv",
    entity_col=0,   # column index for entity ID
    taxon_col=1,    # column index for taxonomy ID
    delimiter="\t",
)
# → {"P12345": {"9606"}, ...}
```

---

## Exceptions

| Exception | When raised |
|---|---|
| `TaxonomyError` | Base exception for all taxonomy errors |
| `TaxonNotFoundError` | Requested taxonomy ID not in the loaded graph |
| `TaxonCountsNotPreparedError` | IC method called before `prepare_taxon_counts()` for that mode |
