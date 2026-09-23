# Agent Code Examples

```bash
python docs/examples/01_esm2_peer_sweep.py --help
```

These scripts correspond to [AgentExamplePrompts.md](../AgentExamplePrompts.md). Each file starts with the original prompt as a comment and uses public `CBBIO` imports where possible.

Browse the corresponding public interfaces in the [BioData API reference](https://biodata.readthedocs.io/en/latest/api/complete/).

## Files

| File | Covers |
|---|---|
| `01_esm2_peer_sweep.py` | ESM-2 model discovery, embedding generation, PEER probing |
| `02_wnt_signaling_graph.py` | BioData SQL, GO descendants, ProstT5 embeddings, graph output |
| `03_cafa_transfer_baseline.py` | CAFA datasets, `TransferProbe`, search backends |
| `04_residue_ptm_probe_matrix.py` | PTM residue datasets, unpooled embeddings, probe comparison |
| `05_go_semantic_neighbor_audit.py` | Nearest neighbors, GO similarity, correlation report |
| `06_taxonomy_embedding_report.py` | Taxonomy IC and embedding distance summaries |
| `07_embedding_store_converter.py` | Embedding IO conversion and filtering |
| `08_custom_residue_dataset_adapter.py` | Minimal custom residue CSV loader |
| `09_new_embedding_model_family_skeleton.py` | Model adapter skeleton |
| `10_cross_modal_similarity_explorer.py` | Alignment, DB search, GO, taxonomy ranking |

## Exceptions

| Exception | When raised |
|---|---|
| `EmbeddingInputError` | Required files, records, layers, or labels are invalid |
| `EmbeddingDependencyError` | Optional runtime dependency is unavailable |
