#!/usr/bin/env python3
"""Create ANN (HNSW) indexes for all embedding types at last layer (0).

Creates one index per (embedding_type_id, metric) on:
  public.sequence_embeddings (embedding)
with a partial predicate:
  embedding_type_id = <id> AND layer_index = 0

Metrics/operator classes:
  - l2 -> halfvec_l2_ops
  - cosine -> halfvec_cosine_ops
  - inner_product -> halfvec_ip_ops
"""

from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Dict, List, Tuple

# Force local repo import before site-packages when running this script directly.
_SCRIPT_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_SCRIPT_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_REPO_ROOT))

from CBBIO.BioData import BioDataClient, _embedding_dimension


METRIC_OPCLASS: Dict[str, str] = {
    "l2": "halfvec_l2_ops",
    "cosine": "halfvec_cosine_ops",
    "inner_product": "halfvec_ip_ops",
}


def _slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.strip().lower()).strip("_")


def main() -> None:
    created: List[Tuple[str, int, str]] = []
    with BioDataClient() as client:
        embedding_types = client.list_embedding_types()
        conn = client._require_connection()  # noqa: SLF001 - local maintenance script
        with conn.cursor() as cur:
            for emb in embedding_types:
                emb_id = int(emb.id)
                emb_slug = _slug(str(emb.name))
                row = client.query_one(
                    """
                    SELECT embedding
                    FROM sequence_embeddings
                    WHERE embedding_type_id = %s
                      AND layer_index = 0
                    LIMIT 1;
                    """,
                    (emb_id,),
                )
                if row is None or row.get("embedding") is None:
                    print(f"- skipping embedding_type_id={emb_id} ({emb.name}): no layer 0 embeddings found")
                    continue
                dim = int(_embedding_dimension(row["embedding"]))
                for metric_name, opclass in METRIC_OPCLASS.items():
                    index_name = f"ix_seqemb_e{emb_id}_{emb_slug}_l0_hnsw_{metric_name}"
                    sql = (
                        f"CREATE INDEX IF NOT EXISTS {index_name} "
                        "ON public.sequence_embeddings "
                        f"USING hnsw ((embedding::halfvec({dim})) {opclass}) "
                        f"WHERE embedding_type_id = {emb_id} AND layer_index = 0;"
                    )
                    cur.execute(sql)
                    created.append((str(emb.name), emb_id, metric_name))

    print("Index creation statements executed:")
    for emb_name, emb_id, metric_name in created:
        print(f"- embedding_type_id={emb_id} ({emb_name}), metric={metric_name}, layer=0")


if __name__ == "__main__":
    main()
