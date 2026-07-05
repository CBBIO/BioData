#!/usr/bin/env python3
"""Create ANN (HNSW) indexes for selected embedding types and metrics.

Creates one index per (embedding_type_id, metric) on:
  public.sequence_embeddings (embedding)
with a partial predicate:
  embedding_type_id = <id> AND layer_index = <layer>

Metrics/operator classes:
  - l2 -> halfvec_l2_ops
  - cosine -> halfvec_cosine_ops
  - inner_product -> halfvec_ip_ops
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Any, Sequence, cast

# Force local repo import before site-packages when running this script directly.
_SCRIPT_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_SCRIPT_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_REPO_ROOT))

from CBBIO.BioData import BioDataClient, BioDataError

try:
    from psycopg.errors import DiskFull
except ModuleNotFoundError:  # pragma: no cover - script dependency path
    DiskFull = None  # type: ignore[assignment]


METRIC_OPCLASS: dict[str, str] = {
    "l2": "halfvec_l2_ops",
    "cosine": "halfvec_cosine_ops",
    "inner_product": "halfvec_ip_ops",
}


def _slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.strip().lower()).strip("_")


def _embedding_dimension(value: Any) -> int:
    dimensions_attr = getattr(value, "dimensions", None)
    if callable(dimensions_attr):
        try:
            dimension = int(cast(Any, dimensions_attr)())
            if dimension >= 1:
                return dimension
        except (TypeError, ValueError):
            pass

    to_list_attr = getattr(value, "to_list", None)
    if callable(to_list_attr):
        try:
            values = cast(Sequence[Any], cast(Any, to_list_attr)())
            dimension = int(len(values))
            if dimension >= 1:
                return dimension
        except (TypeError, ValueError):
            pass

    try:
        dimension = int(len(value))
    except (TypeError, ValueError) as exc:
        raise BioDataError("Could not infer embedding dimension from a sequence_embeddings row.") from exc
    if dimension < 1:
        raise BioDataError("Embedding dimension must be >= 1.")
    return dimension


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create HNSW ANN indexes for selected embedding types and metrics.",
    )
    parser.add_argument(
        "--embedding-type-id",
        type=int,
        action="append",
        default=None,
        help="Limit index creation to one or more embedding_type_id values. Repeat to pass multiple.",
    )
    parser.add_argument(
        "--metric",
        choices=sorted(METRIC_OPCLASS),
        action="append",
        default=None,
        help="Limit index creation to one or more metrics. Repeat to pass multiple.",
    )
    parser.add_argument(
        "--layer-index",
        type=int,
        default=0,
        help="Layer index to index. Default: 0.",
    )
    parser.add_argument(
        "--max-parallel-maintenance-workers",
        type=int,
        default=0,
        help="Session setting for max_parallel_maintenance_workers. Default: 0.",
    )
    parser.add_argument(
        "--max-parallel-workers-per-gather",
        type=int,
        default=0,
        help="Session setting for max_parallel_workers_per_gather. Default: 0.",
    )
    parser.add_argument(
        "--maintenance-work-mem",
        type=str,
        default=None,
        help="Optional session setting for maintenance_work_mem, for example 256MB.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    created: list[tuple[str, int, str]] = []
    selected_embedding_ids = (
        {int(value) for value in args.embedding_type_id}
        if args.embedding_type_id
        else None
    )
    selected_metrics = list(dict.fromkeys(args.metric or METRIC_OPCLASS.keys()))
    layer_index = int(args.layer_index)
    with BioDataClient() as client:
        embedding_types = client.list_embedding_types()
        if selected_embedding_ids is not None:
            embedding_types = [emb for emb in embedding_types if int(emb.id) in selected_embedding_ids]
        client.execute(
            "SELECT set_config('max_parallel_maintenance_workers', %s, false);",
            (str(max(0, int(args.max_parallel_maintenance_workers))),),
        )
        client.execute(
            "SELECT set_config('max_parallel_workers_per_gather', %s, false);",
            (str(max(0, int(args.max_parallel_workers_per_gather))),),
        )
        if args.maintenance_work_mem:
            client.execute(
                "SELECT set_config('maintenance_work_mem', %s, false);",
                (str(args.maintenance_work_mem).strip(),),
            )

        for emb in embedding_types:
            emb_id = int(emb.id)
            emb_slug = _slug(str(emb.name))
            row = client.query_one(
                """
                SELECT embedding
                FROM sequence_embeddings
                WHERE embedding_type_id = %s
                  AND layer_index = %s
                LIMIT 1;
                """,
                (emb_id, layer_index),
            )
            if row is None or row.get("embedding") is None:
                print(
                    f"- skipping embedding_type_id={emb_id} ({emb.name}): "
                    f"no layer {layer_index} embeddings found"
                )
                continue
            dim = int(_embedding_dimension(row["embedding"]))
            for metric_name in selected_metrics:
                opclass = METRIC_OPCLASS[metric_name]
                index_name = f"ix_seqemb_e{emb_id}_{emb_slug}_l{layer_index}_hnsw_{metric_name}"
                sql = (
                    f"CREATE INDEX IF NOT EXISTS {index_name} "
                    "ON public.sequence_embeddings "
                    f"USING hnsw ((embedding::halfvec({dim})) {opclass}) "
                    f"WHERE embedding_type_id = {emb_id} AND layer_index = {layer_index};"
                )
                try:
                    client.execute(sql)
                except Exception as exc:
                    if DiskFull is not None and isinstance(exc, DiskFull):
                        raise SystemExit(
                            "PostgreSQL ran out of Docker shared memory while building the HNSW index.\n"
                            "Try either:\n"
                            "  1. restart the container with a larger shared-memory segment, e.g. --shm-size=1g\n"
                            "  2. rerun this script with parallel workers disabled (already the default here)\n"
                            "  3. optionally lower build memory, e.g. --maintenance-work-mem 128MB\n"
                        ) from exc
                    raise
                created.append((str(emb.name), emb_id, metric_name))

    print("Index creation statements executed:")
    for emb_name, emb_id, metric_name in created:
        print(f"- embedding_type_id={emb_id} ({emb_name}), metric={metric_name}, layer={layer_index}")


if __name__ == "__main__":
    main()
