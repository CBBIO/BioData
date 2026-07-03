# Prompt:
# Write a robust conversion tool for embedding files. It should accept pickle, NumPy,
# or HDF5 inputs through CBBIO.load_embedding_records, optionally mean-pool residue-level
# records with CBBIO.mean_pool_embedding_record, filter by ids, layer_index,
# model_reference, and pool_method, then write HDF5 with CBBIO.save_embedding_records_h5.
# Include a dry-run mode that reports record count, payload shapes, layer distribution,
# model references, and estimated output size without writing files.

from __future__ import annotations

import argparse
import logging
from collections import Counter
from collections.abc import Sequence
from pathlib import Path

from CBBIO import EmbeddingRecord, load_embedding_records, mean_pool_embedding_record, save_embedding_records_h5


LOGGER = logging.getLogger("embedding_store_converter")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--id", action="append")
    parser.add_argument("--layer-index", type=int, action="append")
    parser.add_argument("--model-reference", action="append")
    parser.add_argument("--pool-method", action="append")
    parser.add_argument("--mean-pool", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def keep_record(record: EmbeddingRecord, args: argparse.Namespace) -> bool:
    pool_method = (record.metadata or {}).get("pooling")
    return (
        (args.id is None or record.id in set(args.id))
        and (args.layer_index is None or record.layer_index in set(args.layer_index))
        and (args.model_reference is None or record.model_reference in set(args.model_reference))
        and (args.pool_method is None or str(pool_method or "none") in set(args.pool_method))
    )


def summarize(records: Sequence[EmbeddingRecord]) -> None:
    LOGGER.info("record_count=%s", len(records))
    LOGGER.info("layers=%s", dict(Counter(record.layer_index for record in records)))
    LOGGER.info("models=%s", dict(Counter(record.model_reference for record in records)))
    LOGGER.info("shapes=%s", dict(Counter(str(record.shape) for record in records)))
    estimated = sum(len(record.embedding) if len(record.shape) == 1 else record.shape[0] * record.shape[1] for record in records)
    LOGGER.info("estimated_float_values=%s", estimated)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
    args = parse_args()
    records = [record for record in load_embedding_records(args.input) if keep_record(record, args)]
    if args.mean_pool:
        records = [mean_pool_embedding_record(record) if len(record.shape) == 2 else record for record in records]
    summarize(records)
    if not args.dry_run:
        save_embedding_records_h5(records, Path(args.output))


if __name__ == "__main__":
    main()
