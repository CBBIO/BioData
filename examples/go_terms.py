#!/usr/bin/env python3
"""GO ontology example using CBBIO.GO."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

try:
    from CBBIO.GO import GOError, load_go, read_annotations_tsv
except ModuleNotFoundError:
    PROJECT_ROOT = Path(__file__).resolve().parents[1]
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    from CBBIO.GO import GOError, load_go, read_annotations_tsv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="GO term traversal and similarity example")
    parser.add_argument("--obo", required=True, help="Path to GO OBO file (e.g. go-basic.obo)")
    parser.add_argument("--go-id", required=True, help="Primary GO term ID")
    parser.add_argument("--other-go-id", help="Second GO term ID for similarity")
    parser.add_argument("--annotations", help="TSV mapping entity_id<TAB>go_id for IC/similarity")
    parser.add_argument("--metric", choices=["resnik", "lin", "schlicker"], default="resnik")
    parser.add_argument("--pretty", action="store_true", help="Pretty-print JSON output")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    go = load_go(args.obo)
    payload = {
        "term": go.term(args.go_id),
        "ancestors": go.ancestors(args.go_id),
        "descendants": go.descendants(args.go_id),
    }

    if args.annotations:
        annots = read_annotations_tsv(args.annotations)
        go.prepare_term_counts(annots)
        payload["information_content"] = go.information_content(args.go_id)

        if args.other_go_id:
            payload["similarity"] = {
                "other_go_id": args.other_go_id,
                "metric": args.metric,
                "value": go.semantic_similarity(args.go_id, args.other_go_id, method=args.metric),
            }

    if args.pretty:
        print(json.dumps(payload, indent=2, default=str))
    else:
        print(json.dumps(payload, default=str))

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except GOError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
