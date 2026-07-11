#!/usr/bin/env python3
"""Download all supported embedding model checkpoints except ESM-1b."""

from __future__ import annotations

import argparse
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Any


_SCRIPT_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_SCRIPT_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_REPO_ROOT))

from CBBIO import (  # noqa: E402
    AmplifyEmbeddingGenerator,
    Ankh3EmbeddingGenerator,
    Esm2EmbeddingGenerator,
    EsmcEmbeddingGenerator,
    ProteinGlmEmbeddingGenerator,
    ProstT5EmbeddingGenerator,
    ProtT5EmbeddingGenerator,
    download_generator_model,
)


@dataclass(frozen=True)
class DownloadTarget:
    """One embedding model download target."""

    model_class: str
    name: str
    model_reference: str


_GENERATOR_TYPES: tuple[type[Any], ...] = (
    ProtT5EmbeddingGenerator,
    ProstT5EmbeddingGenerator,
    Ankh3EmbeddingGenerator,
    AmplifyEmbeddingGenerator,
    ProteinGlmEmbeddingGenerator,
    EsmcEmbeddingGenerator,
    Esm2EmbeddingGenerator,
)

MODELS: tuple[DownloadTarget, ...] = ()


def _resolve_advertised_model_name(generator_type: type[Any], model_name: str) -> str:
    aliases = getattr(generator_type, "MODEL_ALIASES", {})
    if isinstance(aliases, dict):
        resolved = aliases.get(str(model_name).strip().lower())
        if isinstance(resolved, str) and resolved.strip():
            return resolved
    return str(model_name).strip()


def _iter_download_targets(model_classes: Sequence[str] | None = None) -> list[DownloadTarget]:
    requested = {str(value).strip().lower() for value in model_classes or () if str(value).strip()}
    targets: list[DownloadTarget] = []
    seen: set[tuple[str, str]] = set()

    for generator_type in _GENERATOR_TYPES:
        model_class = str(getattr(generator_type, "GENERATOR_CLASS", "")).strip()
        if not model_class:
            continue
        if requested and model_class not in requested:
            continue

        family_models = getattr(generator_type, "FAMILY_MODELS", ())
        if not isinstance(family_models, Sequence) or isinstance(family_models, (str, bytes, bytearray)):
            continue

        for advertised_name in family_models:
            name = str(advertised_name).strip()
            if not name:
                continue
            model_reference = _resolve_advertised_model_name(generator_type, name)
            key = (model_class, model_reference)
            if key in seen:
                continue
            seen.add(key)
            targets.append(
                DownloadTarget(
                    model_class=model_class,
                    name=name,
                    model_reference=model_reference,
                )
            )

    return targets


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download all CBBIO embedding model checkpoints except ESM-1b.",
    )
    parser.add_argument(
        "--model-class",
        action="append",
        default=None,
        help=(
            "Restrict downloads to one generator class. Repeat to include several. "
            "Default: all supported classes except esm1b."
        ),
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=None,
        help="Optional Hugging Face cache directory shared by all downloads.",
    )
    parser.add_argument(
        "--local-root",
        type=Path,
        default=None,
        help="Optional directory where each repository is materialized under its own subdirectory.",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        help="Optional Hugging Face revision to download for every repository.",
    )
    parser.add_argument(
        "--token",
        type=str,
        default=None,
        help="Optional Hugging Face token. If omitted, huggingface_hub uses its normal environment/cache config.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the download plan without contacting model providers.",
    )
    parser.add_argument(
        "--keep-going",
        action="store_true",
        help="Continue downloading remaining models after a failure.",
    )
    return parser.parse_args(argv)


def _local_dir_for_target(local_root: Path | None, target: DownloadTarget) -> Path | None:
    if local_root is None:
        return None
    safe_name = target.model_reference.replace("/", "__").replace(":", "__")
    return local_root / safe_name


def _print_plan(targets: Iterable[DownloadTarget]) -> None:
    for target in targets:
        print(f"{target.model_class}\t{target.name}\t{target.model_reference}", flush=True)


def main(argv: Sequence[str] | None = None) -> int:
    """Run model downloads from the command line."""
    args = _parse_args(argv)
    targets = _iter_download_targets(args.model_class)
    if not targets:
        print("No matching model download targets.", file=sys.stderr, flush=True)
        return 1

    if args.dry_run:
        _print_plan(targets)
        return 0

    failures: list[tuple[DownloadTarget, Exception]] = []
    for index, target in enumerate(targets, start=1):
        print(
            f"[{index}/{len(targets)}] downloading {target.model_class}: "
            f"{target.name} -> {target.model_reference}",
            flush=True,
        )
        try:
            result = download_generator_model(
                model_class=target.model_class,
                name=target.name,
                revision=args.revision,
                cache_dir=args.cache_dir,
                local_dir=_local_dir_for_target(args.local_root, target),
                token=args.token,
            )
        except Exception as exc:
            failures.append((target, exc))
            print(f"FAILED {target.model_class}:{target.name}: {exc}", file=sys.stderr, flush=True)
            if not args.keep_going:
                break
            continue
        print(f"cached {result.model_reference} at {result.path}", flush=True)

    if failures:
        print(f"{len(failures)} model download(s) failed.", file=sys.stderr, flush=True)
        return 1
    return 0


MODELS = tuple(_iter_download_targets())


if __name__ == "__main__":
    raise SystemExit(main())
