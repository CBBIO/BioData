"""Embedding generation job orchestration."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
import gc
from pathlib import Path
import time
from typing import Any, Callable, Dict, List, Sequence
import warnings

from .. import (
    EmbeddingGenerator,
    GenerationInput,
    normalize_generation_exception,
)
from .pooler import PoolerInput
from .writer import EmbeddingWriter

ProgressCallback = Callable[[Dict[str, Any]], None]


def _dict_list() -> List[Dict[str, Any]]:
    return []


def _path_list() -> List[Path]:
    return []


@dataclass(frozen=True)
class EmbeddingJobResult:
    """Result metadata for one embedding generation job."""
    record_count: int
    error_count: int = 0
    skipped_count: int = 0
    paths: List[Path] = field(default_factory=_path_list)
    id_paths: List[Path] = field(default_factory=_path_list)
    errors: List[Dict[str, Any]] = field(default_factory=_dict_list)
    skipped: List[Dict[str, Any]] = field(default_factory=_dict_list)
    elapsed_seconds: float = 0.0


def run_embedding_generation(
    generator: EmbeddingGenerator,
    batcher: Iterable[Sequence[GenerationInput]] | None,
    writer: EmbeddingWriter | None,
    *,
    layer_index: int | Sequence[int] | None = 0,
    pooler: PoolerInput = None,
    fail_fast: bool = False,
    progress_callback: ProgressCallback | None = None,
) -> EmbeddingJobResult:
    """Run an embedding job from a batcher into a writer."""
    if batcher is None:
        raise ValueError("batcher is required.")
    if writer is None:
        raise ValueError("writer is required.")

    _warn_for_unbounded_lengths(batcher)

    started = time.perf_counter()
    errors: List[Dict[str, Any]] = []
    error_count = 0
    batch_index = 0

    try:
        for batch in batcher:
            batch_index += 1
            batch_errors = _run_batch_with_recovery(
                generator,
                list(batch),
                writer,
                layer_index=layer_index,
                pooler=pooler,
                fail_fast=fail_fast,
                progress_callback=progress_callback,
            )
            error_count += len(batch_errors)
            _extend_sample(errors, batch_errors, limit=5)
            if progress_callback is not None:
                progress_callback(
                    {
                        "event": "batch_completed",
                        "batch_index": batch_index,
                        "written_count": writer.record_count,
                        "error_count": error_count,
                        "skipped_count": int(getattr(batcher, "skipped_count", 0)),
                        "errors": list(errors),
                        "skipped": list(getattr(batcher, "skipped", [])),
                    }
                )
    finally:
        writer.close()

    return EmbeddingJobResult(
        record_count=writer.record_count,
        error_count=error_count,
        skipped_count=int(getattr(batcher, "skipped_count", 0)),
        paths=list(writer.paths),
        id_paths=list(writer.id_paths),
        errors=errors,
        skipped=list(getattr(batcher, "skipped", [])),
        elapsed_seconds=time.perf_counter() - started,
    )


def _warn_for_unbounded_lengths(batcher: object) -> None:
    max_sequence_length = getattr(batcher, "max_sequence_length", None)
    if max_sequence_length is not None:
        return
    max_batch_tokens = getattr(batcher, "max_batch_tokens", None)
    if max_batch_tokens is None:
        warnings.warn(
            "No max_sequence_length or max_batch_tokens is set. Very long sequences can cause CUDA OOM; "
            "use max_sequence_length/max_batch_tokens for production FASTA jobs or keep fail_fast=False to recover.",
            RuntimeWarning,
            stacklevel=3,
        )
    else:
        warnings.warn(
            "No max_sequence_length is set. A single very long sequence can still cause CUDA OOM; "
            "use max_sequence_length to skip outliers when needed.",
            RuntimeWarning,
            stacklevel=3,
        )


_MAX_OOM_RECOVERY_DEPTH = 12


def _run_batch_with_recovery(
    generator: EmbeddingGenerator,
    batch: List[GenerationInput],
    writer: EmbeddingWriter,
    *,
    layer_index: int | Sequence[int] | None,
    pooler: PoolerInput,
    fail_fast: bool,
    progress_callback: ProgressCallback | None,
    _depth: int = 0,
) -> List[Dict[str, Any]]:
    if not batch:
        return []

    try:
        result = generator.generate(batch, layer_index=layer_index, pooler=pooler, fail_fast=True)
    except Exception as exc:
        if _is_cuda_oom(exc):
            _clear_cuda_cache()
            if progress_callback is not None:
                progress_callback({"event": "batch_retry", "reason": "cuda_oom", "batch_size": len(batch)})
            if len(batch) == 1 or _depth >= _MAX_OOM_RECOVERY_DEPTH:
                if fail_fast:
                    raise
                return [_error_event(record, exc, reason="cuda_oom") for record in batch]
            midpoint = max(1, len(batch) // 2)
            return (
                _run_batch_with_recovery(
                    generator,
                    batch[:midpoint],
                    writer,
                    layer_index=layer_index,
                    pooler=pooler,
                    fail_fast=fail_fast,
                    progress_callback=progress_callback,
                    _depth=_depth + 1,
                )
                + _run_batch_with_recovery(
                    generator,
                    batch[midpoint:],
                    writer,
                    layer_index=layer_index,
                    pooler=pooler,
                    fail_fast=fail_fast,
                    progress_callback=progress_callback,
                    _depth=_depth + 1,
                )
            )

        if fail_fast:
            raise
        if len(batch) == 1:
            return [_error_event(batch[0], exc)]
        midpoint = max(1, len(batch) // 2)
        return (
            _run_batch_with_recovery(
                generator,
                batch[:midpoint],
                writer,
                layer_index=layer_index,
                pooler=pooler,
                fail_fast=fail_fast,
                progress_callback=progress_callback,
                _depth=_depth + 1,
            )
            + _run_batch_with_recovery(
                generator,
                batch[midpoint:],
                writer,
                layer_index=layer_index,
                pooler=pooler,
                fail_fast=fail_fast,
                progress_callback=progress_callback,
                _depth=_depth + 1,
            )
        )

    writer.write(result.records)
    return [dict(error) for error in result.errors]


def _error_event(record: GenerationInput, exc: Exception, *, reason: str | None = None) -> Dict[str, Any]:
    normalized = normalize_generation_exception(exc)
    event: Dict[str, Any] = {
        "id": record.id,
        "error_type": normalized.__class__.__name__,
        "message": str(normalized),
    }
    if reason is not None:
        event["reason"] = reason
    return event


def _is_cuda_oom(exc: BaseException) -> bool:
    try:
        import torch  # type: ignore

        oom_cls = getattr(getattr(torch, "cuda", None), "OutOfMemoryError", None)
        if oom_cls is not None and isinstance(exc, oom_cls):
            return True
    except Exception:
        pass
    text = str(exc).lower()
    return "out of memory" in text and ("cuda" in text or "cudnn" in text or "gpu" in text)


def _clear_cuda_cache() -> None:
    gc.collect()
    try:
        import torch  # type: ignore

        cuda = getattr(torch, "cuda", None)
        empty_cache = getattr(cuda, "empty_cache", None)
        if callable(empty_cache):
            empty_cache()
    except Exception:
        return


def _extend_sample(target: List[Dict[str, Any]], values: Sequence[Dict[str, Any]], *, limit: int) -> None:
    remaining = max(0, int(limit) - len(target))
    if remaining:
        target.extend(dict(value) for value in values[:remaining])


__all__ = ["EmbeddingJobResult", "ProgressCallback", "run_embedding_generation"]
