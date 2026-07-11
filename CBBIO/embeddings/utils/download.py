"""Download helpers for embedding model caches."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any, cast

from .. import EmbeddingDependencyError, ModelDownloadResult


def download_huggingface_snapshot(
    model_reference: str,
    *,
    revision: str | None = None,
    cache_dir: str | Path | None = None,
    local_dir: str | Path | None = None,
    token: str | bool | None = None,
    allow_patterns: str | Sequence[str] | None = None,
    ignore_patterns: str | Sequence[str] | None = None,
    trust_remote_code: bool | None = None,
    **kwargs: Any,
) -> ModelDownloadResult:
    """Download a Hugging Face repository into the local cache."""
    try:
        from huggingface_hub import snapshot_download  # type: ignore
    except ModuleNotFoundError as exc:
        raise EmbeddingDependencyError(
            "huggingface_hub is required for model downloads. Install with: pip install huggingface_hub"
        ) from exc

    _ = trust_remote_code
    path = snapshot_download(
        repo_id=str(model_reference),
        revision=revision,
        cache_dir=str(cache_dir) if cache_dir is not None else None,
        local_dir=str(local_dir) if local_dir is not None else None,
        token=token,
        allow_patterns=_snapshot_patterns(allow_patterns),
        ignore_patterns=_snapshot_patterns(ignore_patterns),
        **kwargs,
    )
    return ModelDownloadResult(
        model_reference=str(model_reference),
        path=Path(path),
        backend="huggingface-hub",
    )


def _snapshot_patterns(value: str | Sequence[str] | None) -> str | list[str] | None:
    if value is None or isinstance(value, str):
        return value
    return [str(item) for item in cast(Sequence[object], value)]


__all__ = [
    "download_huggingface_snapshot",
]
