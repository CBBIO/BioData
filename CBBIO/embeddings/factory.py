"""Factory and registry helpers for model-specific embedding generators."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Dict, List, Type, cast

from . import (
    EmbeddingBackendError,
    EmbeddingGenerator,
    EmbeddingInputError,
    ModelDownloadResult,
)


def Generator(
    *,
    name: str | None = None,
    model_class: str | None = None,
    class_: str | None = None,
    device: str = "cpu",
    **kwargs: Any,
) -> EmbeddingGenerator:
    """Convenience factory for model-specific embedding generators."""
    resolved_class = model_class or class_ or cast(str | None, kwargs.pop("class", None))
    if resolved_class is None or not str(resolved_class).strip():
        raise EmbeddingInputError("Generator requires model_class/class_/class (e.g. 'prott5').")

    registry, aliases = _generator_registry()
    canonical = _normalize_model_class(resolved_class, aliases)
    if canonical is None or canonical not in registry:
        supported = ", ".join(registry.keys())
        raise EmbeddingInputError(
            f"Unknown model class: {resolved_class!r}. Supported values: {supported}."
        )

    generator_cls = registry[canonical]
    resolved_name = str(name).strip() if name is not None else ""
    if not resolved_name:
        default_name = getattr(generator_cls, "DEFAULT_MODEL_NAME", None)
        if isinstance(default_name, str) and default_name.strip():
            resolved_name = default_name.strip()
        else:
            raise EmbeddingInputError(
                f"Generator requires a model name for class {canonical!r}; no DEFAULT_MODEL_NAME is defined."
            )
    return generator_cls.from_pretrained(resolved_name, device=device, **kwargs)


def available_generator_classes() -> List[str]:
    """Return canonical generator class names supported by ``Generator``."""
    registry, _ = _generator_registry()
    return list(registry.keys())


def available_generator_models(model_class: str | None = None) -> Dict[str, List[str]] | List[str]:
    """Return available model identifiers by family."""
    registry, aliases = _generator_registry()
    if model_class is None:
        return {name: _family_models_for_class(name, registry=registry) for name in registry.keys()}

    canonical = _normalize_model_class(model_class, aliases)
    if canonical is None or canonical not in registry:
        supported = ", ".join(registry.keys())
        raise EmbeddingInputError(
            f"Unknown model class: {model_class!r}. Supported values: {supported}."
        )
    return _family_models_for_class(canonical, registry=registry)


def download_generator_model(
    *,
    model_class: str,
    name: str | None = None,
    **kwargs: Any,
) -> ModelDownloadResult:
    """Download a generator model into the local cache without loading it for inference."""
    registry, aliases = _generator_registry()
    canonical = _normalize_model_class(model_class, aliases)
    if canonical is None or canonical not in registry:
        supported = ", ".join(registry.keys())
        raise EmbeddingInputError(
            f"Unknown model class: {model_class!r}. Supported values: {supported}."
        )

    generator_cls = registry[canonical]
    download_fn = getattr(generator_cls, "download", None)
    if not callable(download_fn):
        raise EmbeddingBackendError(f"Generator class {canonical!r} does not support downloads.")
    return cast(ModelDownloadResult, download_fn(name, **kwargs))


def _normalize_model_class(value: str | None, aliases: Dict[str, str]) -> str | None:
    if value is None:
        return None
    key = str(value).strip().lower()
    if not key:
        return None
    return aliases.get(key)


def _family_models_for_class(
    canonical_class: str,
    *,
    registry: Dict[str, Type[EmbeddingGenerator]],
) -> List[str]:
    generator_cls = registry.get(canonical_class)
    if generator_cls is None:
        supported = ", ".join(registry.keys())
        raise EmbeddingInputError(
            f"Unknown model class: {canonical_class!r}. Supported values: {supported}."
        )

    values = getattr(generator_cls, "FAMILY_MODELS", None)
    if isinstance(values, Sequence) and not isinstance(values, (str, bytes, bytearray)):
        family_values = cast(Sequence[object], values)
        models = [str(value) for value in family_values if str(value).strip()]
        if models:
            return models

    default_name = getattr(generator_cls, "DEFAULT_MODEL_NAME", None)
    if isinstance(default_name, str) and default_name.strip():
        return [default_name]
    return []


def _generator_registry() -> TupleRegistry:
    classes = _load_generator_classes()
    registry: Dict[str, Type[EmbeddingGenerator]] = {}
    aliases: Dict[str, str] = {}

    for generator_cls in classes:
        canonical = str(getattr(generator_cls, "GENERATOR_CLASS", "")).strip()
        if not canonical:
            continue
        if canonical in registry:
            raise EmbeddingBackendError(f"Duplicate generator class registration for {canonical!r}.")
        registry[canonical] = generator_cls

        raw_aliases = getattr(generator_cls, "GENERATOR_ALIASES", ())
        alias_values = [canonical]
        if isinstance(raw_aliases, Sequence) and not isinstance(raw_aliases, (str, bytes, bytearray)):
            alias_values.extend(str(value) for value in cast(Sequence[object], raw_aliases))

        for alias in alias_values:
            key = str(alias).strip().lower()
            if not key:
                continue
            existing = aliases.get(key)
            if existing is not None and existing != canonical:
                raise EmbeddingBackendError(
                    f"Alias {alias!r} is defined by multiple generator classes: {existing!r}, {canonical!r}."
                )
            aliases[key] = canonical

    return registry, aliases


TupleRegistry = tuple[Dict[str, Type[EmbeddingGenerator]], Dict[str, str]]


def _load_generator_classes() -> tuple[Type[EmbeddingGenerator], ...]:
    from .models.prott5 import ProtT5EmbeddingGenerator
    from .models.prostt5 import ProstT5EmbeddingGenerator
    from .models.ankh3 import Ankh3EmbeddingGenerator
    from .models.amplify import AmplifyEmbeddingGenerator
    from .models.proteinglm import ProteinGlmEmbeddingGenerator
    from .models.esmc import EsmcEmbeddingGenerator
    from .models.esm2 import Esm2EmbeddingGenerator
    from .models.esm1b import Esm1bEmbeddingGenerator

    return (
        ProtT5EmbeddingGenerator,
        ProstT5EmbeddingGenerator,
        Ankh3EmbeddingGenerator,
        AmplifyEmbeddingGenerator,
        ProteinGlmEmbeddingGenerator,
        EsmcEmbeddingGenerator,
        Esm2EmbeddingGenerator,
        Esm1bEmbeddingGenerator,
    )


__all__ = [
    "Generator",
    "available_generator_classes",
    "available_generator_models",
    "download_generator_model",
]
