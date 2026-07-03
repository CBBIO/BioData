# Prompt:
# Add a new embedding model family named "mymodel" under CBBIO/embeddings/models.
# Follow docs/ModelSpecificEmbeddingModules.md. Implement preprocessor, tokenizer
# adapter, model adapter, postprocessor, and MyModelEmbeddingGenerator. Define
# GENERATOR_CLASS, DEFAULT_MODEL_NAME, FAMILY_MODELS, and SUPPORTED_POOLERS. Import
# optional heavy dependencies lazily and raise CBBIO.EmbeddingDependencyError with
# install guidance. Export the generator from CBBIO, register it in the factory,
# update docs, and add tests for factory discovery, dependency errors, layer selection,
# poolers, and metadata.

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Self

from CBBIO import (
    EmbeddingDependencyError,
    EmbeddingGenerator,
    ModelAdapter,
    PostprocessorAdapter,
    PreprocessorAdapter,
    TokenizerAdapter,
)


class MyModelPreprocessor(PreprocessorAdapter):
    def preprocess(self, raw_sequence: str) -> str:
        return raw_sequence.replace(" ", "").upper()


class MyModelTokenizerAdapter(TokenizerAdapter):
    def tokenize(self, sequence: str) -> Mapping[str, Any]:
        try:
            import transformers  # noqa: F401
        except ModuleNotFoundError as exc:
            raise EmbeddingDependencyError("MyModel requires transformers. Install with: pip install transformers") from exc
        return {"sequence": sequence}


class MyModelAdapter(ModelAdapter):
    def infer(self, tokens: Any, *, layer_index: int | Sequence[int] | None = 0) -> Any:
        raise NotImplementedError("Replace with the MyModel forward pass.")


class MyModelPostprocessor(PostprocessorAdapter):
    def postprocess(self, model_output: Any) -> Sequence[float] | Sequence[Sequence[float]]:
        raise NotImplementedError("Convert backend outputs into one embedding payload.")


class MyModelEmbeddingGenerator(EmbeddingGenerator):
    GENERATOR_CLASS = "mymodel"
    DEFAULT_MODEL_NAME = "org/mymodel-base"
    FAMILY_MODELS = ["org/mymodel-base", "org/mymodel-large"]
    SUPPORTED_POOLERS = ("none", "mean")

    @classmethod
    def from_pretrained(cls, model_name: str, *, device: str = "cpu", **kwargs: Any) -> Self:
        _ = device, kwargs
        return cls(
            model_reference=model_name,
            preprocessor=MyModelPreprocessor(),
            tokenizer=MyModelTokenizerAdapter(),
            model=MyModelAdapter(),
            postprocessor=MyModelPostprocessor(),
        )
