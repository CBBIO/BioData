"""Task specifications for supervised layer probing."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

from CBBIO.embeddings import EmbeddingInputError

from .datasets import ObjectiveName, ProteinDataset, ResidueDataset, TaskLevel


ProbeKind = Literal["linear", "mlp"]


@dataclass(frozen=True)
class PredictionSpec:
    """How a dataset target should be interpreted for probing."""

    target: str
    objective: ObjectiveName
    level: TaskLevel = "protein"
    classes: Sequence[str] | None = None
    metrics: Sequence[str] | None = None

    def __post_init__(self) -> None:
        if not self.target.strip():
            raise EmbeddingInputError("PredictionSpec.target must be non-empty.")
        if self.level not in {"protein", "residue"}:
            raise EmbeddingInputError("PredictionSpec.level must be one of: protein, residue.")
        if self.objective not in {"regression", "binary", "multiclass", "multilabel"}:
            raise EmbeddingInputError(
                "PredictionSpec.objective must be one of: regression, binary, multiclass, multilabel."
            )
        if self.objective == "multiclass" and self.classes is not None and len(self.classes) < 2:
            raise EmbeddingInputError("Multiclass PredictionSpec.classes must contain at least two classes.")


@dataclass(frozen=True)
class ProbeSpec:
    """Architecture and training hyperparameters for a small probe."""

    kind: ProbeKind = "linear"
    epochs: int = 100
    learning_rate: float = 0.01
    batch_size: int | None = None
    weight_decay: float = 0.0
    hidden_dim: int = 64
    seed: int = 7

    def __post_init__(self) -> None:
        if self.kind not in {"linear", "mlp"}:
            raise EmbeddingInputError("ProbeSpec.kind must be one of: linear, mlp.")
        if self.epochs < 1:
            raise EmbeddingInputError("ProbeSpec.epochs must be >= 1.")
        if self.learning_rate <= 0:
            raise EmbeddingInputError("ProbeSpec.learning_rate must be > 0.")
        if self.batch_size is not None and self.batch_size < 1:
            raise EmbeddingInputError("ProbeSpec.batch_size must be >= 1 when provided.")
        if self.weight_decay < 0:
            raise EmbeddingInputError("ProbeSpec.weight_decay must be >= 0.")
        if self.hidden_dim < 1:
            raise EmbeddingInputError("ProbeSpec.hidden_dim must be >= 1.")


@dataclass(frozen=True)
class Task:
    """Complete supervised probing task: dataset + prediction + probe."""

    name: str
    dataset: ProteinDataset | ResidueDataset
    prediction: PredictionSpec
    probe: ProbeSpec = ProbeSpec()

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise EmbeddingInputError("Task.name must be non-empty.")
        if self.prediction.level == "protein" and not isinstance(self.dataset, ProteinDataset):
            raise EmbeddingInputError("Protein-level tasks require a ProteinDataset.")
        if self.prediction.level == "residue" and not isinstance(self.dataset, ResidueDataset):
            raise EmbeddingInputError("Residue-level tasks require a ResidueDataset.")


__all__ = ["PredictionSpec", "ProbeKind", "ProbeSpec", "Task"]
