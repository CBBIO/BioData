# Custom Probes

Plug a custom probe class into `Task.probe` to reuse CBBIO splitting, validation, evaluation,
metrics, and result reporting.

```python
from CBBIO import (
    PredictionSpec,
    ProbeBackend,
    ProbeBackendInput,
    ProbeBackendOutput,
    ProteinDataset,
    ProteinExample,
    Task,
    run_task_on_layer,
)


class FixedBinaryProbe(ProbeBackend):
    """Return deterministic predictions for a minimal custom probe example."""

    def fit_predict(self, data: ProbeBackendInput) -> ProbeBackendOutput:
        """Return one prediction and score for each test example."""
        return ProbeBackendOutput(
            predictions={item: index for index, item in enumerate(data.test_ids)},
            scores={item: float(index) for index, item in enumerate(data.test_ids)},
        )


dataset = ProteinDataset(
    [
        ProteinExample("train_0", "AAAA", {"active": 0}, "train"),
        ProteinExample("train_1", "CCCC", {"active": 1}, "train"),
        ProteinExample("test_0", "AAAC", {"active": 0}, "test"),
        ProteinExample("test_1", "CCCA", {"active": 1}, "test"),
    ]
)
task = Task(
    name="fixed_binary",
    dataset=dataset,
    prediction=PredictionSpec(target="active", objective="binary"),
    probe=FixedBinaryProbe(),
)
result = run_task_on_layer(
    task=task,
    embeddings={item: [float(index)] for index, item in enumerate(dataset.ids())},
)
print(result.metrics["accuracy"])  # 1.0
```

The custom class replaces `ProbeSpec` in the task. The runner creates `ProbeBackendInput` from the
dataset and calls `fit_predict()`. CBBIO validates `ProbeBackendOutput` and returns the normal
`TaskLayerResult`.

## Backend Contract

```python
from CBBIO import ProbeBackend, ProbeBackendInput, ProbeBackendOutput


class CustomProbe(ProbeBackend):
    """Implement a custom supervised estimator."""

    def fit_predict(self, data: ProbeBackendInput) -> ProbeBackendOutput:
        """Fit the training split and predict the test split."""
        return ProbeBackendOutput(predictions={}, scores=None)
```

`ProbeBackendInput` provides the canonical task data:

| Field | Meaning |
|---|---|
| `level` | `"protein"` or `"residue"` |
| `prediction` | Target, objective, classes, and requested metrics |
| `train_ids` | Example IDs selected from the training split |
| `test_ids` | Example IDs selected from the test split |
| `embeddings` | Protein vectors or residue matrices keyed by example ID |
| `labels` | Scalar protein labels or residue-label sequences |
| `masks` | Residue masks keyed by example ID; empty for protein tasks |
| `feature_mean` | Optional built-in residue standardization cache |
| `feature_std` | Optional built-in residue standardization cache |
| `flat_data` | Optional built-in flattened residue cache |

Return predictions only for `test_ids`. Protein predictions are scalar values keyed by protein ID.
Residue predictions are full-length sequences keyed by protein ID. CBBIO applies residue masks and
reports residue predictions with IDs such as `P12345:42`.

Binary probes must return probability-like `scores` as well as class predictions. CBBIO uses those
scores for AUROC and AUPRC. Regression and multiclass probes do not require scores.

Custom probes can also return `metadata` in `ProbeBackendOutput`. CBBIO passes it through to
`TaskLayerResult.metadata`. Use this for per-example diagnostics such as transferred neighbor IDs
or model explanations.

## XGBoost

```python
from collections.abc import Sequence
from typing import cast

from xgboost import XGBClassifier

from CBBIO import (
    PredictionSpec,
    ProbeBackend,
    ProbeBackendInput,
    ProbeBackendOutput,
    ProteinDataset,
    ProteinExample,
    Task,
    run_task_on_layer,
)


class XGBoostProbe(ProbeBackend):
    """Train an XGBoost classifier on pooled protein embeddings."""

    def __init__(self, *, estimators: int = 100, max_depth: int = 4) -> None:
        self.estimators = estimators
        self.max_depth = max_depth

    def fit_predict(self, data: ProbeBackendInput) -> ProbeBackendOutput:
        """Fit the canonical training split and predict the test split."""
        train_features = [
            cast(Sequence[float], data.embeddings[item])
            for item in data.train_ids
        ]
        train_labels = [int(cast(int, data.labels[item])) for item in data.train_ids]
        test_features = [
            cast(Sequence[float], data.embeddings[item])
            for item in data.test_ids
        ]
        model = XGBClassifier(
            n_estimators=self.estimators,
            max_depth=self.max_depth,
            random_state=7,
        )
        model.fit(train_features, train_labels)
        predicted = model.predict(test_features)
        probabilities = model.predict_proba(test_features)[:, 1]
        return ProbeBackendOutput(
            predictions={item: int(value) for item, value in zip(data.test_ids, predicted)},
            scores={item: float(value) for item, value in zip(data.test_ids, probabilities)},
        )


dataset = ProteinDataset(
    [
        ProteinExample("train_0", "AAAA", {"active": 0}, "train"),
        ProteinExample("train_1", "AAAC", {"active": 0}, "train"),
        ProteinExample("train_2", "CCCC", {"active": 1}, "train"),
        ProteinExample("train_3", "CCCA", {"active": 1}, "train"),
        ProteinExample("test_0", "AACC", {"active": 0}, "test"),
        ProteinExample("test_1", "CCAA", {"active": 1}, "test"),
    ]
)
embeddings = {
    "train_0": [0.0, 0.1, 0.0],
    "train_1": [0.1, 0.2, 0.0],
    "train_2": [1.0, 0.9, 1.0],
    "train_3": [0.9, 0.8, 1.0],
    "test_0": [0.2, 0.3, 0.1],
    "test_1": [0.8, 0.7, 0.9],
}
task = Task(
    name="xgboost_activity",
    dataset=dataset,
    prediction=PredictionSpec(target="active", objective="binary"),
    probe=XGBoostProbe(estimators=20, max_depth=2),
)
result = run_task_on_layer(task=task, embeddings=embeddings)
print(result.metrics["auroc"])
```

Install XGBoost in the application environment that runs the probe:

```bash
pip install xgboost
```

XGBoost expects one vector per example. Use pooled embeddings for protein-level tasks.

## Residue CNN

```python
from collections.abc import Sequence
from typing import cast

import torch

from CBBIO import (
    PredictionSpec,
    ProbeBackend,
    ProbeBackendInput,
    ProbeBackendOutput,
    ResidueDataset,
    ResidueExample,
    Task,
    run_task_on_layer,
)


class ResidueNetwork(torch.nn.Module):
    """Predict one binary logit for each residue embedding."""

    def __init__(self, embedding_dim: int) -> None:
        super().__init__()
        self.network = torch.nn.Sequential(
            torch.nn.Conv1d(embedding_dim, 8, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv1d(8, 1, kernel_size=1),
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        """Return one logit per residue."""
        return self.network(values.transpose(0, 1).unsqueeze(0)).squeeze(0).squeeze(0)


class ResidueCnnProbe(ProbeBackend):
    """Train a sequence-aware CNN through the custom probe contract."""

    def __init__(self, *, epochs: int = 50, learning_rate: float = 0.02) -> None:
        self.epochs = epochs
        self.learning_rate = learning_rate

    def fit_predict(self, data: ProbeBackendInput) -> ProbeBackendOutput:
        """Fit masked residue labels and return full-length test predictions."""
        first_matrix = cast(
            Sequence[Sequence[float]],
            data.embeddings[data.train_ids[0]],
        )
        torch.manual_seed(7)
        model = ResidueNetwork(embedding_dim=len(first_matrix[0]))
        optimizer = torch.optim.Adam(model.parameters(), lr=self.learning_rate)
        loss_function = torch.nn.BCEWithLogitsLoss()

        for _epoch in range(self.epochs):
            for item in data.train_ids:
                matrix = cast(Sequence[Sequence[float]], data.embeddings[item])
                labels = cast(Sequence[object], data.labels[item])
                features = torch.tensor(matrix, dtype=torch.float32)
                targets = torch.tensor(labels, dtype=torch.float32)
                included = data.masks.get(item)
                if included is None:
                    included = [True] * len(labels)
                mask = torch.tensor(included, dtype=torch.bool)
                loss = loss_function(model(features)[mask], targets[mask])
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        predictions: dict[str, list[int]] = {}
        scores: dict[str, list[float]] = {}
        model.eval()
        with torch.no_grad():
            for item in data.test_ids:
                matrix = cast(Sequence[Sequence[float]], data.embeddings[item])
                values = torch.sigmoid(model(torch.tensor(matrix, dtype=torch.float32)))
                scores[item] = [float(value) for value in values.tolist()]
                predictions[item] = [int(value >= 0.5) for value in scores[item]]
        return ProbeBackendOutput(predictions=predictions, scores=scores)


dataset = ResidueDataset(
    [
        ResidueExample("train_0", "AAAA", {"site": [0, 0, 1, 0]}, "train"),
        ResidueExample("train_1", "CCCC", {"site": [0, 1, 0, 0]}, "train"),
        ResidueExample("test_0", "ACAC", {"site": [0, 0, 1, 0]}, "test"),
    ]
)
embeddings = {
    "train_0": [[0.0, 0.1], [0.0, 0.2], [1.0, 0.9], [0.1, 0.0]],
    "train_1": [[0.1, 0.0], [0.9, 1.0], [0.2, 0.1], [0.0, 0.1]],
    "test_0": [[0.0, 0.1], [0.1, 0.2], [0.9, 0.8], [0.1, 0.0]],
}
task = Task(
    name="residue_cnn",
    dataset=dataset,
    prediction=PredictionSpec(target="site", objective="binary", level="residue"),
    probe=ResidueCnnProbe(),
)
result = run_task_on_layer(task=task, embeddings=embeddings)
print(result.metrics["f1"])
```

The backend returns full-length residue sequences. CBBIO applies dataset masks before calculating
metrics and converts the output to position-qualified prediction IDs.

## Choosing an Integration

```python
from CBBIO import LinearProbe, MlpProbe, ProbeSpec, TransferProbe

linear = LinearProbe(epochs=100)
mlp = MlpProbe(hidden_dim=128, epochs=100)
transfer = TransferProbe()
compatible = ProbeSpec(kind="mlp", hidden_dim=128)
```

Use `LinearProbe` and `MlpProbe` for built-in Torch probes. Use `TransferProbe` for a no-training
transfer head. `ProbeSpec` remains compatible and resolves to the corresponding Torch backend.
Use a custom backend class for external estimators or sequence-aware architectures.

| Probe | `Task.probe` value | Evaluation |
|---|---|---|
| Linear | `LinearProbe(...)` | Canonical |
| MLP | `MlpProbe(...)` | Canonical |
| Transfer | `TransferProbe()` | Canonical |
| XGBoost, random forest, SVM | Custom backend instance | Canonical |
| CNN, RNN, transformer head | Custom backend instance | Canonical |

Custom backends own fitting and inference. CBBIO continues to own splits, output validation, residue
masking, objective-specific metrics, and `TaskLayerResult` construction.

## Exceptions

| Exception | When raised |
|---|---|
| `EmbeddingInputError` | Backend predictions, scores, labels, masks, or shapes are invalid |
| `ImportError` | Optional custom-estimator package is not installed |
