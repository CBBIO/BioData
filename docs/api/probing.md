# Probing

```python
from CBBIO import LinearProbe, ProbeSpec

spec = ProbeSpec(kind="linear", epochs=100)
probe = LinearProbe(epochs=spec.epochs, learning_rate=spec.learning_rate)
```

## Core interfaces

::: CBBIO.Task

::: CBBIO.ProbeSpec

::: CBBIO.LinearProbe
    options:
      members: null

::: CBBIO.MlpProbe
    options:
      members: null

::: CBBIO.TransferProbe
    options:
      members: null

## Workflows

::: CBBIO.train_and_evaluate_probe

::: CBBIO.train_and_evaluate_residue_probe

::: CBBIO.run_task_on_layer

## Exceptions

| Exception | When raised |
|---|---|
| `EmbeddingDependencyError` | Optional probing dependency is unavailable |
