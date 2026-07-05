"""Benchmark probe training pipeline to identify CPU vs GPU bottlenecks.

Simulates the ESM2-3B secondary_structure scenario:
  - 8678 proteins, mean length 200, hidden_dim 2560
  - 3-class multiclass probe (H/E/C secondary structure)
  - batch_size 8192, 5 epochs

Sections
--------
1. H2D transfer variants (CPU / GPU blocking / GPU pinned)
2. P1 — feature-stats caching: once-per-layer vs once-per-(seed,layer)
3. P2 — multiclass_metrics: O(N) confusion matrix vs O(class_count × N) triple scan

Run with:
    python scripts/benchmark_probe_training.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# ── synthetic dataset parameters matching ESM2-3B secondary_structure ─────────

N_PROTEINS = 8678
MEAN_LENGTH = 200
LENGTH_STD = 80
MIN_LENGTH = 20
MAX_LENGTH = 768
HIDDEN_DIM = 2560
N_CLASSES = 3
BATCH_SIZE = 8192
EPOCHS = 5          # enough to get stable timing without waiting forever
SEED = 42

# ── helpers ───────────────────────────────────────────────────────────────────

def _make_synthetic_dataset(
    n: int,
    *,
    mean_len: int,
    std_len: int,
    min_len: int,
    max_len: int,
    hidden_dim: int,
    n_classes: int,
    rng: np.random.Generator,
) -> Tuple[List[str], Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    ids = [f"P{i:05d}" for i in range(n)]
    lengths = rng.normal(mean_len, std_len, size=n).astype(int).clip(min_len, max_len)
    embeddings: Dict[str, np.ndarray] = {}
    labels: Dict[str, np.ndarray] = {}
    for pid, length in zip(ids, lengths):
        embeddings[pid] = rng.standard_normal((length, hidden_dim)).astype(np.float32)
        labels[pid] = rng.integers(0, n_classes, size=length)
    return ids, embeddings, labels


def _iter_batches(
    ids: Sequence[str],
    embeddings: Dict[str, np.ndarray],
    batch_size: int,
) -> Any:
    """Mirrors _iter_residue_batches from probes.py (no masking for simplicity)."""
    pending: List[torch.Tensor] = []
    pending_count = 0
    for pid in ids:
        tensor = torch.as_tensor(embeddings[pid], dtype=torch.float32)
        rows = int(tensor.shape[0])
        start = 0
        while start < rows:
            available = batch_size - pending_count
            end = min(rows, start + available)
            pending.append(tensor[start:end])
            pending_count += end - start
            start = end
            if pending_count >= batch_size:
                yield torch.cat(pending, dim=0)
                pending = []
                pending_count = 0
    if pending_count:
        yield torch.cat(pending, dim=0)


def _time_event_pair(device: torch.device) -> Tuple[Any, Any]:
    if device.type == "cuda":
        return torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    return None, None


def _elapsed_ms(start_event: Any, end_event: Any, *, wall_start: float, wall_end: float) -> float:
    if start_event is not None:
        start_event.synchronize()
        end_event.synchronize()
        return start_event.elapsed_time(end_event)
    return (wall_end - wall_start) * 1000.0


# ── benchmark runner ───────────────────────────────────────────────────────────

def _make_model_and_stats(
    device: torch.device,
    ids: List[str],
    embeddings: Dict[str, np.ndarray],
) -> Tuple[torch.nn.Module, torch.optim.Optimizer, torch.nn.Module, Any, Any, int]:
    model = torch.nn.Linear(HIDDEN_DIM, N_CLASSES).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = torch.nn.CrossEntropyLoss()
    feat_sum = torch.zeros(1, HIDDEN_DIM)
    feat_sumsq = torch.zeros(1, HIDDEN_DIM)
    n_residues = 0
    for batch in _iter_batches(ids, embeddings, BATCH_SIZE):
        feat_sum += batch.sum(dim=0, keepdim=True)
        feat_sumsq += (batch * batch).sum(dim=0, keepdim=True)
        n_residues += batch.shape[0]
    feature_mean = (feat_sum / n_residues).to(device)
    feature_std = torch.clamp(
        torch.sqrt(torch.clamp(feat_sumsq / n_residues - (feat_sum / n_residues) ** 2, min=0.0)),
        min=1e-6,
    ).to(device)
    return model, optimizer, loss_fn, feature_mean, feature_std, n_residues


def run_benchmark(
    device: torch.device,
    label: str,
    ids: List[str],
    embeddings: Dict[str, np.ndarray],
    *,
    use_pinned: bool = False,
) -> float:
    """Run training loop and return wall time in seconds."""
    print(f"\n{'─' * 60}")
    print(f"  {label}")
    print(f"{'─' * 60}")

    model, optimizer, loss_fn, feature_mean, feature_std, n_residues = _make_model_and_stats(
        device, ids, embeddings
    )

    x_pin: Any = None
    if use_pinned and device.type == "cuda":
        try:
            x_pin = torch.empty(BATCH_SIZE, HIDDEN_DIM, dtype=torch.float32).pin_memory()
        except Exception:
            print("  WARNING: pin_memory() failed — falling back to blocking H2D")

    total_h2d_ms = 0.0
    total_compute_ms = 0.0
    n_batches = 0

    if device.type == "cuda":
        torch.cuda.synchronize()

    wall_total_start = time.perf_counter()

    for epoch in range(EPOCHS):
        model.train()
        rng_labels = np.random.default_rng(SEED + epoch)

        for batch_cpu in _iter_batches(ids, embeddings, BATCH_SIZE):
            bsz = int(batch_cpu.shape[0])
            fake_labels = torch.from_numpy(rng_labels.integers(0, N_CLASSES, size=bsz).astype(np.int64))

            if device.type == "cuda":
                torch.cuda.synchronize()
                e_h2d_start = torch.cuda.Event(enable_timing=True)
                e_h2d_end = torch.cuda.Event(enable_timing=True)
                e_compute_end = torch.cuda.Event(enable_timing=True)
                e_h2d_start.record()

            # ── H2D transfer ──────────────────────────────────────────────────
            if x_pin is not None:
                x_pin[:bsz].copy_(batch_cpu)
                batch_gpu = x_pin[:bsz].to(device, non_blocking=True)
            else:
                batch_gpu = batch_cpu.to(device)
            labels_gpu = fake_labels.to(device)

            if device.type == "cuda":
                e_h2d_end.record()

            # ── GPU compute (standardize + forward + backward) ─────────────────
            batch_norm = (batch_gpu - feature_mean) / feature_std
            optimizer.zero_grad()
            logits = model(batch_norm)
            loss = loss_fn(logits, labels_gpu)
            loss.backward()
            optimizer.step()

            if device.type == "cuda":
                e_compute_end.record()
                torch.cuda.synchronize()
                total_h2d_ms += e_h2d_start.elapsed_time(e_h2d_end)
                total_compute_ms += e_h2d_end.elapsed_time(e_compute_end)

            n_batches += 1

    if device.type == "cuda":
        torch.cuda.synchronize()

    wall_total_ms = (time.perf_counter() - wall_total_start) * 1000.0

    print(f"  Epochs:          {EPOCHS}")
    print(f"  Batches:         {n_batches}  ({n_batches // EPOCHS}/epoch)")
    print(f"  Residues/epoch:  {n_residues:,}")
    print(f"  Wall time total: {wall_total_ms / 1000:.2f}s")
    print(f"  Per epoch:       {wall_total_ms / EPOCHS:.0f}ms")
    if device.type == "cuda":
        cpu_ms = wall_total_ms - total_h2d_ms - total_compute_ms
        print(f"  ── Phase breakdown (GPU-timed, {n_batches} batches) ──")
        print(f"     CPU assembly:  {cpu_ms:>8.1f}ms  ({100 * cpu_ms / wall_total_ms:.1f}%)")
        print(f"     H2D transfer:  {total_h2d_ms:>8.1f}ms  ({100 * total_h2d_ms / wall_total_ms:.1f}%)")
        print(f"     GPU compute:   {total_compute_ms:>8.1f}ms  ({100 * total_compute_ms / wall_total_ms:.1f}%)")
    print()
    return wall_total_ms / 1000.0


def _benchmark_feature_stats_caching(
    ids: List[str],
    embeddings: Dict[str, np.ndarray],
    *,
    n_layers: int = 37,
    n_seeds: int = 3,
    n_warmup: int = 2,
) -> None:
    """P1: compare one stats pass per (seed,layer) vs one pass per layer."""
    print(f"\n{'═' * 60}")
    print("  P1 — Feature stats caching")
    print(f"       {n_layers} layers × {n_seeds} seeds  →  {n_layers * n_seeds} vs {n_layers} passes")
    print(f"{'═' * 60}")

    train_ids = ids[: int(len(ids) * 0.8)]

    def _one_stats_pass() -> None:
        feat_sum = torch.zeros(1, HIDDEN_DIM)
        feat_sumsq = torch.zeros(1, HIDDEN_DIM)
        n = 0
        for batch in _iter_batches(train_ids, embeddings, BATCH_SIZE):
            feat_sum += batch.sum(dim=0, keepdim=True)
            feat_sumsq += (batch * batch).sum(dim=0, keepdim=True)
            n += batch.shape[0]
        _ = feat_sum / n
        _ = torch.clamp(torch.sqrt(torch.clamp(feat_sumsq / n - (feat_sum / n) ** 2, min=0.0)), min=1e-6)

    for _ in range(n_warmup):
        _one_stats_pass()

    t0 = time.perf_counter()
    for _ in range(n_layers * n_seeds):
        _one_stats_pass()
    uncached_s = time.perf_counter() - t0

    t0 = time.perf_counter()
    for _ in range(n_layers):
        _one_stats_pass()
    cached_s = time.perf_counter() - t0

    train_residues = sum(embeddings[pid].shape[0] for pid in train_ids)
    print(f"  Train residues:      {train_residues:,}")
    print(f"  Without caching:     {uncached_s:.2f}s  ({n_layers * n_seeds} passes)")
    print(f"  With caching:        {cached_s:.2f}s  ({n_layers} passes)")
    print(f"  Speedup:             {uncached_s / cached_s:.2f}×  "
          f"(saves {uncached_s - cached_s:.2f}s per sweep)")


def _benchmark_multiclass_metrics(*, n_test_residues: int = 350_000, n_classes: int = 3) -> None:
    """P2: O(N) confusion matrix vs O(class_count × N) triple scan."""
    print(f"\n{'═' * 60}")
    print("  P2 — multiclass_metrics O(N) vs O(class_count × N)")
    print(f"       {n_test_residues:,} residues, {n_classes} classes")
    print(f"{'═' * 60}")

    rng = np.random.default_rng(SEED)
    y_true = rng.integers(0, n_classes, size=n_test_residues).tolist()
    y_pred = rng.integers(0, n_classes, size=n_test_residues).tolist()

    def _old_triple_scan(y_true: list, y_pred: list, *, class_count: int) -> Dict[str, float]:
        correct = sum(t == p for t, p in zip(y_true, y_pred))
        accuracy = float(correct) / float(len(y_true))
        f1_values = []
        for c in range(class_count):
            tp = sum(1 for t, p in zip(y_true, y_pred) if t == c and p == c)
            fp = sum(1 for t, p in zip(y_true, y_pred) if t != c and p == c)
            fn = sum(1 for t, p in zip(y_true, y_pred) if t == c and p != c)
            precision = float(tp) / float(tp + fp) if tp + fp else 0.0
            recall = float(tp) / float(tp + fn) if tp + fn else 0.0
            f1_values.append(2.0 * precision * recall / (precision + recall) if precision + recall else 0.0)
        return {"accuracy": accuracy, "macro_f1": sum(f1_values) / float(len(f1_values))}

    def _new_single_pass(y_true: list, y_pred: list, *, class_count: int) -> Dict[str, float]:
        conf = [[0] * class_count for _ in range(class_count)]
        for t, p in zip(y_true, y_pred):
            conf[int(t)][int(p)] += 1
        correct = sum(conf[c][c] for c in range(class_count))
        accuracy = float(correct) / float(len(y_true))
        f1_values = []
        for c in range(class_count):
            tp = conf[c][c]
            fp = sum(conf[r][c] for r in range(class_count)) - tp
            fn = sum(conf[c][r] for r in range(class_count)) - tp
            precision = float(tp) / float(tp + fp) if tp + fp else 0.0
            recall = float(tp) / float(tp + fn) if tp + fn else 0.0
            f1_values.append(2.0 * precision * recall / (precision + recall) if precision + recall else 0.0)
        return {"accuracy": accuracy, "macro_f1": sum(f1_values) / float(len(f1_values))}

    # warmup
    _old_triple_scan(y_true[:1000], y_pred[:1000], class_count=n_classes)
    _new_single_pass(y_true[:1000], y_pred[:1000], class_count=n_classes)

    t0 = time.perf_counter()
    old_result = _old_triple_scan(y_true, y_pred, class_count=n_classes)
    old_s = time.perf_counter() - t0

    t0 = time.perf_counter()
    new_result = _new_single_pass(y_true, y_pred, class_count=n_classes)
    new_s = time.perf_counter() - t0

    assert abs(old_result["accuracy"] - new_result["accuracy"]) < 1e-9, "accuracy mismatch"
    assert abs(old_result["macro_f1"] - new_result["macro_f1"]) < 1e-9, "macro_f1 mismatch"

    print(f"  Old O(class × N):    {old_s * 1000:.1f}ms")
    print(f"  New O(N):            {new_s * 1000:.1f}ms")
    print(f"  Speedup:             {old_s / new_s:.2f}×")
    print(f"  Results match:       accuracy={new_result['accuracy']:.4f}  macro_f1={new_result['macro_f1']:.4f}")


def main() -> None:
    print(f"PyTorch {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"Device: {torch.cuda.get_device_name(0)}")

    print(f"\nSynthetic dataset: {N_PROTEINS} proteins, mean length {MEAN_LENGTH}, "
          f"hidden_dim {HIDDEN_DIM}, {N_CLASSES} classes")
    print(f"Batch size: {BATCH_SIZE}, epochs: {EPOCHS}")

    rng = np.random.default_rng(SEED)
    ids, embeddings, _ = _make_synthetic_dataset(
        N_PROTEINS,
        mean_len=MEAN_LENGTH,
        std_len=LENGTH_STD,
        min_len=MIN_LENGTH,
        max_len=MAX_LENGTH,
        hidden_dim=HIDDEN_DIM,
        n_classes=N_CLASSES,
        rng=rng,
    )

    total_residues = sum(e.shape[0] for e in embeddings.values())
    total_gb = sum(e.nbytes for e in embeddings.values()) / 1e9
    print(f"Total residues: {total_residues:,}  ({total_gb:.2f} GB numpy float32 in RAM)")

    cpu_s = run_benchmark(torch.device("cpu"), "CPU baseline", ids, embeddings)

    if not torch.cuda.is_available():
        print("No CUDA device — skipping GPU benchmarks.")
    else:
        gpu_blocking_s = run_benchmark(
            torch.device("cuda"), "GPU  —  blocking H2D (pageable memory)", ids, embeddings, use_pinned=False
        )
        gpu_pinned_s = run_benchmark(
            torch.device("cuda"), "GPU  —  pinned memory + non_blocking H2D", ids, embeddings, use_pinned=True
        )

        print("=" * 60)
        print("  H2D Summary")
        print("=" * 60)
        print(f"  CPU baseline:            {cpu_s:.2f}s")
        print(f"  GPU blocking H2D:        {gpu_blocking_s:.2f}s  ({cpu_s / gpu_blocking_s:.2f}× vs CPU)")
        print(f"  GPU pinned non_blocking:  {gpu_pinned_s:.2f}s  ({cpu_s / gpu_pinned_s:.2f}× vs CPU, "
              f"{gpu_blocking_s / gpu_pinned_s:.2f}× vs blocking)")
        print()

    _benchmark_feature_stats_caching(ids, embeddings)
    _benchmark_multiclass_metrics()


if __name__ == "__main__":
    main()
