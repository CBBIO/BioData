"""Benchmark probe training pipeline to identify CPU vs GPU bottlenecks.

Simulates the ESM2-3B secondary_structure scenario:
  - 8678 proteins, mean length 200, hidden_dim 2560
  - 3-class multiclass probe (H/E/C secondary structure)
  - batch_size 8192, 5 epochs

Measures wall time for three variants:
  1. CPU only      — baseline
  2. GPU blocking  — batch_x.to("cuda")  (pageable → staging → DMA, blocks CPU)
  3. GPU pinned    — copy to pre-alloc pinned buf, then non_blocking=True
                     (direct DMA, CPU freed immediately to assemble next batch)

Run with:
    python benchmark_probe_training.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import torch

REPO_ROOT = Path(__file__).parent
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
        return

    gpu_blocking_s = run_benchmark(
        torch.device("cuda"), "GPU  —  blocking H2D (pageable memory)", ids, embeddings, use_pinned=False
    )
    gpu_pinned_s = run_benchmark(
        torch.device("cuda"), "GPU  —  pinned memory + non_blocking H2D", ids, embeddings, use_pinned=True
    )

    print("=" * 60)
    print("  Summary")
    print("=" * 60)
    print(f"  CPU baseline:           {cpu_s:.2f}s")
    print(f"  GPU blocking H2D:       {gpu_blocking_s:.2f}s  ({cpu_s / gpu_blocking_s:.2f}× vs CPU)")
    print(f"  GPU pinned non_blocking: {gpu_pinned_s:.2f}s  ({cpu_s / gpu_pinned_s:.2f}× vs CPU, "
          f"{gpu_blocking_s / gpu_pinned_s:.2f}× vs blocking)")
    print()


if __name__ == "__main__":
    main()
