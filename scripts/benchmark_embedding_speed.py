#!/usr/bin/env python3
"""
Benchmark: residue-level vs pooled embedding speed with CBBIO.embedding.

Both modes go through _generate_from_batched_layer_output_map — the same
tokenisation and the same forward pass.  This script instruments each step
to show exactly where the wall-clock difference comes from.

Usage
-----
Edit the CONFIG block below to plug in your generator, then run:

    python scripts/benchmark_embedding_speed.py

The script prints two outputs:
  1. Black-box timing  – full generate() call for each mode.
  2. Step-level timing – manually times each internal sub-step to isolate
     where the overhead is (tokenise / infer / pool / materialise / record).
"""
from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Any, List, Sequence

from tqdm import tqdm

# ── CONFIG ───────────────────────────────────────────────────────────────────
REPO_ROOT         = Path(__file__).resolve().parent.parent
FASTA_PATH        = REPO_ROOT / "scripts" / "data" / "random_protein_ids_10000_lenle1000.fasta"
LAYER_INDEX       = -1      # last transformer layer; change as needed
BATCH_SIZE          = 2    # max proteins per forward pass
MAX_BATCH_TOKENS    = 32768/2  # mirrors notebook usage; None = fixed BATCH_SIZE only
MAX_PROTEINS        = 500   # None = all
MAX_SEQUENCE_LENGTH = 1024  # skip proteins longer than this — avoids O(L²) attention OOM

# Plug in your own generator here, or leave None to auto-build a demo one.
# Example:
#   import sys; sys.path.insert(0, "/home/icases/BioData")
#   from CBBIO.embeddings.factory import Generator
#   GENERATOR: Any = Generator(model_class="esm2", device="cuda")
GENERATOR: Any = None

# Used only when GENERATOR is None:
MODEL_CLASS = "prot_t5"
MODEL_NAME  = None  # smallest ESM2 — fast demo
DEVICE      = "cuda"
# ─────────────────────────────────────────────────────────────────────────────


# ── helpers ──────────────────────────────────────────────────────────────────

def _sync(device: str) -> None:
    """GPU barrier — no-op on CPU."""
    if "cuda" in str(device):
        try:
            import torch  # type: ignore
            torch.cuda.synchronize()
        except Exception:
            pass


class Timer:
    """Accumulates wall-clock seconds across multiple calls."""

    def __init__(self) -> None:
        self.total = 0.0
        self._t0: float | None = None

    def start(self) -> None:
        self._t0 = time.perf_counter()

    def stop(self) -> None:
        if self._t0 is not None:
            self.total += time.perf_counter() - self._t0
            self._t0 = None


def _fmt(seconds: float, n: int, n_batches: int) -> str:
    ms = seconds * 1e3
    return f"{ms:8.1f} ms  ({ms/n_batches:6.1f} ms/batch  {ms/n:.2f} ms/prot)"


def _data_volume(records: list, mode: str) -> str:
    total = sum(
        (r.shape[0] * r.shape[1] if len(r.shape) == 2 else r.shape[0]) * 4
        for r in records
    )
    if total >= 1_000_000:
        return f"{total/1e6:.1f} MB"
    return f"{total/1e3:.1f} kB"


# ── step-level benchmark ─────────────────────────────────────────────────────

def _iter_batches(inputs: list, batch_size: int, max_batch_tokens: int | None):
    from CBBIO.embeddings import batch_generation_inputs
    yield from batch_generation_inputs(iter(inputs), batch_size=batch_size, max_batch_tokens=max_batch_tokens)


def _count_batches(inputs: list, batch_size: int, max_batch_tokens: int | None) -> int:
    return sum(1 for _ in _iter_batches(inputs, batch_size, max_batch_tokens))


def run_step_benchmark(
    generator: Any,
    inputs: list,
    *,
    layer_index: Any,
    device: str,
    batch_size: int,
    max_batch_tokens: int | None = None,
) -> tuple[dict[str, dict[str, Timer]], int]:
    """
    Manually call each internal sub-step and time them separately.
    Returns a dict of Timer objects, keyed by step name, for each mode.
    """
    import sys, os
    sys.path.insert(0, str(REPO_ROOT))
    from CBBIO.embeddings import (                       # private helpers
        _sample_spans_from_model_output,
        _slice_batched_layer_tensor,
    )
    from CBBIO.embeddings.utils.pooler import (
        MeanPooler,
        materialize_embedding_payload,
        resolve_pooler,
    )

    preprocessor  = generator.preprocessor
    tokenizer     = generator.tokenizer
    model_adapter = generator.model

    mean_pooler = MeanPooler()

    step_names = ["preprocess", "tokenise", "infer", "slice", "pool", "materialise", "record"]
    timers: dict[str, dict[str, Timer]] = {
        "residue": {s: Timer() for s in step_names},
        "pooled":  {s: Timer() for s in step_names},
    }

    resolved_layer = layer_index
    n_batches = 0
    total_batches = _count_batches(inputs, batch_size, max_batch_tokens)

    with tqdm(total=total_batches, unit="batch", desc="step benchmark") as bar:
        for batch in _iter_batches(inputs, batch_size, max_batch_tokens):
            n_batches += 1
            bar.set_description(f"step benchmark  [batch {n_batches}/{total_batches}]")

            # ── shared steps: run ONCE per batch, results apply to both modes ──

            bar.set_postfix_str("preprocess")
            t_pre = Timer(); t_pre.start()
            prepared = [preprocessor.preprocess(r.sequence) for r in batch]
            t_pre.stop()

            bar.set_postfix_str("tokenise")
            t_tok = Timer(); t_tok.start()
            tokenized = tokenizer.tokenize_many(prepared)
            _sync(device)
            t_tok.stop()

            bar.set_postfix_str("infer ★")
            t_inf = Timer(); t_inf.start()
            _sync(device)
            model_output = model_adapter.infer(tokenized, layer_index=resolved_layer)
            _sync(device)
            t_inf.stop()

            # copy shared timings into both mode accumulators
            for mode in ("residue", "pooled"):
                timers[mode]["preprocess"].total += t_pre.total
                timers[mode]["tokenise"].total   += t_tok.total
                timers[mode]["infer"].total      += t_inf.total

            # ── mode-specific steps: both modes read the same GPU tensors ──
            model_output_map = model_output if isinstance(model_output, dict) else {}
            layers_source: dict = model_output_map.get("layers", {})
            spans, explicit_spans = _sample_spans_from_model_output(
                model_output_map, sample_count=len(batch)
            )

            for layer_id in sorted(int(k) for k in layers_source.keys()):
                layer_tensor = layers_source[layer_id]  # shared; do NOT pop yet

                for mode in ("residue", "pooled"):
                    T = timers[mode]

                    for row_index in range(len(batch)):
                        start, end = spans[row_index]

                        # 4 · slice  (view — no copy yet)
                        T["slice"].start()
                        sample_tensor = _slice_batched_layer_tensor(
                            layer_tensor,
                            row_index=row_index,
                            start=start,
                            end=end,
                            singleton_count=len(batch),
                            explicit_spans=explicit_spans,
                        )
                        T["slice"].stop()

                        # 5 · pool
                        bar.set_postfix_str(f"{mode}: pool{'↓' if mode=='pooled' else '―'}")
                        T["pool"].start()
                        if mode == "pooled":
                            payload = mean_pooler(sample_tensor)
                        else:
                            payload = sample_tensor
                        _sync(device)
                        T["pool"].stop()

                        # 6 · materialise  (GPU → CPU, .numpy())
                        bar.set_postfix_str(f"{mode}: materialise ◄")
                        T["materialise"].start()
                        embedding, shape = materialize_embedding_payload(payload)
                        T["materialise"].stop()

                        # 7 · record construction
                        T["record"].start()
                        _ = {
                            "id": batch[row_index].id,
                            "embedding": embedding,
                            "shape": shape,
                        }
                        T["record"].stop()

                # free the layer tensor after both modes are done with it
                del layer_tensor
                layers_source.pop(layer_id, None)

            bar.update(1)

    return timers, n_batches


# ── black-box benchmark ───────────────────────────────────────────────────────

def run_blackbox_benchmark(
    generator: Any,
    inputs: list,
    *,
    layer_index: Any,
    device: str,
    batch_size: int,
    max_batch_tokens: int | None = None,
    n_repeats: int = 1,
) -> tuple[dict[str, float], int]:
    results: dict[str, list[float]] = {"residue": [], "pooled": []}

    def _empty_cache() -> None:
        try:
            import torch; torch.cuda.empty_cache()
        except Exception:
            pass

    n_batches = _count_batches(inputs, batch_size, max_batch_tokens)
    for rep in range(n_repeats):
        for mode, pooler in (("residue", None), ("pooled", "mean")):
            _sync(device)
            _empty_cache()
            t0 = time.perf_counter()
            with tqdm(total=n_batches, desc=f"black-box [{rep+1}/{n_repeats}] {mode:<7}", unit="batch", leave=False) as bar:
                for batch in _iter_batches(inputs, batch_size, max_batch_tokens):
                    generator.generate(batch, layer_index=layer_index, pooler=pooler)
                    bar.update(1)
            _sync(device)
            results[mode].append(time.perf_counter() - t0)

    return {mode: min(times) for mode, times in results.items()}, n_batches


# ── main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    sys.path.insert(0, str(REPO_ROOT))
    from CBBIO.embeddings import load_fasta_inputs
    from CBBIO.embeddings.factory import Generator

    # ── build generator if not provided
    generator = GENERATOR
    if generator is None:
        print(f"Building generator: class={MODEL_CLASS!r}, name={MODEL_NAME!r}, device={DEVICE!r}")
        generator = Generator(model_class=MODEL_CLASS, name=MODEL_NAME, device=DEVICE)
    device = getattr(getattr(generator, "model_metadata", None), "device", DEVICE) or DEVICE

    # ── load and filter sequences
    all_inputs = load_fasta_inputs(FASTA_PATH)
    candidate = all_inputs[:MAX_PROTEINS] if MAX_PROTEINS else all_inputs
    if MAX_SEQUENCE_LENGTH:
        inputs  = [r for r in candidate if len(r.sequence) <= MAX_SEQUENCE_LENGTH]
        skipped = len(candidate) - len(inputs)
    else:
        inputs, skipped = candidate, 0
    seq_lens = [len(r.sequence) for r in inputs]
    n = len(inputs)

    # ── batch distribution analysis (done once, before any benchmarking)
    batch_sizes   = [len(b) for b in _iter_batches(inputs, BATCH_SIZE, MAX_BATCH_TOKENS)]
    batch_maxlens = [max(len(r.sequence) for r in b)
                     for b in _iter_batches(inputs, BATCH_SIZE, MAX_BATCH_TOKENS)]
    n_batches_total   = len(batch_sizes)
    singletons        = sum(1 for s in batch_sizes if s == 1)
    mean_batch_size   = sum(batch_sizes) / n_batches_total
    mean_maxlen       = sum(batch_maxlens) / n_batches_total
    fixed_n_batches   = -(-n // BATCH_SIZE)

    print(f"\nDataset          : {FASTA_PATH.name}")
    print(f"Proteins         : {n}  (skipped {skipped} > {MAX_SEQUENCE_LENGTH} residues)" if skipped else f"Proteins         : {n}")
    print(f"Lengths          : min={min(seq_lens)}  max={max(seq_lens)}  mean={sum(seq_lens)//n}")
    print(f"Device           : {device}")
    print(f"BATCH_SIZE       : {BATCH_SIZE}")
    print(f"MAX_BATCH_TOKENS : {MAX_BATCH_TOKENS if MAX_BATCH_TOKENS else 'None (fixed batches)'}")
    print(f"MAX_SEQ_LENGTH   : {MAX_SEQUENCE_LENGTH if MAX_SEQUENCE_LENGTH else 'None'}")
    print()
    print("  BATCH DISTRIBUTION")
    print(f"  Fixed-size batches (no token cap) : {fixed_n_batches}")
    print(f"  Actual batches with token cap     : {n_batches_total}  "
          f"({'same' if n_batches_total == fixed_n_batches else f'+{n_batches_total - fixed_n_batches} extra'})")
    print(f"  Mean proteins/batch               : {mean_batch_size:.1f}")
    print(f"  Singleton batches (size=1)        : {singletons} "
          f"({100*singletons/n_batches_total:.0f}%)")
    print(f"  Mean max-length within batch      : {mean_maxlen:.0f} residues")
    print()

    # ─────────────────────────────────────────────────────────────────────
    # BLACK-BOX TIMING
    # ─────────────────────────────────────────────────────────────────────
    print("=" * 62)
    print("  BLACK-BOX TIMING  (full generate() call)")
    print("=" * 62)
    bb, bb_batches = run_blackbox_benchmark(
        generator, inputs, layer_index=LAYER_INDEX, device=device,
        batch_size=BATCH_SIZE, max_batch_tokens=MAX_BATCH_TOKENS, n_repeats=1,
    )
    for mode in ("residue", "pooled"):
        print(f"  {mode:<8}  {_fmt(bb[mode], n, bb_batches)}")
    ratio = bb["residue"] / bb["pooled"] if bb["pooled"] > 0 else float("inf")
    print(f"\n  residue / pooled  speedup ratio : {ratio:.1f}×")

    # ─────────────────────────────────────────────────────────────────────
    # STEP-LEVEL TIMING
    # ─────────────────────────────────────────────────────────────────────
    # Release memory reserved by PyTorch allocator from the black-box run
    # before starting a new forward pass; avoids false OOM from fragmentation.
    try:
        import torch
        torch.cuda.empty_cache()
    except Exception:
        pass

    print()
    print("=" * 62)
    print("  STEP-LEVEL TIMING  (manually instrumented sub-steps)")
    print("=" * 62)
    timers, n_batches = run_step_benchmark(
        generator, inputs, layer_index=LAYER_INDEX, device=device,
        batch_size=BATCH_SIZE, max_batch_tokens=MAX_BATCH_TOKENS,
    )

    steps = ["preprocess", "tokenise", "infer", "slice", "pool", "materialise", "record"]
    labels = {
        "preprocess":   "preprocess  (strip, uppercase)",
        "tokenise":     "tokenise    (tokenize_many)",
        "infer":        "infer       ★ forward pass ★",
        "slice":        "slice       (tensor view, no copy)",
        "pool":         "pool        (mean dim=0 | identity)",
        "materialise":  "materialise (GPU→CPU + numpy)",
        "record":       "record      (dict/object build)",
    }
    col = 30
    print(f"\n  {'step':<35} {'residue':>{col}}   {'pooled':>{col}}")
    print(f"  {'-'*35} {'-'*col}   {'-'*col}")
    for step in steps:
        r = _fmt(timers["residue"][step].total, n, n_batches)
        p = _fmt(timers["pooled"][step].total, n, n_batches)
        shared_note = "  (shared)" if step in ("preprocess", "tokenise", "infer") else ""
        marker = " ◄" if step == "materialise" else ""
        print(f"  {labels[step]:<35} {r}   {p}{marker}{shared_note}")

    total_r = sum(timers["residue"][s].total for s in steps)
    total_p = sum(timers["pooled"][s].total for s in steps)
    print(f"  {'-'*35} {'-'*col}   {'-'*col}")
    print(f"  {'TOTAL (sum of steps)':<35} {_fmt(total_r, n, n_batches)}   {_fmt(total_p, n, n_batches)}")
    print(f"\n  Wall-clock total  residue : {total_r*1e3:.0f} ms  |  pooled : {total_p*1e3:.0f} ms"
          f"  |  ratio : {total_r/total_p:.1f}×")

    # ─────────────────────────────────────────────────────────────────────
    # DATA VOLUME ANALYSIS
    # ─────────────────────────────────────────────────────────────────────
    try:
        from CBBIO.embeddings import batch_generation_inputs
        sample_batch = next(iter(batch_generation_inputs(iter(inputs[:BATCH_SIZE]), batch_size=BATCH_SIZE)))
        model_out = generator.model.infer(
            generator.tokenizer.tokenize_many([generator.preprocessor.preprocess(r.sequence) for r in sample_batch]),
            layer_index=LAYER_INDEX,
        )
        layers_obj = model_out.get("layers", {})
        hidden_dim = next(iter(layers_obj.values())).shape[-1]
        n_layers = len(layers_obj)
    except Exception:
        hidden_dim = "?"
        n_layers = "?"

    total_residues = sum(seq_lens)
    residue_bytes = total_residues * (hidden_dim if isinstance(hidden_dim, int) else 0) * 4
    pooled_bytes  = n             * (hidden_dim if isinstance(hidden_dim, int) else 0) * 4

    print()
    print("=" * 62)
    print("  DATA VOLUME  (bytes moved GPU → CPU  per full run)")
    print("=" * 62)
    print(f"  Hidden dim             : {hidden_dim}")
    print(f"  Total residues (sum L) : {total_residues:,}")
    if isinstance(hidden_dim, int):
        print(f"  residue  : {residue_bytes/1e6:.1f} MB  ({total_residues:,} × {hidden_dim} × float32)")
        print(f"  pooled   : {pooled_bytes/1e3:.1f} kB  ({n} × {hidden_dim} × float32)")
        print(f"  ratio    : {residue_bytes//max(pooled_bytes,1):,}×  more data to transfer for residue")

    # ─────────────────────────────────────────────────────────────────────
    # EXPLANATION
    # ─────────────────────────────────────────────────────────────────────
    print()
    print("=" * 62)
    print("  WHY IS RESIDUE-LEVEL SLOWER?")
    print("=" * 62)
    print("""
  Both modes share the EXACT same code path:
    _generate_from_batched_layer_output_map()
    → one tokenize_many() call  (full batch)
    → one model.infer()         (one forward pass)

  The only difference is AFTER the forward pass:

  1. pool step
       pooled  → tensor.mean(dim=0) reduces [L, D] to [D] ON THE GPU
                 (cheap: D multiplies, no data movement yet)
       residue → identity, [L, D] tensor stays as-is

  2. materialise step  ← dominant cost
       pooled  → .detach().cpu().float().numpy() on shape (D,)
                   transfers D × 4 bytes per protein
       residue → same call on shape (L, D)
                   transfers L × D × 4 bytes per protein
                   = L times more data GPU→CPU per sequence

  3. memory pressure
       Storing (L, D) numpy arrays in EmbeddingRecords for 100 proteins
       can be hundreds of MB; allocation and GC pressure add up.

  Rule of thumb: if your mean sequence length is L, residue-level
  embedding collection will cost ~L× more time in materialise, even
  though the forward pass is identical.
    """)


if __name__ == "__main__":
    main()
