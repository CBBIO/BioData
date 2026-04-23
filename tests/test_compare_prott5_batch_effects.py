from __future__ import annotations

import importlib.util
import math
from pathlib import Path

from CBBIO.types import Neighbor


_SCRIPT_PATH = Path(__file__).resolve().parent.parent / "scripts" / "compare_prott5_batch_effects.py"
_SPEC = importlib.util.spec_from_file_location("compare_prott5_batch_effects", _SCRIPT_PATH)
if _SPEC is None or _SPEC.loader is None:
    raise RuntimeError(f"Could not load script module from {_SCRIPT_PATH}")
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)


def test_vector_diff_metrics_identity() -> None:
    metrics = _MODULE._vector_diff_metrics([1.0, 2.0], [1.0, 2.0])
    assert metrics["cosine_similarity"] == 1.0
    assert metrics["max_abs_diff"] == 0.0
    assert metrics["mean_abs_diff"] == 0.0
    assert metrics["l2_diff"] == 0.0


def test_vector_diff_metrics_detects_difference() -> None:
    metrics = _MODULE._vector_diff_metrics([1.0, 0.0], [0.0, 1.0])
    assert math.isclose(metrics["cosine_similarity"], 0.0, abs_tol=1e-12)
    assert metrics["max_abs_diff"] == 1.0
    assert metrics["mean_abs_diff"] == 1.0


def test_neighbor_overlap_metrics() -> None:
    baseline = {
        "q1": [Neighbor("A", 0, 0.1), Neighbor("B", 0, 0.2)],
        "q2": [Neighbor("C", 0, 0.1), Neighbor("D", 0, 0.2)],
    }
    candidate = {
        "q1": [Neighbor("A", 0, 0.1), Neighbor("B", 0, 0.2)],
        "q2": [Neighbor("D", 0, 0.2), Neighbor("C", 0, 0.1)],
    }
    metrics = _MODULE._neighbor_overlap_metrics(baseline, candidate, k=2)
    assert metrics["exact_top1_fraction"] == 0.5
    assert metrics["exact_topk_fraction"] == 0.5
    assert metrics["mean_overlap_at_k"] == 1.0
    assert metrics["mean_jaccard_at_k"] == 1.0


def test_sort_inputs_by_length_orders_shorter_first() -> None:
    records = [
        _MODULE.GenerationInput(id="b", sequence="AAAA"),
        _MODULE.GenerationInput(id="a", sequence="AA"),
        _MODULE.GenerationInput(id="c", sequence="AA"),
    ]
    ordered = _MODULE._sort_inputs_by_length(records)
    assert [record.id for record in ordered] == ["a", "c", "b"]


def test_chunked_by_token_budget_groups_to_length_budget() -> None:
    records = [
        _MODULE.GenerationInput(id="a", sequence="AA"),
        _MODULE.GenerationInput(id="b", sequence="AAA"),
        _MODULE.GenerationInput(id="c", sequence="AAAA"),
        _MODULE.GenerationInput(id="d", sequence="AAAAA"),
    ]
    chunks = list(_MODULE._chunked_by_token_budget(records, max_tokens_per_batch=8))
    assert [[record.id for record in chunk] for chunk in chunks] == [["a", "b"], ["c"], ["d"]]


def test_setting_key_supports_token_budget() -> None:
    assert _MODULE._setting_key(batch_size=16, max_tokens_per_batch=None) == "16"
    assert _MODULE._setting_key(batch_size=None, max_tokens_per_batch=2048) == "tokens:2048"
