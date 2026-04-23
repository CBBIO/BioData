from __future__ import annotations

import importlib.util
import math
from pathlib import Path


_SCRIPT_PATH = Path(__file__).resolve().parent.parent / "scripts" / "benchmark_local_models_token_budgets.py"
_SPEC = importlib.util.spec_from_file_location("benchmark_local_models_token_budgets", _SCRIPT_PATH)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)


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


def test_setting_key_supports_baseline_and_token_budget() -> None:
    assert _MODULE._setting_key(batch_size=1, max_tokens_per_batch=None) == "1"
    assert _MODULE._setting_key(batch_size=None, max_tokens_per_batch=4096) == "tokens:4096"


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


def test_compare_to_baseline_summarizes_shared_ids_only() -> None:
    baseline = {"a": [1.0, 0.0], "b": [0.0, 1.0]}
    candidate = {"a": [1.0, 0.0], "b": [0.0, 1.0], "c": [5.0, 5.0]}
    summary = _MODULE._compare_to_baseline(baseline, candidate)
    assert summary["cosine_similarity"]["mean"] == 1.0
    assert summary["max_abs_diff"]["mean"] == 0.0


def test_prott5_is_included_in_supported_model_sets() -> None:
    assert "protT5" in _MODULE.DEFAULT_MODELS
    assert "prott5" in _MODULE._TRUE_BATCH_TOKEN_BUDGET_MODELS
