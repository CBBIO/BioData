from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any

import pytest


_SCRIPT_PATH = Path(__file__).resolve().parent.parent / "scripts" / "benchmark_neighbor_search.py"
_SPEC = importlib.util.spec_from_file_location("benchmark_neighbor_search", _SCRIPT_PATH)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)


def test_parse_args_accepts_ann_options(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "benchmark_neighbor_search.py",
            "--k",
            "10",
            "--use-ann",
            "--ann-ef-search",
            "300",
            "--ann-candidate-pool",
            "500",
        ],
    )

    args = _MODULE._parse_args()

    assert args.use_ann
    assert args.ann_ef_search == 300
    assert args.ann_candidate_pool == 500


def test_parse_args_rejects_ann_candidate_pool_smaller_than_k(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        ["benchmark_neighbor_search.py", "--k", "10", "--ann-candidate-pool", "9"],
    )

    with pytest.raises(SystemExit, match="2"):
        _MODULE._parse_args()


def test_run_backend_once_passes_ann_options_to_client() -> None:
    captured: dict[str, Any] = {}

    class _Client:
        last_search_diagnostics: dict[str, Any] = {}

        def find_nearest_neighbors_for_proteins(self, protein_ids: list[str], **kwargs: Any) -> dict[str, list[Any]]:
            captured["protein_ids"] = protein_ids
            captured.update(kwargs)
            return {protein_id: [] for protein_id in protein_ids}

    result = _MODULE._run_backend_once(
        _Client(),
        backend="pgvector",
        protein_ids=["P12345"],
        embedding_type_id=3,
        layer_index=0,
        metric="cosine",
        k=10,
        device=None,
        use_ann=True,
        ann_ef_search=300,
        ann_candidate_pool=500,
    )

    assert captured["protein_ids"] == ["P12345"]
    assert captured["use_ann"]
    assert captured["ann_ef_search"] == 300
    assert captured["ann_candidate_pool"] == 500
    assert result["neighbor_rows"] == 0
