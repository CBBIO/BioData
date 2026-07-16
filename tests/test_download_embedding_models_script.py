from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import types


_SCRIPT_PATH = Path(__file__).resolve().parent.parent / "scripts" / "download_embedding_models.py"
_SPEC = importlib.util.spec_from_file_location("download_embedding_models", _SCRIPT_PATH)
if _SPEC is None or _SPEC.loader is None:
    raise RuntimeError(f"Could not load script module from {_SCRIPT_PATH}")
_MODULE = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _MODULE
_SPEC.loader.exec_module(_MODULE)


def test_default_model_download_targets_skip_esm1b_and_deduplicate_aliases() -> None:
    targets = _MODULE._iter_download_targets()
    pairs = [(target.model_class, target.model_reference) for target in targets]

    assert "esm1b" not in {target.model_class for target in targets}
    assert len(pairs) == len(set(pairs))
    assert ("esmc", "biohub/ESMC-300M") in pairs
    assert pairs.count(("esmc", "biohub/ESMC-300M")) == 1


def test_model_download_targets_can_filter_by_class() -> None:
    targets = _MODULE._iter_download_targets(["esm2"])

    assert targets
    assert {target.model_class for target in targets} == {"esm2"}


def test_dry_run_prints_download_plan(capsys) -> None:
    exit_code = _MODULE.main(["--model-class", "esm2", "--dry-run"])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "esm2\tfacebook/esm2_t6_8M_UR50D\tfacebook/esm2_t6_8M_UR50D" in captured.out


def test_download_target_falls_back_to_huggingface_hub(
    monkeypatch,
    tmp_path: Path,
) -> None:
    calls = []

    def _snapshot_download(**kwargs):
        calls.append(dict(kwargs))
        return str(tmp_path / "snapshot")

    monkeypatch.setitem(
        sys.modules,
        "huggingface_hub",
        types.SimpleNamespace(snapshot_download=_snapshot_download),
    )
    target = _MODULE.DownloadTarget(
        model_class="esm2",
        name="esm2_8m",
        model_reference="facebook/esm2_t6_8M_UR50D",
    )

    result = _MODULE._download_target(
        target,
        revision="main",
        cache_dir=tmp_path / "cache",
        local_dir=tmp_path / "local",
        token=None,
        force_download=True,
        download_generator_model_fn=None,
    )

    assert result.model_reference == "facebook/esm2_t6_8M_UR50D"
    assert result.path == tmp_path / "snapshot"
    assert calls == [
        {
            "repo_id": "facebook/esm2_t6_8M_UR50D",
            "revision": "main",
            "cache_dir": str(tmp_path / "cache"),
            "local_dir": str(tmp_path / "local"),
            "token": None,
            "force_download": True,
        }
    ]


def test_force_download_flag_reaches_download_helper(monkeypatch) -> None:
    calls = []

    def _download_generator_model(**kwargs):
        calls.append(dict(kwargs))
        return types.SimpleNamespace(model_reference="ok", path=None)

    monkeypatch.setattr(_MODULE, "_download_generator_model", _download_generator_model)
    exit_code = _MODULE.main(["--model-class", "esm2", "-f"])

    assert exit_code == 0
    assert calls
    assert all(call["force_download"] is True for call in calls)
