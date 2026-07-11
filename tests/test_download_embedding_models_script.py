from __future__ import annotations

import importlib.util
from pathlib import Path
import sys


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
