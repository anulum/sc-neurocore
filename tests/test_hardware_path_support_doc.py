# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — The tracked hardware-path supported set is the code's derivation

"""The supported-set document is derived from the capability matrix and kept current.

Deriving the whole catalogue computes every model's numeric contracts, which
takes minutes, so the derivation runs once for this module.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

from sc_neurocore.neurons.model_registry import MODEL_MODULES
from sc_neurocore.neurons.readiness import REPO_ROOT
from sc_neurocore.studio.hardware_path import supported_models

TOOL = REPO_ROOT / "tools" / "hardware_path_support.py"


def _tool() -> ModuleType:
    spec = importlib.util.spec_from_file_location("hardware_path_support", TOOL)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def catalogue_rows() -> list[dict[str, Any]]:
    """Derive the supported set of the whole catalogue once."""
    return supported_models(tool_status={})


def test_the_tracked_document_is_the_derivation_now(catalogue_rows: list[dict[str, Any]]) -> None:
    """A model whose capabilities change fails here until the table is rewritten."""
    tool = _tool()
    rendered = tool.render_support(catalogue_rows)
    assert tool.document_problems(tool.DOCUMENT, rendered) == []


def test_every_catalogue_model_has_a_verdict(catalogue_rows: list[dict[str, Any]]) -> None:
    """Each registered model is judged once; a refused one names its operation and reason."""
    assert [row["model"] for row in catalogue_rows] == sorted(MODEL_MODULES)
    supported = [row for row in catalogue_rows if row["model_supported"]]
    assert {"LapicqueNeuron", "AdaptiveThresholdIFNeuron"} <= {row["model"] for row in supported}
    for row in supported:
        assert row["combinations"] and row["model_refusal"] is None
    for row in catalogue_rows:
        if not row["model_supported"]:
            assert row["model_refusal"]["operation"] in {"compile", "cosimulate"}
            assert row["model_refusal"]["reason"]


def test_the_tool_writes_and_checks_a_document(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """``--write`` then ``--check`` passes; an edited or missing document fails the check."""
    tool = _tool()
    output = tmp_path / "support.md"
    selection = ["--output", str(output), "--models", "LapicqueNeuron,ATypeKNeuron"]
    assert tool.main(["--write", *selection]) == 0
    text = output.read_text(encoding="utf-8")
    assert "Models: **2**; supported: **1**; refused: **1**." in text
    assert "| LapicqueNeuron | supported | map @ Q8.8, map @ Q16.16 |" in text
    assert tool.main(["--check", *selection]) == 0

    output.write_text(text.replace("supported: **1**", "supported: **2**"), encoding="utf-8")
    assert tool.main(["--check", *selection]) == 1
    assert "stale hardware path support table" in capsys.readouterr().err
    output.unlink()
    assert tool.main(["--check", *selection]) == 1
    assert "missing hardware path support table" in capsys.readouterr().err
