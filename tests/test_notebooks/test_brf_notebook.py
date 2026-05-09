# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


REPO = Path(__file__).resolve().parents[2]
NOTEBOOK = REPO / "notebooks/31_balanced_resonate_and_fire_evidence.ipynb"


def _load_notebook() -> dict[str, Any]:
    return json.loads(NOTEBOOK.read_text(encoding="utf-8"))


def test_brf_notebook_declares_scientific_boundary_and_has_no_saved_outputs() -> None:
    notebook = _load_notebook()
    sources = ["".join(cell.get("source", [])) for cell in notebook["cells"]]
    text = "\n".join(sources)

    assert "Evidence Boundary" in text
    assert "does not claim reproduction of the full BRF-RSNN ICML training experiments" in text
    assert "sc-neurocore.brf-evidence.v1" in text
    assert all(cell.get("outputs", []) == [] for cell in notebook["cells"] if cell["cell_type"] == "code")
    assert all(cell.get("execution_count") is None for cell in notebook["cells"] if cell["cell_type"] == "code")


def test_brf_notebook_code_executes_against_committed_artifacts() -> None:
    notebook = _load_notebook()
    namespace: dict[str, Any] = {"__name__": "__notebook_test__"}

    for cell in notebook["cells"]:
        if cell["cell_type"] != "code":
            continue
        source = "".join(cell["source"])
        exec(compile(source, str(NOTEBOOK), "exec"), namespace)

    manifest = namespace["manifest"]
    assert manifest["schema_version"] == "sc-neurocore.brf-evidence.v1"
    assert manifest["benchmark_artifact"]["schema_version"] == 1
    assert manifest["one_step"]["q"] >= 0.0
    assert manifest["boundary"]["invalid_boundary_guard"] == "dt * omega > 1 raises ValueError"
    assert manifest["benchmark_artifact"]["step_ns"]["python_step_ns"] > 0.0
