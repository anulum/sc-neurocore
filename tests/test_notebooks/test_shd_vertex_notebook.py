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
NOTEBOOK = REPO / "notebooks/30_shd_vertex_deployable_evidence.ipynb"


def _load_notebook() -> dict[str, Any]:
    return json.loads(NOTEBOOK.read_text(encoding="utf-8"))


def test_shd_vertex_notebook_declares_partial_artifact_boundary() -> None:
    notebook = _load_notebook()
    sources = ["".join(cell.get("source", [])) for cell in notebook["cells"]]
    text = "\n".join(sources)

    assert "Evidence Boundary" in text
    assert "reports only artifact directories present" in text
    assert "does not claim that every intended Vertex seed has been downloaded" in text
    assert "sc-neurocore.shd-vertex-notebook-evidence.v1" in text
    assert all(cell.get("outputs", []) == [] for cell in notebook["cells"] if cell["cell_type"] == "code")
    assert all(cell.get("execution_count") is None for cell in notebook["cells"] if cell["cell_type"] == "code")


def test_shd_vertex_notebook_code_executes_against_available_artifacts() -> None:
    notebook = _load_notebook()
    namespace: dict[str, Any] = {"__name__": "__notebook_test__"}

    for cell in notebook["cells"]:
        if cell["cell_type"] != "code":
            continue
        source = "".join(cell["source"])
        exec(compile(source, str(NOTEBOOK), "exec"), namespace)

    manifest = namespace["manifest"]
    aggregate = manifest["aggregate"]
    assert manifest["schema_version"] == "sc-neurocore.shd-vertex-notebook-evidence.v1"
    assert aggregate["available_runs"] >= 1
    assert aggregate["available_runs"] == len(manifest["runs"])
    assert all(0.0 <= run["deployable_test"] <= 100.0 for run in manifest["runs"])
    assert isinstance(aggregate["missing_expected_seeds"], list)
