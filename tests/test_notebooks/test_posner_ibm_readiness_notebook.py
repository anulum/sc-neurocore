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
NOTEBOOK = REPO / "notebooks/32_posner_ibm_readiness_evidence.ipynb"


def _load_notebook() -> dict[str, Any]:
    return json.loads(NOTEBOOK.read_text(encoding="utf-8"))


def test_posner_readiness_notebook_declares_boundary_and_has_no_saved_outputs() -> None:
    notebook = _load_notebook()
    sources = ["".join(cell.get("source", [])) for cell in notebook["cells"]]
    text = "\n".join(sources)

    assert "Evidence Boundary" in text
    assert "does not claim that ORCA-derived `hf.json`" in text
    assert "sc-neurocore.posner-ibm-readiness-evidence.v1" in text
    assert "minimum planned QPU shot budget" in text
    assert all(
        cell.get("outputs", []) == [] for cell in notebook["cells"] if cell["cell_type"] == "code"
    )
    assert all(
        cell.get("execution_count") is None
        for cell in notebook["cells"]
        if cell["cell_type"] == "code"
    )


def test_posner_readiness_notebook_code_executes_and_refuses_incomplete_runtime_data() -> None:
    notebook = _load_notebook()
    namespace: dict[str, Any] = {"__name__": "__notebook_test__"}

    for cell in notebook["cells"]:
        if cell["cell_type"] != "code":
            continue
        source = "".join(cell["source"])
        exec(compile(source, str(NOTEBOOK), "exec"), namespace)

    manifest = namespace["manifest"]
    gates = manifest["gates"]
    budget = manifest["qpu_budget_estimate"]

    assert manifest["schema_version"] == "sc-neurocore.posner-ibm-readiness-evidence.v1"
    assert gates["runtime_validator_rejects_incomplete_json"] is True
    assert gates["verification_runner_requires_hf_json"] is True
    assert gates["verification_runner_requires_extended_json_for_submission"] is True
    assert budget["minimum_shot_circuits"] == budget["circuit_count"] * budget["shots_per_circuit"]
    assert budget["minimum_shot_circuits"] == 20480
