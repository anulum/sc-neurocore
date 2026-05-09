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
NOTEBOOK = REPO / "notebooks/38_formal_snn_verification_standard_evidence.ipynb"


def _load_notebook() -> dict[str, Any]:
    return json.loads(NOTEBOOK.read_text(encoding="utf-8"))


def test_formal_snn_standard_notebook_declares_boundary_and_has_no_saved_outputs() -> None:
    notebook = _load_notebook()
    sources = ["".join(cell.get("source", [])) for cell in notebook["cells"]]
    text = "\n".join(sources)

    assert "Evidence Boundary" in text
    assert "does not generate an external theorem-prover proof" in text
    assert "sc-neurocore.formal-snn-verification-standard-evidence.v1" in text
    assert all(cell.get("outputs", []) == [] for cell in notebook["cells"] if cell["cell_type"] == "code")
    assert all(cell.get("execution_count") is None for cell in notebook["cells"] if cell["cell_type"] == "code")


def test_formal_snn_standard_notebook_code_executes() -> None:
    notebook = _load_notebook()
    namespace: dict[str, Any] = {"__name__": "__notebook_test__"}

    for cell in notebook["cells"]:
        if cell["cell_type"] != "code":
            continue
        source = "".join(cell["source"])
        exec(compile(source, str(NOTEBOOK), "exec"), namespace)

    manifest = namespace["manifest"]
    assert manifest["schema_version"] == "sc-neurocore.formal-snn-verification-standard-evidence.v1"
    assert manifest["profile_summary"]["profile_id"] == "publication-grade-snn-v1"
    assert manifest["passing_summary"]["passed"] is True
    assert "external_formal_proof" in manifest["missing_external_summary"]["missing_mandatory"]
    assert "implementation_equivalence" in manifest["failed_equivalence_summary"]["failed_mandatory"]
    assert "external_formal_proof" in manifest["wrong_kind_summary"]["missing_mandatory"]
    assert "evidence_id" in manifest["guardrails"]["empty_id_refusal"]
