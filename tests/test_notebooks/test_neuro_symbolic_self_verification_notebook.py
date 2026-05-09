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
NOTEBOOK = REPO / "notebooks/37_neuro_symbolic_self_verification_evidence.ipynb"


def _load_notebook() -> dict[str, Any]:
    return json.loads(NOTEBOOK.read_text(encoding="utf-8"))


def test_neuro_symbolic_notebook_declares_boundary_and_has_no_saved_outputs() -> None:
    notebook = _load_notebook()
    sources = ["".join(cell.get("source", [])) for cell in notebook["cells"]]
    text = "\n".join(sources)

    assert "Evidence Boundary" in text
    assert "verifies internal consistency" in text
    assert "does not prove that the symbolic interpretation is externally true" in text
    assert "sc-neurocore.neuro-symbolic-self-verification-evidence.v1" in text
    assert all(
        cell.get("outputs", []) == [] for cell in notebook["cells"] if cell["cell_type"] == "code"
    )
    assert all(
        cell.get("execution_count") is None
        for cell in notebook["cells"]
        if cell["cell_type"] == "code"
    )


def test_neuro_symbolic_notebook_code_executes() -> None:
    notebook = _load_notebook()
    namespace: dict[str, Any] = {"__name__": "__notebook_test__"}

    for cell in notebook["cells"]:
        if cell["cell_type"] != "code":
            continue
        source = "".join(cell["source"])
        exec(compile(source, str(NOTEBOOK), "exec"), namespace)

    manifest = namespace["manifest"]
    assert manifest["schema_version"] == "sc-neurocore.neuro-symbolic-self-verification-evidence.v1"
    assert manifest["pass_summary"]["passed"] is True
    assert manifest["pass_summary"]["digest_length"] == 64
    assert manifest["determinism_summary"]["same_payload"] is True
    assert "sc_signature_consistency" in manifest["tamper_summary"]["failed_obligations"]
    assert "symbol_score_ordering" in manifest["symbol_summary"]["failed_obligations"]
    assert "one-dimensional" in manifest["guardrails"]["shape_refusal"]
