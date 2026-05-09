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
NOTEBOOK = REPO / "notebooks/39_self_hosted_hub_evidence.ipynb"


def _load_notebook() -> dict[str, Any]:
    return json.loads(NOTEBOOK.read_text(encoding="utf-8"))


def test_self_hosted_hub_notebook_declares_boundary_and_has_no_saved_outputs() -> None:
    notebook = _load_notebook()
    sources = ["".join(cell.get("source", [])) for cell in notebook["cells"]]
    text = "\n".join(sources)

    assert "Evidence Boundary" in text
    assert "does not start Docker" in text
    assert "sc-neurocore.self-hosted-hub-evidence.v1" in text
    assert all(
        cell.get("outputs", []) == [] for cell in notebook["cells"] if cell["cell_type"] == "code"
    )
    assert all(
        cell.get("execution_count") is None
        for cell in notebook["cells"]
        if cell["cell_type"] == "code"
    )


def test_self_hosted_hub_notebook_code_executes() -> None:
    notebook = _load_notebook()
    namespace: dict[str, Any] = {"__name__": "__notebook_test__"}

    for cell in notebook["cells"]:
        if cell["cell_type"] != "code":
            continue
        source = "".join(cell["source"])
        exec(compile(source, str(NOTEBOOK), "exec"), namespace)

    manifest = namespace["manifest_evidence"]
    assert manifest["schema_version"] == "sc-neurocore.self-hosted-hub-evidence.v1"
    assert manifest["model_zoo_summary"]["plugins"] == [
        "AdEx",
        "Hodgkin-Huxley",
        "Izhikevich",
        "LIF",
    ]
    assert manifest["manifest_summary"]["ingress_scope"] == "loopback"
    assert manifest["manifest_summary"]["external_egress_required"] is False
    assert manifest["bundle_summary"]["studio_read_only"] is True
    assert manifest["bundle_summary"]["benchmark_profiles"] == ["benchmark"]
    assert any("bind_host" in key for key in manifest["guardrails"])
