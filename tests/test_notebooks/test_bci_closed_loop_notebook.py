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
NOTEBOOK = REPO / "notebooks/35_bci_closed_loop_evidence.ipynb"


def _load_notebook() -> dict[str, Any]:
    return json.loads(NOTEBOOK.read_text(encoding="utf-8"))


def test_bci_closed_loop_notebook_declares_boundary_and_has_no_saved_outputs() -> None:
    notebook = _load_notebook()
    sources = ["".join(cell.get("source", [])) for cell in notebook["cells"]]
    text = "\n".join(sources)

    assert "Evidence Boundary" in text
    assert "synthetic waveform windows and an in-process implant emulator only" in text
    assert "does not claim clinical safety" in text
    assert "sc-neurocore.bci-closed-loop-evidence.v1" in text
    assert all(cell.get("outputs", []) == [] for cell in notebook["cells"] if cell["cell_type"] == "code")
    assert all(cell.get("execution_count") is None for cell in notebook["cells"] if cell["cell_type"] == "code")


def test_bci_closed_loop_notebook_code_executes() -> None:
    notebook = _load_notebook()
    namespace: dict[str, Any] = {"__name__": "__notebook_test__"}

    for cell in notebook["cells"]:
        if cell["cell_type"] != "code":
            continue
        source = "".join(cell["source"])
        exec(compile(source, str(NOTEBOOK), "exec"), namespace)

    manifest = namespace["manifest"]
    assert manifest["schema_version"] == "sc-neurocore.bci-closed-loop-evidence.v1"
    assert manifest["closed_loop_summary"]["detected_spikes"] == 3
    assert manifest["closed_loop_summary"]["aer_magic"] == "AERX"
    assert manifest["decoder_sink_summary"]["clipped_feedback"] == [0.5, 1.0, -1.0]
    assert manifest["reference_manifest_summary"]["hardware_required"] is False
    assert manifest["reference_manifest_summary"]["no_stimulation_without_sink_override"] is True
    assert "expected 4" in manifest["guardrails"]["channel_mismatch_refusal"]
