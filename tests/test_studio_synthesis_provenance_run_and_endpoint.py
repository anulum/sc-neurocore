# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Focused free-test suite (run_and_endpoint) from former test_studio_synthesis_provenance.py

from __future__ import annotations

from tests.studio_synthesis_provenance_support import *  # noqa: F403

def test_run_synthesis_includes_target_provenance_when_yosys_is_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Synthesis failures still carry target provenance for operator evidence."""

    import sc_neurocore.studio.synthesis as synthesis

    def missing_tool(_name: str) -> None:
        return None

    monkeypatch.setattr(synthesis, "_resolve_eda_tool", missing_tool)

    result = run_synthesis(
        "module t(); endmodule",
        "ice40",
        tool_status=_tool_status(),
    )

    assert result["success"] is False
    provenance = result["target_provenance"]
    assert provenance["schema_version"] == STUDIO_SYNTHESIS_TARGET_PROVENANCE_SCHEMA_VERSION
    assert provenance["target"] == "ice40"
    assert provenance["status"] == "completed"
    assert provenance["synthesis_ready"] is True
    assert provenance["pnr_ready"] is False

def test_synthesis_endpoint_returns_target_provenance(client: TestClient) -> None:
    """The public synthesis endpoint returns path-free target provenance."""

    response = client.post(
        "/api/synth/run",
        json={"verilog": "module test(); endmodule", "target": "ice40"},
    )

    assert response.status_code == 200
    data = response.json()
    provenance = data["target_provenance"]
    assert provenance["schema_version"] == STUDIO_SYNTHESIS_TARGET_PROVENANCE_SCHEMA_VERSION
    assert provenance["target"] == "ice40"
    assert provenance["status"] == "completed"
    assert re.fullmatch(r"[a-z0-9_]+", provenance["evidence_classification"])
