# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Studio synthesis provenance tests

"""Tests for Studio synthesis target provenance contracts."""

from __future__ import annotations

import hashlib
import json
import re
from typing import cast

import pytest
from starlette.testclient import TestClient

from sc_neurocore.studio.app import create_app
from sc_neurocore.studio.synthesis import run_synthesis
from sc_neurocore.studio.synthesis_provenance import (
    STUDIO_SYNTHESIS_TARGET_PROVENANCE_MATRIX_SCHEMA_VERSION,
    STUDIO_SYNTHESIS_TARGET_PROVENANCE_SCHEMA_VERSION,
    JsonValue,
    StudioSynthesisTargetProvenance,
    StudioSynthesisToolProvenance,
    ToolStatusMap,
    build_synthesis_target_provenance,
    build_synthesis_target_provenance_matrix,
)


@pytest.fixture
def client() -> TestClient:
    """Return a Studio test client."""

    return TestClient(create_app(), base_url="http://127.0.0.1")


def _tool_status() -> ToolStatusMap:
    """Return deterministic path-free EDA tool status."""

    return cast(
        ToolStatusMap,
        {
            "yosys": {"available": True, "version": "Yosys 0.test"},
            "nextpnr_ice40": {"available": False, "version": None},
            "nextpnr_ecp5": {"available": True, "version": "nextpnr-ecp5 test"},
            "firtool": {"available": False, "version": 123},
        },
    )


def test_build_synthesis_target_provenance_for_pnr_target() -> None:
    """Target provenance records synthesis and PnR tool state without paths."""

    provenance = build_synthesis_target_provenance(
        "ice40",
        target_config={"synth_cmd": "synth_ice40", "pnr": "nextpnr-ice40", "device": "up5k"},
        capacity={"luts": 5280, "ffs": 5280, "brams": 30, "dsps": 0},
        tool_status=_tool_status(),
    ).to_public_dict()

    assert provenance["schema_version"] == STUDIO_SYNTHESIS_TARGET_PROVENANCE_SCHEMA_VERSION
    assert provenance["target"] == "ice40"
    assert provenance["synthesis_command"] == "synth_ice40"
    assert provenance["synthesis_ready"] is True
    assert provenance["pnr_ready"] is False
    assert provenance["evidence_classification"] == "synthesis"
    assert provenance["status"] == "completed"
    assert "path" not in provenance
    tools = cast(list[dict[str, JsonValue]], provenance["tools"])
    assert tools == [
        {
            "available": True,
            "executable": "yosys",
            "key": "yosys",
            "role": "synthesis",
            "version": "Yosys 0.test",
        },
        {
            "available": False,
            "executable": "nextpnr-ice40",
            "key": "nextpnr_ice40",
            "role": "place_and_route",
            "version": None,
        },
    ]


def test_build_synthesis_target_provenance_for_synthesis_only_target() -> None:
    """Targets without PnR still record synthesis readiness."""

    provenance = build_synthesis_target_provenance(
        "gowin",
        target_config={"synth_cmd": "synth_gowin", "pnr": None, "device": None},
        capacity={"luts": 20736, "ffs": 20736, "brams": 41, "dsps": 0},
        tool_status=_tool_status(),
    ).to_public_dict()

    assert provenance["pnr_tool"] is None
    assert provenance["pnr_ready"] is True
    assert cast(list[dict[str, JsonValue]], provenance["tools"])[0]["version"] == "Yosys 0.test"


def test_build_synthesis_target_provenance_rejects_missing_command() -> None:
    """Target provenance fails closed when synthesis command metadata is absent."""

    with pytest.raises(ValueError, match="no synthesis command"):
        build_synthesis_target_provenance(
            "broken",
            target_config={"synth_cmd": None, "pnr": None, "device": None},
            capacity={},
            tool_status=_tool_status(),
        )


def test_synthesis_target_provenance_rejects_unknown_evidence_classification() -> None:
    """Synthesis provenance uses the shared Studio evidence-class contract."""

    provenance = StudioSynthesisTargetProvenance(
        target="ice40",
        capacity={"luts": 5280},
        synthesis_command="synth_ice40",
        pnr_tool=None,
        device=None,
        tools=(
            StudioSynthesisToolProvenance(
                key="yosys",
                executable="yosys",
                role="synthesis",
                available=True,
                version="Yosys 0.test",
            ),
        ),
        evidence_classification="screenshots",  # type: ignore[arg-type]  # Invalid by design.
    )

    with pytest.raises(ValueError, match="classification"):
        provenance.to_public_dict()


def test_synthesis_target_provenance_rejects_unknown_status() -> None:
    """Synthesis provenance uses the shared terminal-status contract."""

    provenance = StudioSynthesisTargetProvenance(
        target="ice40",
        capacity={"luts": 5280},
        synthesis_command="synth_ice40",
        pnr_tool=None,
        device=None,
        tools=(
            StudioSynthesisToolProvenance(
                key="yosys",
                executable="yosys",
                role="synthesis",
                available=True,
                version="Yosys 0.test",
            ),
        ),
        status="running",  # type: ignore[arg-type]  # Invalid by design.
    )

    with pytest.raises(ValueError, match="status"):
        provenance.to_public_dict()


def test_build_synthesis_target_provenance_matrix_has_stable_digest() -> None:
    """The target matrix has deterministic target membership and digest."""

    matrix = build_synthesis_target_provenance_matrix(
        targets={
            "ice40": {
                "synth_cmd": "synth_ice40",
                "pnr": "nextpnr-ice40",
                "device": "up5k",
            },
            "gowin": {"synth_cmd": "synth_gowin", "pnr": None, "device": None},
        },
        capacities={
            "ice40": {"luts": 5280, "ffs": 5280, "brams": 30, "dsps": 0},
            "gowin": {"luts": 20736, "ffs": 20736, "brams": 41, "dsps": 0},
        },
        tool_status=_tool_status(),
    )
    matrix_without_digest = {
        "evidence_classification": "synthesis",
        "provenance_grade": matrix["provenance_grade"],
        "schema_version": STUDIO_SYNTHESIS_TARGET_PROVENANCE_MATRIX_SCHEMA_VERSION,
        "status": "completed",
        "targets": matrix["targets"],
    }

    assert matrix["evidence_classification"] == "synthesis"
    assert matrix["schema_version"] == STUDIO_SYNTHESIS_TARGET_PROVENANCE_MATRIX_SCHEMA_VERSION
    assert matrix["status"] == "completed"
    assert set(cast(dict[str, JsonValue], matrix["targets"])) == {"ice40", "gowin"}
    assert (
        matrix["matrix_sha256"]
        == hashlib.sha256(
            json.dumps(
                matrix_without_digest,
                allow_nan=False,
                separators=(",", ":"),
                sort_keys=True,
            ).encode("utf-8")
        ).hexdigest()
    )


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


def test_target_provenance_grade_tool_backed_when_tools_available_and_versioned() -> None:
    grade = build_synthesis_target_provenance(
        "gowin",
        target_config={"synth_cmd": "synth_gowin", "pnr": None, "device": None},
        capacity={},
        tool_status=_tool_status(),
    ).provenance_grade

    assert grade == "tool_backed"


def test_target_provenance_grade_unverified_when_required_tool_missing() -> None:
    provenance = build_synthesis_target_provenance(
        "ice40",
        target_config={"synth_cmd": "synth_ice40", "pnr": "nextpnr-ice40", "device": "up5k"},
        capacity={},
        tool_status=_tool_status(),
    )

    assert provenance.provenance_grade == "unverified"
    assert provenance.to_public_dict()["provenance_grade"] == "unverified"


def test_target_provenance_grade_unverified_when_version_missing() -> None:
    tool_status = cast(ToolStatusMap, {"yosys": {"available": True, "version": None}})
    grade = build_synthesis_target_provenance(
        "gowin",
        target_config={"synth_cmd": "synth_gowin", "pnr": None, "device": None},
        capacity={},
        tool_status=tool_status,
    ).provenance_grade

    assert grade == "unverified"


def test_target_provenance_grade_unverified_when_no_tools() -> None:
    provenance = StudioSynthesisTargetProvenance(
        target="empty",
        capacity={},
        synthesis_command="synth",
        pnr_tool=None,
        device=None,
        tools=(),
    )

    assert provenance.provenance_grade == "unverified"


def test_matrix_provenance_grade_tool_backed_when_every_target_backed() -> None:
    matrix = build_synthesis_target_provenance_matrix(
        targets={
            "gowin": {"synth_cmd": "synth_gowin", "pnr": None, "device": None},
            "ecp5": {"synth_cmd": "synth_ecp5", "pnr": "nextpnr-ecp5", "device": "25k"},
        },
        capacities={},
        tool_status=_tool_status(),
    )

    assert matrix["provenance_grade"] == "tool_backed"


def test_matrix_provenance_grade_unverified_when_any_target_unverified() -> None:
    matrix = build_synthesis_target_provenance_matrix(
        targets={
            "gowin": {"synth_cmd": "synth_gowin", "pnr": None, "device": None},
            "ice40": {"synth_cmd": "synth_ice40", "pnr": "nextpnr-ice40", "device": "up5k"},
        },
        capacities={},
        tool_status=_tool_status(),
    )

    assert matrix["provenance_grade"] == "unverified"


def test_matrix_provenance_grade_unverified_when_empty() -> None:
    matrix = build_synthesis_target_provenance_matrix(
        targets={},
        capacities={},
        tool_status=_tool_status(),
    )

    assert matrix["provenance_grade"] == "unverified"
