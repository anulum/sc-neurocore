# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Focused free-test suite (target_build) from former test_studio_synthesis_provenance.py

from __future__ import annotations

from tests.studio_synthesis_provenance_support import *  # noqa: F403


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
