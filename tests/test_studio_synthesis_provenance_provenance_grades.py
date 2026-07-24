# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Focused free-test suite (provenance_grades) from former test_studio_synthesis_provenance.py

from __future__ import annotations

from tests.studio_synthesis_provenance_support import *  # noqa: F403

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
