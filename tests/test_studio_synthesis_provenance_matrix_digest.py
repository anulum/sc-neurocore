# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Focused free-test suite (matrix_digest) from former test_studio_synthesis_provenance.py

from __future__ import annotations

from tests.studio_synthesis_provenance_support import *  # noqa: F403


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
