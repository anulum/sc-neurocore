# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Focused free-test suite (silicon_axis) from former test_descriptor_tiers.py

from __future__ import annotations

from tests.descriptor_tiers_support import *  # noqa: F403

def test_silicon_tier_none_without_compile_clean_rtl() -> None:
    """No compile-clean RTL means no silicon evidence at all."""
    assert silicon_tier(parse_model_descriptor(_validated_payload())) is None
    assert silicon_tier(_silicon(compiles=False, cosim_validated=True)) is None


def test_silicon_ladder_climbs_one_rung_per_anchor() -> None:
    """Each rung is reached only when its flag and proof anchor are both present."""
    h0 = _silicon(compiles=True)
    assert silicon_tier(h0) == 0

    h1 = _silicon(compiles=True, cosim_validated=True, cosim_evidence="cosim.log")
    assert silicon_tier(h1) == 1

    h2 = _silicon(
        compiles=True,
        cosim_validated=True,
        cosim_evidence="cosim.log",
        synthesised=True,
        synth_report="yosys.json",
    )
    assert silicon_tier(h2) == 2

    h3 = _silicon(
        compiles=True,
        cosim_validated=True,
        cosim_evidence="cosim.log",
        synthesised=True,
        synth_report="yosys.json",
        timing_closed=True,
        timing_report="sta.rpt",
        clock_mhz=100.0,
    )
    assert silicon_tier(h3) == 3

    h4 = _silicon(
        compiles=True,
        cosim_validated=True,
        cosim_evidence="cosim.log",
        synthesised=True,
        synth_report="yosys.json",
        timing_closed=True,
        timing_report="sta.rpt",
        clock_mhz=100.0,
        formally_equivalent=True,
        equivalence_proof="miter.smt2",
    )
    assert silicon_tier(h4) == 4

    h5 = _silicon(
        compiles=True,
        cosim_validated=True,
        cosim_evidence="cosim.log",
        synthesised=True,
        synth_report="yosys.json",
        timing_closed=True,
        timing_report="sta.rpt",
        clock_mhz=100.0,
        formally_equivalent=True,
        equivalence_proof="miter.smt2",
        ppa_signed=True,
        ppa_report="openlane.json",
    )
    assert silicon_tier(h5) == 5


def test_silicon_rung_withheld_when_anchor_missing() -> None:
    """A flag without its proof anchor does not advance the tier."""
    # cosim flag, no evidence -> stays H0.
    assert silicon_tier(_silicon(compiles=True, cosim_validated=True)) == 0
    # synth flag, no report -> stays H1.
    assert (
        silicon_tier(
            _silicon(
                compiles=True,
                cosim_validated=True,
                cosim_evidence="cosim.log",
                synthesised=True,
            )
        )
        == 1
    )
    base_h2 = {
        "compiles": True,
        "cosim_validated": True,
        "cosim_evidence": "cosim.log",
        "synthesised": True,
        "synth_report": "yosys.json",
    }
    # timing flag, no report -> stays H2.
    assert silicon_tier(_silicon(**base_h2, timing_closed=True)) == 2
    # timing flag + report, no clock -> stays H2.
    assert silicon_tier(_silicon(**base_h2, timing_closed=True, timing_report="sta.rpt")) == 2
    base_h3 = {**base_h2, "timing_closed": True, "timing_report": "sta.rpt", "clock_mhz": 50.0}
    # formal flag, no proof -> stays H3.
    assert silicon_tier(_silicon(**base_h3, formally_equivalent=True)) == 3
    base_h4 = {**base_h3, "formally_equivalent": True, "equivalence_proof": "miter.smt2"}
    # ppa flag, no report -> stays H4.
    assert silicon_tier(_silicon(**base_h4, ppa_signed=True)) == 4
