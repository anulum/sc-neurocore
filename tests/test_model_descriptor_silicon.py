# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Focused free-test suite (silicon) from former test_model_descriptor.py

from __future__ import annotations

from tests.model_descriptor_support import *  # noqa: F403

def test_silicon_defaults_are_empty() -> None:
    """An absent [silicon] section yields the below-H0 default facet."""
    descriptor = parse_model_descriptor(_minimal_payload())
    assert descriptor.silicon == Silicon()
    assert descriptor.silicon.compiles is False
    assert descriptor.silicon.clock_mhz is None
    assert descriptor.silicon.target_tier == ""


def test_parse_silicon_section_reads_every_field() -> None:
    """The [silicon] section carries the realisation ladder and its anchors."""
    payload = _minimal_payload()
    payload["silicon"] = {
        "compiles": True,
        "cosim_validated": True,
        "synthesised": True,
        "timing_closed": True,
        "formally_equivalent": True,
        "ppa_signed": True,
        "cosim_evidence": "cosim.log",
        "synth_report": "yosys.json",
        "timing_report": "sta.rpt",
        "equivalence_proof": "miter.smt2",
        "ppa_report": "openlane.json",
        "target_device": "xc7a35t",
        "clock_mhz": 125,
        "target_tier": "H3",
        "terminal_reason": "point neuron, deployable to H3",
    }
    silicon = parse_model_descriptor(payload).silicon
    assert silicon.compiles is True
    assert silicon.cosim_validated is True
    assert silicon.target_device == "xc7a35t"
    assert silicon.clock_mhz == pytest.approx(125.0)
    assert isinstance(silicon.clock_mhz, float)
    assert silicon.target_tier == "H3"
    assert silicon.terminal_reason == "point neuron, deployable to H3"


def test_parse_rejects_unknown_silicon_target_tier() -> None:
    payload = _minimal_payload()
    payload["silicon"] = {"target_tier": "H9"}
    with pytest.raises(ModelDescriptorError, match="target_tier"):
        parse_model_descriptor(payload)


def test_parse_rejects_non_numeric_clock() -> None:
    payload = _minimal_payload()
    payload["silicon"] = {"clock_mhz": "fast"}
    with pytest.raises(ModelDescriptorError, match="clock_mhz"):
        parse_model_descriptor(payload)


def test_parse_rejects_boolean_clock() -> None:
    payload = _minimal_payload()
    payload["silicon"] = {"clock_mhz": True}
    with pytest.raises(ModelDescriptorError, match="clock_mhz"):
        parse_model_descriptor(payload)


