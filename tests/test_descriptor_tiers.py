# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Dual-axis catalogue-to-silicon tier scoring tests

"""Tests for the science (S0-S5) and silicon (H0-H5) tier scoring.

The scoring must be a faithful, evidence-gated derivation: S0-S3 track the
curation kernel, S4/S5 open only on recorded validation evidence, and every
silicon rung is credited only alongside its proof anchor so a tier can never be
inflated ahead of the committed evidence (master plan invariant I7).
"""

from __future__ import annotations

from typing import Any

import pytest

from sc_neurocore.neurons.descriptor_tiers import (
    SILICON_RUNGS,
    CompletenessTiers,
    completeness_tiers,
    is_perfect,
    science_tier,
    silicon_tier,
)
from sc_neurocore.neurons.model_descriptor import (
    descriptor_completeness_tier,
    parse_model_descriptor,
)


def _metadata(**extra: Any) -> dict[str, Any]:
    base = {
        "schema_version": 2,
        "name": "AdEx",
        "class_name": "AdExNeuron",
        "module": "adex",
    }
    base.update(extra)
    return base


def _s0_payload() -> dict[str, Any]:
    """Structure only: identifies a real model, no taxonomy (science tier 0)."""
    return {
        "metadata": _metadata(),
        "state": {"v": {"init": -65.0}},
        "parameters": {"tau": {"default": 20.0}},
        "integration": {"dt": 0.1},
    }


def _s3_payload() -> dict[str, Any]:
    """A descriptor curated and engineering-verified to exactly science tier 3.

    Carries dynamics as well, so adding validation evidence can lift it to S4/S5.
    """
    return {
        "metadata": _metadata(family="Integrate-and-Fire", category="adaptive"),
        "provenance": {"authors": ["Brette"], "year": 2005, "doi": "10.1152/jn.00686.2005"},
        "state": {"v": {"init": -65.0}},
        "parameters": {
            "tau": {
                "default": 20.0,
                "unit": "ms",
                "range": [1.0, 100.0],
                "meaning": "time constant",
            }
        },
        "integration": {"dt": 0.1, "method": "euler"},
        "dynamics": {"v": "(-v + tau * I) / tau"},
        "backends": {
            "python": {"status": "implemented"},
            "rust": {"status": "implemented", "parity": "ulp-bounded"},
        },
        "reproducibility": {
            "reference_config": "golden/adex.json",
            "golden_trace_sha256": "a" * 64,
        },
    }


def _validated_payload() -> dict[str, Any]:
    """An S3 payload with full faithful-dynamics + class-validation evidence (S5)."""
    payload = _s3_payload()
    payload["validation"] = {
        "dynamics_faithful": True,
        "metric": "parity",
        "operating_point": "I = 5",
        "tolerance": "0 spikes",
        "evidence": "tests/test_cosimulation.py::TestRK4Emitter",
    }
    return payload


# --------------------------------------------------------------------------- #
# Science axis
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("expected", [0, 1, 2, 3])
def test_science_axis_matches_kernel_through_s3(expected: int) -> None:
    """S0-S3 track the curation kernel exactly, with no validation evidence."""
    payload = _s3_payload()
    payload.pop("validation", None)
    if expected < 3:
        payload["backends"] = {}  # drop the engineering-verified rung
        payload["reproducibility"] = {}
    if expected < 2:
        payload["provenance"] = {}  # drop citeable provenance
    if expected < 1:
        del payload["metadata"]["family"]
        del payload["metadata"]["category"]

    descriptor = parse_model_descriptor(payload)
    assert descriptor_completeness_tier(descriptor) == expected
    assert science_tier(descriptor) == expected


def test_science_tier_stays_at_base_when_below_three_despite_validation() -> None:
    """S4/S5 are gated behind the S3 kernel: validation alone cannot lift S0."""
    payload = _s0_payload()
    payload["dynamics"] = {"v": "-v"}
    payload["validation"] = {
        "dynamics_faithful": True,
        "metric": "parity",
        "evidence": "somewhere",
    }
    descriptor = parse_model_descriptor(payload)
    assert science_tier(descriptor) == 0


def test_science_tier_s4_requires_faithful_dynamics() -> None:
    """S4 opens only when dynamics are declared and confirmed faithful."""
    at_s3 = parse_model_descriptor(_s3_payload())
    assert science_tier(at_s3) == 3  # dynamics present but not confirmed faithful

    faithful = _s3_payload()
    faithful["validation"] = {"dynamics_faithful": True}
    assert science_tier(parse_model_descriptor(faithful)) == 4


def test_science_tier_s4_needs_declared_dynamics() -> None:
    """A faithful flag without any declared dynamics cannot reach S4."""
    payload = _s3_payload()
    payload["dynamics"] = {}
    payload["validation"] = {"dynamics_faithful": True}
    assert science_tier(parse_model_descriptor(payload)) == 3


def test_science_tier_s5_requires_metric_and_evidence() -> None:
    """S5 opens only with a non-trivial metric and committed evidence."""
    assert science_tier(parse_model_descriptor(_validated_payload())) == 5

    metric_only = _validated_payload()
    metric_only["validation"]["evidence"] = ""
    assert science_tier(parse_model_descriptor(metric_only)) == 4

    evidence_only = _validated_payload()
    evidence_only["validation"]["metric"] = "none"
    assert science_tier(parse_model_descriptor(evidence_only)) == 4


# --------------------------------------------------------------------------- #
# Silicon axis
# --------------------------------------------------------------------------- #


def _silicon(**flags: Any) -> Any:
    payload = _validated_payload()
    payload["silicon"] = flags
    return parse_model_descriptor(payload)


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


# --------------------------------------------------------------------------- #
# Combined view + perfection contract
# --------------------------------------------------------------------------- #


def test_completeness_tiers_reports_both_axes_and_labels() -> None:
    """The combined view pairs the two axes and formats honest labels."""
    tiers = completeness_tiers(parse_model_descriptor(_validated_payload()))
    assert tiers == CompletenessTiers(science=5, silicon=None)
    assert tiers.science_label == "S5"
    assert tiers.silicon_label == "none"

    on_silicon = _silicon(compiles=True, cosim_validated=True, cosim_evidence="cosim.log")
    tiers = completeness_tiers(on_silicon)
    assert (tiers.science, tiers.silicon) == (5, 1)
    assert tiers.silicon_label == "H1"


def test_silicon_rungs_index_matches_label() -> None:
    """The silicon-rung ordering is the numeric tier index."""
    assert SILICON_RUNGS == ("H0", "H1", "H2", "H3", "H4", "H5")
    assert SILICON_RUNGS.index("H3") == 3


def _perfect_payload(target_tier: str = "H2") -> dict[str, Any]:
    payload = _validated_payload()
    payload["silicon"] = {
        "compiles": True,
        "cosim_validated": True,
        "cosim_evidence": "cosim.log",
        "synthesised": True,
        "synth_report": "yosys.json",
        "target_tier": target_tier,
        "terminal_reason": "point neuron; H2 is the deployable terminal",
    }
    return payload


def test_is_perfect_true_when_s5_and_terminal_met() -> None:
    """S5 science plus reaching the declared terminal H-tier is perfection."""
    assert is_perfect(parse_model_descriptor(_perfect_payload("H2"))) is True
    # Reaching beyond the declared terminal is also perfect.
    assert is_perfect(parse_model_descriptor(_perfect_payload("H1"))) is True


def test_is_perfect_false_when_science_below_s5() -> None:
    """A silicon-strong model that is not S5 is not perfect."""
    payload = _perfect_payload("H2")
    payload["validation"]["evidence"] = ""  # drops science to S4
    assert is_perfect(parse_model_descriptor(payload)) is False


def test_is_perfect_false_when_terminal_undeclared() -> None:
    """Without a declared deployability class, perfection cannot be certified."""
    payload = _perfect_payload("")
    assert is_perfect(parse_model_descriptor(payload)) is False


def test_is_perfect_false_when_silicon_below_terminal() -> None:
    """S5 with a terminal above the reached silicon tier is not perfect."""
    payload = _perfect_payload("H4")  # only H2 evidence present
    assert is_perfect(parse_model_descriptor(payload)) is False


def test_is_perfect_false_when_no_silicon_evidence() -> None:
    """S5 with a declared terminal but no compile-clean RTL is not perfect."""
    payload = _validated_payload()
    payload["silicon"] = {"target_tier": "H2"}
    assert is_perfect(parse_model_descriptor(payload)) is False
