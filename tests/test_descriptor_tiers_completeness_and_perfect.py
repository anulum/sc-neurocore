# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Focused free-test suite (completeness_and_perfect) from former test_descriptor_tiers.py

from __future__ import annotations

from tests.descriptor_tiers_support import *  # noqa: F403

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
