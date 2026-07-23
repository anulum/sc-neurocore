# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Module-level tests from former test_arcane_zenith.py

"""Module-level tests from former test_arcane_zenith.py."""

from __future__ import annotations

from tests.test_arcane_zenith.arcane_zenith_support import *  # noqa: F403

def test_level_band_classifies_below_and_above_thresholds() -> None:
    assert ArcaneZenithCognitiveCore._level(0.0, low=0.33, high=0.66) == "low"
    assert ArcaneZenithCognitiveCore._level(1.0, low=0.33, high=0.66) == "high"
    assert ArcaneZenithCognitiveCore._level(0.5, low=0.33, high=0.66) == "medium"
def test_pathway_bitstreams_are_all_zero_for_silent_rates() -> None:
    # When no channel carries a positive rate the maximum is zero, so every
    # pathway probability collapses to zero rather than dividing by it.
    bitstreams = ArcaneZenithCognitiveCore._pathway_bitstreams(
        {0: 0.0, 1: 0.0}, bitstream_length=16, seed=0
    )
    assert bitstreams.shape == (2, 16)
    assert int(bitstreams.sum()) == 0
