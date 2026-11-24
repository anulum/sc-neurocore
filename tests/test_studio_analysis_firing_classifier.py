# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Studio analysis firing classifier

"""Focused suite: TestFiringClassifier from former test_studio_analysis.py."""

from __future__ import annotations

from tests.studio_analysis_support import (
    classify_firing_pattern,
)


class TestFiringClassifier:
    """Classification of firing patterns."""

    def test_silent(self) -> None:
        """No spikes are classed as silent."""
        r = classify_firing_pattern([], 1000, 0.1)
        assert r["pattern"] == "silent"

    def test_tonic(self) -> None:
        """Regular spikes are classed as tonic."""
        spikes = list(range(100, 1000, 100))
        r = classify_firing_pattern(spikes, 1000, 0.1)
        assert r["pattern"] == "tonic"

    def test_single_spike(self) -> None:
        """One spike is classed as a single spike."""
        r = classify_firing_pattern([500], 1000, 0.1)
        assert r["pattern"] == "single_spike"
