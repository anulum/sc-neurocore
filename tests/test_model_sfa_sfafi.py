# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestSFAFI from former test_model_sfa.py

"""Focused suite: TestSFAFI from former test_model_sfa.py."""

from __future__ import annotations

from tests.model_sfa_support import *  # noqa: F403

class TestSFAFI:
    def test_subthreshold_no_spikes(self):
        """Low current → no spikes."""
        n = SFANeuron()
        spikes = len(_run(n, current=10.0, steps=10000))
        assert spikes == 0

    def test_suprathreshold_fires(self):
        n = SFANeuron()
        spikes = len(_run(n, current=50.0, steps=10000))
        assert spikes > 10

    def test_rate_increases_with_current(self):
        n30 = SFANeuron()
        n100 = SFANeuron()
        s30 = len(_run(n30, current=30.0, steps=10000))
        s100 = len(_run(n100, current=100.0, steps=10000))
        assert s100 > s30

    def test_zero_current_silent(self):
        n = SFANeuron()
        assert len(_run(n, current=0.0, steps=10000)) == 0
