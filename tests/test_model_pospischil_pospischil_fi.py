# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestPospischilFI from former test_model_pospischil.py

"""Focused suite: TestPospischilFI from former test_model_pospischil.py."""

from __future__ import annotations

from tests.model_pospischil_support import *  # noqa: F403

class TestPospischilFI:
    def test_subthreshold_no_spikes(self):
        """Low current (I<2) → no sustained spiking."""
        n = PospischilNeuron()
        spikes = _run(n, current=1.0, steps=50000)
        assert len(spikes) == 0

    def test_suprathreshold_spiking(self):
        """Moderate current (I=5–10) → sustained regular spiking."""
        for I in [5.0, 10.0]:
            n = PospischilNeuron()
            spikes = _run(n, current=I, steps=50000)
            assert len(spikes) >= 100, f"I={I}: only {len(spikes)} spikes"

    def test_rate_increases_with_current(self):
        """Monotonic f–I: more current → more spikes."""
        n5 = PospischilNeuron()
        n10 = PospischilNeuron()
        n20 = PospischilNeuron()
        s5 = len(_run(n5, current=5.0, steps=50000))
        s10 = len(_run(n10, current=10.0, steps=50000))
        s20 = len(_run(n20, current=20.0, steps=50000))
        assert s5 < s10 < s20
