# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestDurstewitzDynamics from former test_model_durstewitz_dopamine.py

"""Focused suite: TestDurstewitzDynamics from former test_model_durstewitz_dopamine.py."""

from __future__ import annotations

from tests.model_durstewitz_dopamine_support import *  # noqa: F403

class TestDurstewitzDynamics:
    def test_spontaneous_firing(self):
        """Fires at I=0 (dopaminergic tonic activity)."""
        n = DurstewitzDopamineNeuron()
        spikes = _run(n, current=0.0, steps=10000)
        assert len(spikes) >= 5

    def test_rate_increases_with_current(self):
        n0 = DurstewitzDopamineNeuron()
        n50 = DurstewitzDopamineNeuron()
        s0 = len(_run(n0, current=0.0, steps=10000))
        s50 = len(_run(n50, current=50.0, steps=10000))
        assert s50 > s0

    def test_monotonic_fi(self):
        rates = []
        for I in [0.0, 10.0, 30.0, 50.0]:
            n = DurstewitzDopamineNeuron()
            rates.append(len(_run(n, current=I, steps=10000)))
        assert all(rates[i] <= rates[i + 1] for i in range(len(rates) - 1))

    def test_deterministic(self):
        traces = []
        for _ in range(2):
            n = DurstewitzDopamineNeuron()
            trace = [(n.step(10.0), n.v) for _ in range(200)]
            traces.append(trace)
        assert traces[0] == traces[1]
