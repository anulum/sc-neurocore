# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — End-to-end test: MarderSTGNeuron

"""Full pipeline test for MarderSTGNeuron (Marder & Selverston 1992).

Stomatogastric ganglion: 8 ionic currents, Ca dynamics, CPG oscillator.
Fires intrinsically even at I=0."""

from __future__ import annotations

import numpy as np

from sc_neurocore.neurons.models.marder_stg import MarderSTGNeuron
from sc_neurocore.network.population import Population
from sc_neurocore.analysis.spike_stats.basic import spike_count


class TestSTGIsolation:
    def test_construction(self):
        n = MarderSTGNeuron()
        assert n.v == -60.0
        assert n.ca == 0.05

    def test_step_returns_binary(self):
        assert MarderSTGNeuron().step(0.0) in (0, 1)

    def test_intrinsic_oscillation(self):
        """CPG fires at I=0."""
        n = MarderSTGNeuron()
        assert sum(n.step(0.0) for _ in range(10000)) > 10

    def test_calcium_dynamics(self):
        """Ca should accumulate during spiking."""
        n = MarderSTGNeuron()
        for _ in range(5000):
            n.step(0.0)
        assert n.ca > 0.05

    def test_ca_non_negative(self):
        n = MarderSTGNeuron()
        for _ in range(10000):
            n.step(2.0)
        assert n.ca >= 0.0

    def test_eight_currents(self):
        """All gating variables should evolve."""
        n = MarderSTGNeuron()
        for _ in range(5000):
            n.step(0.0)
        assert n.m_na > 0.0
        assert n.m_h > 0.0

    def test_boltzmann(self):
        n = MarderSTGNeuron()
        assert 0.0 < n._boltz(-25.0, -25.5, 5.29) < 1.0

    def test_eleven_state_variables(self):
        n = MarderSTGNeuron()
        states = ["v", "m_na", "h_na", "m_cat", "h_cat", "m_cas", "m_a", "h_a", "m_kd", "m_h", "ca"]
        for s in states:
            assert hasattr(n, s)

    def test_numerical_stability(self):
        for I in [0.0, 2.0, 5.0]:
            n = MarderSTGNeuron()
            for _ in range(10000):
                n.step(I)
            assert np.isfinite(n.v), f"v NaN at I={I}"
            assert np.isfinite(n.ca), f"ca NaN at I={I}"

    def test_reset(self):
        n = MarderSTGNeuron()
        for _ in range(3000):
            n.step(0.0)
        n.reset()
        assert n.v == -60.0
        assert n.ca == 0.05
        assert n.m_na == 0.0

    def test_deterministic(self):
        n1 = MarderSTGNeuron()
        n2 = MarderSTGNeuron()
        for _ in range(200):
            assert n1.step(0.0) == n2.step(0.0)


class TestSTGNetwork:
    def test_population(self):
        assert Population(MarderSTGNeuron, n=5, label="stg").n == 5


class TestSTGAnalysis:
    def test_spike_count(self):
        n = MarderSTGNeuron()
        train = np.zeros(10000, dtype=np.int8)
        for t in range(10000):
            train[t] = n.step(0.0)
        assert spike_count(train) > 10
