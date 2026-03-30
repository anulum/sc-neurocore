# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — End-to-end test: HuberBraunNeuron

"""Full pipeline test for HuberBraunNeuron (Braun, Huber et al. 1998).

Cold receptor model: slow depolarising + slow repolarising currents + noise.
Default params produce a single spike then settle to depolarised equilibrium
— sustained oscillation requires parameter tuning (temperature-dependent)."""

from __future__ import annotations

import numpy as np

from sc_neurocore.neurons.models.huber_braun import HuberBraunNeuron
from sc_neurocore.network.population import Population
from sc_neurocore.analysis.spike_stats.basic import spike_count


class TestHBIsolation:
    def test_construction(self):
        n = HuberBraunNeuron()
        assert n.v == -50.0
        assert n.a_sd == 0.0
        assert n.a_sr == 0.0

    def test_step_returns_binary(self):
        assert HuberBraunNeuron().step(0.0) in (0, 1)

    def test_initial_spike(self):
        """Default params produce exactly 1 spike (upward crossing)."""
        n = HuberBraunNeuron()
        s = sum(n.step(5.0) for _ in range(10000))
        assert s >= 1

    def test_sd_gating_activates(self):
        """Slow depolarising gate should activate under drive."""
        n = HuberBraunNeuron()
        for _ in range(500):
            n.step(5.0)
        assert n.a_sd > 0.1

    def test_sr_gating(self):
        """Slow repolarising gate should respond to voltage."""
        n = HuberBraunNeuron()
        a_sr_init = n.a_sr
        for _ in range(500):
            n.step(5.0)
        assert n.a_sr != a_sr_init

    def test_noise_present(self):
        """Two runs with eta > 0 should differ (stochastic noise)."""
        n1 = HuberBraunNeuron(eta=0.1)
        n2 = HuberBraunNeuron(eta=0.1)
        v1 = [n1.step(1.0) or n1.v for _ in range(100)]
        v2 = [n2.step(1.0) or n2.v for _ in range(100)]
        assert v1 != v2

    def test_no_noise(self):
        """eta=0 should make the model deterministic."""
        np.random.seed(42)
        n1 = HuberBraunNeuron(eta=0.0)
        np.random.seed(42)
        n2 = HuberBraunNeuron(eta=0.0)
        for _ in range(100):
            s1 = n1.step(1.0)
            s2 = n2.step(1.0)
            assert s1 == s2

    def test_numerical_stability(self):
        for I in [0.0, 5.0, 20.0]:
            n = HuberBraunNeuron()
            for _ in range(5000):
                n.step(I)
            assert np.isfinite(n.v), f"v NaN at I={I}"
            assert np.isfinite(n.a_sd), f"a_sd NaN at I={I}"
            assert np.isfinite(n.a_sr), f"a_sr NaN at I={I}"

    def test_gating_bounded(self):
        """Sigmoid gating should stay in [0, 1]."""
        n = HuberBraunNeuron()
        for _ in range(5000):
            n.step(5.0)
        assert 0.0 <= n.a_sd <= 1.0
        assert 0.0 <= n.a_sr <= 1.0

    def test_reset(self):
        n = HuberBraunNeuron()
        for _ in range(1000):
            n.step(5.0)
        n.reset()
        assert n.v == -50.0
        assert n.a_sd == 0.0
        assert n.a_sr == 0.0

    def test_depolarised_equilibrium(self):
        """Default params → v settles to depolarised state (v > 0)."""
        n = HuberBraunNeuron(eta=0.0)
        for _ in range(10000):
            n.step(5.0)
        assert n.v > 0.0


class TestHBNetwork:
    def test_population(self):
        assert Population(HuberBraunNeuron, n=10, label="hb").n == 10


class TestHBAnalysis:
    def test_spike_count_initial(self):
        """At least 1 spike from the initial crossing."""
        n = HuberBraunNeuron()
        train = np.zeros(5000, dtype=np.int8)
        for t in range(5000):
            train[t] = n.step(5.0)
        assert spike_count(train) >= 1
