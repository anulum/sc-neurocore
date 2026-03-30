# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — End-to-end test: MainenSejnowskiNeuron

"""Full pipeline test for MainenSejnowskiNeuron (Mainen & Sejnowski 1996).

2-compartment: passive soma + active axon (Na/K). 20 sub-steps.
Produces single spike per drive episode — axon Na saturates without
external reset. safe_exp guards prevent overflow."""

from __future__ import annotations

import numpy as np

from sc_neurocore.neurons.models.mainen_sejnowski import MainenSejnowskiNeuron
from sc_neurocore.network.population import Population


class TestMSIsolation:
    def test_construction(self):
        n = MainenSejnowskiNeuron()
        assert n.vs == -65.0
        assert n.va == -65.0

    def test_step_returns_binary(self):
        assert MainenSejnowskiNeuron().step(0.0) in (0, 1)

    def test_single_transient_spike(self):
        """Initial conditions produce 1 transient spike; no sustained firing at I=0."""
        n = MainenSejnowskiNeuron()
        s = sum(n.step(0.0) for _ in range(1000))
        assert s == 1

    def test_spike_under_strong_drive(self):
        """Strong drive produces at least 1 spike."""
        n = MainenSejnowskiNeuron()
        s = sum(n.step(200.0) for _ in range(5000))
        assert s >= 1

    def test_two_compartments_differ(self):
        """Soma and axon voltages should diverge under drive."""
        n = MainenSejnowskiNeuron()
        for _ in range(100):
            n.step(50.0)
        assert n.vs != n.va

    def test_gating_bounded(self):
        """m, h, n must stay in [0, 1] (clipped)."""
        n = MainenSejnowskiNeuron()
        for _ in range(2000):
            n.step(200.0)
        assert 0.0 <= n.m <= 1.0
        assert 0.0 <= n.h <= 1.0
        assert 0.0 <= n.n <= 1.0

    def test_voltage_clamped(self):
        """Voltage should be clamped to [-200, 200]."""
        n = MainenSejnowskiNeuron()
        for _ in range(3000):
            n.step(500.0)
        assert n.vs >= -200.0 and n.vs <= 200.0
        assert n.va >= -200.0 and n.va <= 200.0

    def test_numerical_stability(self):
        for I in [0.0, 50.0, 200.0]:
            n = MainenSejnowskiNeuron()
            for _ in range(2000):
                n.step(I)
            assert np.isfinite(n.vs), f"vs NaN at I={I}"
            assert np.isfinite(n.va), f"va NaN at I={I}"

    def test_reset(self):
        n = MainenSejnowskiNeuron()
        for _ in range(1000):
            n.step(200.0)
        n.reset()
        assert n.vs == -65.0
        assert n.va == -65.0
        assert n.m == 0.05

    def test_deterministic(self):
        n1 = MainenSejnowskiNeuron()
        n2 = MainenSejnowskiNeuron()
        for _ in range(200):
            assert n1.step(100.0) == n2.step(100.0)


class TestMSNetwork:
    def test_population(self):
        assert Population(MainenSejnowskiNeuron, n=5, label="ms").n == 5
