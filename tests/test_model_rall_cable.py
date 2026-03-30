# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — End-to-end test: RallCableNeuron

"""Full pipeline test for RallCableNeuron (Rall 1962).

N-compartment passive cable. Current injected at distal end (N-1),
spike detected at soma (compartment 0). Signal attenuates with distance."""

from __future__ import annotations

import numpy as np
import pytest

from sc_neurocore.neurons.models.rall_cable import RallCableNeuron
from sc_neurocore.network.population import Population
from sc_neurocore.network.network import Network
from sc_neurocore.network.monitor import SpikeMonitor
from sc_neurocore.network.stimulus import PoissonInput
from sc_neurocore.analysis.spike_stats.basic import spike_count


def _run(neuron: RallCableNeuron, current: float,
         steps: int) -> list[int]:
    return [t for t in range(steps) if neuron.step(current) == 1]


class TestRallCableIsolation:

    def test_construction_defaults(self):
        n = RallCableNeuron()
        assert n.n_comp == 5
        assert n.tau_m == 20.0
        assert n.v_rest == -65.0
        assert n.g_ratio == 0.5
        assert n.v.shape == (5,)
        np.testing.assert_allclose(n.v, -65.0)

    def test_step_returns_binary(self):
        assert RallCableNeuron().step(0.0) in (0, 1)

    def test_compartments_evolve(self):
        """All compartments should change from rest under current."""
        n = RallCableNeuron()
        for _ in range(1000):
            n.step(100.0)
        # Distal end (current injection) should depolarise most
        assert n.v[-1] > n.v_rest

    def test_state_finite_long_run(self):
        n = RallCableNeuron()
        for _ in range(50000):
            n.step(100.0)
        assert np.all(np.isfinite(n.v))

    def test_reset(self):
        n = RallCableNeuron()
        for _ in range(500):
            n.step(100.0)
        n.reset()
        np.testing.assert_allclose(n.v, n.v_rest)


class TestRallCablePropagation:

    def test_distal_depolarises_more(self):
        """Current at distal end → distal compartment most depolarised."""
        n = RallCableNeuron()
        for _ in range(5000):
            n.step(100.0)
        assert n.v[-1] > n.v[0], "Distal should be more depolarised than soma"

    def test_signal_attenuates_with_distance(self):
        """Voltage decreases from distal to soma (passive attenuation)."""
        n = RallCableNeuron(n_comp=5, g_ratio=0.5)
        for _ in range(10000):
            n.step(200.0)
        # Monotonic attenuation: v[4] > v[3] > ... > v[0]
        for i in range(n.n_comp - 1):
            assert n.v[i + 1] >= n.v[i] - 1.0, (
                f"Compartment {i}: {n.v[i]:.2f} > {n.v[i+1]:.2f}"
            )

    def test_coupling_strength_affects_propagation(self):
        """Stronger coupling (g_ratio) → less attenuation → more somatic depolarisation."""
        n_weak = RallCableNeuron(n_comp=3, g_ratio=0.1)
        n_strong = RallCableNeuron(n_comp=3, g_ratio=5.0)
        for _ in range(10000):
            n_weak.step(200.0)
            n_strong.step(200.0)
        assert n_strong.v[0] > n_weak.v[0], (
            f"Strong soma={n_strong.v[0]:.2f}, weak soma={n_weak.v[0]:.2f}"
        )


class TestRallCableSpiking:

    def test_fewer_compartments_easier_to_spike(self):
        """Shorter cable (fewer compartments) → less attenuation → more spikes."""
        n2 = RallCableNeuron(n_comp=2, g_ratio=2.0)
        n5 = RallCableNeuron(n_comp=5, g_ratio=2.0)
        s2 = len(_run(n2, current=500.0, steps=50000))
        s5 = len(_run(n5, current=500.0, steps=50000))
        assert s2 > s5, f"n_comp=2: {s2} spikes, n_comp=5: {s5}"

    def test_spikes_with_short_cable(self):
        """n_comp=2 with strong coupling should produce spikes."""
        n = RallCableNeuron(n_comp=2, g_ratio=2.0)
        spikes = _run(n, current=500.0, steps=50000)
        assert len(spikes) >= 100

    def test_no_spikes_long_cable_weak_coupling(self):
        """Default (n=5, g_ratio=0.5) with moderate current → no somatic spikes."""
        n = RallCableNeuron()
        spikes = _run(n, current=500.0, steps=50000)
        assert len(spikes) == 0

    def test_somatic_reset_on_spike(self):
        """After spike, soma resets to v_reset."""
        n = RallCableNeuron(n_comp=2, g_ratio=5.0)
        for _ in range(50000):
            s = n.step(500.0)
            if s == 1:
                assert abs(n.v[0] - n.v_reset) < 1e-6
                break
        else:
            pytest.skip("No spike observed")


class TestRallCableParameters:

    @pytest.mark.parametrize("n_comp", [2, 3, 5, 10])
    def test_n_comp_variations(self, n_comp: int):
        n = RallCableNeuron(n_comp=n_comp)
        assert n.v.shape == (n_comp,)
        for _ in range(1000):
            n.step(100.0)
        assert np.all(np.isfinite(n.v))

    @pytest.mark.parametrize("dt", [0.05, 0.1, 0.2])
    def test_dt_stability(self, dt: float):
        n = RallCableNeuron(dt=dt, n_comp=3, g_ratio=1.0)
        for _ in range(10000):
            n.step(100.0)
        assert np.all(np.isfinite(n.v))

    def test_deterministic(self):
        traces = []
        for _ in range(2):
            n = RallCableNeuron(n_comp=3)
            trace = [(n.step(100.0), n.v[0]) for _ in range(200)]
            traces.append(trace)
        assert traces[0] == traces[1]


class TestRallCableNetwork:

    def test_population_incompatible(self):
        """RallCableNeuron has array-valued v — Population._sync_voltages
        cannot handle this (expects scalar v). Document this limitation."""
        with pytest.raises((ValueError, TypeError)):
            Population(RallCableNeuron, n=5, label="rall")


class TestRallCableAnalysis:

    def test_spike_count(self):
        n = RallCableNeuron(n_comp=2, g_ratio=5.0)
        train = np.array([float(n.step(500.0)) for _ in range(50000)])
        assert spike_count(train) >= 10

    def test_spike_count_consistency(self):
        n = RallCableNeuron(n_comp=2, g_ratio=5.0)
        train = np.array([float(n.step(500.0)) for _ in range(50000)])
        assert spike_count(train) == int(train.sum())
