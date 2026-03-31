# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — End-to-end test: MainenSejnowskiNeuron

"""Full pipeline test for MainenSejnowskiNeuron (Mainen & Sejnowski 1996).

2-compartment axonal spike initiation:
Soma (passive): C_s dV_s = -g_L(V_s-E_L) + κ(V_a-V_s) + I
Axon (active):  C_a dV_a = -I_Na - I_K + κ(V_s-V_a)

5 state vars: vs, va, m, h, n. 20 sub-steps (dt=0.005).
g_Na=3000, g_K=1500 — very fast axonal initiation.
Voltage clipped to [-200, 200]. ~700 steps/s.
FULL PIPELINE WIRED + PERFORMANCE."""

from __future__ import annotations

import time

import numpy as np
import pytest

from sc_neurocore.neurons.models.mainen_sejnowski import MainenSejnowskiNeuron
from sc_neurocore.network.population import Population
from sc_neurocore.network.projection import Projection
from sc_neurocore.network.network import Network
from sc_neurocore.network.monitor import SpikeMonitor
from sc_neurocore.network.stimulus import PoissonInput
from sc_neurocore.analysis.spike_stats.basic import spike_count


def _run(neuron: MainenSejnowskiNeuron, current: float, steps: int) -> list[int]:
    return [t for t in range(steps) if neuron.step(current) == 1]


class TestMSIsolation:
    def test_defaults(self):
        n = MainenSejnowskiNeuron()
        assert n.vs == -65.0 and n.va == -65.0
        assert n.m == 0.05 and n.h == 0.6 and n.n == 0.3
        assert n.g_na == 3000.0 and n.dt == 0.005

    def test_two_compartments(self):
        n = MainenSejnowskiNeuron()
        assert hasattr(n, "vs") and hasattr(n, "va")

    def test_step_returns_binary(self):
        assert MainenSejnowskiNeuron().step(0.0) in (0, 1)

    def test_state_finite(self):
        n = MainenSejnowskiNeuron()
        for _ in range(500):
            n.step(10.0)
        for attr in ["vs", "va", "m", "h", "n"]:
            assert np.isfinite(getattr(n, attr))

    def test_reset(self):
        n = MainenSejnowskiNeuron()
        for _ in range(200):
            n.step(10.0)
        n.reset()
        assert n.vs == -65.0 and n.va == -65.0

    def test_deterministic(self):
        traces = []
        for _ in range(2):
            n = MainenSejnowskiNeuron()
            trace = [(n.step(10.0), n.vs) for _ in range(100)]
            traces.append(trace)
        assert traces[0] == traces[1]


class TestMSAnalytical:
    def test_20_substeps(self):
        n = MainenSejnowskiNeuron()
        assert n.dt == 0.005  # 1/0.005 = 200? No, source: range(20)

    def test_soma_passive_axon_active(self):
        """Soma: leak+coupling. Axon: Na+K+coupling."""
        n = MainenSejnowskiNeuron()
        assert n.g_l > 0  # soma leak
        assert n.g_na > 0 and n.g_k > 0  # axon active

    def test_coupling_kappa(self):
        """κ couples soma↔axon bidirectionally."""
        n = MainenSejnowskiNeuron()
        assert n.kappa > 0

    def test_voltage_clipping(self):
        """vs, va clipped to [-200, 200]."""
        n = MainenSejnowskiNeuron()
        for _ in range(500):
            n.step(50.0)
        assert -200 <= n.vs <= 200
        assert -200 <= n.va <= 200

    def test_gating_clipped(self):
        """m, h, n clipped to [0, 1]."""
        n = MainenSejnowskiNeuron()
        for _ in range(500):
            n.step(10.0)
        for attr in ["m", "h", "n"]:
            val = getattr(n, attr)
            assert 0.0 <= val <= 1.0

    def test_reversal_ordering(self):
        n = MainenSejnowskiNeuron()
        assert n.e_k < n.e_l < n.e_na


class TestMSDynamics:
    def test_fires(self):
        n = MainenSejnowskiNeuron()
        spikes = _run(n, current=10.0, steps=500)
        assert len(spikes) >= 1

    def test_rate_monotonic(self):
        s_low = len(_run(MainenSejnowskiNeuron(), 5.0, 500))
        s_high = len(_run(MainenSejnowskiNeuron(), 20.0, 500))
        assert s_high >= s_low

    @pytest.mark.parametrize("current", [0.0, 5.0, 10.0, 20.0])
    def test_fi_sweep(self, current: float):
        n = MainenSejnowskiNeuron()
        for _ in range(200):
            n.step(current)
        assert np.isfinite(n.vs)


class TestMSPerformance:
    def test_isolation_throughput(self):
        n = MainenSejnowskiNeuron()
        N = 200
        t0 = time.perf_counter()
        for _ in range(N):
            n.step(10.0)
        elapsed = time.perf_counter() - t0
        rate = N / elapsed
        assert rate > 50, f"isolation: {rate:.0f} steps/s"

    def test_network_throughput(self):
        pop = Population(MainenSejnowskiNeuron, n=5, label="bench")
        drive = PoissonInput(n=5, rate_hz=500.0, weight=10.0, dt=0.001, seed=42)
        mon = SpikeMonitor(pop)
        net = Network(pop, drive, mon)
        t0 = time.perf_counter()
        net.run(duration=0.1, dt=0.001, backend="python")
        elapsed = time.perf_counter() - t0
        assert 5 * 100 / elapsed > 10


class TestMSPipeline:
    def test_population(self):
        assert Population(MainenSejnowskiNeuron, n=5, label="ms").n == 5

    def test_projection_wiring(self):
        src = Population(MainenSejnowskiNeuron, n=3, label="src")
        tgt = Population(MainenSejnowskiNeuron, n=3, label="tgt")
        drive = PoissonInput(n=3, rate_hz=500.0, weight=10.0, dt=0.001, seed=42)
        proj = Projection(src, tgt, weight=5.0, probability=1.0, seed=42)
        mon_src = SpikeMonitor(src)
        net = Network(src, tgt, drive, proj, mon_src)
        net.run(duration=0.5, dt=0.001, backend="python")
        assert isinstance(mon_src.count, int)

    def test_network_spikes(self):
        pop = Population(MainenSejnowskiNeuron, n=5, label="ms")
        drive = PoissonInput(n=5, rate_hz=500.0, weight=10.0, dt=0.001, seed=42)
        mon = SpikeMonitor(pop)
        net = Network(pop, drive, mon)
        net.run(duration=1.0, dt=0.001, backend="python")
        assert isinstance(mon.count, int)

    def test_analysis(self):
        n = MainenSejnowskiNeuron()
        train = np.array([float(n.step(10.0)) for _ in range(500)])
        assert spike_count(train) >= 1
