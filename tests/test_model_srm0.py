# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — End-to-end test: SRM0Neuron

"""Full pipeline test for SRM0Neuron (Gerstner & Kistler 2002).

SRM zeroth order: LIF-like integration + refractory kernel eta.
eta decays exp(-dt/tau_eta) and provides afterhyperpolarisation.
Unlike SpikeResponseNeuron, this has actual voltage accumulation.
Performance: ~389K isolation steps/s."""

from __future__ import annotations

import time

import numpy as np

from sc_neurocore.neurons.models.srm0 import SRM0Neuron
from sc_neurocore.network.population import Population
from sc_neurocore.network.projection import Projection
from sc_neurocore.network.network import Network
from sc_neurocore.network.monitor import SpikeMonitor
from sc_neurocore.network.stimulus import PoissonInput
from sc_neurocore.analysis.spike_stats.basic import spike_count, isi, firing_rate


def _run(neuron: SRM0Neuron, current: float, steps: int) -> list[int]:
    return [t for t in range(steps) if neuron.step(current) == 1]


class TestSRM0Isolation:
    def test_defaults(self):
        n = SRM0Neuron()
        assert n.v == 0.0 and n.v_threshold == 1.0 and n.tau_m == 20.0
        assert n.tau_eta == 50.0 and n.eta_reset == 5.0

    def test_step_returns_binary(self):
        assert SRM0Neuron().step(0.0) in (0, 1)

    def test_state_finite(self):
        n = SRM0Neuron()
        for _ in range(50000):
            n.step(2.0)
        assert np.isfinite(n.v)

    def test_reset(self):
        n = SRM0Neuron()
        for _ in range(100):
            n.step(5.0)
        n.reset()
        assert n.v == n.v_rest and n._eta == 0.0

    def test_get_state(self):
        n = SRM0Neuron()
        n.step(1.0)
        state = n.get_state()
        assert "v" in state and "eta" in state and "t" in state


class TestSRM0EtaKernel:
    def test_eta_set_on_spike(self):
        """After spike: eta = -eta_reset (negative = afterhyperpolarisation)."""
        n = SRM0Neuron()
        for _ in range(10000):
            if n.step(5.0) == 1:
                assert n._eta == -n.eta_reset
                break
        else:
            raise AssertionError("No spike")

    def test_eta_decays_exponentially(self):
        """eta *= exp(-dt/tau_eta) each step."""
        n = SRM0Neuron()
        n._eta = -5.0
        eta0 = n._eta
        n.step(0.0)
        expected = eta0 * np.exp(-n.dt / n.tau_eta)
        assert abs(n._eta - expected) < 1e-10

    def test_eta_provides_afterhyperpolarisation(self):
        """Negative eta shifts effective rest downward, slowing next spike."""
        n = SRM0Neuron()
        for _ in range(10000):
            if n.step(5.0) == 1:
                # eta is now -5.0 → effective_rest = v_rest + (-5) = -5
                # V was just reset to 0. Next step: V will be pulled down.
                n.step(5.0)
                # V should be below what it would be without eta
                assert n.v < 0.5  # much less than 5.0 * dt/tau_m
                break

    def test_eta_zero_long_after_spike(self):
        """eta decays to ~0 long after spike."""
        n = SRM0Neuron()
        n._eta = -5.0
        for _ in range(500):
            n.step(0.0)
        assert abs(n._eta) < 0.001


class TestSRM0FI:
    def test_subthreshold_silent(self):
        n = SRM0Neuron()
        assert len(_run(n, current=0.5, steps=10000)) == 0

    def test_suprathreshold_fires(self):
        n = SRM0Neuron()
        assert len(_run(n, current=2.0, steps=10000)) >= 20

    def test_monotonic_fi(self):
        rates = []
        for I in [2.0, 3.0, 5.0, 10.0]:
            n = SRM0Neuron()
            rates.append(len(_run(n, current=I, steps=10000)))
        assert all(rates[i] <= rates[i + 1] for i in range(len(rates) - 1))

    def test_isi_regularity(self):
        n = SRM0Neuron()
        spikes = _run(n, current=5.0, steps=10000)
        if len(spikes) >= 20:
            isis = np.diff(spikes[5:]).astype(float)
            cv = np.std(isis) / np.mean(isis)
            assert cv < 0.1


class TestSRM0Dynamics:
    def test_voltage_integrates(self):
        """Unlike SpikeResponseNeuron, SRM0 accumulates V over steps."""
        n = SRM0Neuron()
        v_prev = n.v
        for _ in range(5):
            n.step(0.5)
        assert n.v > v_prev  # V grew from integration

    def test_steady_state_subthreshold(self):
        """V_ss = R·I when eta=0 (no recent spike)."""
        n = SRM0Neuron()
        for _ in range(10000):
            n.step(0.5)
        v_ss = n.resistance * 0.5
        assert abs(n.v - v_ss) < 0.01

    def test_refractory_lengthens_isi(self):
        """eta_reset > 0 → longer ISI than pure LIF."""
        n_refrac = SRM0Neuron(eta_reset=5.0)
        n_norefrac = SRM0Neuron(eta_reset=0.0)
        s_refrac = len(_run(n_refrac, current=5.0, steps=10000))
        s_norefrac = len(_run(n_norefrac, current=5.0, steps=10000))
        assert s_norefrac > s_refrac


class TestSRM0Performance:
    def test_isolation_throughput(self):
        n = SRM0Neuron()
        N = 50000
        t0 = time.perf_counter()
        for _ in range(N):
            n.step(2.0)
        elapsed = time.perf_counter() - t0
        assert N / elapsed > 50000

    def test_network_throughput(self):
        pop = Population(SRM0Neuron, n=50, label="bench")
        drive = PoissonInput(n=50, rate_hz=500.0, weight=5.0, dt=0.001, seed=42)
        mon = SpikeMonitor(pop)
        net = Network(pop, drive, mon)
        t0 = time.perf_counter()
        net.run(duration=0.5, dt=0.001, backend="python")
        elapsed = time.perf_counter() - t0
        assert 50 * 500 / elapsed > 5000


class TestSRM0Pipeline:
    def test_population(self):
        assert Population(SRM0Neuron, n=10, label="srm0").n == 10

    def test_network_spikes(self):
        pop = Population(SRM0Neuron, n=10, label="srm0")
        drive = PoissonInput(n=10, rate_hz=500.0, weight=5.0, dt=0.001, seed=42)
        mon = SpikeMonitor(pop)
        net = Network(pop, drive, mon)
        net.run(duration=1.0, dt=0.001, backend="python")
        assert mon.count > 0

    def test_projection_wiring(self):
        src = Population(SRM0Neuron, n=10, label="src")
        tgt = Population(SRM0Neuron, n=10, label="tgt")
        drive = PoissonInput(n=10, rate_hz=500.0, weight=5.0, dt=0.001, seed=42)
        proj = Projection(src, tgt, weight=3.0, probability=1.0, seed=42)
        mon = SpikeMonitor(src)
        net = Network(src, tgt, drive, proj, mon)
        net.run(duration=1.0, dt=0.001, backend="python")
        assert mon.count > 0

    def test_analysis(self):
        n = SRM0Neuron()
        train = np.array([float(n.step(5.0)) for _ in range(10000)])
        sc = spike_count(train)
        assert sc >= 50
        isis = isi(train, dt=0.001)
        assert len(isis) >= 10
        rate = firing_rate(train, dt=0.001)
        assert rate > 0

    def test_deterministic(self):
        traces = []
        for _ in range(2):
            n = SRM0Neuron()
            trace = [(n.step(3.0), n.v) for _ in range(200)]
            traces.append(trace)
        assert traces[0] == traces[1]
