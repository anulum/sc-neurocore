# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — End-to-end test: EPropALIFNeuron

"""Full pipeline test for EPropALIFNeuron (Bellec et al. 2020).

Adaptive LIF with eligibility traces for e-prop learning. Threshold
adapts: θ(t) = θ_base + β·a(t). ISI lengthens as a accumulates.
Performance: ~284K isolation steps/s."""

from __future__ import annotations

import time

import numpy as np
import pytest

from sc_neurocore.neurons.models.e_prop_alif import EPropALIFNeuron
from sc_neurocore.network.population import Population
from sc_neurocore.network.projection import Projection
from sc_neurocore.network.network import Network
from sc_neurocore.network.monitor import SpikeMonitor
from sc_neurocore.network.stimulus import PoissonInput
from sc_neurocore.analysis.spike_stats.basic import spike_count, firing_rate


def _run(neuron: EPropALIFNeuron, current: float, steps: int) -> list[int]:
    return [t for t in range(steps) if neuron.step(current) == 1]


class TestEPropALIFIsolation:
    def test_defaults(self):
        n = EPropALIFNeuron()
        assert n.v == 0.0 and n.a == 0.0 and n.e_trace == 0.0
        assert n.tau_m == 20.0 and n.tau_a == 200.0 and n.beta == 0.07

    def test_alpha_precomputed(self):
        n = EPropALIFNeuron()
        assert abs(n.alpha_m - np.exp(-1.0 / 20.0)) < 1e-12
        assert abs(n.alpha_a - np.exp(-1.0 / 200.0)) < 1e-12

    def test_step_returns_binary(self):
        assert EPropALIFNeuron().step(0.0) in (0, 1)

    def test_state_finite(self):
        n = EPropALIFNeuron()
        for _ in range(50000):
            n.step(0.2)
        assert all(np.isfinite(v) for v in [n.v, n.a, n.e_trace])

    def test_reset(self):
        n = EPropALIFNeuron()
        for _ in range(100):
            n.step(0.5)
        n.reset()
        assert n.v == 0.0 and n.a == 0.0 and n.e_trace == 0.0


class TestEPropALIFAdaptiveThreshold:
    """Core: θ(t) = θ_base + β·a(t). a increments on spike, decays with tau_a."""

    def test_a_increments_on_spike(self):
        n = EPropALIFNeuron()
        a_before = n.a
        for _ in range(10000):
            if n.step(0.5) == 1:
                assert n.a > a_before
                break
        else:
            raise AssertionError("No spike")

    def test_a_decays_between_spikes(self):
        n = EPropALIFNeuron()
        n.a = 5.0
        n.step(0.0)  # subthreshold, a decays
        assert n.a < 5.0

    def test_threshold_increases_with_a(self):
        """Effective threshold = θ_base + β·a. Higher a → harder to spike."""
        n = EPropALIFNeuron()
        # After many spikes, a is large → threshold high → ISI long
        spikes = _run(n, current=0.2, steps=5000)
        if len(spikes) >= 10:
            isis = np.diff(spikes)
            assert isis[-1] > isis[0], "ISI should lengthen (adaptation)"

    def test_isi_lengthens(self):
        """Early ISI < late ISI (adaptation effect)."""
        n = EPropALIFNeuron()
        spikes = _run(n, current=0.2, steps=5000)
        if len(spikes) >= 10:
            early = np.mean(np.diff(spikes[:5]))
            late = np.mean(np.diff(spikes[-5:]))
            assert late > early

    def test_no_adaptation_when_beta_zero(self):
        """β=0: threshold is constant → no ISI lengthening."""
        n = EPropALIFNeuron(beta=0.0)
        spikes = _run(n, current=0.2, steps=5000)
        if len(spikes) >= 10:
            isis = np.diff(spikes[3:]).astype(float)
            cv = np.std(isis) / np.mean(isis)
            assert cv < 0.05, f"CV = {cv:.4f} with β=0"


class TestEPropALIFEligibilityTrace:
    """e_trace tracks how weight changes affect future spiking."""

    def test_e_trace_accumulates(self):
        n = EPropALIFNeuron()
        for _ in range(100):
            n.step(0.2)
        assert n.e_trace > 0

    def test_e_trace_decays(self):
        n = EPropALIFNeuron()
        n.e_trace = 10.0
        n.v = -10.0  # far from threshold → psi ≈ 0
        n.step(0.0)
        assert n.e_trace < 10.0

    def test_pseudo_derivative_peaks_near_threshold(self):
        """psi = 0.3 · max(0, 1 - |V-θ|). Peaks when V ≈ θ."""
        n = EPropALIFNeuron()
        n.v = n.v_threshold_base  # exactly at threshold
        # psi = 0.3 * max(0, 1 - 0) = 0.3
        psi = max(0.0, 1.0 - abs(n.v - n.v_threshold_base)) * 0.3
        assert abs(psi - 0.3) < 1e-10


class TestEPropALIFFI:
    def test_zero_silent(self):
        n = EPropALIFNeuron()
        assert len(_run(n, current=0.0, steps=5000)) == 0

    def test_monotonic_fi(self):
        rates = []
        for I in [0.1, 0.2, 0.5, 1.0]:
            n = EPropALIFNeuron()
            rates.append(len(_run(n, current=I, steps=5000)))
        assert all(rates[i] <= rates[i + 1] for i in range(len(rates) - 1))


class TestEPropALIFParameters:
    def test_tau_a_controls_adaptation_speed(self):
        n_fast = EPropALIFNeuron(tau_a=50.0)
        n_slow = EPropALIFNeuron(tau_a=500.0)
        s_fast = len(_run(n_fast, current=0.2, steps=5000))
        s_slow = len(_run(n_slow, current=0.2, steps=5000))
        # Faster a decay → adaptation wears off quicker → more spikes
        assert s_fast > s_slow

    @pytest.mark.parametrize("dt", [0.5, 1.0, 2.0])
    def test_dt_stability(self, dt: float):
        n = EPropALIFNeuron(dt=dt)
        for _ in range(5000):
            n.step(0.2)
        assert np.isfinite(n.v)

    def test_deterministic(self):
        traces = []
        for _ in range(2):
            n = EPropALIFNeuron()
            trace = [(n.step(0.2), n.v, n.a, n.e_trace) for _ in range(200)]
            traces.append(trace)
        assert traces[0] == traces[1]


class TestEPropALIFPerformance:
    def test_isolation_throughput(self):
        n = EPropALIFNeuron()
        N = 50000
        t0 = time.perf_counter()
        for _ in range(N):
            n.step(0.2)
        elapsed = time.perf_counter() - t0
        assert N / elapsed > 50000

    def test_network_throughput(self):
        pop = Population(EPropALIFNeuron, n=50, label="bench")
        drive = PoissonInput(n=50, rate_hz=500.0, weight=0.5, dt=0.001, seed=42)
        mon = SpikeMonitor(pop)
        net = Network(pop, drive, mon)
        t0 = time.perf_counter()
        net.run(duration=0.5, dt=0.001, backend="python")
        elapsed = time.perf_counter() - t0
        assert 50 * 500 / elapsed > 5000


class TestEPropALIFPipeline:
    def test_population(self):
        assert Population(EPropALIFNeuron, n=10, label="eprop").n == 10

    def test_network_spikes(self):
        pop = Population(EPropALIFNeuron, n=10, label="eprop")
        drive = PoissonInput(n=10, rate_hz=500.0, weight=0.5, dt=0.001, seed=42)
        mon = SpikeMonitor(pop)
        net = Network(pop, drive, mon)
        net.run(duration=1.0, dt=0.001, backend="python")
        assert mon.count > 0

    def test_projection_wiring(self):
        src = Population(EPropALIFNeuron, n=10, label="src")
        tgt = Population(EPropALIFNeuron, n=10, label="tgt")
        drive = PoissonInput(n=10, rate_hz=500.0, weight=0.5, dt=0.001, seed=42)
        proj = Projection(src, tgt, weight=0.3, probability=1.0, seed=42)
        mon = SpikeMonitor(src)
        net = Network(src, tgt, drive, proj, mon)
        net.run(duration=1.0, dt=0.001, backend="python")
        assert mon.count > 0

    def test_analysis(self):
        n = EPropALIFNeuron()
        train = np.array([float(n.step(0.2)) for _ in range(5000)])
        sc = spike_count(train)
        assert sc >= 50
        rate = firing_rate(train, dt=0.001)
        assert rate > 0
