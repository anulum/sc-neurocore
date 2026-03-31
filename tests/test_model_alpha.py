# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — End-to-end test: AlphaNeuron

"""Full pipeline test for AlphaNeuron (Rall 1967).

Dual exc/inh alpha-synapse currents. step(exc_current, inh_current).
Inhibition suppresses excitatory drive. Performance: ~488K steps/s."""

from __future__ import annotations

import time

import numpy as np
import pytest

from sc_neurocore.neurons.models.alpha import AlphaNeuron
from sc_neurocore.network.population import Population
from sc_neurocore.network.projection import Projection
from sc_neurocore.network.network import Network
from sc_neurocore.network.monitor import SpikeMonitor
from sc_neurocore.network.stimulus import PoissonInput
from sc_neurocore.analysis.spike_stats.basic import spike_count, firing_rate


def _run(neuron: AlphaNeuron, exc: float, steps: int, inh: float = 0.0) -> list[int]:
    return [t for t in range(steps) if neuron.step(exc, inh) == 1]


class TestAlphaIsolation:
    def test_defaults(self):
        n = AlphaNeuron()
        assert n.v == 0.0 and n.i_exc == 0.0 and n.i_inh == 0.0
        assert n.tau_v == 20.0 and n.tau_exc == 5.0 and n.tau_inh == 10.0

    def test_step_returns_binary(self):
        assert AlphaNeuron().step(0.0) in (0, 1)

    def test_dual_input_signature(self):
        n = AlphaNeuron()
        s = n.step(1.0, 0.5)
        assert s in (0, 1)

    def test_three_variables_evolve(self):
        n = AlphaNeuron()
        for _ in range(100):
            n.step(1.0, 0.3)
        assert n.v != 0.0 and n.i_exc != 0.0 and n.i_inh != 0.0

    def test_state_finite(self):
        n = AlphaNeuron()
        for _ in range(50000):
            n.step(1.0, 0.3)
        assert all(np.isfinite(v) for v in [n.v, n.i_exc, n.i_inh])

    def test_reset(self):
        n = AlphaNeuron()
        for _ in range(100):
            n.step(2.0)
        n.reset()
        assert n.v == n.v_rest and n.i_exc == 0.0 and n.i_inh == 0.0


class TestAlphaSynapticCurrents:
    def test_exc_charges_i_exc(self):
        n = AlphaNeuron(v_threshold=100.0)
        n.step(1.0, 0.0)
        assert n.i_exc > 0.0 and n.i_inh == 0.0

    def test_inh_charges_i_inh(self):
        n = AlphaNeuron(v_threshold=100.0)
        n.step(0.0, 1.0)
        assert n.i_inh > 0.0 and n.i_exc == 0.0

    def test_exc_drives_v_up(self):
        n = AlphaNeuron(v_threshold=100.0)
        for _ in range(100):
            n.step(1.0, 0.0)
        assert n.v > 0.0

    def test_inh_drives_v_down(self):
        """Inhibition opposes excitation: net V = i_exc - i_inh."""
        n = AlphaNeuron(v_threshold=100.0)
        for _ in range(100):
            n.step(0.0, 1.0)
        assert n.v < 0.0

    def test_inhibition_suppresses_spiking(self):
        """Strong inhibition prevents spikes even with strong excitation."""
        n_exc = AlphaNeuron()
        n_bal = AlphaNeuron()
        s_exc = len(_run(n_exc, exc=2.0, steps=5000))
        s_bal = len(_run(n_bal, exc=2.0, steps=5000, inh=2.0))
        assert s_exc > s_bal, f"Exc only: {s_exc}, balanced: {s_bal}"

    def test_i_exc_decays_with_tau_exc(self):
        """i_exc decays with tau_exc when input is removed."""
        n = AlphaNeuron(v_threshold=100.0)
        for _ in range(100):
            n.step(1.0)
        i_exc_charged = n.i_exc
        n.step(0.0)  # remove input
        assert n.i_exc < i_exc_charged  # decayed

    def test_alpha_function_dynamics(self):
        """di_exc/dt = -i_exc/tau_exc + I_ext. Verify one step."""
        n = AlphaNeuron(v_threshold=100.0)
        I = 1.0
        n.step(I)
        # i_exc = 0 + (-0/5 + 1) * 1 = 1.0
        assert abs(n.i_exc - I * n.dt) < 1e-10


class TestAlphaFI:
    def test_zero_silent(self):
        n = AlphaNeuron()
        assert len(_run(n, exc=0.0, steps=5000)) == 0

    def test_monotonic_fi(self):
        rates = []
        for I in [0.5, 1.0, 2.0, 5.0]:
            n = AlphaNeuron()
            rates.append(len(_run(n, exc=I, steps=5000)))
        assert all(rates[i] <= rates[i + 1] for i in range(len(rates) - 1))

    def test_suprathreshold_fires(self):
        n = AlphaNeuron()
        assert len(_run(n, exc=2.0, steps=5000)) >= 100


class TestAlphaParameters:
    def test_tau_exc_affects_integration(self):
        n_fast = AlphaNeuron(tau_exc=2.0)
        n_slow = AlphaNeuron(tau_exc=20.0)
        s_fast = len(_run(n_fast, exc=1.0, steps=5000))
        s_slow = len(_run(n_slow, exc=1.0, steps=5000))
        assert s_fast != s_slow

    @pytest.mark.parametrize("dt", [0.5, 1.0, 2.0])
    def test_dt_stability(self, dt: float):
        n = AlphaNeuron(dt=dt)
        for _ in range(5000):
            n.step(1.0)
        assert np.isfinite(n.v)

    def test_deterministic(self):
        traces = []
        for _ in range(2):
            n = AlphaNeuron()
            trace = [(n.step(1.0, 0.3), n.v, n.i_exc, n.i_inh) for _ in range(200)]
            traces.append(trace)
        assert traces[0] == traces[1]


class TestAlphaPerformance:
    def test_isolation_throughput(self):
        n = AlphaNeuron()
        N = 50000
        t0 = time.perf_counter()
        for _ in range(N):
            n.step(1.0)
        elapsed = time.perf_counter() - t0
        assert N / elapsed > 50000

    def test_network_throughput(self):
        pop = Population(AlphaNeuron, n=50, label="bench")
        drive = PoissonInput(n=50, rate_hz=500.0, weight=2.0, dt=0.001, seed=42)
        mon = SpikeMonitor(pop)
        net = Network(pop, drive, mon)
        t0 = time.perf_counter()
        net.run(duration=0.5, dt=0.001, backend="python")
        elapsed = time.perf_counter() - t0
        assert 50 * 500 / elapsed > 5000


class TestAlphaPipeline:
    def test_population(self):
        assert Population(AlphaNeuron, n=10, label="alpha").n == 10

    def test_network_spikes(self):
        pop = Population(AlphaNeuron, n=10, label="alpha")
        drive = PoissonInput(n=10, rate_hz=500.0, weight=2.0, dt=0.001, seed=42)
        mon = SpikeMonitor(pop)
        net = Network(pop, drive, mon)
        net.run(duration=1.0, dt=0.001, backend="python")
        assert mon.count > 0

    def test_projection_wiring(self):
        src = Population(AlphaNeuron, n=10, label="src")
        tgt = Population(AlphaNeuron, n=10, label="tgt")
        drive = PoissonInput(n=10, rate_hz=500.0, weight=2.0, dt=0.001, seed=42)
        proj = Projection(src, tgt, weight=1.0, probability=1.0, seed=42)
        mon = SpikeMonitor(src)
        net = Network(src, tgt, drive, proj, mon)
        net.run(duration=1.0, dt=0.001, backend="python")
        assert mon.count > 0

    def test_analysis(self):
        n = AlphaNeuron()
        train = np.array([float(n.step(1.0)) for _ in range(5000)])
        sc = spike_count(train)
        assert sc >= 50
        rate = firing_rate(train, dt=0.001)
        assert rate > 0
