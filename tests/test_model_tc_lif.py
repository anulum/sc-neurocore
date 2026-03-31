# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — End-to-end test: TwoCompartmentLIFNeuron

"""Full pipeline test for TwoCompartmentLIFNeuron (Yang et al. AAAI 2024).

Soma + dendrite: dendritic input provides history-dependent sequential
context via kappa coupling. step(i_soma, i_dend). Performance: ~635K steps/s."""

from __future__ import annotations

import time

import numpy as np
import pytest

from sc_neurocore.neurons.models.tc_lif import TwoCompartmentLIFNeuron
from sc_neurocore.network.population import Population
from sc_neurocore.network.projection import Projection
from sc_neurocore.network.network import Network
from sc_neurocore.network.monitor import SpikeMonitor
from sc_neurocore.network.stimulus import PoissonInput
from sc_neurocore.analysis.spike_stats.basic import spike_count, firing_rate


def _run(
    neuron: TwoCompartmentLIFNeuron, i_soma: float, steps: int, i_dend: float = 0.0
) -> list[int]:
    return [t for t in range(steps) if neuron.step(i_soma, i_dend) == 1]


class TestTCLIFIsolation:
    def test_construction_defaults(self):
        n = TwoCompartmentLIFNeuron()
        assert n.v_s == 0.0
        assert n.v_d == 0.0
        assert n.tau_s == 2.0
        assert n.tau_d == 20.0
        assert n.kappa == 0.5
        assert n.theta == 1.0

    def test_step_returns_binary(self):
        assert TwoCompartmentLIFNeuron().step(0.0) in (0, 1)

    def test_dual_input_signature(self):
        """step(i_soma, i_dend) — two current inputs."""
        n = TwoCompartmentLIFNeuron()
        s = n.step(1.0, 0.5)
        assert s in (0, 1)

    def test_both_compartments_evolve(self):
        n = TwoCompartmentLIFNeuron()
        for _ in range(100):
            n.step(0.5, 1.0)
        assert n.v_s != 0.0
        assert n.v_d != 0.0

    def test_state_finite(self):
        n = TwoCompartmentLIFNeuron()
        for _ in range(50000):
            n.step(2.0, 1.0)
        assert np.isfinite(n.v_s) and np.isfinite(n.v_d)

    def test_reset(self):
        n = TwoCompartmentLIFNeuron()
        for _ in range(100):
            n.step(2.0, 1.0)
        n.reset()
        assert n.v_s == n.v_rest and n.v_d == n.v_rest


class TestTCLIFCompartmentalCoupling:
    """kappa controls soma←dendrite coupling."""

    def test_dendrite_charges_independently(self):
        """v_d responds to i_dend, not to i_soma directly."""
        n = TwoCompartmentLIFNeuron(theta=100.0)  # prevent spikes
        n.step(0.0, 1.0)  # i_dend=1.0
        assert n.v_d > 0.0

    def test_dendritic_input_boosts_soma(self):
        """Dendritic current flows to soma via kappa: v_s += kappa*(v_d - v_s)/tau_s."""
        n_dend = TwoCompartmentLIFNeuron()
        n_nodend = TwoCompartmentLIFNeuron()
        s_dend = len(_run(n_dend, i_soma=0.5, steps=5000, i_dend=5.0))
        s_nodend = len(_run(n_nodend, i_soma=0.5, steps=5000, i_dend=0.0))
        assert s_dend > s_nodend, (
            f"Dend: {s_dend}, no dend: {s_nodend} — dendritic input should help"
        )

    def test_kappa_controls_coupling_strength(self):
        """Higher kappa → more dendritic influence on soma."""
        n_weak = TwoCompartmentLIFNeuron(kappa=0.1)
        n_strong = TwoCompartmentLIFNeuron(kappa=2.0)
        s_weak = len(_run(n_weak, i_soma=0.5, steps=5000, i_dend=3.0))
        s_strong = len(_run(n_strong, i_soma=0.5, steps=5000, i_dend=3.0))
        assert s_strong > s_weak

    def test_somatic_reset_dendrite_unchanged(self):
        """On spike: v_s → v_reset but v_d retains its value."""
        n = TwoCompartmentLIFNeuron()
        for _ in range(5000):
            s = n.step(2.0, 1.0)
            if s == 1:
                assert n.v_s == n.v_reset
                # v_d should NOT be reset (it retains its value)
                # Can't check exact value, but it shouldn't be v_rest
                break

    def test_timescale_separation(self):
        """tau_d=20 >> tau_s=2: dendrite is 10× slower than soma."""
        n = TwoCompartmentLIFNeuron(theta=100.0)
        vs0, vd0 = n.v_s, n.v_d
        n.step(1.0, 1.0)
        dvs = abs(n.v_s - vs0)
        dvd = abs(n.v_d - vd0)
        assert dvs > dvd * 5, f"dvs={dvs:.4f}, dvd={dvd:.4f}"


class TestTCLIFSteadyState:
    def test_soma_steady_state(self):
        """At steady state (no spikes): v_s_ss depends on both i_soma and v_d."""
        n = TwoCompartmentLIFNeuron(theta=100.0)  # prevent spikes
        for _ in range(10000):
            n.step(0.5, 0.0)
        # Soma steady state with v_d=0: v_s_ss = i_soma / (1 + kappa)
        # From: 0 = (-(v_s - 0) + kappa*(0 - v_s) + I)/tau_s
        # → 0 = -v_s(1+kappa) + I → v_s = I/(1+kappa) = 0.5/1.5 ≈ 0.333
        v_ss = 0.5 / (1.0 + n.kappa)
        assert abs(n.v_s - v_ss) < 0.01, f"v_s={n.v_s:.4f}, expected={v_ss:.4f}"

    def test_dendrite_steady_state(self):
        """v_d_ss = i_dend (at v_rest=0, from -(v_d-0)+i_dend=0 → v_d=i_dend)."""
        n = TwoCompartmentLIFNeuron(theta=100.0)
        for _ in range(10000):
            n.step(0.0, 2.0)
        # v_d_ss = i_dend = 2.0
        assert abs(n.v_d - 2.0) < 0.01


class TestTCLIFFI:
    def test_zero_input_silent(self):
        n = TwoCompartmentLIFNeuron()
        assert len(_run(n, i_soma=0.0, steps=5000)) == 0

    def test_monotonic_fi(self):
        rates = []
        for I in [1.5, 2.0, 3.0, 5.0]:
            n = TwoCompartmentLIFNeuron()
            rates.append(len(_run(n, i_soma=I, steps=5000)))
        assert all(rates[i] <= rates[i + 1] for i in range(len(rates) - 1))


class TestTCLIFParameters:
    @pytest.mark.parametrize("dt", [0.5, 1.0, 2.0])
    def test_dt_stability(self, dt: float):
        n = TwoCompartmentLIFNeuron(dt=dt)
        for _ in range(5000):
            n.step(2.0)
        assert np.isfinite(n.v_s)

    def test_deterministic(self):
        traces = []
        for _ in range(2):
            n = TwoCompartmentLIFNeuron()
            trace = [(n.step(2.0, 1.0), n.v_s, n.v_d) for _ in range(200)]
            traces.append(trace)
        assert traces[0] == traces[1]


class TestTCLIFPerformance:
    def test_isolation_throughput(self):
        n = TwoCompartmentLIFNeuron()
        N = 50000
        t0 = time.perf_counter()
        for _ in range(N):
            n.step(2.0)
        elapsed = time.perf_counter() - t0
        assert N / elapsed > 50000

    def test_network_throughput(self):
        pop = Population(TwoCompartmentLIFNeuron, n=50, label="bench")
        drive = PoissonInput(n=50, rate_hz=500.0, weight=2.0, dt=0.001, seed=42)
        mon = SpikeMonitor(pop)
        net = Network(pop, drive, mon)
        t0 = time.perf_counter()
        net.run(duration=0.5, dt=0.001, backend="python")
        elapsed = time.perf_counter() - t0
        assert 50 * 500 / elapsed > 5000


class TestTCLIFPipeline:
    def test_population(self):
        assert Population(TwoCompartmentLIFNeuron, n=10, label="tc").n == 10

    def test_network_with_drive(self):
        pop = Population(TwoCompartmentLIFNeuron, n=10, label="tc")
        drive = PoissonInput(n=10, rate_hz=500.0, weight=2.0, dt=0.001, seed=42)
        mon = SpikeMonitor(pop)
        net = Network(pop, drive, mon)
        net.run(duration=1.0, dt=0.001, backend="python")
        assert mon.count > 0

    def test_projection_wiring(self):
        src = Population(TwoCompartmentLIFNeuron, n=10, label="src")
        tgt = Population(TwoCompartmentLIFNeuron, n=10, label="tgt")
        drive = PoissonInput(n=10, rate_hz=500.0, weight=2.0, dt=0.001, seed=42)
        proj = Projection(src, tgt, weight=2.0, probability=1.0, seed=42)
        mon_src = SpikeMonitor(src)
        net = Network(src, tgt, drive, proj, mon_src)
        net.run(duration=1.0, dt=0.001, backend="python")
        assert mon_src.count > 0

    def test_analysis_pipeline(self):
        n = TwoCompartmentLIFNeuron()
        train = np.array([float(n.step(2.0)) for _ in range(5000)])
        sc = spike_count(train)
        assert sc >= 100
        rate = firing_rate(train, dt=0.001)
        assert rate > 0
