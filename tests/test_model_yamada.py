# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — End-to-end test: YamadaNeuron

"""Full pipeline test for YamadaNeuron (Yamada, Kashimori & Kambara 1989).

Subcritical Hopf burster: 3 ODEs (V, n, q). q is ultra-slow (tau_q=300ms)
controlling burst envelope. Square-wave bursting via slow modulation."""

from __future__ import annotations

import numpy as np
import pytest

from sc_neurocore.neurons.models.yamada import YamadaNeuron
from sc_neurocore.network.population import Population
from sc_neurocore.network.projection import Projection
from sc_neurocore.network.network import Network
from sc_neurocore.network.monitor import SpikeMonitor
from sc_neurocore.network.stimulus import PoissonInput
from sc_neurocore.analysis.spike_stats.basic import spike_count, firing_rate


def _run(neuron: YamadaNeuron, current: float, steps: int) -> list[int]:
    return [t for t in range(steps) if neuron.step(current) == 1]


class TestYamadaIsolation:
    def test_construction_defaults(self):
        n = YamadaNeuron()
        assert n.v == -60.0
        assert n.n == 0.1
        assert n.q == 0.0
        assert n.tau_q == 300.0
        assert n.dt == 0.05

    def test_step_returns_binary(self):
        assert YamadaNeuron().step(0.0) in (0, 1)

    def test_three_variables_evolve(self):
        n = YamadaNeuron()
        initial = (n.v, n.n, n.q)
        for _ in range(500):
            n.step(50.0)
        for name, v0, v1 in zip(["v", "n", "q"], initial, (n.v, n.n, n.q)):
            assert v0 != v1, f"{name} didn't evolve"

    def test_state_finite(self):
        n = YamadaNeuron()
        for _ in range(200000):
            n.step(50.0)
        assert all(np.isfinite(v) for v in [n.v, n.n, n.q])

    def test_reset(self):
        n = YamadaNeuron()
        for _ in range(1000):
            n.step(50.0)
        n.reset()
        assert n.v == -60.0 and n.n == 0.1 and n.q == 0.0


class TestYamadaSlowDynamics:
    def test_q_evolves_slowly(self):
        """q (tau_q=300) evolves much slower than n."""
        n = YamadaNeuron()
        n0, q0 = n.n, n.q
        for _ in range(100):
            n.step(50.0)
        dn = abs(n.n - n0)
        dq = abs(n.q - q0)
        assert dn > 5 * dq, f"dn={dn:.6f}, dq={dq:.6f}"

    def test_q_accumulates_with_current(self):
        """Higher current → V spends more time depolarised → q_inf → q grows."""
        n_low = YamadaNeuron()
        n_high = YamadaNeuron()
        for _ in range(200000):
            n_low.step(10.0)
            n_high.step(100.0)
        assert n_high.q > n_low.q

    def test_q_modulates_excitability(self):
        """Higher g_q → stronger q current → different firing."""
        n_weak = YamadaNeuron(g_q=1.0)
        n_strong = YamadaNeuron(g_q=10.0)
        s_weak = len(_run(n_weak, current=50.0, steps=200000))
        s_strong = len(_run(n_strong, current=50.0, steps=200000))
        assert s_weak != s_strong

    def test_tau_q_controls_convergence_speed(self):
        """Faster tau_q → q converges to q_inf faster."""
        n_fast = YamadaNeuron(tau_q=100.0)
        n_slow = YamadaNeuron(tau_q=1000.0)
        # Check after SHORT run (not enough for both to reach steady state)
        for _ in range(5000):
            n_fast.step(50.0)
            n_slow.step(50.0)
        # Fast tau_q should have moved q further from initial 0.0
        assert n_fast.q > n_slow.q, (
            f"fast q={n_fast.q:.6f}, slow q={n_slow.q:.6f}"
        )


class TestYamadaFI:
    def test_silent_at_zero(self):
        n = YamadaNeuron()
        assert len(_run(n, current=0.0, steps=50000)) == 0

    def test_fires_at_high_current(self):
        n = YamadaNeuron()
        assert len(_run(n, current=50.0, steps=200000)) >= 10

    def test_rate_increases_with_current(self):
        n1 = YamadaNeuron()
        n2 = YamadaNeuron()
        s1 = len(_run(n1, current=30.0, steps=200000))
        s2 = len(_run(n2, current=100.0, steps=200000))
        assert s2 > s1


class TestYamadaHHProperties:
    def test_gating_bounded(self):
        n = YamadaNeuron()
        for _ in range(200000):
            n.step(50.0)
        assert -0.01 <= n.n <= 1.01, f"n = {n.n}"
        assert -0.01 <= n.q <= 1.01, f"q = {n.q}"

    def test_sigmoid_half_activations(self):
        """m_inf(-30) = 0.5, n_inf(-30) = 0.5, q_inf(-50) = 0.5."""
        m_inf = 1.0 / (1.0 + np.exp(-(-30.0 + 30.0) / 9.5))
        assert abs(m_inf - 0.5) < 1e-10
        n_inf = 1.0 / (1.0 + np.exp(-(-30.0 + 30.0) / 10.0))
        assert abs(n_inf - 0.5) < 1e-10
        q_inf = 1.0 / (1.0 + np.exp(-(-50.0 + 50.0) / 10.0))
        assert abs(q_inf - 0.5) < 1e-10

    def test_na_inactivation_via_n(self):
        """Na current uses (1-n) as inactivation: I_Na = g_Na·m_inf³·(1-n)·(V-E_Na)."""
        n = YamadaNeuron()
        m_inf = 1.0 / (1.0 + np.exp(-(n.v + 30.0) / 9.5))
        i_na = n.g_na * m_inf**3 * (1.0 - n.n) * (n.v - n.e_na)
        # At rest V=-60 < E_Na=60: (V-E_Na) < 0, m_inf small, (1-n)=0.9
        assert i_na < 0  # inward

    @pytest.mark.parametrize("dt", [0.02, 0.05, 0.1])
    def test_dt_stability(self, dt: float):
        n = YamadaNeuron(dt=dt)
        for _ in range(100000):
            n.step(50.0)
        assert np.isfinite(n.v)

    def test_deterministic(self):
        traces = []
        for _ in range(2):
            n = YamadaNeuron()
            trace = [(n.step(50.0), n.v) for _ in range(200)]
            traces.append(trace)
        assert traces[0] == traces[1]


class TestYamadaPipeline:
    def test_population(self):
        assert Population(YamadaNeuron, n=5, label="yam").n == 5

    def test_network_with_drive(self):
        pop = Population(YamadaNeuron, n=5, label="yam")
        drive = PoissonInput(n=5, rate_hz=500.0, weight=50.0, dt=0.001, seed=42)
        mon = SpikeMonitor(pop)
        net = Network(pop, drive, mon)
        net.run(duration=5.0, dt=0.001, backend="python")
        assert mon.count > 0

    def test_projection_wiring(self):
        """Projection from strong source adds current to target population.

        Yamada needs high sustained current to fire, so we verify the
        projection is wired by checking the source fires and using
        strong enough drive + projection weight.
        """
        src = Population(YamadaNeuron, n=10, label="src")
        tgt = Population(YamadaNeuron, n=10, label="tgt")
        drive_src = PoissonInput(n=10, rate_hz=500.0, weight=50.0, dt=0.001, seed=42)
        drive_tgt = PoissonInput(n=10, rate_hz=500.0, weight=50.0, dt=0.001, seed=99)
        proj = Projection(src, tgt, weight=30.0, probability=1.0, seed=42)
        mon_src = SpikeMonitor(src)
        mon_tgt = SpikeMonitor(tgt)
        net = Network(src, tgt, drive_src, drive_tgt, proj, mon_src, mon_tgt)
        net.run(duration=5.0, dt=0.001, backend="python")
        assert mon_src.count > 0, "Source should fire"
        # Target has both its own drive and projection input
        assert mon_tgt.count > 0, "Target should fire with drive + projection"

    def test_analysis_pipeline(self):
        n = YamadaNeuron()
        train = np.array([float(n.step(100.0)) for _ in range(200000)])
        sc = spike_count(train)
        assert sc >= 10
        rate = firing_rate(train, dt=0.00005)  # dt=0.05ms per step
        assert rate > 0
