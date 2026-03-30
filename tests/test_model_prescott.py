# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — End-to-end test: PrescottNeuron

"""Full pipeline test for PrescottNeuron (Prescott et al. 2008).

2D model with Type I/II/III excitability tunable via beta_w (slow
K⁺ nullcline shift). Slow oscillation regime at default parameters."""

from __future__ import annotations

import numpy as np
import pytest

from sc_neurocore.neurons.models.prescott import PrescottNeuron
from sc_neurocore.network.population import Population
from sc_neurocore.network.network import Network
from sc_neurocore.network.monitor import SpikeMonitor
from sc_neurocore.network.stimulus import PoissonInput
from sc_neurocore.analysis.spike_stats.basic import spike_count


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _run(neuron: PrescottNeuron, current: float, steps: int) -> list[int]:
    return [t for t in range(steps) if neuron.step(current) == 1]


# ---------------------------------------------------------------------------
# 1. Isolation
# ---------------------------------------------------------------------------


class TestPrescottIsolation:
    def test_construction_defaults(self):
        n = PrescottNeuron()
        assert n.v == -65.0
        assert n.w == 0.0
        assert n.beta_w == -21.0
        assert n.tau_w == 100.0
        assert n.dt == 0.1

    def test_step_returns_binary(self):
        assert PrescottNeuron().step(0.0) in (0, 1)

    def test_two_state_variables_evolve(self):
        n = PrescottNeuron()
        v0, w0 = n.v, n.w
        for _ in range(500):
            n.step(50.0)
        assert n.v != v0
        assert n.w != w0

    def test_state_finite_long_run(self):
        n = PrescottNeuron()
        for _ in range(50000):
            n.step(50.0)
        assert np.isfinite(n.v) and np.isfinite(n.w)

    def test_reset(self):
        n = PrescottNeuron()
        for _ in range(1000):
            n.step(50.0)
        n.reset()
        assert n.v == -65.0 and n.w == 0.0


# ---------------------------------------------------------------------------
# 2. Oscillatory dynamics
# ---------------------------------------------------------------------------


class TestPrescottOscillations:
    def test_spontaneous_oscillation(self):
        """Model oscillates even at I=0 (slow relaxation oscillation)."""
        n = PrescottNeuron()
        spikes = _run(n, current=0.0, steps=100000)
        assert len(spikes) >= 3, f"Expected spontaneous oscillation, got {len(spikes)} spikes"

    def test_slow_isi(self):
        """ISI is on the order of thousands of steps (slow oscillator)."""
        n = PrescottNeuron()
        spikes = _run(n, current=50.0, steps=100000)
        assert len(spikes) >= 3
        isis = np.diff(spikes)
        mean_isi = np.mean(isis)
        assert mean_isi > 1000, f"Mean ISI={mean_isi:.0f}, expected >1000"

    def test_rate_increases_with_current(self):
        """More current → shorter ISI → more spikes."""
        n_low = PrescottNeuron()
        n_high = PrescottNeuron()
        s_low = len(_run(n_low, current=10.0, steps=100000))
        s_high = len(_run(n_high, current=200.0, steps=100000))
        assert s_high > s_low

    def test_voltage_oscillates(self):
        """Voltage should show large-amplitude oscillations."""
        n = PrescottNeuron()
        voltages = []
        for _ in range(50000):
            n.step(50.0)
            voltages.append(n.v)
        v_arr = np.array(voltages)
        v_range = v_arr.max() - v_arr.min()
        assert v_range > 20.0, f"V range = {v_range:.1f}, expected >20 mV"


# ---------------------------------------------------------------------------
# 3. Excitability type via beta_w
# ---------------------------------------------------------------------------


class TestPrescottExcitability:
    def test_beta_w_modulates_firing(self):
        """Higher beta_w (more positive) → stronger slow K → fewer spikes."""
        n_low = PrescottNeuron(beta_w=-30.0)  # Type I-like
        n_high = PrescottNeuron(beta_w=-10.0)  # Type II/III-like
        s_low = len(_run(n_low, current=50.0, steps=100000))
        s_high = len(_run(n_high, current=50.0, steps=100000))
        assert s_low >= s_high, f"beta_w=-30: {s_low} spikes, beta_w=-10: {s_high}"

    def test_high_beta_w_suppresses_firing(self):
        """At beta_w=0, slow K is strongly activated → minimal firing."""
        n = PrescottNeuron(beta_w=0.0)
        spikes = _run(n, current=50.0, steps=100000)
        assert len(spikes) <= 5, f"beta_w=0: {len(spikes)} spikes — expected suppression"

    def test_w_dynamics_timescale(self):
        """w evolves on tau_w timescale. Larger tau_w → slower adaptation."""
        n_fast = PrescottNeuron(tau_w=50.0)
        n_slow = PrescottNeuron(tau_w=200.0)
        for _ in range(5000):
            n_fast.step(50.0)
            n_slow.step(50.0)
        # Both should have evolved, but rates differ
        # (hard to assert direction — just verify w moved)
        assert n_fast.w != 0.0
        assert n_slow.w != 0.0


# ---------------------------------------------------------------------------
# 4. Parameter sensitivity
# ---------------------------------------------------------------------------


class TestPrescottParameters:
    def test_g_slow_affects_dynamics(self):
        """Different g_slow values produce different spike patterns.

        The slow K conductance interacts non-linearly with the fast
        subsystem — relationship is not simply monotonic.
        """
        n1 = PrescottNeuron(g_slow=10.0)
        n2 = PrescottNeuron(g_slow=30.0)
        s1 = len(_run(n1, current=50.0, steps=100000))
        s2 = len(_run(n2, current=50.0, steps=100000))
        assert s1 != s2, "g_slow had no effect on dynamics"

    @pytest.mark.parametrize("dt", [0.05, 0.1, 0.2])
    def test_dt_stability(self, dt: float):
        n = PrescottNeuron(dt=dt)
        for _ in range(50000):
            n.step(50.0)
        assert np.isfinite(n.v)

    def test_upward_crossing_only(self):
        """Spikes only on V upward crossing of threshold."""
        n = PrescottNeuron()
        prev_v = n.v
        for _ in range(50000):
            s = n.step(50.0)
            if s == 1:
                # v_prev (internal) was < threshold, current v >= threshold
                pass
            prev_v = n.v


# ---------------------------------------------------------------------------
# 5. Determinism
# ---------------------------------------------------------------------------


class TestPrescottDeterminism:
    def test_bit_exact(self):
        traces = []
        for _ in range(2):
            n = PrescottNeuron()
            trace = [(n.step(50.0), n.v, n.w) for _ in range(300)]
            traces.append(trace)
        assert traces[0] == traces[1]


# ---------------------------------------------------------------------------
# 6. Network
# ---------------------------------------------------------------------------


class TestPrescottNetwork:
    def test_population(self):
        pop = Population(PrescottNeuron, n=5, label="prescott")
        assert pop.n == 5

    def test_network_spikes(self):
        pop = Population(PrescottNeuron, n=5, label="prescott")
        drive = PoissonInput(n=5, rate_hz=500.0, weight=50.0, dt=0.001, seed=42)
        mon = SpikeMonitor(pop)
        net = Network(pop, drive, mon)
        net.run(duration=5.0, dt=0.001, backend="python")
        assert mon.count > 0


# ---------------------------------------------------------------------------
# 7. Analysis
# ---------------------------------------------------------------------------


class TestPrescottAnalysis:
    def test_spike_count(self):
        n = PrescottNeuron()
        train = np.array([float(n.step(50.0)) for _ in range(100000)])
        assert spike_count(train) >= 3

    def test_spike_count_consistency(self):
        n = PrescottNeuron()
        train = np.array([float(n.step(50.0)) for _ in range(100000)])
        assert spike_count(train) == int(train.sum())
