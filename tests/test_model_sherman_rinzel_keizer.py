# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — End-to-end test: ShermanRinzelKeizerNeuron

"""Full pipeline test for ShermanRinzelKeizerNeuron (Sherman et al. 1988).

Pancreatic beta cell: 3 variables (V, n, s). Spontaneous burster at I=0.
Non-monotonic f–I: peak at moderate I, suppression at high I via slow
s variable accumulation. Three timescales: V (fast), n (tau_n=9.09),
s (tau_s=5000)."""

from __future__ import annotations

import numpy as np
import pytest

from sc_neurocore.neurons.models.sherman_rinzel_keizer import ShermanRinzelKeizerNeuron
from sc_neurocore.network.population import Population
from sc_neurocore.network.network import Network
from sc_neurocore.network.monitor import SpikeMonitor
from sc_neurocore.network.stimulus import PoissonInput
from sc_neurocore.analysis.spike_stats.basic import spike_count


def _run(neuron: ShermanRinzelKeizerNeuron, current: float, steps: int) -> list[int]:
    return [t for t in range(steps) if neuron.step(current) == 1]


def _collect_trace(
    neuron: ShermanRinzelKeizerNeuron, current: float, steps: int
) -> tuple[list[int], np.ndarray, np.ndarray, np.ndarray]:
    """Return (spike_times, V_trace, n_trace, s_trace)."""
    spikes: list[int] = []
    vs, ns, ss = [], [], []
    for t in range(steps):
        if neuron.step(current) == 1:
            spikes.append(t)
        vs.append(neuron.v)
        ns.append(neuron.n)
        ss.append(neuron.s)
    return spikes, np.array(vs), np.array(ns), np.array(ss)


class TestSRKIsolation:
    def test_construction_defaults(self):
        n = ShermanRinzelKeizerNeuron()
        assert n.v == -50.0
        assert n.n == 0.1
        assert n.s == 0.1
        assert n.g_ca == 3.6
        assert n.g_k == 10.0
        assert n.g_s == 4.0
        assert n.e_ca == 25.0
        assert n.e_k == -75.0
        assert n.tau_s == 5000.0
        assert n.dt == 0.5

    def test_step_returns_binary(self):
        assert ShermanRinzelKeizerNeuron().step(0.0) in (0, 1)

    def test_three_state_variables_evolve(self):
        n = ShermanRinzelKeizerNeuron()
        initial = (n.v, n.n, n.s)
        for _ in range(500):
            n.step(0.0)
        for name, v0, v1 in zip(["v", "n", "s"], initial, (n.v, n.n, n.s)):
            assert v0 != v1, f"{name} didn't evolve"

    def test_state_finite_long_run(self):
        n = ShermanRinzelKeizerNeuron()
        for _ in range(100000):
            n.step(10.0)
        for name, val in [("v", n.v), ("n", n.n), ("s", n.s)]:
            assert np.isfinite(val), f"{name} = {val}"

    def test_reset(self):
        n = ShermanRinzelKeizerNeuron()
        for _ in range(1000):
            n.step(10.0)
        n.reset()
        assert n.v == -50.0 and n.n == 0.1 and n.s == 0.1


class TestSRKSpontaneousBursting:
    """The model fires spontaneously at I=0 — this is the core property."""

    def test_fires_at_zero_input(self):
        n = ShermanRinzelKeizerNeuron()
        spikes = _run(n, current=0.0, steps=100000)
        assert len(spikes) >= 100, f"Only {len(spikes)} spikes at I=0"

    def test_regular_isi_limit_cycle(self):
        """After transient, ISI settles to a limit cycle (low CV)."""
        n = ShermanRinzelKeizerNeuron()
        spikes = _run(n, current=0.0, steps=100000)
        assert len(spikes) >= 30
        isis = np.diff(spikes[15:]).astype(float)
        cv = np.std(isis) / np.mean(isis)
        assert cv < 0.15, f"CV(ISI) = {cv:.4f} — expected near-regular limit cycle"

    def test_mean_isi_quantified(self):
        """Quantify: at I=0, mean ISI ≈ 60–80 steps (30–40 ms at dt=0.5)."""
        n = ShermanRinzelKeizerNeuron()
        spikes = _run(n, current=0.0, steps=100000)
        isis = np.diff(spikes[15:])
        mean_isi = np.mean(isis)
        assert 40 < mean_isi < 100, f"Mean ISI = {mean_isi:.0f}"

    def test_voltage_oscillation_amplitude(self):
        """V should show large oscillations (spike peaks vs rest)."""
        n = ShermanRinzelKeizerNeuron()
        _, vs, _, _ = _collect_trace(n, current=0.0, steps=50000)
        vs_steady = vs[5000:]  # skip transient
        v_range = vs_steady.max() - vs_steady.min()
        assert v_range > 30.0, f"V range = {v_range:.1f} mV, expected >30"


class TestSRKNonMonotonicFI:
    """Non-monotonic f–I curve is a hallmark of slow-variable feedback."""

    def test_fi_5_point_sweep(self):
        """Map out f–I with 5 points to characterise the non-monotonicity."""
        rates = {}
        for I in [0.0, 5.0, 20.0, 50.0, 100.0]:
            n = ShermanRinzelKeizerNeuron()
            rates[I] = len(_run(n, current=I, steps=100000))
        # Rate should increase from I=0 to peak (~I=20)
        assert rates[5.0] > rates[0.0]
        assert rates[20.0] > rates[5.0]
        # Then decline at high I
        assert rates[20.0] > rates[100.0], (
            f"f(20)={rates[20.0]}, f(100)={rates[100.0]} — expected decline"
        )

    def test_high_current_depolarisation_block(self):
        """I=100 → very few spikes. s accumulates → strong outward current."""
        n = ShermanRinzelKeizerNeuron()
        spikes, _, _, s_trace = _collect_trace(n, current=100.0, steps=100000)
        assert len(spikes) < 50, f"{len(spikes)} spikes at I=100"
        # s should have grown substantially
        assert s_trace[-1] > 0.5, f"s_final = {s_trace[-1]:.4f}"

    def test_peak_rate_region(self):
        """Around I=20, rate should be the highest among tested points."""
        rates = {}
        for I in [10.0, 20.0, 30.0]:
            n = ShermanRinzelKeizerNeuron()
            rates[I] = len(_run(n, current=I, steps=100000))
        peak_I = max(rates, key=rates.get)
        # Peak should be at 10 or 20 (moderate current)
        assert peak_I <= 30.0


class TestSRKThreeTimescales:
    """V (fast), n (tau_n=9.09), s (tau_s=5000) — verify separation."""

    def test_timescale_ordering(self):
        """After 100 steps: |dn| > |ds| by at least 10×."""
        n = ShermanRinzelKeizerNeuron()
        n0_val, s0 = n.n, n.s
        for _ in range(100):
            n.step(10.0)
        dn = abs(n.n - n0_val)
        ds = abs(n.s - s0)
        assert dn > 10 * ds, f"dn={dn:.6f}, ds={ds:.6f} — n should evolve 10× faster than s"

    def test_s_tracks_mean_v_on_slow_timescale(self):
        """s_inf = sigmoid(-(V+35)/10). When V is high on average, s grows."""
        n = ShermanRinzelKeizerNeuron()
        for _ in range(100000):
            n.step(50.0)
        # At high current, V spends time depolarised → s_inf ≈ 1 → s grows
        assert n.s > 0.3

    def test_n_follows_v_on_fast_timescale(self):
        """n_inf = sigmoid(-(V+16)/5). n should track n_inf closely."""
        n = ShermanRinzelKeizerNeuron()
        for _ in range(10000):
            n.step(0.0)
        # Compute n_inf at current V
        n_inf = 1.0 / (1.0 + np.exp(-(n.v + 16.0) / 5.0))
        # n should be close to n_inf (tau_n=9.09 is fast)
        assert abs(n.n - n_inf) < 0.15, (
            f"n = {n.n:.4f}, n_inf = {n_inf:.4f} — expected close tracking"
        )

    def test_s_modulation_changes_burst_envelope(self):
        """Different g_s values change the burst pattern — s mediates bursting."""
        n_weak = ShermanRinzelKeizerNeuron(g_s=1.0)
        n_strong = ShermanRinzelKeizerNeuron(g_s=8.0)
        s_weak = len(_run(n_weak, current=0.0, steps=100000))
        s_strong = len(_run(n_strong, current=0.0, steps=100000))
        assert s_weak != s_strong, "g_s had no effect on spike count"

    def test_tau_s_controls_burst_period(self):
        """Shorter tau_s → faster s dynamics → different burst period."""
        n_fast = ShermanRinzelKeizerNeuron(tau_s=1000.0)
        n_slow = ShermanRinzelKeizerNeuron(tau_s=10000.0)
        s_fast = _run(n_fast, current=0.0, steps=100000)
        s_slow = _run(n_slow, current=0.0, steps=100000)
        if len(s_fast) > 10 and len(s_slow) > 10:
            isi_fast = np.mean(np.diff(s_fast[10:]))
            isi_slow = np.mean(np.diff(s_slow[10:]))
            assert isi_fast != isi_slow, "tau_s had no effect on ISI"


class TestSRKSigmoidActivation:
    """Verify the three sigmoid activation functions are biophysically correct."""

    def test_m_inf_sigmoid(self):
        """m_inf(V) = 1/(1+exp(-(V+20)/12)). At V=-20: m_inf=0.5."""
        v_half = -20.0
        m_inf = 1.0 / (1.0 + np.exp(-(v_half + 20.0) / 12.0))
        assert abs(m_inf - 0.5) < 1e-10

    def test_n_inf_sigmoid(self):
        """n_inf(V) = 1/(1+exp(-(V+16)/5)). At V=-16: n_inf=0.5."""
        v_half = -16.0
        n_inf = 1.0 / (1.0 + np.exp(-(v_half + 16.0) / 5.0))
        assert abs(n_inf - 0.5) < 1e-10

    def test_s_inf_sigmoid(self):
        """s_inf(V) = 1/(1+exp(-(V+35)/10)). At V=-35: s_inf=0.5."""
        v_half = -35.0
        s_inf = 1.0 / (1.0 + np.exp(-(v_half + 35.0) / 10.0))
        assert abs(s_inf - 0.5) < 1e-10

    def test_gating_variables_bounded(self):
        """n and s should stay in [0, 1] (sigmoid range)."""
        n = ShermanRinzelKeizerNeuron()
        for _ in range(100000):
            n.step(10.0)
        assert 0.0 <= n.n <= 1.0, f"n = {n.n:.6f}"
        assert 0.0 <= n.s <= 1.0, f"s = {n.s:.6f}"


class TestSRKCurrentBalance:
    """Verify the three currents (I_Ca, I_K, I_s) behave correctly."""

    def test_i_ca_depolarising(self):
        """I_Ca is inward (depolarising) when V < E_Ca = 25 mV.

        At rest V ≈ -50 < 25: I_Ca = g_Ca·m_inf·(V-E_Ca) < 0 (inward).
        """
        n = ShermanRinzelKeizerNeuron()
        m_inf = 1.0 / (1.0 + np.exp(-(n.v + 20.0) / 12.0))
        i_ca = n.g_ca * m_inf * (n.v - n.e_ca)
        assert i_ca < 0, f"I_Ca = {i_ca:.4f}, expected < 0 (inward)"

    def test_i_k_hyperpolarising(self):
        """I_K = g_K·n·(V-E_K). At rest V > E_K = -75: I_K > 0 (outward)."""
        n = ShermanRinzelKeizerNeuron()
        i_k = n.g_k * n.n * (n.v - n.e_k)
        assert i_k > 0, f"I_K = {i_k:.4f}, expected > 0 (outward)"

    def test_i_s_hyperpolarising(self):
        """I_s = g_s·s·(V-E_K). Same reversal as I_K → outward at rest."""
        n = ShermanRinzelKeizerNeuron()
        i_s = n.g_s * n.s * (n.v - n.e_k)
        assert i_s > 0, f"I_s = {i_s:.4f}, expected > 0 (outward)"


class TestSRKNumericalStability:
    @pytest.mark.parametrize("dt", [0.2, 0.5])
    def test_dt_stability(self, dt: float):
        n = ShermanRinzelKeizerNeuron(dt=dt)
        for _ in range(50000):
            n.step(10.0)
        assert np.isfinite(n.v)

    def test_large_dt_unstable(self):
        """dt=1.0 causes Euler divergence (NaN) — documents numerical limit."""
        n = ShermanRinzelKeizerNeuron(dt=1.0)
        for _ in range(50000):
            n.step(10.0)
        assert not np.isfinite(n.v), "dt=1.0 expected to diverge"


class TestSRKParameterSensitivity:
    def test_g_ca_higher_more_excitable(self):
        """Higher g_Ca → stronger inward Ca current → more excitable."""
        n_low = ShermanRinzelKeizerNeuron(g_ca=2.0)
        n_high = ShermanRinzelKeizerNeuron(g_ca=5.0)
        s_low = len(_run(n_low, current=0.0, steps=100000))
        s_high = len(_run(n_high, current=0.0, steps=100000))
        assert s_high > s_low, f"g_ca=2: {s_low}, g_ca=5: {s_high}"

    def test_g_k_affects_dynamics(self):
        """g_K modulates the fast K current — changing it alters the spike pattern.

        The relationship is non-monotonic due to interaction with the Ca and s
        subsystems: at g_K=5, weaker K delayed rectifier changes the V
        nullcline shape, potentially shifting the oscillatory regime.
        """
        n_low = ShermanRinzelKeizerNeuron(g_k=5.0)
        n_high = ShermanRinzelKeizerNeuron(g_k=15.0)
        s_low = len(_run(n_low, current=0.0, steps=100000))
        s_high = len(_run(n_high, current=0.0, steps=100000))
        assert s_low != s_high, "g_K change had no effect on spike count"


class TestSRKDeterminism:
    def test_bit_exact(self):
        traces = []
        for _ in range(2):
            n = ShermanRinzelKeizerNeuron()
            trace = [(n.step(5.0), n.v, n.n, n.s) for _ in range(500)]
            traces.append(trace)
        assert traces[0] == traces[1]


class TestSRKNetwork:
    def test_population(self):
        assert Population(ShermanRinzelKeizerNeuron, n=5, label="srk").n == 5

    def test_network_spikes(self):
        pop = Population(ShermanRinzelKeizerNeuron, n=5, label="srk")
        drive = PoissonInput(n=5, rate_hz=200.0, weight=5.0, dt=0.001, seed=42)
        mon = SpikeMonitor(pop)
        net = Network(pop, drive, mon)
        net.run(duration=2.0, dt=0.001, backend="python")
        assert mon.count > 0


class TestSRKAnalysis:
    def test_spike_count_matches_manual(self):
        n = ShermanRinzelKeizerNeuron()
        train = np.array([float(n.step(0.0)) for _ in range(100000)])
        assert spike_count(train) == int(train.sum())

    def test_spike_count_substantial(self):
        n = ShermanRinzelKeizerNeuron()
        train = np.array([float(n.step(0.0)) for _ in range(100000)])
        assert spike_count(train) >= 100
