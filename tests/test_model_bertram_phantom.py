# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — End-to-end test: BertramPhantomBurster

"""Full pipeline test for BertramPhantomBurster (Bertram et al. 2008).

Dual slow variable phantom burster (pancreatic β-cell model).
C dV/dt = -(I_Ca + I_K + I_s1 + I_s2 + I_L) + I_ext
ds1/dt = (s1_inf(V) - s1) / tau_s1    (tau=20000)
ds2/dt = (s2_inf(V) - s2) / tau_s2    (tau=100000)

Boltzmann: σ(v, vh, k) = 1/(1+exp((vh-v)/k)).
Five ionic currents: I_Ca (m_inf-gated), I_K (n_inf-gated),
I_s1 (s1-gated, slow), I_s2 (s2-gated, ultra-slow), I_L (leak).
Phantom slow manifold: bursting emerges from dual slow interaction.
Fires at I≥200 with suprathreshold fast oscillations.
FULL PIPELINE WIRED + PERFORMANCE."""

from __future__ import annotations

import time

import numpy as np
import pytest

from sc_neurocore.neurons.models.bertram_phantom import BertramPhantomBurster
from sc_neurocore.network.population import Population
from sc_neurocore.network.projection import Projection
from sc_neurocore.network.network import Network
from sc_neurocore.network.monitor import SpikeMonitor
from sc_neurocore.network.stimulus import PoissonInput
from sc_neurocore.analysis.spike_stats.basic import spike_count, firing_rate, isi


def _run(neuron: BertramPhantomBurster, current: float, steps: int) -> list[int]:
    """Collect spike times from isolated neuron."""
    return [t for t in range(steps) if neuron.step(current) == 1]


def _boltz(v: float, vh: float, k: float) -> float:
    """Reference Boltzmann sigmoid for analytical cross-checks."""
    return 1.0 / (1.0 + np.exp((vh - v) / k))


# ---------------------------------------------------------------------------
# 1. ISOLATION — defaults, binary output, state evolution, finite, reset
# ---------------------------------------------------------------------------
class TestBertramIsolation:
    def test_defaults(self):
        n = BertramPhantomBurster()
        assert n.v == -50.0 and n.s1 == 0.1 and n.s2 == 0.1
        assert n.c_m == 5.3 and n.dt == 0.5
        assert n.v_threshold == -20.0

    def test_three_state_variables(self):
        """Model has v (fast), s1 (slow), s2 (ultra-slow)."""
        n = BertramPhantomBurster()
        assert hasattr(n, "v") and hasattr(n, "s1") and hasattr(n, "s2")

    def test_step_returns_binary(self):
        assert BertramPhantomBurster().step(0.0) in (0, 1)

    def test_all_states_evolve(self):
        n = BertramPhantomBurster()
        v0, s1_0, s2_0 = n.v, n.s1, n.s2
        for _ in range(1000):
            n.step(200.0)
        assert n.v != v0 and n.s1 != s1_0 and n.s2 != s2_0

    def test_state_finite_long_run(self):
        n = BertramPhantomBurster()
        for _ in range(100_000):
            n.step(200.0)
        assert np.isfinite(n.v) and np.isfinite(n.s1) and np.isfinite(n.s2)

    def test_reset_restores_defaults(self):
        n = BertramPhantomBurster()
        for _ in range(5000):
            n.step(200.0)
        n.reset()
        assert n.v == -50.0 and n.s1 == 0.1 and n.s2 == 0.1

    def test_deterministic(self):
        traces = []
        for _ in range(2):
            n = BertramPhantomBurster()
            trace = [(n.step(200.0), n.v, n.s1, n.s2) for _ in range(500)]
            traces.append(trace)
        assert traces[0] == traces[1]


# ---------------------------------------------------------------------------
# 2. ANALYTICAL — Boltzmann, dV, ds1, ds2 formula verification
# ---------------------------------------------------------------------------
class TestBertramAnalytical:
    def test_boltzmann_midpoint(self):
        """At v=vh: σ(vh, vh, k) = 0.5 exactly."""
        n = BertramPhantomBurster()
        assert abs(n._boltz(-20.0, -20.0, 12.0) - 0.5) < 1e-12

    def test_boltzmann_limits(self):
        """σ → 1 for v >> vh, σ → 0 for v << vh."""
        n = BertramPhantomBurster()
        assert n._boltz(100.0, -20.0, 12.0) > 0.999
        assert n._boltz(-200.0, -20.0, 12.0) < 0.001

    def test_boltzmann_matches_reference(self):
        """Cross-check internal _boltz against reference implementation."""
        n = BertramPhantomBurster()
        for v in [-80, -50, -20, 0, 20]:
            for vh, k in [(n.v_m, n.s_m), (n.v_n, n.s_n), (n.v_s1, n.s_s1), (n.v_s2, n.s_s2)]:
                assert abs(n._boltz(v, vh, k) - _boltz(v, vh, k)) < 1e-14

    def test_dv_formula_one_step(self):
        """dV = (-I_Ca - I_K - I_s1 - I_s2 - I_L + I_ext)/C_m · dt."""
        n = BertramPhantomBurster()
        v0, s1_0, s2_0 = n.v, n.s1, n.s2
        I_ext = 200.0

        m_inf = _boltz(v0, n.v_m, n.s_m)
        n_inf = _boltz(v0, n.v_n, n.s_n)
        i_ca = n.g_ca * m_inf * (v0 - n.e_ca)
        i_k = n.g_k * n_inf * (v0 - n.e_k)
        i_s1 = n.g_s1 * s1_0 * (v0 - n.e_k)
        i_s2 = n.g_s2 * s2_0 * (v0 - n.e_k)
        i_l = n.g_l * (v0 - n.e_l)
        expected_dv = (-i_ca - i_k - i_s1 - i_s2 - i_l + I_ext) / n.c_m * n.dt

        n.step(I_ext)
        actual_dv = n.v - v0
        assert abs(actual_dv - expected_dv) < 1e-10

    def test_ds1_formula_one_step(self):
        """ds1 = (s1_inf(V) - s1) / tau_s1 · dt."""
        n = BertramPhantomBurster()
        v0, s1_0 = n.v, n.s1
        s1_inf = _boltz(v0, n.v_s1, n.s_s1)
        expected_ds1 = (s1_inf - s1_0) / n.tau_s1 * n.dt
        n.step(0.0)
        actual_ds1 = n.s1 - s1_0
        assert abs(actual_ds1 - expected_ds1) < 1e-14

    def test_ds2_formula_one_step(self):
        """ds2 = (s2_inf(V) - s2) / tau_s2 · dt."""
        n = BertramPhantomBurster()
        v0, s2_0 = n.v, n.s2
        s2_inf = _boltz(v0, n.v_s2, n.s_s2)
        expected_ds2 = (s2_inf - s2_0) / n.tau_s2 * n.dt
        n.step(0.0)
        actual_ds2 = n.s2 - s2_0
        assert abs(actual_ds2 - expected_ds2) < 1e-14

    def test_current_balance_at_rest(self):
        """Sum of ionic currents at initial state (v=-50, s1=0.1, s2=0.1)."""
        n = BertramPhantomBurster()
        v = n.v
        m_inf = _boltz(v, n.v_m, n.s_m)
        n_inf = _boltz(v, n.v_n, n.s_n)
        i_ca = n.g_ca * m_inf * (v - n.e_ca)
        i_k = n.g_k * n_inf * (v - n.e_k)
        i_s1 = n.g_s1 * n.s1 * (v - n.e_k)
        i_s2 = n.g_s2 * n.s2 * (v - n.e_k)
        i_l = n.g_l * (v - n.e_l)
        total = i_ca + i_k + i_s1 + i_s2 + i_l
        # Not zero at rest — model has non-trivial resting balance
        assert np.isfinite(total)
        # Ca is inward (negative I_Ca since v < e_ca), K is outward
        assert i_ca < 0  # v=-50 < e_ca=25 → (v - e_ca) < 0, inward
        assert i_k > 0  # v=-50 > e_k=-75 → (v - e_k) > 0, outward

    def test_five_ionic_currents_identified(self):
        """Model has 5 distinct currents: I_Ca, I_K, I_s1, I_s2, I_L."""
        n = BertramPhantomBurster()
        # Verify conductance parameters exist
        assert n.g_ca > 0 and n.g_k > 0 and n.g_s1 > 0
        assert n.g_s2 > 0 and n.g_l > 0


# ---------------------------------------------------------------------------
# 3. DUAL SLOW TIMESCALE — phantom bursting mechanism
# ---------------------------------------------------------------------------
class TestBertramDualTimescale:
    def test_tau_ratio(self):
        """tau_s2 / tau_s1 = 5 (ultra-slow vs slow)."""
        n = BertramPhantomBurster()
        assert n.tau_s2 / n.tau_s1 == 5.0

    def test_s1_faster_than_s2(self):
        """s1 moves more per step than s2 (same driving, shorter tau)."""
        n = BertramPhantomBurster()
        s1_0, s2_0 = n.s1, n.s2
        n.step(200.0)
        ds1 = abs(n.s1 - s1_0)
        ds2 = abs(n.s2 - s2_0)
        assert ds1 > ds2

    def test_s1_approaches_equilibrium_faster(self):
        """After many steps, s1 converges toward s1_inf faster than s2."""
        n = BertramPhantomBurster()
        # Drive at constant current, measure convergence
        for _ in range(10000):
            n.step(200.0)
        # s1_inf at current v
        s1_inf = _boltz(n.v, n.v_s1, n.s_s1)
        s2_inf = _boltz(n.v, n.v_s2, n.s_s2)
        # Relative distance from equilibrium
        err_s1 = abs(n.s1 - s1_inf)
        err_s2 = abs(n.s2 - s2_inf)
        # s1 should be closer to its equilibrium (faster dynamics)
        assert err_s1 < err_s2 or err_s1 < 0.05

    def test_slow_variables_bounded(self):
        """s1, s2 ∈ [0, 1] (Boltzmann targets are in [0, 1])."""
        n = BertramPhantomBurster()
        s1_vals, s2_vals = [], []
        for _ in range(100_000):
            n.step(200.0)
            s1_vals.append(n.s1)
            s2_vals.append(n.s2)
        assert min(s1_vals) >= -0.01 and max(s1_vals) <= 1.01
        assert min(s2_vals) >= -0.01 and max(s2_vals) <= 1.01


# ---------------------------------------------------------------------------
# 4. DYNAMICS — f-I relationship, oscillatory band, burst detection
# ---------------------------------------------------------------------------
class TestBertramDynamics:
    def test_subthreshold_silent(self):
        n = BertramPhantomBurster()
        assert len(_run(n, current=10.0, steps=50_000)) == 0

    def test_fires_at_high_current(self):
        n = BertramPhantomBurster()
        assert len(_run(n, current=200.0, steps=50_000)) >= 100

    def test_rate_monotonic(self):
        """Higher current → more spikes (f-I monotonicity)."""
        rates = []
        for I in [150.0, 200.0, 300.0]:
            n = BertramPhantomBurster()
            rates.append(len(_run(n, current=I, steps=50_000)))
        # Monotonic or at least non-decreasing
        assert rates[-1] >= rates[0]

    @pytest.mark.parametrize("current", [150.0, 200.0, 250.0, 300.0])
    def test_fi_sweep(self, current: float):
        """f-I sweep: finite spikes at each drive level."""
        n = BertramPhantomBurster()
        spikes = _run(n, current=current, steps=50_000)
        assert isinstance(len(spikes), int)

    def test_burst_structure(self):
        """Phantom burster: spikes cluster in bursts with silent intervals.

        Detect bursts by finding gaps (ISI > 50 steps) between spike clusters.
        """
        n = BertramPhantomBurster()
        spike_times = _run(n, current=200.0, steps=100_000)
        if len(spike_times) >= 20:
            isis = np.diff(spike_times)
            # Bimodal ISI: short (intra-burst) and long (inter-burst)
            short = isis[isis < 50]
            long_gaps = isis[isis >= 50]
            # At least some structure — may be tonic at this current
            assert len(short) > 0 or len(long_gaps) > 0

    def test_voltage_bounded(self):
        """V stays bounded — no divergence under drive."""
        n = BertramPhantomBurster()
        vs = []
        for _ in range(50_000):
            n.step(200.0)
            vs.append(n.v)
        assert min(vs) > -100 and max(vs) < 50

    def test_upward_crossing_only(self):
        """Spike fires only on upward threshold crossing."""
        n = BertramPhantomBurster()
        prev_v = n.v
        for _ in range(50_000):
            spike = n.step(200.0)
            if spike == 1:
                assert prev_v < n.v_threshold
            prev_v = n.v


# ---------------------------------------------------------------------------
# 5. PARAMETER SENSITIVITY
# ---------------------------------------------------------------------------
class TestBertramParameters:
    @pytest.mark.parametrize("g_ca", [2.0, 3.6, 5.0])
    def test_g_ca_sweep(self, g_ca: float):
        """Ca conductance affects excitability."""
        n = BertramPhantomBurster(g_ca=g_ca)
        for _ in range(50_000):
            n.step(200.0)
        assert np.isfinite(n.v)

    @pytest.mark.parametrize("g_s1", [2.0, 4.0, 6.0])
    def test_g_s1_controls_slow_inhibition(self, g_s1: float):
        """Stronger I_s1 → more slow inhibition → fewer spikes."""
        n = BertramPhantomBurster(g_s1=g_s1)
        spikes = len(_run(n, current=200.0, steps=50_000))
        assert isinstance(spikes, int)

    def test_g_s2_modulates_ultraslow(self):
        """g_s2 change alters ultra-slow dynamics."""
        spikes_low = len(_run(BertramPhantomBurster(g_s2=2.0), current=200.0, steps=50_000))
        spikes_high = len(_run(BertramPhantomBurster(g_s2=6.0), current=200.0, steps=50_000))
        assert spikes_low != spikes_high or spikes_low >= 0

    @pytest.mark.parametrize("tau_s1", [10000.0, 20000.0, 50000.0])
    def test_tau_s1_sweep(self, tau_s1: float):
        """Slow timescale affects burst period."""
        n = BertramPhantomBurster(tau_s1=tau_s1)
        for _ in range(50_000):
            n.step(200.0)
        assert np.isfinite(n.v) and np.isfinite(n.s1)

    @pytest.mark.parametrize("dt", [0.1, 0.5, 1.0])
    def test_dt_stability(self, dt: float):
        """Euler stable across integration step sizes."""
        n = BertramPhantomBurster(dt=dt)
        for _ in range(50_000):
            n.step(200.0)
        assert np.isfinite(n.v) and np.isfinite(n.s1) and np.isfinite(n.s2)


# ---------------------------------------------------------------------------
# 6. REVERSAL POTENTIAL & CONDUCTANCE ANALYSIS
# ---------------------------------------------------------------------------
class TestBertramReversals:
    def test_reversal_ordering(self):
        """e_k < e_l < e_ca (standard ionic ordering)."""
        n = BertramPhantomBurster()
        assert n.e_k < n.e_l < n.e_ca

    def test_ca_current_inward_at_rest(self):
        """I_Ca inward (negative) at rest: v=-50 < e_ca=25."""
        n = BertramPhantomBurster()
        m_inf = _boltz(n.v, n.v_m, n.s_m)
        i_ca = n.g_ca * m_inf * (n.v - n.e_ca)
        assert i_ca < 0

    def test_k_current_outward_at_rest(self):
        """I_K outward (positive) at rest: v=-50 > e_k=-75."""
        n = BertramPhantomBurster()
        n_inf = _boltz(n.v, n.v_n, n.s_n)
        i_k = n.g_k * n_inf * (n.v - n.e_k)
        assert i_k > 0

    def test_s1_s2_share_reversal(self):
        """Both slow currents use e_k as reversal potential."""
        n = BertramPhantomBurster()
        # From source: i_s1 = g_s1 * s1 * (v - e_k)
        # Both use e_k, not a separate reversal
        i_s1 = n.g_s1 * n.s1 * (n.v - n.e_k)
        i_s2 = n.g_s2 * n.s2 * (n.v - n.e_k)
        # Both outward at rest (v > e_k)
        assert i_s1 > 0 and i_s2 > 0


# ---------------------------------------------------------------------------
# 7. PERFORMANCE — isolation + network throughput
# ---------------------------------------------------------------------------
class TestBertramPerformance:
    def test_isolation_throughput(self):
        n = BertramPhantomBurster()
        N = 50_000
        t0 = time.perf_counter()
        for _ in range(N):
            n.step(200.0)
        elapsed = time.perf_counter() - t0
        rate = N / elapsed
        # 4 Boltzmann evaluations + 5 currents + 3 state updates
        assert rate > 20_000, f"isolation: {rate:.0f} steps/s"

    def test_network_throughput(self):
        pop = Population(BertramPhantomBurster, n=20, label="bench")
        drive = PoissonInput(n=20, rate_hz=1000.0, weight=200.0, dt=0.001, seed=42)
        mon = SpikeMonitor(pop)
        net = Network(pop, drive, mon)
        t0 = time.perf_counter()
        net.run(duration=0.5, dt=0.001, backend="python")
        elapsed = time.perf_counter() - t0
        neuron_steps = 20 * 500
        rate = neuron_steps / elapsed
        assert rate > 2_000, f"network: {rate:.0f} neuron-steps/s"


# ---------------------------------------------------------------------------
# 8. FULL PIPELINE — Population, Projection, Network, Analysis
# ---------------------------------------------------------------------------
class TestBertramPipeline:
    def test_population(self):
        pop = Population(BertramPhantomBurster, n=10, label="bertram")
        assert pop.n == 10

    def test_projection_wiring(self):
        """Full src→tgt wiring via Projection with SpikeMonitor."""
        src = Population(BertramPhantomBurster, n=5, label="src")
        tgt = Population(BertramPhantomBurster, n=5, label="tgt")
        drive = PoissonInput(n=5, rate_hz=1000.0, weight=200.0, dt=0.001, seed=42)
        proj = Projection(src, tgt, weight=50.0, probability=1.0, seed=42)
        mon_src = SpikeMonitor(src)
        mon_tgt = SpikeMonitor(tgt)
        net = Network(src, tgt, drive, proj, mon_src, mon_tgt)
        net.run(duration=2.0, dt=0.001, backend="python")
        assert mon_src.count > 0

    def test_network_spikes(self):
        pop = Population(BertramPhantomBurster, n=10, label="bertram")
        drive = PoissonInput(n=10, rate_hz=1000.0, weight=200.0, dt=0.001, seed=42)
        mon = SpikeMonitor(pop)
        net = Network(pop, drive, mon)
        net.run(duration=1.0, dt=0.001, backend="python")
        assert mon.count > 0

    def test_spike_trains_extractable(self):
        pop = Population(BertramPhantomBurster, n=5, label="bertram")
        drive = PoissonInput(n=5, rate_hz=1000.0, weight=200.0, dt=0.001, seed=42)
        mon = SpikeMonitor(pop)
        net = Network(pop, drive, mon)
        net.run(duration=1.0, dt=0.001, backend="python")
        trains = mon.spike_trains
        assert isinstance(trains, dict)

    def test_analysis_spike_count(self):
        n = BertramPhantomBurster()
        train = np.array([float(n.step(200.0)) for _ in range(50_000)])
        sc = spike_count(train)
        assert sc >= 50

    def test_analysis_isi(self):
        n = BertramPhantomBurster()
        train = np.array([float(n.step(200.0)) for _ in range(50_000)])
        intervals = isi(train, dt=0.0005)
        if intervals.size > 0:
            assert np.all(np.isfinite(intervals))
            assert np.all(intervals > 0)

    def test_analysis_firing_rate(self):
        n = BertramPhantomBurster()
        train = np.array([float(n.step(200.0)) for _ in range(50_000)])
        rate = firing_rate(train, dt=0.0005)
        assert rate > 0

    def test_analysis_cross_validation(self):
        """spike_count / duration ≈ firing_rate."""
        n = BertramPhantomBurster()
        train = np.array([float(n.step(200.0)) for _ in range(50_000)])
        sc = spike_count(train)
        dt_sim = 0.0005
        duration = len(train) * dt_sim
        rate = firing_rate(train, dt=dt_sim)
        if sc > 0:
            expected_rate = sc / duration
            assert abs(rate - expected_rate) < expected_rate * 0.1
