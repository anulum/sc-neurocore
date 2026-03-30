# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — End-to-end test: SpikeResponseNeuron

"""Full pipeline test for SpikeResponseNeuron (SRM0, Gerstner 1995).

Kernel-based: v(t) = η(tss) + κ(I). η is refractory afterpotential
(decays from eta_reset), κ is instantaneous input kernel.
No voltage accumulation — v computed fresh each step.

Timing: eta uses time_since_spike BEFORE increment. After spike,
tss=0; next step uses eta(0) = eta_reset, then tss becomes dt."""

from __future__ import annotations

import numpy as np
import pytest

from sc_neurocore.neurons.models.spike_response import SpikeResponseNeuron
from sc_neurocore.network.population import Population
from sc_neurocore.network.projection import Projection
from sc_neurocore.network.network import Network
from sc_neurocore.network.monitor import SpikeMonitor
from sc_neurocore.network.stimulus import PoissonInput
from sc_neurocore.analysis.spike_stats.basic import spike_count, isi, firing_rate


def _kappa(I: float, dt: float, tau_kappa: float) -> float:
    return I * (1.0 - np.exp(-dt / tau_kappa))


def _eta(tss: float, eta_reset: float, tau_eta: float) -> float:
    if tss >= 100.0:
        return 0.0
    return eta_reset * np.exp(-tss / tau_eta)


def _run(neuron: SpikeResponseNeuron, current: float, steps: int) -> list[int]:
    return [t for t in range(steps) if neuron.step(current) == 1]


class TestSRMIsolation:
    def test_construction_defaults(self):
        n = SpikeResponseNeuron()
        assert n.v == 0.0
        assert n.v_threshold == 1.0
        assert n.tau_eta == 10.0
        assert n.tau_kappa == 5.0
        assert n.eta_reset == -5.0
        assert n.time_since_spike == 1000.0
        assert n.dt == 1.0

    def test_step_returns_binary(self):
        assert SpikeResponseNeuron().step(0.0) in (0, 1)

    def test_v_no_accumulation(self):
        """v = η + κ, computed fresh. No memory of previous v."""
        n = SpikeResponseNeuron()
        n.step(5.0)
        v1 = n.v
        n.step(5.0)
        v2 = n.v
        # v should be nearly identical (both have tss > 1000, eta ≈ 0)
        assert abs(v2 - v1) < 0.01

    def test_reset(self):
        n = SpikeResponseNeuron()
        for _ in range(50):
            n.step(10.0)
        n.reset()
        assert n.v == 0.0
        assert n.time_since_spike == 1000.0


class TestSRMRefractoryKernel:
    def test_eta_at_tss_zero(self):
        """After spike, tss=0. Next step: η uses tss=0 → η = eta_reset exactly."""
        n = SpikeResponseNeuron()
        n.step(10.0)  # spike, tss → 0
        assert n.time_since_spike == 0.0
        # Next step: eta(tss=0) = eta_reset = -5.0
        n.step(0.0)  # zero input → v = eta(0) + 0 = -5.0
        assert abs(n.v - n.eta_reset) < 1e-10, f"v={n.v}, expected eta_reset={n.eta_reset}"

    def test_eta_decays_step_by_step(self):
        """Track η decay: at step k after spike, η uses tss = k-1."""
        n = SpikeResponseNeuron()
        n.step(10.0)  # spike → tss = 0
        for k in range(1, 15):
            n.step(0.0)
            # eta was computed at tss = k-1 (before increment)
            expected_eta = _eta(float(k - 1), n.eta_reset, n.tau_eta)
            assert abs(n.v - expected_eta) < 1e-6, (
                f"Step {k} after spike: v={n.v:.6f}, eta(tss={k - 1})={expected_eta:.6f}"
            )

    def test_eta_zero_beyond_100(self):
        """η clipped to 0 when tss ≥ 100 (code optimisation)."""
        n = SpikeResponseNeuron()
        n.time_since_spike = 100.0
        n.step(0.0)
        assert n.v == 0.0

    def test_refractory_prevents_immediate_respike(self):
        """Strong η suppression prevents re-spike even with strong input."""
        n = SpikeResponseNeuron()
        n.step(10.0)  # spike
        s = n.step(10.0)  # eta(0) = -5, kappa = 1.81 → v ≈ -3.19
        assert s == 0

    def test_v_after_spike_equals_eta_plus_kappa(self):
        """Verify v = η(tss) + κ(I) exactly for several steps post-spike."""
        n = SpikeResponseNeuron()
        n.step(10.0)  # spike
        I = 8.0
        for k in range(1, 10):
            n.step(I)
            expected = _eta(float(k - 1), n.eta_reset, n.tau_eta) + _kappa(I, n.dt, n.tau_kappa)
            assert abs(n.v - expected) < 1e-6, f"k={k}: v={n.v:.6f}, expected={expected:.6f}"


class TestSRMInputKernel:
    def test_kappa_formula_exact(self):
        """κ = I·(1 - exp(-dt/tau_kappa))."""
        n = SpikeResponseNeuron()
        n.time_since_spike = 1000.0
        n.step(5.0)
        expected = _kappa(5.0, n.dt, n.tau_kappa)
        assert abs(n.v - expected) < 1e-10

    def test_kappa_linear_in_I(self):
        """κ(2I) = 2·κ(I) — linearity."""
        k3 = _kappa(3.0, 1.0, 5.0)
        k6 = _kappa(6.0, 1.0, 5.0)
        assert abs(k6 - 2 * k3) < 1e-10

    def test_kappa_decreases_with_tau_kappa(self):
        """Larger tau_kappa → smaller κ (slower integration)."""
        k_small_tau = _kappa(10.0, 1.0, 1.0)
        k_large_tau = _kappa(10.0, 1.0, 20.0)
        assert k_small_tau > k_large_tau

    def test_critical_current(self):
        """I_crit = θ / (1 - exp(-dt/tau_kappa)). Verified above/below."""
        n = SpikeResponseNeuron()
        I_crit = n.v_threshold / (1.0 - np.exp(-n.dt / n.tau_kappa))
        n_below = SpikeResponseNeuron()
        n_below.time_since_spike = 1000.0
        assert n_below.step(I_crit * 0.9) == 0
        n_above = SpikeResponseNeuron()
        n_above.time_since_spike = 1000.0
        assert n_above.step(I_crit * 1.1) == 1


class TestSRMISI:
    def test_constant_isi(self):
        """At constant suprathreshold input, ISI is perfectly constant."""
        n = SpikeResponseNeuron()
        spikes = _run(n, current=10.0, steps=10000)
        assert len(spikes) >= 20
        isis = np.diff(spikes[2:])
        assert np.all(isis == isis[0]), f"Non-constant ISI: {np.unique(isis)}"

    def test_isi_from_simulation(self):
        """Measure ISI at I=10.0: should be 20 steps (from probing)."""
        n = SpikeResponseNeuron()
        spikes = _run(n, current=10.0, steps=5000)
        measured_isi = int(np.median(np.diff(spikes[2:])))
        # ISI = 20 from probing (refractory recovery takes 19 steps + spike step)
        assert 18 <= measured_isi <= 22, f"ISI = {measured_isi}"

    def test_isi_shortens_with_stronger_input(self):
        n_weak = SpikeResponseNeuron()
        n_strong = SpikeResponseNeuron()
        s_weak = _run(n_weak, current=8.0, steps=5000)
        s_strong = _run(n_strong, current=15.0, steps=5000)
        if len(s_weak) > 5 and len(s_strong) > 5:
            isi_weak = np.median(np.diff(s_weak[2:]))
            isi_strong = np.median(np.diff(s_strong[2:]))
            assert isi_strong < isi_weak

    def test_cv_isi_zero(self):
        n = SpikeResponseNeuron()
        spikes = _run(n, current=10.0, steps=5000)
        isis_arr = np.diff(spikes[2:]).astype(float)
        cv = np.std(isis_arr) / np.mean(isis_arr) if len(isis_arr) > 5 else 0
        assert cv < 0.01


class TestSRMFI:
    def test_subthreshold_silent(self):
        n = SpikeResponseNeuron()
        assert len(_run(n, current=5.0, steps=10000)) == 0

    def test_suprathreshold_fires(self):
        n = SpikeResponseNeuron()
        assert len(_run(n, current=10.0, steps=10000)) >= 100

    def test_monotonic_fi(self):
        rates = []
        for I in [8.0, 10.0, 15.0, 20.0]:
            n = SpikeResponseNeuron()
            rates.append(len(_run(n, current=I, steps=5000)))
        assert all(rates[i] <= rates[i + 1] for i in range(len(rates) - 1))

    def test_fi_cross_current_comparison(self):
        """Verify rate ratios are consistent: stronger input → proportionally more spikes."""
        n10 = SpikeResponseNeuron()
        n20 = SpikeResponseNeuron()
        s10 = len(_run(n10, current=10.0, steps=10000))
        s20 = len(_run(n20, current=20.0, steps=10000))
        # Ratio should be > 1 (monotonic)
        assert s20 > s10
        # Not exactly 2× because ISI depends nonlinearly on η recovery
        ratio = s20 / s10
        assert 1.2 < ratio < 3.0, f"Rate ratio f(20)/f(10) = {ratio:.2f}"


class TestSRMParameters:
    def test_tau_eta_controls_refractory_duration(self):
        n_fast = SpikeResponseNeuron(tau_eta=5.0)
        n_slow = SpikeResponseNeuron(tau_eta=20.0)
        s_fast = len(_run(n_fast, current=10.0, steps=5000))
        s_slow = len(_run(n_slow, current=10.0, steps=5000))
        assert s_fast > s_slow

    def test_eta_reset_controls_suppression_depth(self):
        n_shallow = SpikeResponseNeuron(eta_reset=-2.0)
        n_deep = SpikeResponseNeuron(eta_reset=-10.0)
        s_shallow = len(_run(n_shallow, current=10.0, steps=5000))
        s_deep = len(_run(n_deep, current=10.0, steps=5000))
        assert s_shallow > s_deep

    def test_threshold_controls_sensitivity(self):
        n_low = SpikeResponseNeuron(v_threshold=0.5)
        n_high = SpikeResponseNeuron(v_threshold=2.0)
        s_low = len(_run(n_low, current=10.0, steps=5000))
        s_high = len(_run(n_high, current=10.0, steps=5000))
        assert s_low > s_high

    @pytest.mark.parametrize("dt", [0.5, 1.0, 2.0])
    def test_dt_stability(self, dt: float):
        n = SpikeResponseNeuron(dt=dt)
        for _ in range(10000):
            n.step(10.0)
        assert np.isfinite(n.v)


class TestSRMEdgeCases:
    def test_zero_input_silent(self):
        n = SpikeResponseNeuron()
        assert all(n.step(0.0) == 0 for _ in range(1000))

    def test_negative_input(self):
        n = SpikeResponseNeuron()
        n.time_since_spike = 1000.0
        n.step(-10.0)
        assert n.v < 0

    def test_time_since_spike_increments(self):
        n = SpikeResponseNeuron()
        tss0 = n.time_since_spike
        n.step(0.0)
        assert n.time_since_spike == tss0 + n.dt

    def test_spike_resets_tss_to_zero(self):
        n = SpikeResponseNeuron()
        n.step(10.0)
        assert n.time_since_spike == 0.0

    def test_deterministic(self):
        traces = []
        for _ in range(2):
            n = SpikeResponseNeuron()
            trace = [(n.step(10.0), n.v, n.time_since_spike) for _ in range(100)]
            traces.append(trace)
        assert traces[0] == traces[1]


class TestSRMPipeline:
    def test_population(self):
        assert Population(SpikeResponseNeuron, n=10, label="srm").n == 10

    def test_network_with_drive(self):
        pop = Population(SpikeResponseNeuron, n=10, label="srm")
        drive = PoissonInput(n=10, rate_hz=500.0, weight=10.0, dt=0.001, seed=42)
        mon = SpikeMonitor(pop)
        net = Network(pop, drive, mon)
        net.run(duration=1.0, dt=0.001, backend="python")
        assert mon.count > 0

    def test_projection_propagates(self):
        """Source → Projection → Target. Target fires from projected spikes."""
        src = Population(SpikeResponseNeuron, n=10, label="src")
        tgt = Population(SpikeResponseNeuron, n=10, label="tgt")
        drive = PoissonInput(n=10, rate_hz=500.0, weight=10.0, dt=0.001, seed=42)
        proj = Projection(src, tgt, weight=8.0, probability=0.5, seed=42)
        mon_src = SpikeMonitor(src)
        mon_tgt = SpikeMonitor(tgt)
        net = Network(src, tgt, drive, proj, mon_src, mon_tgt)
        net.run(duration=1.0, dt=0.001, backend="python")
        assert mon_src.count > 0
        assert mon_tgt.count > 0

    def test_analysis_pipeline(self):
        """spike_count, isi, firing_rate — all work on SRM output."""
        n = SpikeResponseNeuron()
        train = np.array([float(n.step(10.0)) for _ in range(10000)])
        sc = spike_count(train)
        assert sc >= 100
        isis = isi(train, dt=0.001)
        assert len(isis) >= 10
        rate = firing_rate(train, dt=0.001)
        assert rate > 0
        duration = 10000 * 0.001
        assert abs(rate - sc / duration) < 1.0
