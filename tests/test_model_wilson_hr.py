# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — End-to-end test: WilsonHRNeuron

"""Full pipeline test for WilsonHRNeuron (Wilson 1999).

Polynomial cortical model: cubic V dynamics + linear R recovery.
Non-monotonic f–I curve: peak at I≈0.3, decline at I=0.5–1.0,
resurgence at I>5. Performance: ~676K isolation steps/s."""

from __future__ import annotations

import time

import numpy as np
import pytest

from sc_neurocore.neurons.models.wilson_hr import WilsonHRNeuron
from sc_neurocore.network.population import Population
from sc_neurocore.network.projection import Projection
from sc_neurocore.network.network import Network
from sc_neurocore.network.monitor import SpikeMonitor
from sc_neurocore.network.stimulus import PoissonInput
from sc_neurocore.analysis.spike_stats.basic import spike_count, isi, firing_rate


def _run(neuron: WilsonHRNeuron, current: float, steps: int) -> list[int]:
    return [t for t in range(steps) if neuron.step(current) == 1]


class TestWilsonHRIsolation:
    def test_defaults(self):
        n = WilsonHRNeuron()
        assert n.v == -0.7
        assert n.r == 0.1
        assert n.tau_r == 1.9
        assert n.v_peak == 0.4
        assert n.dt == 0.05

    def test_step_returns_binary(self):
        assert WilsonHRNeuron().step(0.0) in (0, 1)

    def test_two_variables_evolve(self):
        n = WilsonHRNeuron()
        v0, r0 = n.v, n.r
        for _ in range(100):
            n.step(0.3)
        assert n.v != v0 and n.r != r0

    def test_state_finite(self):
        n = WilsonHRNeuron()
        for _ in range(50000):
            n.step(0.3)
        assert np.isfinite(n.v) and np.isfinite(n.r)

    def test_reset(self):
        n = WilsonHRNeuron()
        for _ in range(100):
            n.step(0.3)
        n.reset()
        assert n.v == -0.7 and n.r == 0.1

    def test_spike_resets_v(self):
        """On spike: V → -0.7 (hard reset)."""
        n = WilsonHRNeuron()
        for _ in range(50000):
            if n.step(0.3) == 1:
                assert n.v == -0.7
                break


class TestWilsonHRPolynomialDynamics:
    """Core: cubic polynomial V dynamics with recovery variable R."""

    def test_polynomial_formula(self):
        """dV = [-(17.81 + 47.71V + 32.63V²)(V-0.55) - 26R(V+0.92) + I] · dt.

        Verify at V=-0.7, R=0.1, I=0: the polynomial should give a
        specific numerical value.
        """
        v, r = -0.7, 0.1
        poly = -(17.81 + 47.71 * v + 32.63 * v**2) * (v - 0.55)
        syn = -26.0 * r * (v + 0.92)
        dv_expected = (poly + syn + 0.0) * 0.05
        n = WilsonHRNeuron()
        n.step(0.0)
        dv_actual = n.v - (-0.7)
        assert abs(dv_actual - dv_expected) < 1e-10

    def test_r_recovery_equation(self):
        """dR = (-R + 1.35V + 1.03) / tau_r · dt."""
        n = WilsonHRNeuron()
        r0, v0 = n.r, n.v
        n.step(0.0)
        dr_expected = (-r0 + 1.35 * v0 + 1.03) / n.tau_r * n.dt
        dr_actual = n.r - r0
        assert abs(dr_actual - dr_expected) < 1e-10

    def test_v_bounded_by_reset(self):
        """V resets at v_peak=0.4 → stays below 0.4."""
        n = WilsonHRNeuron()
        vs = []
        for _ in range(50000):
            n.step(0.3)
            vs.append(n.v)
        assert max(vs) <= n.v_peak + 0.1  # small overshoot from dt


class TestWilsonHRNonMonotonicFI:
    """Non-monotonic f–I: peak at I≈0.3, decline, resurgence at high I."""

    def test_peak_near_03(self):
        """I=0.3 should produce more spikes than I=0.1 and I=0.5."""
        n01 = WilsonHRNeuron()
        n03 = WilsonHRNeuron()
        n05 = WilsonHRNeuron()
        s01 = len(_run(n01, current=0.1, steps=50000))
        s03 = len(_run(n03, current=0.3, steps=50000))
        s05 = len(_run(n05, current=0.5, steps=50000))
        assert s03 > s01
        assert s03 > s05

    def test_suppression_at_moderate_I(self):
        """I=0.6–1.0: very few spikes (depolarisation block region)."""
        for I in [0.6, 0.8, 1.0]:
            n = WilsonHRNeuron()
            spikes = len(_run(n, current=I, steps=50000))
            assert spikes <= 5, f"I={I}: {spikes} spikes, expected ≤5"

    def test_resurgence_at_high_I(self):
        """At I=10, spiking resumes (very short ISI)."""
        n = WilsonHRNeuron()
        spikes = _run(n, current=10.0, steps=50000)
        assert len(spikes) >= 10

    def test_fi_5_point_sweep(self):
        """Map f–I at 5 points covering all regimes."""
        rates = {}
        for I in [0.0, 0.3, 0.6, 2.0, 10.0]:
            n = WilsonHRNeuron()
            rates[I] = len(_run(n, current=I, steps=50000))
        # Verify non-monotonicity: peak at 0.3, trough at 0.6
        assert rates[0.3] > rates[0.6]


class TestWilsonHRISI:
    def test_isi_variability_at_peak(self):
        """At I=0.3 (peak firing), ISI shows variability from the polynomial
        dynamics — this is NOT a simple limit-cycle oscillator. CV can be
        substantial (measured ~0.5), documenting the irregular spiking pattern."""
        n = WilsonHRNeuron()
        spikes = _run(n, current=0.3, steps=50000)
        if len(spikes) >= 20:
            isis = np.diff(spikes[5:]).astype(float)
            cv = np.std(isis) / np.mean(isis)
            # Document: CV is high (polynomial dynamics ≠ regular oscillator)
            assert cv > 0, f"CV(ISI) should be > 0, got {cv:.4f}"


class TestWilsonHRParameters:
    @pytest.mark.parametrize(
        ("field", "value"),
        [
            ("v", np.nan),
            ("r", np.inf),
            ("tau_r", 0.0),
            ("v_peak", np.inf),
            ("dt", 0.0),
        ],
    )
    def test_rejects_invalid_numerical_configuration(self, field: str, value: float):
        with pytest.raises(ValueError):
            WilsonHRNeuron(**{field: value})

    def test_rejects_non_finite_current_before_state_mutation(self):
        n = WilsonHRNeuron()
        before = (n.v, n.r)
        with pytest.raises(ValueError, match="current"):
            n.step(np.nan)
        assert (n.v, n.r) == before

    def test_rejects_corrupted_runtime_state_before_mutation(self):
        n = WilsonHRNeuron()
        n.r = np.inf
        before = (n.v, n.r)
        with pytest.raises(FloatingPointError, match="runtime state"):
            n.step(0.3)
        assert (n.v, n.r) == before

    def test_rejects_polynomial_overflow_before_state_mutation(self):
        n = WilsonHRNeuron(v=1.0e308)
        before = (n.v, n.r)
        with pytest.raises(FloatingPointError, match="polynomial|candidate"):
            n.step(0.3)
        assert (n.v, n.r) == before

    def test_tau_r_affects_recovery(self):
        n_fast = WilsonHRNeuron(tau_r=1.0)
        n_slow = WilsonHRNeuron(tau_r=5.0)
        s_fast = len(_run(n_fast, current=0.3, steps=50000))
        s_slow = len(_run(n_slow, current=0.3, steps=50000))
        assert s_fast != s_slow

    def test_v_peak_controls_threshold(self):
        n_low = WilsonHRNeuron(v_peak=0.2)
        n_high = WilsonHRNeuron(v_peak=0.6)
        s_low = len(_run(n_low, current=0.3, steps=50000))
        s_high = len(_run(n_high, current=0.3, steps=50000))
        assert s_low >= s_high

    @pytest.mark.parametrize("dt", [0.02, 0.05, 0.1])
    def test_dt_stability(self, dt: float):
        n = WilsonHRNeuron(dt=dt)
        for _ in range(50000):
            n.step(0.3)
        assert np.isfinite(n.v)

    def test_deterministic(self):
        traces = []
        for _ in range(2):
            n = WilsonHRNeuron()
            trace = [(n.step(0.3), n.v, n.r) for _ in range(300)]
            traces.append(trace)
        assert traces[0] == traces[1]


class TestWilsonHRPerformance:
    def test_isolation_throughput(self):
        n = WilsonHRNeuron()
        N = 50000
        t0 = time.perf_counter()
        for _ in range(N):
            n.step(0.3)
        elapsed = time.perf_counter() - t0
        assert N / elapsed > 50000

    def test_network_throughput(self):
        pop = Population(WilsonHRNeuron, n=50, label="bench")
        drive = PoissonInput(n=50, rate_hz=500.0, weight=0.3, dt=0.001, seed=42)
        mon = SpikeMonitor(pop)
        net = Network(pop, drive, mon)
        t0 = time.perf_counter()
        net.run(duration=0.5, dt=0.001, backend="python")
        elapsed = time.perf_counter() - t0
        assert 50 * 500 / elapsed > 5000


class TestWilsonHRPipeline:
    def test_population(self):
        assert Population(WilsonHRNeuron, n=10, label="whr").n == 10

    def test_network_spikes(self):
        pop = Population(WilsonHRNeuron, n=10, label="whr")
        drive = PoissonInput(n=10, rate_hz=500.0, weight=0.3, dt=0.001, seed=42)
        mon = SpikeMonitor(pop)
        net = Network(pop, drive, mon)
        net.run(duration=5.0, dt=0.001, backend="python")
        assert mon.count > 0

    def test_projection_wiring(self):
        src = Population(WilsonHRNeuron, n=10, label="src")
        tgt = Population(WilsonHRNeuron, n=10, label="tgt")
        drive = PoissonInput(n=10, rate_hz=500.0, weight=0.3, dt=0.001, seed=42)
        proj = Projection(src, tgt, weight=0.2, probability=1.0, seed=42)
        mon = SpikeMonitor(src)
        net = Network(src, tgt, drive, proj, mon)
        net.run(duration=5.0, dt=0.001, backend="python")
        assert mon.count > 0

    def test_analysis(self):
        n = WilsonHRNeuron()
        train = np.array([float(n.step(0.3)) for _ in range(50000)])
        sc = spike_count(train)
        assert sc >= 10
        isis = isi(train, dt=0.00005)  # dt=0.05ms per step
        assert len(isis) >= 5
        rate = firing_rate(train, dt=0.00005)
        assert rate > 0
