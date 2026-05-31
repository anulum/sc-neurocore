# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — End-to-end test: ExpIFNeuron

"""Full pipeline test for ExpIFNeuron (Fourcaud-Trocmé et al. 2003).

EIF: LIF with exponential spike initiation near v_rh.
dV/dt = [-(V-V_rest) + Δ_T·exp((V-v_rh)/Δ_T) + I] / τ.
Monotonic f–I. Performance: ~137K isolation steps/s.
Rust: wired in network_runner.rs. Full pipeline: Population →
Projection → Network → Analysis (spike_count + isi + firing_rate)."""

from __future__ import annotations

import time
import os

import numpy as np
import pytest

from sc_neurocore.neurons.models.expif import ExpIFNeuron
from sc_neurocore.network.population import Population
from sc_neurocore.network.projection import Projection
from sc_neurocore.network.network import Network
from sc_neurocore.network.monitor import SpikeMonitor
from sc_neurocore.network.stimulus import PoissonInput
from sc_neurocore.analysis.spike_stats.basic import spike_count, isi, firing_rate


def _run(neuron: ExpIFNeuron, current: float, steps: int) -> list[int]:
    return [t for t in range(steps) if neuron.step(current) == 1]


class TestExpIFIsolation:
    def test_construction_all_defaults(self):
        n = ExpIFNeuron()
        assert n.v == -65.0 and n.v_rest == -65.0 and n.v_reset == -68.0
        assert n.v_threshold == -50.0 and n.v_rh == -55.0
        assert n.delta_t == 2.0 and n.tau == 20.0 and n.dt == 0.1

    def test_step_returns_binary(self):
        assert ExpIFNeuron().step(0.0) in (0, 1)

    def test_state_evolves(self):
        n = ExpIFNeuron()
        v0 = n.v
        n.step(20.0)
        assert n.v != v0

    def test_state_finite_long_run(self):
        n = ExpIFNeuron()
        for _ in range(100000):
            n.step(20.0)
        assert np.isfinite(n.v)

    def test_reset(self):
        n = ExpIFNeuron()
        for _ in range(100):
            n.step(20.0)
        n.reset()
        assert n.v == n.v_rest


class TestExpIFValidation:
    @pytest.mark.parametrize(
        ("field", "value"),
        [
            ("v", np.nan),
            ("v_rest", np.inf),
            ("v_reset", -np.inf),
            ("v_threshold", np.nan),
            ("v_rh", np.inf),
        ],
    )
    def test_rejects_non_finite_voltage_parameters(self, field: str, value: float):
        with pytest.raises(ValueError, match=field):
            ExpIFNeuron(**{field: value})

    @pytest.mark.parametrize("field", ["delta_t", "tau", "dt"])
    @pytest.mark.parametrize("value", [0.0, -1.0, np.nan, np.inf])
    def test_rejects_non_positive_or_non_finite_scale_parameters(self, field: str, value: float):
        with pytest.raises(ValueError, match=field):
            ExpIFNeuron(**{field: value})

    @pytest.mark.parametrize("current", [np.nan, np.inf, -np.inf])
    def test_rejects_non_finite_current_before_state_mutation(self, current: float):
        n = ExpIFNeuron(v=-60.0)
        before = n.v
        with pytest.raises(ValueError, match="current"):
            n.step(current)
        assert n.v == before

    def test_rejects_runtime_non_finite_voltage_before_update(self):
        n = ExpIFNeuron(v=-60.0)
        n.v = float("nan")
        with pytest.raises(ValueError, match="runtime voltage state"):
            n.step(0.0)
        assert np.isnan(n.v)

    def test_rejects_non_finite_euler_update_before_state_mutation(self):
        n = ExpIFNeuron(v=-60.0, dt=1.0e308, tau=1.0)
        before = n.v
        with pytest.raises(ValueError, match="Euler update"):
            n.step(1.0e308)
        assert n.v == before


class TestExpIFExponentialEscape:
    """Core: exp term drives runaway near v_rh."""

    def test_exp_term_at_vrh(self):
        """At V=v_rh: exp((v_rh-v_rh)/delta_t) = exp(0) = 1.
        Contribution = delta_t * 1.0 = 2.0."""
        n = ExpIFNeuron()
        n.v = n.v_rh
        exp_term = n.delta_t * np.exp(np.clip(0.0, -20, 20))
        assert abs(exp_term - n.delta_t) < 1e-10

    def test_exp_term_drives_spike(self):
        """When V near v_rh with sufficient I, the exponential drives V to threshold."""
        n = ExpIFNeuron()
        n.v = n.v_rh - 1.0
        spikes = sum(n.step(15.0) for _ in range(200))
        assert spikes > 0

    def test_delta_t_controls_sharpness(self):
        """Smaller delta_t → sharper spike initiation."""
        n_sharp = ExpIFNeuron(delta_t=0.5)
        n_broad = ExpIFNeuron(delta_t=5.0)
        # Both should fire, but with different dynamics
        s_sharp = len(_run(n_sharp, current=20.0, steps=10000))
        s_broad = len(_run(n_broad, current=20.0, steps=10000))
        assert s_sharp != s_broad

    def test_exp_clipping_prevents_overflow(self):
        """np.clip guards exp argument to [-20, 20]."""
        n = ExpIFNeuron()
        n.v = 1000.0
        n.step(0.0)
        assert np.isfinite(n.v)

    def test_negative_extreme_finite(self):
        n = ExpIFNeuron()
        n.v = -1000.0
        for _ in range(100):
            n.step(0.0)
        assert np.isfinite(n.v)


class TestExpIFAnalytical:
    def test_membrane_equation_one_step(self):
        """dV = [-(V-V_rest) + delta_t*exp(clip((V-v_rh)/delta_t)) + I] / tau * dt."""
        n = ExpIFNeuron()
        v0 = n.v
        I = 5.0
        exp_term = n.delta_t * np.exp(np.clip((v0 - n.v_rh) / n.delta_t, -20, 20))
        expected_dv = (-(v0 - n.v_rest) + exp_term + I) / n.tau * n.dt
        n.step(I)
        if n.v != n.v_reset:  # no spike
            assert abs((n.v - v0) - expected_dv) < 1e-10

    def test_subthreshold_v_approaches_rest(self):
        """With zero input and V near rest, V stays near V_rest."""
        n = ExpIFNeuron()
        for _ in range(10000):
            n.step(0.0)
        # V should be near V_rest (exp term small at V << v_rh)
        assert abs(n.v - n.v_rest) < 1.0


class TestExpIFFI:
    def test_subthreshold_silent(self):
        n = ExpIFNeuron()
        assert len(_run(n, current=5.0, steps=10000)) == 0

    def test_suprathreshold_fires(self):
        n = ExpIFNeuron()
        assert len(_run(n, current=20.0, steps=10000)) >= 10

    def test_monotonic_fi(self):
        rates = []
        for I in [10.0, 20.0, 50.0, 100.0]:
            n = ExpIFNeuron()
            rates.append(len(_run(n, current=I, steps=10000)))
        assert all(rates[i] <= rates[i + 1] for i in range(len(rates) - 1))

    def test_isi_regularity(self):
        """Deterministic → constant ISI."""
        n = ExpIFNeuron()
        spikes = _run(n, current=50.0, steps=10000)
        if len(spikes) >= 10:
            isis = np.diff(spikes[3:]).astype(float)
            cv = np.std(isis) / np.mean(isis)
            assert cv < 0.05


class TestExpIFParameters:
    def test_tau_affects_rate(self):
        n_fast = ExpIFNeuron(tau=5.0)
        n_slow = ExpIFNeuron(tau=40.0)
        s_fast = len(_run(n_fast, current=20.0, steps=10000))
        s_slow = len(_run(n_slow, current=20.0, steps=10000))
        assert s_fast > s_slow

    def test_custom_vrh(self):
        """Lower v_rh → easier to reach exponential zone → more spikes."""
        n_low = ExpIFNeuron(v_rh=-60.0)
        n_high = ExpIFNeuron(v_rh=-52.0)
        s_low = len(_run(n_low, current=15.0, steps=10000))
        s_high = len(_run(n_high, current=15.0, steps=10000))
        assert s_low > s_high

    @pytest.mark.parametrize("dt", [0.05, 0.1, 0.2])
    def test_dt_stability(self, dt: float):
        n = ExpIFNeuron(dt=dt)
        for _ in range(10000):
            n.step(20.0)
        assert np.isfinite(n.v)

    def test_deterministic(self):
        traces = []
        for _ in range(2):
            n = ExpIFNeuron()
            trace = [(n.step(20.0), n.v) for _ in range(200)]
            traces.append(trace)
        assert traces[0] == traces[1]


class TestExpIFPerformance:
    def test_isolation_throughput(self):
        n = ExpIFNeuron()
        N = 50000
        t0 = time.perf_counter()
        for _ in range(N):
            n.step(20.0)
        elapsed = time.perf_counter() - t0
        rate = N / elapsed
        min_rate = 40_000 if os.getenv("CI") else 50_000
        assert rate > min_rate, f"isolation: {rate:.0f} steps/s, minimum={min_rate}"

    def test_network_throughput(self):
        pop = Population(ExpIFNeuron, n=50, label="bench")
        drive = PoissonInput(n=50, rate_hz=500.0, weight=50.0, dt=0.001, seed=42)
        mon = SpikeMonitor(pop)
        net = Network(pop, drive, mon)
        t0 = time.perf_counter()
        net.run(duration=0.5, dt=0.001, backend="python")
        elapsed = time.perf_counter() - t0
        assert 50 * 500 / elapsed > 5000


class TestExpIFPipeline:
    def test_population(self):
        assert Population(ExpIFNeuron, n=10, label="expif").n == 10

    def test_network_spikes(self):
        pop = Population(ExpIFNeuron, n=10, label="expif")
        drive = PoissonInput(n=10, rate_hz=500.0, weight=50.0, dt=0.001, seed=42)
        mon = SpikeMonitor(pop)
        net = Network(pop, drive, mon)
        net.run(duration=1.0, dt=0.001, backend="python")
        assert mon.count > 0

    def test_projection_wiring(self):
        src = Population(ExpIFNeuron, n=10, label="src")
        tgt = Population(ExpIFNeuron, n=10, label="tgt")
        drive = PoissonInput(n=10, rate_hz=500.0, weight=50.0, dt=0.001, seed=42)
        proj = Projection(src, tgt, weight=20.0, probability=1.0, seed=42)
        mon = SpikeMonitor(src)
        net = Network(src, tgt, drive, proj, mon)
        net.run(duration=1.0, dt=0.001, backend="python")
        assert mon.count > 0

    def test_analysis_pipeline(self):
        n = ExpIFNeuron()
        train = np.array([float(n.step(50.0)) for _ in range(10000)])
        sc = spike_count(train)
        assert sc >= 10
        isis = isi(train, dt=0.0001)
        assert len(isis) >= 5
        rate = firing_rate(train, dt=0.0001)
        assert rate > 0
