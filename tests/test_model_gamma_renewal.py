# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — End-to-end test: GammaRenewalNeuron

"""Hazard-based gamma renewal. Rate-driven. ~83K steps/s."""

from __future__ import annotations

import time
import warnings

import numpy as np
import pytest

from sc_neurocore.neurons.models.gamma_renewal import GammaRenewalNeuron
from sc_neurocore.network.population import Population
from sc_neurocore.network.projection import Projection
from sc_neurocore.network.network import Network
from sc_neurocore.network.monitor import SpikeMonitor
from sc_neurocore.network.stimulus import PoissonInput
from sc_neurocore.analysis.spike_stats.basic import spike_count, firing_rate


def _run(neuron: GammaRenewalNeuron, current: float, steps: int) -> list[int]:
    return [t for t in range(steps) if neuron.step(current) == 1]


class TestIsolation:
    def test_step_returns_binary(self):
        assert GammaRenewalNeuron().step(0.0) in (0, 1)

    def test_state_finite(self):
        n = GammaRenewalNeuron()
        for _ in range(5000):
            n.step(100.0)
        assert np.isfinite(getattr(n, "v", 0.0))

    def test_reset(self):
        n = GammaRenewalNeuron()
        for _ in range(100):
            n.step(100.0)
        n.reset()


class TestDynamics:
    def test_fires_at_test_current(self):
        n = GammaRenewalNeuron()
        spikes = _run(n, current=100.0, steps=5000)
        assert len(spikes) >= 10

    def test_rate_increases_with_current(self):
        n_low = GammaRenewalNeuron()
        n_high = GammaRenewalNeuron()
        s_low = len(_run(n_low, current=50.0, steps=5000))
        s_high = len(_run(n_high, current=500.0, steps=5000))
        assert s_high >= s_low

    def test_two_runs_differ(self):
        n1 = GammaRenewalNeuron()
        n2 = GammaRenewalNeuron()
        t1 = [n1.step(100.0) for _ in range(1000)]
        t2 = [n2.step(100.0) for _ in range(1000)]
        assert t1 != t2


class TestValidation:
    @pytest.mark.parametrize("rate_hz", [-1.0, np.nan, np.inf, -np.inf])
    def test_rejects_negative_or_non_finite_baseline_rate(self, rate_hz: float):
        with pytest.raises(ValueError, match="rate_hz"):
            GammaRenewalNeuron(rate_hz=rate_hz)

    @pytest.mark.parametrize("shape_k", [0, -1, 1.5])
    def test_rejects_non_positive_or_non_integer_shape(self, shape_k):
        with pytest.raises(ValueError, match="shape_k"):
            GammaRenewalNeuron(shape_k=shape_k)

    @pytest.mark.parametrize("dt_ms", [0.0, -1.0, np.nan, np.inf])
    def test_rejects_non_positive_or_non_finite_dt(self, dt_ms: float):
        with pytest.raises(ValueError, match="dt_ms"):
            GammaRenewalNeuron(dt_ms=dt_ms)

    @pytest.mark.parametrize("_time_since_spike", [-1.0, np.nan, np.inf, -np.inf])
    def test_rejects_negative_or_non_finite_elapsed_state(self, _time_since_spike: float):
        with pytest.raises(ValueError, match="time_since_spike"):
            GammaRenewalNeuron(_time_since_spike=_time_since_spike)

    @pytest.mark.parametrize("rate_override", [np.nan, np.inf, -np.inf])
    def test_rejects_non_finite_rate_override_before_elapsed_mutation(self, rate_override: float):
        n = GammaRenewalNeuron(_time_since_spike=0.125)
        before = n._time_since_spike
        with pytest.raises(ValueError, match="rate_override"):
            n.step(rate_override=rate_override)
        assert n._time_since_spike == before

    def test_zero_rate_path_is_silent_and_never_spikes(self):
        n = GammaRenewalNeuron(rate_hz=50.0)
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            spikes = [n.step(rate_override=0.0) for _ in range(8)]
        assert spikes == [0] * 8
        assert n._time_since_spike > 0.0


class TestPerformance:
    def test_isolation_throughput(self):
        n = GammaRenewalNeuron()
        N = 20000
        t0 = time.perf_counter()
        for _ in range(N):
            n.step(100.0)
        elapsed = time.perf_counter() - t0
        assert N / elapsed > 20000

    def test_network_throughput(self):
        pop = Population(GammaRenewalNeuron, n=50, label="bench")
        drive = PoissonInput(n=50, rate_hz=500.0, weight=100.0, dt=0.001, seed=42)
        mon = SpikeMonitor(pop)
        net = Network(pop, drive, mon)
        t0 = time.perf_counter()
        net.run(duration=0.5, dt=0.001, backend="python")
        elapsed = time.perf_counter() - t0
        assert 50 * 500 / elapsed > 5000


class TestPipeline:
    def test_population(self):
        assert Population(GammaRenewalNeuron, n=10, label="test").n == 10

    def test_network_spikes(self):
        pop = Population(GammaRenewalNeuron, n=10, label="test")
        drive = PoissonInput(n=10, rate_hz=500.0, weight=100.0, dt=0.001, seed=42)
        mon = SpikeMonitor(pop)
        net = Network(pop, drive, mon)
        net.run(duration=2.0, dt=0.001, backend="python")
        assert mon.count > 0

    def test_projection_wiring(self):
        src = Population(GammaRenewalNeuron, n=5, label="src")
        tgt = Population(GammaRenewalNeuron, n=5, label="tgt")
        drive = PoissonInput(n=5, rate_hz=500.0, weight=100.0, dt=0.001, seed=42)
        proj = Projection(src, tgt, weight=100.0, probability=1.0, seed=42)
        mon = SpikeMonitor(src)
        net = Network(src, tgt, drive, proj, mon)
        net.run(duration=2.0, dt=0.001, backend="python")
        assert mon.count > 0

    def test_analysis(self):
        n = GammaRenewalNeuron()
        train = np.array([float(n.step(100.0)) for _ in range(5000)])
        sc = spike_count(train)
        assert sc >= 5
        rate = firing_rate(train, dt=0.001)
        assert rate > 0
