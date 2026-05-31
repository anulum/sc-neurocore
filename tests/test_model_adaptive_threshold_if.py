# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — End-to-end test: AdaptiveThresholdIFNeuron

"""Full pipeline test for AdaptiveThresholdIFNeuron (Platkiewicz & Bhatt 2010).

Verifies: import → isolation → Population → Projection → Network →
SpikeMonitor → analysis toolkit → reset. No shortcuts."""

from __future__ import annotations

import numpy as np
import pytest

from sc_neurocore.neurons.models.adaptive_threshold_if import AdaptiveThresholdIFNeuron
from sc_neurocore.network.population import Population
from sc_neurocore.network.projection import Projection
from sc_neurocore.network.network import Network
from sc_neurocore.network.monitor import SpikeMonitor
from sc_neurocore.network.stimulus import PoissonInput
from sc_neurocore.analysis.spike_stats.basic import firing_rate, spike_count, isi


class TestAdaptiveThresholdIFIsolation:
    """Model works in isolation."""

    def test_construction(self):
        n = AdaptiveThresholdIFNeuron()
        assert n.v == -65.0
        assert n.theta == -50.0

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"v": float("nan")},
            {"theta": float("inf")},
            {"v_rest": float("nan")},
            {"v_reset": float("inf")},
            {"theta_rest": float("nan")},
            {"theta_rest": -70.0},
            {"v_reset": -45.0},
            {"delta_theta": -0.1},
            {"delta_theta": float("inf")},
            {"tau_m": 0.0},
            {"tau_m": float("nan")},
            {"tau_theta": 0.0},
            {"tau_theta": float("inf")},
            {"dt": 0.0},
            {"dt": float("nan")},
        ],
    )
    def test_rejects_non_physical_configuration(self, kwargs):
        with pytest.raises(ValueError):
            AdaptiveThresholdIFNeuron(**kwargs)

    def test_subthreshold_step_matches_exact_relaxation(self):
        n = AdaptiveThresholdIFNeuron(v=-70.0, theta=-40.0, dt=0.25)
        expected_v = n.v_rest + 12.0 + (n.v - (n.v_rest + 12.0)) * np.exp(-n.dt / n.tau_m)
        expected_theta = n.theta_rest + (n.theta - n.theta_rest) * np.exp(-n.dt / n.tau_theta)

        assert n.step(12.0) == 0

        assert n.v == pytest.approx(expected_v, rel=1e-14, abs=1e-14)
        assert n.theta == pytest.approx(expected_theta, rel=1e-14, abs=1e-14)

    def test_large_timestep_exact_relaxation_remains_bounded(self):
        n = AdaptiveThresholdIFNeuron(v=-70.0, theta=-30.0, tau_m=0.04, tau_theta=0.04, dt=1.0)

        assert n.step(0.0) == 0

        assert n.v == pytest.approx(n.v_rest, rel=0.0, abs=1e-8)
        assert n.theta == pytest.approx(n.theta_rest, rel=0.0, abs=1e-8)

    def test_subthreshold_relaxation_is_monotone_toward_rest(self):
        n = AdaptiveThresholdIFNeuron(v=-70.0, theta=-40.0)
        v_before = n.v
        theta_before = n.theta

        assert n.step(0.0) == 0

        assert v_before < n.v < n.v_rest
        assert n.theta_rest < n.theta < theta_before

    def test_step_returns_binary(self):
        n = AdaptiveThresholdIFNeuron()
        result = n.step(0.0)
        assert result in (0, 1)

    @pytest.mark.parametrize("current", [float("nan"), float("inf"), -float("inf")])
    def test_rejects_non_finite_current(self, current):
        n = AdaptiveThresholdIFNeuron()
        with pytest.raises(ValueError, match="current"):
            n.step(current)

    def test_rejects_non_finite_runtime_voltage_before_update(self):
        n = AdaptiveThresholdIFNeuron(v=-60.0, theta=-45.0)
        n.v = float("nan")
        with pytest.raises(ValueError, match="runtime voltage state"):
            n.step(0.0)
        assert np.isnan(n.v)

    def test_rejects_non_finite_runtime_threshold_before_update(self):
        n = AdaptiveThresholdIFNeuron(v=-60.0, theta=-45.0)
        n.theta = float("nan")
        with pytest.raises(ValueError, match="runtime threshold state"):
            n.step(0.0)
        assert np.isnan(n.theta)

    def test_rejects_non_finite_relaxation_update_before_state_mutation(self):
        n = AdaptiveThresholdIFNeuron(v=1.0e308, theta=-45.0)
        before = (n.v, n.theta)
        with pytest.raises(ValueError, match="exact relaxation"):
            n.step(-1.0e308)
        assert (n.v, n.theta) == before

    def test_rejects_non_finite_threshold_jump_before_state_mutation(self):
        n = AdaptiveThresholdIFNeuron(v=-49.0, theta=-50.0, delta_theta=1.0e308)
        n.theta = 1.0e308
        n.v = 1.7e308
        before = (n.v, n.theta)
        with pytest.raises(ValueError, match="threshold jump"):
            n.step(0.0)
        assert (n.v, n.theta) == before

    def test_spikes_under_drive(self):
        n = AdaptiveThresholdIFNeuron()
        spikes = sum(n.step(100.0) for _ in range(2000))
        assert spikes > 0, "no spikes at I=100"

    def test_threshold_adapts(self):
        n = AdaptiveThresholdIFNeuron()
        theta_init = n.theta
        for _ in range(2000):
            n.step(100.0)
        assert n.theta > theta_init, "threshold did not increase after spiking"

    def test_state_finite(self):
        n = AdaptiveThresholdIFNeuron()
        for _ in range(5000):
            n.step(200.0)
        assert np.isfinite(n.v)
        assert np.isfinite(n.theta)

    def test_reset(self):
        n = AdaptiveThresholdIFNeuron()
        for _ in range(100):
            n.step(100.0)
        n.reset()
        assert n.v == n.v_rest
        assert n.theta == n.theta_rest


class TestAdaptiveThresholdIFNetwork:
    """Model works in the full SC-NeuroCore network pipeline."""

    def test_population_creation(self):
        pop = Population(AdaptiveThresholdIFNeuron, n=10, label="atif")
        assert pop.n == 10
        assert pop.model_name == "AdaptiveThresholdIFNeuron"

    def test_network_produces_spikes(self):
        pop = Population(AdaptiveThresholdIFNeuron, n=20, label="atif")
        proj = Projection(pop, pop, weight=1.0, probability=0.2, seed=42)
        drive = PoissonInput(n=20, rate_hz=500.0, weight=100.0, dt=0.001, seed=42)
        mon = SpikeMonitor(pop)
        net = Network(pop, proj, drive, mon)
        net.run(duration=0.5, dt=0.001, backend="python")
        assert mon.count > 0, "network produced zero spikes"

    def test_spike_trains_extractable(self):
        pop = Population(AdaptiveThresholdIFNeuron, n=10, label="atif")
        drive = PoissonInput(n=10, rate_hz=500.0, weight=100.0, dt=0.001, seed=42)
        mon = SpikeMonitor(pop)
        net = Network(pop, drive, mon)
        net.run(duration=0.3, dt=0.001, backend="python")
        trains = mon.spike_trains
        assert isinstance(trains, dict)
        assert len(trains) > 0, "no spike trains recorded"


class TestAdaptiveThresholdIFAnalysis:
    """Analysis toolkit works on spikes from this model."""

    def _get_binary_train(self):
        n = AdaptiveThresholdIFNeuron()
        train = np.zeros(5000, dtype=np.int8)
        for t in range(5000):
            train[t] = n.step(80.0)
        return train

    def test_firing_rate(self):
        train = self._get_binary_train()
        rate = firing_rate(train, dt=0.0001)  # dt=0.1ms (model dt)
        assert rate > 0

    def test_spike_count(self):
        train = self._get_binary_train()
        assert spike_count(train) > 0

    def test_isi(self):
        train = self._get_binary_train()
        intervals = isi(train, dt=0.0001)
        if intervals.size > 0:
            assert np.all(intervals > 0)
            assert np.all(np.isfinite(intervals))
