# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — AdaptiveThresholdIFNeuron model-unit contracts

"""Model-unit contracts for the composite reduced adaptive-threshold LIF."""

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


class TestConstructionAndValidation:
    """Construction normalises fields and rejects invalid configurations."""

    def test_catalogue_defaults(self) -> None:
        n = AdaptiveThresholdIFNeuron()
        assert (n.v, n.theta) == (-65.0, -50.0)
        assert (n.v_rest, n.v_reset, n.theta_rest) == (-65.0, -65.0, -50.0)
        assert (n.delta_theta, n.tau_m, n.tau_theta, n.dt) == (5.0, 10.0, 50.0, 0.1)

    def test_scalar_fields_are_normalised_to_float(self) -> None:
        n = AdaptiveThresholdIFNeuron(v=-60, tau_m=8)  # type: ignore[arg-type]
        assert isinstance(n.v, float) and isinstance(n.tau_m, float)

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
    def test_rejects_non_physical_configuration(self, kwargs: dict[str, float]) -> None:
        with pytest.raises(ValueError):
            AdaptiveThresholdIFNeuron(**kwargs)

    @pytest.mark.parametrize("field", ["v", "theta", "tau_m"])
    def test_rejects_non_numeric_fields(self, field: str) -> None:
        with pytest.raises(ValueError, match="must be numeric"):
            AdaptiveThresholdIFNeuron(**{field: "fast"})  # type: ignore[arg-type]


class TestExactRelaxationDynamics:
    """Each step is the exact constant-input relaxation, never an Euler step."""

    def test_subthreshold_step_matches_exact_relaxation(self) -> None:
        n = AdaptiveThresholdIFNeuron(v=-70.0, theta=-40.0, dt=0.25)
        expected_v = n.v_rest + 12.0 + (n.v - (n.v_rest + 12.0)) * np.exp(-n.dt / n.tau_m)
        expected_theta = n.theta_rest + (n.theta - n.theta_rest) * np.exp(-n.dt / n.tau_theta)

        assert n.step(12.0) == 0

        assert n.v == pytest.approx(expected_v, rel=1e-14, abs=1e-14)
        assert n.theta == pytest.approx(expected_theta, rel=1e-14, abs=1e-14)

    def test_step_is_not_forward_euler(self) -> None:
        n = AdaptiveThresholdIFNeuron(v=-70.0, theta=-40.0, dt=0.5)
        n.step(12.0)
        euler_v = -70.0 + (-(-70.0 - (-65.0)) + 12.0) / 10.0 * 0.5
        assert abs(n.v - euler_v) > 1.0e-3

    def test_large_timestep_exact_relaxation_remains_bounded(self) -> None:
        n = AdaptiveThresholdIFNeuron(v=-70.0, theta=-30.0, tau_m=0.04, tau_theta=0.04, dt=1.0)

        assert n.step(0.0) == 0

        assert n.v == pytest.approx(n.v_rest, rel=0.0, abs=1e-8)
        assert n.theta == pytest.approx(n.theta_rest, rel=0.0, abs=1e-8)

    def test_subthreshold_relaxation_is_monotone_toward_rest(self) -> None:
        n = AdaptiveThresholdIFNeuron(v=-70.0, theta=-40.0)
        v_before = n.v
        theta_before = n.theta

        assert n.step(0.0) == 0

        assert v_before < n.v < n.v_rest
        assert n.theta_rest < n.theta < theta_before

    def test_steady_state_is_a_fixed_point(self) -> None:
        n = AdaptiveThresholdIFNeuron(v=-53.0, theta=-50.0)
        n.v = n.v_rest + 12.0
        n.theta = n.theta_rest
        assert n.step(12.0) == 0
        assert n.v == pytest.approx(n.v_rest + 12.0, rel=0.0, abs=1e-14)
        assert n.theta == pytest.approx(n.theta_rest, rel=0.0, abs=1e-14)


class TestSpikeSemantics:
    """Candidate crossing, reset, fixed shift, and adaptation."""

    def test_step_returns_binary(self) -> None:
        n = AdaptiveThresholdIFNeuron()
        assert n.step(0.0) in (0, 1)

    def test_crossing_installs_reset_and_fixed_shift(self) -> None:
        n = AdaptiveThresholdIFNeuron(v=-50.5, theta=-51.0)
        assert n.step(0.0) == 1
        assert n.v == -65.0
        relaxed = -50.0 + (-51.0 + 50.0) * np.exp(-0.1 / 50.0)
        assert n.theta == pytest.approx(relaxed + 5.0, rel=0.0, abs=1e-14)

    def test_shifted_threshold_does_not_immediately_retrigger(self) -> None:
        n = AdaptiveThresholdIFNeuron(v=-50.5, theta=-51.0)
        assert n.step(0.0) == 1
        assert n.step(0.0) == 0

    def test_spikes_under_drive(self) -> None:
        n = AdaptiveThresholdIFNeuron()
        spikes = sum(n.step(100.0) for _ in range(2000))
        assert spikes > 0, "no spikes at I=100"

    def test_threshold_adapts(self) -> None:
        n = AdaptiveThresholdIFNeuron()
        theta_init = n.theta
        for _ in range(2000):
            n.step(100.0)
        assert n.theta > theta_init, "threshold did not increase after spiking"

    def test_adaptation_accumulates_per_spike_with_decay(self) -> None:
        n = AdaptiveThresholdIFNeuron()
        first_spike = None
        for index in range(2000):
            if n.step(100.0) == 1:
                first_spike = index
                break
        assert first_spike is not None
        theta_after_first = n.theta
        for _ in range(2000):
            if n.step(100.0) == 1:
                break
        assert n.theta > theta_after_first - 5.0

    def test_state_finite(self) -> None:
        n = AdaptiveThresholdIFNeuron()
        for _ in range(5000):
            n.step(200.0)
        assert np.isfinite(n.v)
        assert np.isfinite(n.theta)

    def test_reset_restores_documented_state_preserving_configuration(self) -> None:
        n = AdaptiveThresholdIFNeuron()
        for _ in range(100):
            n.step(100.0)
        n.reset()
        assert n.v == n.v_rest
        assert n.theta == n.theta_rest
        assert (n.delta_theta, n.tau_m, n.tau_theta, n.dt) == (5.0, 10.0, 50.0, 0.1)


class TestAtomicity:
    """Rejected steps leave both dynamic states unchanged."""

    @pytest.mark.parametrize("current", [float("nan"), float("inf"), -float("inf")])
    def test_rejects_non_finite_current(self, current: float) -> None:
        n = AdaptiveThresholdIFNeuron()
        before = (n.v, n.theta)
        with pytest.raises(ValueError, match="current"):
            n.step(current)
        assert (n.v, n.theta) == before

    def test_rejects_non_finite_runtime_voltage_before_update(self) -> None:
        n = AdaptiveThresholdIFNeuron(v=-60.0, theta=-45.0)
        n.v = float("nan")
        with pytest.raises(ValueError, match="state"):
            n.step(0.0)
        assert np.isnan(n.v)

    def test_rejects_non_finite_runtime_threshold_before_update(self) -> None:
        n = AdaptiveThresholdIFNeuron(v=-60.0, theta=-45.0)
        n.theta = float("nan")
        with pytest.raises(ValueError, match="state"):
            n.step(0.0)
        assert np.isnan(n.theta)

    def test_rejects_non_finite_relaxation_update_before_state_mutation(self) -> None:
        n = AdaptiveThresholdIFNeuron(v=-1.0e308, theta=-45.0)
        before = (n.v, n.theta)
        with pytest.raises(FloatingPointError, match="exact relaxation"):
            n.step(1.0e308)
        assert (n.v, n.theta) == before

    def test_rejects_non_finite_threshold_jump_before_state_mutation(self) -> None:
        n = AdaptiveThresholdIFNeuron(v=1.2e308, theta=1.0e308, delta_theta=1.0e308)
        before = (n.v, n.theta)
        with pytest.raises(FloatingPointError, match="threshold jump"):
            n.step(0.0)
        assert (n.v, n.theta) == before

    def test_rejects_invalid_runtime_configuration_before_mutation(self) -> None:
        n = AdaptiveThresholdIFNeuron()
        n.tau_m = 0.0
        before = (n.v, n.theta)
        with pytest.raises(ValueError, match="tau_m"):
            n.step(1.0)
        assert (n.v, n.theta) == before


class TestBatchAndDispatch:
    """The maintained batch lane matches the scalar golden loop."""

    def test_batch_matches_scalar_step_loop(self) -> None:
        drive = 12.0 + 6.0 * np.sin(np.arange(256) * 0.037)
        scalar = AdaptiveThresholdIFNeuron()
        expected_v = []
        expected_theta = []
        expected_spikes = 0
        for value in drive:
            expected_spikes += scalar.step(float(value))
            expected_v.append(scalar.v)
            expected_theta.append(scalar.theta)
        batch = AdaptiveThresholdIFNeuron().simulate(drive, backend="python")
        np.testing.assert_allclose(batch["v"], expected_v, rtol=0.0, atol=0.0)
        np.testing.assert_allclose(batch["theta"], expected_theta, rtol=0.0, atol=0.0)
        assert batch["spike_count"] == expected_spikes

    def test_empty_batch_returns_initial_state(self) -> None:
        result = AdaptiveThresholdIFNeuron(v=-60.0, theta=-48.0).simulate([], backend="python")
        assert result["v"].size == 0
        assert result["v_final"] == -60.0
        assert result["theta_final"] == -48.0
        assert result["spike_count"] == 0

    def test_simulate_writes_back_final_state(self) -> None:
        n = AdaptiveThresholdIFNeuron()
        result = n.simulate(np.full(200, 20.0), backend="python")
        assert n.v == result["v_final"]
        assert n.theta == result["theta_final"]

    def test_long_varied_run_is_finite_and_deterministic(self) -> None:
        drive = 14.0 + 5.0 * np.sin(np.arange(20_000, dtype=np.float64) * 0.013)
        first = AdaptiveThresholdIFNeuron()
        second = AdaptiveThresholdIFNeuron()
        trace_first = first.simulate(drive, backend="python")
        trace_second = second.simulate(drive, backend="python")
        assert np.isfinite(trace_first["v"]).all()
        assert np.isfinite(trace_first["theta"]).all()
        np.testing.assert_array_equal(trace_first["v"], trace_second["v"])
        np.testing.assert_array_equal(trace_first["theta"], trace_second["theta"])


class TestAdaptiveThresholdIFNetwork:
    """Model works in the full SC-NeuroCore network pipeline."""

    def test_population_creation(self) -> None:
        pop = Population(AdaptiveThresholdIFNeuron, n=10, label="atif")
        assert pop.n == 10
        assert pop.model_name == "AdaptiveThresholdIFNeuron"

    def test_network_produces_spikes(self) -> None:
        pop = Population(AdaptiveThresholdIFNeuron, n=20, label="atif")
        proj = Projection(pop, pop, weight=1.0, probability=0.2, seed=42)
        drive = PoissonInput(n=20, rate_hz=500.0, weight=100.0, dt=0.001, seed=42)
        mon = SpikeMonitor(pop)
        net = Network(pop, proj, drive, mon)
        net.run(duration=0.5, dt=0.001, backend="python")
        assert mon.count > 0, "network produced zero spikes"

    def test_spike_trains_extractable(self) -> None:
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

    def _get_binary_train(self) -> np.ndarray:
        n = AdaptiveThresholdIFNeuron()
        train = np.zeros(5000, dtype=np.int8)
        for t in range(5000):
            train[t] = n.step(80.0)
        return train

    def test_firing_rate(self) -> None:
        train = self._get_binary_train()
        rate = firing_rate(train, dt=0.0001)  # dt=0.1ms (model dt)
        assert rate > 0

    def test_spike_count(self) -> None:
        train = self._get_binary_train()
        assert spike_count(train) > 0

    def test_isi(self) -> None:
        train = self._get_binary_train()
        intervals = isi(train, dt=0.0001)
        if intervals.size > 0:
            assert np.all(intervals > 0)
