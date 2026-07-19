# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — AlphaNeuron model-unit contracts

"""Model-unit contracts for the dual alpha-synapse LIF."""

from __future__ import annotations

import math
from typing import cast

import numpy as np
import numpy.typing as npt
import pytest

from sc_neurocore.neurons.models.alpha import AlphaNeuron
from sc_neurocore.network.population import Population
from sc_neurocore.network.projection import Projection
from sc_neurocore.network.network import Network
from sc_neurocore.network.monitor import SpikeMonitor
from sc_neurocore.network.stimulus import PoissonInput
from sc_neurocore.analysis.spike_stats.basic import spike_count, firing_rate


class TestConstructionAndValidation:
    """Construction normalises fields and rejects invalid configurations."""

    def test_catalogue_defaults(self) -> None:
        n = AlphaNeuron()
        assert (n.v, n.a_exc, n.i_exc, n.a_inh, n.i_inh) == (0.0, 0.0, 0.0, 0.0, 0.0)
        assert (n.v_rest, n.v_threshold) == (0.0, 1.0)
        assert (n.tau_v, n.tau_exc, n.tau_inh, n.dt) == (20.0, 5.0, 10.0, 1.0)

    def test_scalar_fields_are_normalised_to_float(self) -> None:
        n = AlphaNeuron(v=1, tau_v=15)
        assert isinstance(n.v, float) and isinstance(n.tau_v, float)

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"v": float("nan")},
            {"a_exc": float("inf")},
            {"i_exc": float("nan")},
            {"a_inh": float("inf")},
            {"i_inh": float("nan")},
            {"v_rest": float("nan")},
            {"v_threshold": float("inf")},
            {"v_threshold": -1.0},
            {"tau_v": 0.0},
            {"tau_exc": -1.0},
            {"tau_inh": 0.0},
            {"dt": 0.0},
            {"dt": float("nan")},
        ],
    )
    def test_rejects_non_physical_configuration(self, kwargs: dict[str, float]) -> None:
        with pytest.raises(ValueError):
            AlphaNeuron(**kwargs)

    @pytest.mark.parametrize("field", ["v", "a_exc", "tau_v"])
    def test_rejects_non_numeric_fields(self, field: str) -> None:
        with pytest.raises(ValueError, match="must be numeric"):
            AlphaNeuron(**cast(dict[str, float], {field: "fast"}))


class TestExactFlowDynamics:
    """Each step is the exact constant-input flow, never an Euler step."""

    def test_filter_matches_exact_alpha_cascade(self) -> None:
        n = AlphaNeuron(a_exc=0.25, i_exc=0.1, v_threshold=100.0, dt=0.5)
        steady = 5.0 * 2.0
        decay = math.exp(-0.5 / 5.0)
        expected_a = steady + (0.25 - steady) * decay
        expected_i = steady + decay * ((0.1 - steady) + (0.25 - steady) * 0.5 / 5.0)
        assert n.step(2.0) == 0
        assert n.a_exc == pytest.approx(expected_a, rel=0.0, abs=1e-14)
        assert n.i_exc == pytest.approx(expected_i, rel=0.0, abs=1e-14)

    def test_step_is_not_forward_euler(self) -> None:
        n = AlphaNeuron(v=0.5, v_threshold=100.0, dt=0.5)
        n.step(0.0)
        euler_v = 0.5 + (-(0.5 - 0.0)) / 20.0 * 0.5
        assert abs(n.v - euler_v) > 1.0e-4

    def test_equal_time_constant_limit_is_analytic(self) -> None:
        n = AlphaNeuron(i_exc=0.3, a_exc=0.2, tau_v=20.0, tau_exc=20.0, v_threshold=100.0, dt=0.5)
        rate = 1.0 / 20.0
        decay = math.exp(-0.5 / 20.0)
        contribution = rate * decay * (0.3 * 0.5 + 0.2 * 0.5 * 0.5 / (2.0 * 20.0))
        expected_v = n.v * decay + contribution
        assert n.step(0.0) == 0
        assert n.v == pytest.approx(expected_v, rel=0.0, abs=1e-14)

    def test_large_timestep_remains_bounded(self) -> None:
        n = AlphaNeuron(v=0.5, tau_v=0.04, tau_exc=0.04, tau_inh=0.04, dt=1.0)
        assert n.step(0.0) == 0
        assert n.v == pytest.approx(n.v_rest, rel=0.0, abs=1e-8)

    def test_steady_state_is_a_fixed_point(self) -> None:
        n = AlphaNeuron(v_threshold=100.0)
        n.v = n.v_rest + 5.0 * 2.0 - 10.0 * 1.0
        n.a_exc = 5.0 * 2.0
        n.i_exc = 5.0 * 2.0
        n.a_inh = 10.0 * 1.0
        n.i_inh = 10.0 * 1.0
        assert n.step(2.0, 1.0) == 0
        assert n.v == pytest.approx(0.0, rel=0.0, abs=1e-13)


class TestSpikeSemantics:
    """Candidate crossing, somatic reset, cascade preservation."""

    def test_step_returns_binary(self) -> None:
        n = AlphaNeuron()
        assert n.step(0.0) in (0, 1)

    def test_spike_resets_only_the_membrane(self) -> None:
        n = AlphaNeuron(v=0.9, a_exc=0.4, i_exc=0.6, a_inh=0.2, i_inh=0.1, v_threshold=0.5)
        before = (n.a_exc, n.i_exc, n.a_inh, n.i_inh)
        assert n.step(0.0) == 1
        assert n.v == n.v_rest
        decay_exc = math.exp(-1.0 / 5.0)
        decay_inh = math.exp(-1.0 / 10.0)
        assert n.a_exc == pytest.approx(before[0] * decay_exc, rel=0.0, abs=1e-14)
        assert n.i_exc == pytest.approx(
            decay_exc * (before[1] + before[0] * 1.0 / 5.0), rel=0.0, abs=1e-14
        )
        assert n.a_inh == pytest.approx(before[2] * decay_inh, rel=0.0, abs=1e-14)
        assert n.i_inh == pytest.approx(
            decay_inh * (before[3] + before[2] * 1.0 / 10.0), rel=0.0, abs=1e-14
        )

    def test_spikes_under_excitatory_drive(self) -> None:
        n = AlphaNeuron()
        spikes = sum(n.step(3.0) for _ in range(2000))
        assert spikes > 0

    def test_inhibition_suppresses_excitatory_drive(self) -> None:
        exc_only = AlphaNeuron()
        dual = AlphaNeuron()
        exc_spikes = sum(exc_only.step(2.5) for _ in range(500))
        dual_spikes = sum(dual.step(2.5, 1.5) for _ in range(500))
        assert dual_spikes < exc_spikes

    def test_state_finite(self) -> None:
        n = AlphaNeuron()
        for index in range(5000):
            n.step(3.0 + math.sin(index * 0.01), 0.5)
        assert np.isfinite(n.v)
        assert np.isfinite(n.a_exc)
        assert np.isfinite(n.i_exc)
        assert np.isfinite(n.a_inh)
        assert np.isfinite(n.i_inh)

    def test_reset_restores_documented_state_preserving_configuration(self) -> None:
        n = AlphaNeuron()
        for _ in range(100):
            n.step(3.0)
        n.reset()
        assert n.v == n.v_rest
        assert (n.a_exc, n.i_exc, n.a_inh, n.i_inh) == (0.0, 0.0, 0.0, 0.0)
        assert (n.tau_v, n.tau_exc, n.tau_inh, n.dt) == (20.0, 5.0, 10.0, 1.0)


class TestAtomicity:
    """Rejected steps leave every dynamic state unchanged."""

    @pytest.mark.parametrize(
        ("field", "message"),
        (("v", "state must be numeric"), ("tau_v", "parameters must be numeric")),
    )
    def test_rejects_non_numeric_runtime_fields(self, field: str, message: str) -> None:
        n = AlphaNeuron()
        setattr(n, field, "invalid")
        with pytest.raises(ValueError, match=message):
            n.step(0.0)

    def test_rejects_non_numeric_runtime_current(self) -> None:
        n = AlphaNeuron()
        before = (n.v, n.a_exc, n.i_exc, n.a_inh, n.i_inh)
        with pytest.raises(ValueError, match="current values must be numeric"):
            n.step(cast(float, "invalid"))
        assert (n.v, n.a_exc, n.i_exc, n.a_inh, n.i_inh) == before

    @pytest.mark.parametrize("current", [float("nan"), float("inf"), -float("inf")])
    def test_rejects_non_finite_current(self, current: float) -> None:
        n = AlphaNeuron()
        before = (n.v, n.a_exc, n.i_exc, n.a_inh, n.i_inh)
        with pytest.raises(ValueError, match="current"):
            n.step(current)
        assert (n.v, n.a_exc, n.i_exc, n.a_inh, n.i_inh) == before

    def test_rejects_non_finite_runtime_state_before_update(self) -> None:
        n = AlphaNeuron()
        n.v = float("nan")
        with pytest.raises(ValueError, match="state"):
            n.step(0.0)
        assert np.isnan(n.v)

    def test_rejects_invalid_runtime_configuration_before_mutation(self) -> None:
        n = AlphaNeuron()
        n.tau_exc = 0.0
        before = (n.v, n.a_exc, n.i_exc, n.a_inh, n.i_inh)
        with pytest.raises(ValueError, match="tau_exc"):
            n.step(1.0)
        assert (n.v, n.a_exc, n.i_exc, n.a_inh, n.i_inh) == before

    def test_rejects_non_finite_update_before_state_mutation(self) -> None:
        n = AlphaNeuron(v=-1.0e308)
        before = (n.v, n.a_exc, n.i_exc, n.a_inh, n.i_inh)
        with pytest.raises((FloatingPointError, ValueError), match="finite"):
            n.step(1.0e308)
        assert (n.v, n.a_exc, n.i_exc, n.a_inh, n.i_inh) == before


class TestBatchAndDispatch:
    """The maintained batch lane matches the scalar golden loop."""

    def test_batch_matches_scalar_step_loop(self) -> None:
        exc = 1.5 + 0.8 * np.sin(np.arange(256) * 0.037)
        inh = 0.6 + 0.3 * np.cos(np.arange(256) * 0.021)
        scalar = AlphaNeuron()
        expected: dict[str, list[float]] = {
            key: [] for key in ("v", "a_exc", "i_exc", "a_inh", "i_inh")
        }
        expected_spikes = 0
        for exc_value, inh_value in zip(exc, inh):
            expected_spikes += scalar.step(float(exc_value), float(inh_value))
            for key in expected:
                expected[key].append(getattr(scalar, key))
        batch = AlphaNeuron().simulate(exc, inh, backend="python")
        for key in expected:
            np.testing.assert_allclose(batch[key], expected[key], rtol=0.0, atol=0.0)
        assert batch["spike_count"] == expected_spikes

    def test_scalar_inhibitory_broadcast_matches_vector(self) -> None:
        exc = np.full(64, 2.0)
        vector = AlphaNeuron().simulate(exc, np.full(64, 0.5), backend="python")
        scalar = AlphaNeuron().simulate(exc, 0.5, backend="python")
        np.testing.assert_array_equal(vector["v"], scalar["v"])

    def test_empty_batch_returns_initial_state(self) -> None:
        result = AlphaNeuron(v=0.1, a_exc=0.2).simulate([], backend="python")
        assert cast(npt.NDArray[np.float64], result["v"]).size == 0
        assert result["v_final"] == 0.1
        assert result["a_exc_final"] == 0.2
        assert result["spike_count"] == 0

    def test_simulate_writes_back_final_state(self) -> None:
        n = AlphaNeuron()
        result = n.simulate(np.full(200, 2.0), backend="python")
        assert n.v == result["v_final"]
        assert n.a_exc == result["a_exc_final"]

    def test_long_varied_run_is_finite_and_deterministic(self) -> None:
        exc = 2.0 + 0.5 * np.sin(np.arange(20_000, dtype=np.float64) * 0.013)
        inh = 0.8 + 0.2 * np.cos(np.arange(20_000, dtype=np.float64) * 0.007)
        first = AlphaNeuron().simulate(exc, inh, backend="python")
        second = AlphaNeuron().simulate(exc, inh, backend="python")
        assert np.isfinite(first["v"]).all()
        np.testing.assert_array_equal(first["v"], second["v"])
        np.testing.assert_array_equal(first["theta" if False else "i_exc"], second["i_exc"])


class TestAlphaNetwork:
    """Model works in the full SC-NeuroCore network pipeline."""

    def test_population_creation(self) -> None:
        pop = Population(AlphaNeuron, n=10, label="alpha")
        assert pop.n == 10
        assert pop.model_name == "AlphaNeuron"

    def test_network_produces_spikes(self) -> None:
        pop = Population(AlphaNeuron, n=20, label="alpha")
        proj = Projection(pop, pop, weight=1.0, probability=0.2, seed=42)
        drive = PoissonInput(n=20, rate_hz=500.0, weight=3.0, dt=0.001, seed=42)
        mon = SpikeMonitor(pop)
        net = Network(pop, proj, drive, mon)
        net.run(duration=0.5, dt=0.001, backend="python")
        assert mon.count > 0, "network produced zero spikes"

    def test_spike_trains_extractable(self) -> None:
        pop = Population(AlphaNeuron, n=10, label="alpha")
        drive = PoissonInput(n=10, rate_hz=500.0, weight=3.0, dt=0.001, seed=42)
        mon = SpikeMonitor(pop)
        net = Network(pop, drive, mon)
        net.run(duration=0.3, dt=0.001, backend="python")
        trains = mon.spike_trains
        assert isinstance(trains, dict)
        assert len(trains) > 0, "no spike trains recorded"


class TestAlphaAnalysis:
    """Analysis toolkit works on spikes from this model."""

    def _get_binary_train(self) -> npt.NDArray[np.int8]:
        n = AlphaNeuron()
        train = np.zeros(5000, dtype=np.int8)
        for t in range(5000):
            train[t] = n.step(2.5)
        return train

    def test_firing_rate(self) -> None:
        train = self._get_binary_train()
        rate = firing_rate(train, dt=0.001)
        assert rate > 0

    def test_spike_count(self) -> None:
        train = self._get_binary_train()
        assert spike_count(train) > 0
