# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — End-to-end test: EnergyLIFNeuron

"""Full pipeline test for EnergyLIFNeuron exact-flow hardening.

LIF with metabolic energy constraint ε. Spike cost depletes ε."""

from __future__ import annotations

import numpy as np
import pytest

from sc_neurocore.neurons.models.energy_lif import EnergyLIFNeuron
from sc_neurocore.network.population import Population
from sc_neurocore.network.projection import Projection
from sc_neurocore.network.network import Network
from sc_neurocore.network.monitor import SpikeMonitor
from sc_neurocore.network.stimulus import PoissonInput
from sc_neurocore.analysis.spike_stats.basic import firing_rate, spike_count


class TestEnergyLIFIsolation:
    def test_construction(self):
        n = EnergyLIFNeuron()
        assert n.epsilon == 1.0

    def test_step_returns_binary(self):
        assert EnergyLIFNeuron().step(0.0) in (0, 1)

    def test_subthreshold(self):
        n = EnergyLIFNeuron()
        assert sum(n.step(10.0) for _ in range(5000)) == 0

    def test_spikes(self):
        n = EnergyLIFNeuron()
        assert sum(n.step(30.0) for _ in range(5000)) > 10

    def test_energy_depletes(self):
        """ε should decrease after spiking."""
        n = EnergyLIFNeuron()
        for _ in range(5000):
            n.step(50.0)
        assert n.epsilon < 1.0, "energy did not deplete"

    def test_energy_recovers(self):
        """ε should recover toward ε₀ without spiking."""
        n = EnergyLIFNeuron()
        n.epsilon = 0.1
        for _ in range(5000):
            n.step(0.0)
        assert n.epsilon > 0.1, "energy did not recover"

    def test_energy_gates_spiking(self):
        """When ε < 0.1, neuron cannot spike (energy gate)."""
        n = EnergyLIFNeuron()
        n.epsilon = 0.05
        spikes = sum(n.step(50.0) for _ in range(100))
        assert spikes == 0, "spiked with depleted energy"

    def test_energy_nonnegative(self):
        n = EnergyLIFNeuron()
        for _ in range(10000):
            n.step(50.0)
        assert n.epsilon >= 0.0

    def test_reset(self):
        n = EnergyLIFNeuron()
        for _ in range(100):
            n.step(50.0)
        n.reset()
        assert n.v == n.v_rest
        assert n.epsilon == n.epsilon_0


class TestEnergyLIFValidation:
    @pytest.mark.parametrize("field", ["v", "v_rest", "v_reset", "v_threshold"])
    @pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
    def test_rejects_non_finite_voltage_parameters(self, field: str, value: float):
        with pytest.raises(ValueError, match=field):
            EnergyLIFNeuron(**{field: value})

    @pytest.mark.parametrize("field", ["epsilon", "epsilon_0"])
    @pytest.mark.parametrize("value", [-1.0, np.nan, np.inf, -np.inf])
    def test_rejects_negative_or_non_finite_energy_parameters(self, field: str, value: float):
        with pytest.raises(ValueError, match=field):
            EnergyLIFNeuron(**{field: value})

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"epsilon": 1.1},
            {"epsilon": 0.2, "epsilon_0": 0.1},
            {"v_threshold": -75.0},
            {"v_reset": -45.0},
            {"dt": 11.0},
            {"dt": 501.0},
        ],
    )
    def test_rejects_non_physical_energy_geometry_or_timestep(self, kwargs):
        with pytest.raises(ValueError):
            EnergyLIFNeuron(**kwargs)

    def test_energy_recovery_is_monotone_and_bounded_without_spike(self):
        n = EnergyLIFNeuron(epsilon=0.2)
        before = n.epsilon

        assert n.step(0.0) == 0

        assert before < n.epsilon < n.epsilon_0

    def test_exact_candidate_commit(self):
        n = EnergyLIFNeuron(epsilon=0.5)
        expected_v, expected_epsilon = n._exact_candidate(10.0)

        assert n.step(10.0) == 0

        assert abs(n.v - expected_v) < 1.0e-12
        assert abs(n.epsilon - expected_epsilon) < 1.0e-12

    def test_exact_flow_separates_from_forward_euler(self):
        n = EnergyLIFNeuron(v=-65.0, epsilon=0.5, dt=2.0)
        euler_v = n.v + (-(n.v - n.v_rest) + n.resistance * n.epsilon * 10.0) / n.tau_m * n.dt
        exact_v, _ = n._exact_candidate(10.0)

        assert abs(exact_v - euler_v) > 1.0e-3

    def test_spike_uses_energy_candidate(self):
        n = EnergyLIFNeuron()
        _, epsilon_candidate = n._exact_candidate(250.0)

        assert n.step(250.0) == 1

        assert n.v == n.v_reset
        assert abs(n.epsilon - max(0.0, epsilon_candidate - n.alpha)) < 1.0e-12

    @pytest.mark.parametrize("field", ["tau_m", "tau_e", "resistance", "dt"])
    @pytest.mark.parametrize("value", [0.0, -1.0, np.nan, np.inf])
    def test_rejects_non_positive_or_non_finite_scale_parameters(self, field: str, value: float):
        with pytest.raises(ValueError, match=field):
            EnergyLIFNeuron(**{field: value})

    @pytest.mark.parametrize("alpha", [-1.0, np.nan, np.inf, -np.inf])
    def test_rejects_negative_or_non_finite_spike_cost(self, alpha: float):
        with pytest.raises(ValueError, match="alpha"):
            EnergyLIFNeuron(alpha=alpha)

    @pytest.mark.parametrize("current", [np.nan, np.inf, -np.inf])
    def test_rejects_non_finite_current_before_state_mutation(self, current: float):
        n = EnergyLIFNeuron(v=-65.0, epsilon=0.5)
        before = (n.v, n.epsilon)
        with pytest.raises(ValueError, match="current"):
            n.step(current)
        assert (n.v, n.epsilon) == before

    def test_rejects_corrupted_runtime_state_before_mutation(self):
        n = EnergyLIFNeuron(v=-65.0, epsilon=0.5)
        n.epsilon = -1.0
        before = (n.v, n.epsilon)

        with pytest.raises(ValueError, match="epsilon"):
            n.step(10.0)

        assert (n.v, n.epsilon) == before


class TestEnergyLIFNetwork:
    def test_population(self):
        assert Population(EnergyLIFNeuron, n=10, label="elif").n == 10

    def test_network_spikes(self):
        pop = Population(EnergyLIFNeuron, n=20, label="elif")
        drive = PoissonInput(n=20, rate_hz=500.0, weight=30.0, dt=0.001, seed=42)
        mon = SpikeMonitor(pop)
        net = Network(pop, drive, mon)
        net.run(duration=0.5, dt=0.001, backend="python")
        assert mon.count > 0

    def test_with_projection(self):
        pop = Population(EnergyLIFNeuron, n=10, label="elif")
        proj = Projection(pop, pop, weight=5.0, probability=0.3, seed=42)
        drive = PoissonInput(n=10, rate_hz=500.0, weight=30.0, dt=0.001, seed=42)
        mon = SpikeMonitor(pop)
        net = Network(pop, proj, drive, mon)
        net.run(duration=0.3, dt=0.001, backend="python")
        assert isinstance(mon.spike_trains, dict)


class TestEnergyLIFAnalysis:
    def _get_train(self):
        n = EnergyLIFNeuron()
        train = np.zeros(5000, dtype=np.int8)
        for t in range(5000):
            train[t] = n.step(40.0)
        return train

    def test_firing_rate(self):
        assert firing_rate(self._get_train(), dt=0.001) > 0

    def test_spike_count(self):
        assert spike_count(self._get_train()) > 10
