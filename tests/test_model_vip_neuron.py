# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — End-to-end test: VIPNeuron

"""Full pipeline test for VIPNeuron (Porter 1998 + Kv4 A-current).

VIP irregular-spiking interneuron: the slow-inactivating A-type current produces
firing accommodation. The candidate-first RK4 integrator advances the five-state
``(V, h, n, a, b)`` system."""

from __future__ import annotations

import numpy as np
import pytest

from sc_neurocore.neurons.models.vip_neuron import VIPNeuron
from sc_neurocore.network.population import Population
from sc_neurocore.network.network import Network
from sc_neurocore.network.monitor import SpikeMonitor
from sc_neurocore.network.stimulus import PoissonInput


def _spikes(neuron: VIPNeuron, current: float, steps: int) -> int:
    return sum(neuron.step(current) for _ in range(steps))


class TestVIPIsolation:
    def test_construction_defaults(self):
        n = VIPNeuron()
        assert n.v == -65.0
        assert n.g_a == 8.0
        assert n.c_m == 0.5
        assert n.dt == 0.025

    def test_step_returns_binary(self):
        assert VIPNeuron().step(1.0) in (0, 1)

    def test_quiescent_without_drive(self):
        assert _spikes(VIPNeuron(), 0.0, 20000) == 0

    def test_suprathreshold_spiking(self):
        assert _spikes(VIPNeuron(), 1.0, 40000) >= 10

    def test_rate_increases_with_current(self):
        s1 = _spikes(VIPNeuron(), 0.6, 30000)
        s2 = _spikes(VIPNeuron(), 2.0, 30000)
        assert s1 < s2

    def test_state_finite_long_run(self):
        n = VIPNeuron()
        for _ in range(50000):
            n.step(1.0)
        for value in (n.v, n.h, n.n, n.a, n.b):
            assert np.isfinite(value)

    def test_reset_restores_initial(self):
        n = VIPNeuron()
        for _ in range(1000):
            n.step(1.0)
        n.reset()
        assert n.v == -65.0
        assert (n.h, n.n, n.a, n.b) == (0.8, 0.1, 0.0, 0.9)


class TestVIPACurrent:
    def test_a_current_block_changes_firing(self):
        intact = _spikes(VIPNeuron(), 1.0, 40000)
        blocked = _spikes(VIPNeuron(g_a=0.0), 1.0, 40000)
        assert intact != blocked

    def test_a_gate_inactivates_during_drive(self):
        # The b inactivation gate falls from its rested 0.9 under sustained drive.
        n = VIPNeuron()
        for _ in range(2000):
            n.step(1.0)
        assert n.b < 0.9


class TestVIPIntegrator:
    def test_default_integrator_is_rk4(self):
        assert VIPNeuron().integrator == "rk4"

    def test_rejects_unknown_integrator(self):
        with pytest.raises(ValueError, match="Unsupported integrator"):
            VIPNeuron(integrator="midpoint")  # type: ignore[arg-type]

    def test_baseline_euler_path_runs_and_diverges_from_rk4(self):
        rk4 = VIPNeuron()
        euler = VIPNeuron(integrator="baseline_euler")
        assert _spikes(rk4, 1.0, 40000) > 0
        assert _spikes(euler, 1.0, 40000) > 0
        assert rk4.v != euler.v

    def test_rk4_and_euler_agree_to_first_order_at_tiny_dt(self):
        rk4 = VIPNeuron(dt=1e-5)
        euler = VIPNeuron(dt=1e-5, integrator="baseline_euler")
        for _ in range(200):
            rk4.step(1.0)
            euler.step(1.0)
        assert abs(rk4.v - euler.v) < 1e-2


class TestVIPValidation:
    @pytest.mark.parametrize(
        "kwargs",
        [
            {"g_na": -1.0},
            {"g_k": 0.0},
            {"g_l": -0.1},
            {"c_m": 0.0},
            {"dt": 0.0},
            {"g_a": -1.0},
        ],
    )
    def test_rejects_invalid_parameters(self, kwargs: dict[str, float]):
        with pytest.raises(ValueError):
            VIPNeuron(**kwargs)

    def test_accepts_zero_a_conductance(self):
        assert VIPNeuron(g_a=0.0).g_a == 0.0

    @pytest.mark.parametrize("field", ["v", "e_na", "e_k", "e_l"])
    def test_rejects_non_finite_field(self, field: str):
        with pytest.raises(ValueError, match="must be finite"):
            VIPNeuron(**{field: float("nan")})

    def test_rejects_boolean_field(self):
        with pytest.raises(ValueError, match="must be finite"):
            VIPNeuron(v=True)  # type: ignore[arg-type]

    def test_rejects_non_finite_current(self):
        with pytest.raises(ValueError, match="must be finite"):
            VIPNeuron().step(float("inf"))

    def test_runtime_validation_catches_corrupted_state(self):
        n = VIPNeuron()
        n.dt = -1.0
        with pytest.raises(ValueError, match="dt must be positive"):
            n.step(0.0)

    def test_non_finite_candidate_fails_closed(self):
        n = VIPNeuron()
        with pytest.raises((FloatingPointError, OverflowError)):
            for _ in range(60):
                n.step(1e308)


class TestVIPNetwork:
    def test_population_size(self):
        assert Population(VIPNeuron, n=8, label="vip").n == 8

    def test_population_drives_spikes(self):
        pop = Population(VIPNeuron, n=5, label="vip")
        drive = PoissonInput(n=5, rate_hz=500.0, weight=4.0, dt=0.001, seed=42)
        mon = SpikeMonitor(pop)
        net = Network(pop, drive, mon)
        net.run(duration=2.0, dt=0.001, backend="python")
        assert mon.count > 0
