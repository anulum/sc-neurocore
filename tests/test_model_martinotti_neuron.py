# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — End-to-end test: MartinottiNeuron

"""Full pipeline test for MartinottiNeuron (Pospischil 2008 adapting interneuron).

Six-state (V, m, h, n, p, s) Martinotti cell integrated with candidate-first RK4.
Includes a regression for the β_m offset: the earlier ``-17`` numerator (shared
with α_h) drove the cell into depolarisation block — two or three spikes then a
fixed point near threshold for any stimulus — while the published ``-40`` offset
restores a monotone frequency-current relation under the strong M-current."""

from __future__ import annotations

import numpy as np
import pytest

from sc_neurocore.neurons.models.martinotti_neuron import MartinottiNeuron
from sc_neurocore.network.population import Population
from sc_neurocore.network.network import Network
from sc_neurocore.network.monitor import SpikeMonitor
from sc_neurocore.network.stimulus import PoissonInput


def _spikes(neuron: MartinottiNeuron, current: float, steps: int) -> int:
    return sum(neuron.step(current) for _ in range(steps))


class TestMartinottiIsolation:
    def test_construction_defaults(self):
        n = MartinottiNeuron()
        assert n.v == -65.0
        assert n.g_na == 40.0
        assert n.g_m == 0.25
        assert n.c_m == 0.8
        assert n.dt == 0.025

    def test_step_returns_binary(self):
        assert MartinottiNeuron().step(5.0) in (0, 1)

    def test_quiescent_without_drive(self):
        assert _spikes(MartinottiNeuron(), 0.0, 20000) == 0

    def test_suprathreshold_spiking(self):
        assert _spikes(MartinottiNeuron(), 5.0, 40000) > 20

    def test_state_finite_long_run(self):
        n = MartinottiNeuron()
        for _ in range(50000):
            n.step(5.0)
        for value in (n.v, n.m, n.h, n.n, n.p, n.s):
            assert np.isfinite(value)

    def test_reset_restores_initial(self):
        n = MartinottiNeuron()
        for _ in range(1000):
            n.step(5.0)
        n.reset()
        assert n.v == -65.0
        assert (n.m, n.h, n.n, n.p, n.s) == (0.02, 0.8, 0.2, 0.0, 0.9)


class TestMartinottiBetaMRegression:
    """Guards the corrected β_m offset against the depolarisation-block bug."""

    def test_firing_rate_increases_with_current(self):
        low = _spikes(MartinottiNeuron(), 2.0, 30000)
        mid = _spikes(MartinottiNeuron(), 5.0, 30000)
        high = _spikes(MartinottiNeuron(), 10.0, 30000)
        assert low < mid < high

    def test_no_depolarisation_block_at_strong_drive(self):
        # The bug capped firing at two or three spikes and pinned V near threshold
        # for any stimulus; healthy kinetics fire repetitively under strong drive.
        assert _spikes(MartinottiNeuron(), 10.0, 40000) > 100

    def test_membrane_recovers_below_threshold_after_drive(self):
        n = MartinottiNeuron()
        for _ in range(40000):
            n.step(5.0)
        assert n.v < n.v_threshold


class TestMartinottiAdaptation:
    def test_m_current_block_changes_firing(self):
        intact = _spikes(MartinottiNeuron(), 5.0, 40000)
        blocked = _spikes(MartinottiNeuron(g_m=0.0), 5.0, 40000)
        assert intact != blocked


class TestMartinottiIntegrator:
    def test_default_integrator_is_rk4(self):
        assert MartinottiNeuron().integrator == "rk4"

    def test_rejects_unknown_integrator(self):
        with pytest.raises(ValueError, match="Unsupported integrator"):
            MartinottiNeuron(integrator="midpoint")  # type: ignore[arg-type]

    def test_baseline_euler_path_runs_and_diverges_from_rk4(self):
        rk4 = MartinottiNeuron()
        euler = MartinottiNeuron(integrator="baseline_euler")
        assert _spikes(rk4, 5.0, 40000) > 0
        assert _spikes(euler, 5.0, 40000) > 0
        assert rk4.v != euler.v

    def test_rk4_and_euler_agree_to_first_order_at_tiny_dt(self):
        rk4 = MartinottiNeuron(dt=1e-5)
        euler = MartinottiNeuron(dt=1e-5, integrator="baseline_euler")
        for _ in range(200):
            rk4.step(5.0)
            euler.step(5.0)
        assert abs(rk4.v - euler.v) < 1e-2


class TestMartinottiSingularRates:
    def test_alpha_singular_returns_limit_at_singularity(self):
        from sc_neurocore.neurons.models.martinotti_neuron import _alpha_singular

        assert _alpha_singular(0.0, -4.0, -4.0) == -4.0
        assert _alpha_singular(0.0, 5.0, 5.0) == 5.0

    def test_alpha_singular_continuous_across_singularity(self):
        from sc_neurocore.neurons.models.martinotti_neuron import _alpha_singular

        left = _alpha_singular(-1e-7, 5.0, 5.0)
        right = _alpha_singular(1e-7, 5.0, 5.0)
        assert abs(left - 5.0) < 1e-3
        assert abs(right - 5.0) < 1e-3


class TestMartinottiValidation:
    @pytest.mark.parametrize(
        "kwargs",
        [
            {"g_na": -1.0},
            {"c_m": 0.0},
            {"dt": 0.0},
            {"g_m": -0.1},
            {"g_t": -1.0},
        ],
    )
    def test_rejects_invalid_parameters(self, kwargs: dict[str, float]):
        with pytest.raises(ValueError):
            MartinottiNeuron(**kwargs)

    def test_accepts_zero_conductances(self):
        assert MartinottiNeuron(g_t=0.0, g_m=0.0).g_t == 0.0

    @pytest.mark.parametrize("field", ["v", "e_na", "e_k", "e_ca"])
    def test_rejects_non_finite_field(self, field: str):
        with pytest.raises(ValueError, match="must be finite"):
            MartinottiNeuron(**{field: float("nan")})

    def test_rejects_boolean_field(self):
        with pytest.raises(ValueError, match="must be finite"):
            MartinottiNeuron(v=True)  # type: ignore[arg-type]

    def test_rejects_non_finite_current(self):
        with pytest.raises(ValueError, match="must be finite"):
            MartinottiNeuron().step(float("inf"))

    def test_runtime_validation_catches_corrupted_state(self):
        n = MartinottiNeuron()
        n.dt = -1.0
        with pytest.raises(ValueError, match="dt must be positive"):
            n.step(0.0)

    def test_non_finite_candidate_fails_closed(self):
        n = MartinottiNeuron()
        with pytest.raises((FloatingPointError, OverflowError)):
            for _ in range(80):
                n.step(1e308)


class TestMartinottiNetwork:
    def test_population_size(self):
        assert Population(MartinottiNeuron, n=8, label="martinotti").n == 8

    def test_population_drives_spikes(self):
        pop = Population(MartinottiNeuron, n=5, label="martinotti")
        drive = PoissonInput(n=5, rate_hz=600.0, weight=8.0, dt=0.001, seed=42)
        mon = SpikeMonitor(pop)
        net = Network(pop, drive, mon)
        net.run(duration=2.0, dt=0.001, backend="python")
        assert mon.count > 0
