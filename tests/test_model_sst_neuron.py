# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — End-to-end test: SSTNeuron

"""Full pipeline test for SSTNeuron (Pospischil 2008 LTS parameterisation).

Seven-state (V, m, h, n, p, s, r) somatostatin low-threshold spiking interneuron
integrated with candidate-first RK4. Includes a regression for the β_m offset:
the earlier ``-17`` numerator (shared with α_h) drove the cell into depolarisation
block — exactly three spikes then a fixed point near threshold for any stimulus —
while the published ``-40`` offset restores a monotone frequency-current relation."""

from __future__ import annotations

import numpy as np
import pytest

from sc_neurocore.neurons.models.sst_neuron import SSTNeuron
from sc_neurocore.network.population import Population
from sc_neurocore.network.network import Network
from sc_neurocore.network.monitor import SpikeMonitor
from sc_neurocore.network.stimulus import PoissonInput


def _spikes(neuron: SSTNeuron, current: float, steps: int) -> int:
    return sum(neuron.step(current) for _ in range(steps))


class TestSSTIsolation:
    def test_construction_defaults(self):
        n = SSTNeuron()
        assert n.v == -65.0
        assert n.g_na == 50.0
        assert n.g_m == 0.12
        assert n.c_m == 1.0
        assert n.dt == 0.025

    def test_step_returns_binary(self):
        assert SSTNeuron().step(2.0) in (0, 1)

    def test_quiescent_without_drive(self):
        assert _spikes(SSTNeuron(), 0.0, 20000) == 0

    def test_suprathreshold_spiking(self):
        assert _spikes(SSTNeuron(), 2.0, 40000) > 20

    def test_state_finite_long_run(self):
        n = SSTNeuron()
        for _ in range(50000):
            n.step(2.0)
        for value in (n.v, n.m, n.h, n.n, n.p, n.s, n.r):
            assert np.isfinite(value)

    def test_reset_restores_initial(self):
        n = SSTNeuron()
        for _ in range(1000):
            n.step(2.0)
        n.reset()
        assert n.v == -65.0
        assert (n.m, n.h, n.n, n.p, n.s, n.r) == (0.02, 0.8, 0.2, 0.0, 0.9, 0.1)


class TestSSTBetaMRegression:
    """Guards the corrected β_m offset against the depolarisation-block bug."""

    def test_firing_rate_increases_with_current(self):
        low = _spikes(SSTNeuron(), 0.5, 30000)
        mid = _spikes(SSTNeuron(), 2.0, 30000)
        high = _spikes(SSTNeuron(), 5.0, 30000)
        assert low < mid < high

    def test_no_depolarisation_block_at_strong_drive(self):
        # The bug stuck V near threshold and capped firing at three spikes for any
        # stimulus; healthy kinetics fire repetitively under strong drive.
        assert _spikes(SSTNeuron(), 5.0, 40000) > 100

    def test_membrane_recovers_below_threshold_after_drive(self):
        n = SSTNeuron()
        for _ in range(40000):
            n.step(2.0)
        assert n.v < n.v_threshold


class TestSSTAdaptation:
    def test_m_current_block_changes_firing(self):
        intact = _spikes(SSTNeuron(), 2.0, 40000)
        blocked = _spikes(SSTNeuron(g_m=0.0), 2.0, 40000)
        assert intact != blocked


class TestSSTIntegrator:
    def test_default_integrator_is_rk4(self):
        assert SSTNeuron().integrator == "rk4"

    def test_rejects_unknown_integrator(self):
        with pytest.raises(ValueError, match="Unsupported integrator"):
            SSTNeuron(integrator="midpoint")  # type: ignore[arg-type]

    def test_baseline_euler_path_runs_and_diverges_from_rk4(self):
        rk4 = SSTNeuron()
        euler = SSTNeuron(integrator="baseline_euler")
        assert _spikes(rk4, 5.0, 40000) > 0
        assert _spikes(euler, 5.0, 40000) > 0
        assert rk4.v != euler.v

    def test_rk4_and_euler_agree_to_first_order_at_tiny_dt(self):
        rk4 = SSTNeuron(dt=1e-5)
        euler = SSTNeuron(dt=1e-5, integrator="baseline_euler")
        for _ in range(200):
            rk4.step(2.0)
            euler.step(2.0)
        assert abs(rk4.v - euler.v) < 1e-2


class TestSSTSingularRates:
    def test_alpha_singular_returns_limit_at_singularity(self):
        from sc_neurocore.neurons.models.sst_neuron import _alpha_singular

        # At numerator 0 the closed-form L'Hôpital limit (equal to slope) is used.
        assert _alpha_singular(0.0, -4.0, -4.0) == -4.0
        assert _alpha_singular(0.0, 5.0, 5.0) == 5.0

    def test_alpha_singular_continuous_across_singularity(self):
        from sc_neurocore.neurons.models.sst_neuron import _alpha_singular

        left = _alpha_singular(-1e-7, 5.0, 5.0)
        right = _alpha_singular(1e-7, 5.0, 5.0)
        assert abs(left - 5.0) < 1e-3
        assert abs(right - 5.0) < 1e-3


class TestSSTValidation:
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
            SSTNeuron(**kwargs)

    def test_accepts_zero_conductances(self):
        assert SSTNeuron(g_t=0.0, g_h=0.0, g_m=0.0).g_t == 0.0

    @pytest.mark.parametrize("field", ["v", "e_na", "e_k", "e_ca"])
    def test_rejects_non_finite_field(self, field: str):
        with pytest.raises(ValueError, match="must be finite"):
            SSTNeuron(**{field: float("nan")})

    def test_rejects_boolean_field(self):
        with pytest.raises(ValueError, match="must be finite"):
            SSTNeuron(v=True)  # type: ignore[arg-type]

    def test_rejects_non_finite_current(self):
        with pytest.raises(ValueError, match="must be finite"):
            SSTNeuron().step(float("inf"))

    def test_runtime_validation_catches_corrupted_state(self):
        n = SSTNeuron()
        n.dt = -1.0
        with pytest.raises(ValueError, match="dt must be positive"):
            n.step(0.0)

    def test_non_finite_candidate_fails_closed(self):
        n = SSTNeuron()
        with pytest.raises((FloatingPointError, OverflowError)):
            for _ in range(80):
                n.step(1e308)


class TestSSTNetwork:
    def test_population_size(self):
        assert Population(SSTNeuron, n=8, label="sst").n == 8

    def test_population_drives_spikes(self):
        pop = Population(SSTNeuron, n=5, label="sst")
        drive = PoissonInput(n=5, rate_hz=500.0, weight=6.0, dt=0.001, seed=42)
        mon = SpikeMonitor(pop)
        net = Network(pop, drive, mon)
        net.run(duration=2.0, dt=0.001, backend="python")
        assert mon.count > 0
