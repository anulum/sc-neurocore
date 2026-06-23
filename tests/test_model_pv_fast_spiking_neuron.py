# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — End-to-end test: PVFastSpikingNeuron

"""Full pipeline test for PVFastSpikingNeuron (Wang-Buzsáki 1996 + Kv3.1).

Parvalbumin fast-spiking interneuron: high-frequency, non-adapting discharge
sharpened by the Kv3.1 current. The candidate-first RK4 integrator advances the
four-state ``(V, h, n, p)`` system."""

from __future__ import annotations

import numpy as np
import pytest

from sc_neurocore.neurons.models.pv_fast_spiking_neuron import (
    PVFastSpikingNeuron,
    _safe_rate,
)
from sc_neurocore.network.population import Population
from sc_neurocore.network.network import Network
from sc_neurocore.network.monitor import SpikeMonitor
from sc_neurocore.network.stimulus import PoissonInput


def _spikes(neuron: PVFastSpikingNeuron, current: float, steps: int) -> int:
    return sum(neuron.step(current) for _ in range(steps))


class TestPVFastSpikingIsolation:
    def test_construction_defaults(self):
        n = PVFastSpikingNeuron()
        assert n.v == -65.0
        assert n.g_kv3 == 5.0
        assert n.phi == 5.0
        assert n.dt == 0.01

    def test_step_returns_binary(self):
        assert PVFastSpikingNeuron().step(2.0) in (0, 1)

    def test_quiescent_without_drive(self):
        assert _spikes(PVFastSpikingNeuron(), 0.0, 20000) == 0

    def test_suprathreshold_high_frequency_firing(self):
        # The defining FS feature: sustained high-rate discharge under drive.
        assert _spikes(PVFastSpikingNeuron(), 2.0, 40000) >= 200

    def test_rate_increases_with_current(self):
        s1 = _spikes(PVFastSpikingNeuron(), 1.0, 30000)
        s2 = _spikes(PVFastSpikingNeuron(), 3.0, 30000)
        assert s1 < s2

    def test_no_spike_frequency_adaptation(self):
        n = PVFastSpikingNeuron()
        spike_times = [t for t in range(40000) if n.step(2.0)]
        assert len(spike_times) >= 20
        intervals = np.diff(spike_times)
        early = float(np.mean(intervals[:5]))
        late = float(np.mean(intervals[-5:]))
        # FS cells do not adapt: late inter-spike intervals stay close to early.
        assert late < early * 1.3

    def test_state_finite_long_run(self):
        n = PVFastSpikingNeuron()
        for _ in range(50000):
            n.step(2.0)
        for value in (n.v, n.h, n.n, n.p):
            assert np.isfinite(value)

    def test_reset_restores_initial(self):
        n = PVFastSpikingNeuron()
        for _ in range(1000):
            n.step(2.0)
        n.reset()
        assert n.v == -65.0
        assert (n.h, n.n, n.p) == (0.8, 0.1, 0.0)


class TestPVFastSpikingKv3:
    def test_kv3_block_changes_firing(self):
        intact = _spikes(PVFastSpikingNeuron(), 2.0, 40000)
        blocked = _spikes(PVFastSpikingNeuron(g_kv3=0.0), 2.0, 40000)
        assert intact != blocked


class TestPVFastSpikingIntegrator:
    def test_default_integrator_is_rk4(self):
        assert PVFastSpikingNeuron().integrator == "rk4"

    def test_rejects_unknown_integrator(self):
        with pytest.raises(ValueError, match="Unsupported integrator"):
            PVFastSpikingNeuron(integrator="midpoint")  # type: ignore[arg-type]

    def test_baseline_euler_path_runs_and_diverges_from_rk4(self):
        rk4 = PVFastSpikingNeuron()
        euler = PVFastSpikingNeuron(integrator="baseline_euler")
        assert _spikes(rk4, 2.0, 40000) > 0
        assert _spikes(euler, 2.0, 40000) > 0
        assert rk4.v != euler.v

    def test_rk4_and_euler_agree_to_first_order_at_tiny_dt(self):
        rk4 = PVFastSpikingNeuron(dt=1e-5)
        euler = PVFastSpikingNeuron(dt=1e-5, integrator="baseline_euler")
        for _ in range(50):
            rk4.step(2.0)
            euler.step(2.0)
        assert abs(rk4.v - euler.v) < 1e-2


class TestPVFastSpikingSafeRate:
    def test_fallback_returned_at_singularity(self):
        # v + vhalf = 0 -> the L'Hôpital limit a*k is returned.
        assert _safe_rate(0.1, 35.0, -35.0, 10.0, 1.0) == 1.0

    def test_regular_branch_matches_hodgkin_huxley_ratio(self):
        v, a, vhalf, k = -30.0, 0.1, 35.0, 10.0
        d = v + vhalf
        expected = a * d / (1.0 - np.exp(-d / k))
        assert _safe_rate(a, vhalf, v, k, 1.0) == pytest.approx(expected)


class TestPVFastSpikingValidation:
    @pytest.mark.parametrize(
        "kwargs",
        [
            {"g_na": -1.0},
            {"g_k": 0.0},
            {"g_l": -0.1},
            {"c_m": 0.0},
            {"phi": 0.0},
            {"dt": 0.0},
            {"g_kv3": -1.0},
        ],
    )
    def test_rejects_invalid_parameters(self, kwargs: dict[str, float]):
        with pytest.raises(ValueError):
            PVFastSpikingNeuron(**kwargs)

    def test_accepts_zero_kv3_conductance(self):
        assert PVFastSpikingNeuron(g_kv3=0.0).g_kv3 == 0.0

    @pytest.mark.parametrize("field", ["v", "e_na", "e_k", "e_l"])
    def test_rejects_non_finite_field(self, field: str):
        with pytest.raises(ValueError, match="must be finite"):
            PVFastSpikingNeuron(**{field: float("nan")})

    def test_rejects_boolean_field(self):
        with pytest.raises(ValueError, match="must be finite"):
            PVFastSpikingNeuron(v=True)  # type: ignore[arg-type]

    def test_rejects_non_finite_current(self):
        with pytest.raises(ValueError, match="must be finite"):
            PVFastSpikingNeuron().step(float("inf"))

    def test_runtime_validation_catches_corrupted_state(self):
        n = PVFastSpikingNeuron()
        n.dt = -1.0
        with pytest.raises(ValueError, match="dt must be positive"):
            n.step(0.0)

    def test_non_finite_candidate_fails_closed(self):
        n = PVFastSpikingNeuron()
        with pytest.raises((FloatingPointError, OverflowError)):
            for _ in range(60):
                n.step(1e308)


class TestPVFastSpikingNetwork:
    def test_population_size(self):
        assert Population(PVFastSpikingNeuron, n=8, label="pv").n == 8

    def test_population_drives_spikes(self):
        pop = Population(PVFastSpikingNeuron, n=5, label="pv")
        drive = PoissonInput(n=5, rate_hz=500.0, weight=6.0, dt=0.001, seed=42)
        mon = SpikeMonitor(pop)
        net = Network(pop, drive, mon)
        net.run(duration=2.0, dt=0.001, backend="python")
        assert mon.count > 0
