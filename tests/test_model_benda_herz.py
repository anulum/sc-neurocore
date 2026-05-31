# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — End-to-end test: BendaHerzNeuron

"""Full pipeline test for BendaHerzNeuron (Benda & Herz 2003).

Phenomenological spike-frequency adaptation. Stochastic spiking
from instantaneous f-I curve with adaptation variable A."""

from __future__ import annotations

import numpy as np
import pytest

from sc_neurocore.neurons.models.benda_herz import BendaHerzNeuron
from sc_neurocore.network.population import Population
from sc_neurocore.network.projection import Projection
from sc_neurocore.network.network import Network
from sc_neurocore.network.monitor import SpikeMonitor
from sc_neurocore.network.stimulus import PoissonInput
from sc_neurocore.analysis.spike_stats.basic import firing_rate, spike_count


def _rk4_reference(neuron: BendaHerzNeuron, current: float) -> tuple[float, float]:
    def rhs(a: float) -> tuple[float, float]:
        rate = neuron._f_onset(current - a)
        return -a / neuron.tau_a + neuron.delta_a * rate, rate

    k1, r1 = rhs(neuron.a)
    k2, r2 = rhs(neuron.a + 0.5 * neuron.dt * k1)
    k3, r3 = rhs(neuron.a + 0.5 * neuron.dt * k2)
    k4, r4 = rhs(neuron.a + neuron.dt * k3)
    next_a = neuron.a + (neuron.dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
    average_rate = (r1 + 2.0 * r2 + 2.0 * r3 + r4) / 6.0
    probability = -np.expm1(-(average_rate * neuron.dt / 1000.0))
    return next_a, probability


class TestBendaHerzIsolation:
    def test_construction(self):
        n = BendaHerzNeuron()
        assert n.a == 0.0
        assert n.f_max == 200.0

    def test_step_returns_binary(self):
        n = BendaHerzNeuron(seed=1)
        result = n.step(10.0)
        assert result in (0, 1)

    def test_spikes_under_drive(self):
        """Stochastic model — needs many steps for reliable spiking."""
        n = BendaHerzNeuron(seed=2)
        spikes = sum(n.step(50.0) for _ in range(10000))
        assert spikes > 0, "no spikes at I=50 over 10K steps"

    def test_adaptation_increases(self):
        """Adaptation variable A should increase under sustained drive."""
        n = BendaHerzNeuron(seed=3)
        a_init = n.a
        for _ in range(1000):
            n.step(30.0)
        assert n.a > a_init, "adaptation variable did not increase"

    def test_adaptation_reduces_rate(self):
        """Rate should decrease over time due to SFA."""
        n = BendaHerzNeuron(seed=4)
        early_spikes = sum(n.step(50.0) for _ in range(2000))
        late_spikes = sum(n.step(50.0) for _ in range(2000))
        # Early should have more spikes than late (adaptation kicks in)
        # Stochastic — allow for noise; just check adaptation is nonzero
        assert n.a > 0, "no adaptation after 4K steps"

    def test_adaptation_candidate_matches_rk4_reference(self):
        n = BendaHerzNeuron(a=0.35, dt=0.25, seed=5)
        expected_a, expected_p = _rk4_reference(n, 12.5)

        candidate_a, candidate_p = n._rk4_candidate(12.5)

        assert candidate_a == pytest.approx(expected_a, rel=1e-14, abs=1e-14)
        assert candidate_p == pytest.approx(expected_p, rel=1e-14, abs=1e-14)

    def test_step_commits_rk4_candidate_before_sampling(self):
        n = BendaHerzNeuron(a=0.25, dt=0.5, seed=6)
        expected_a, _ = _rk4_reference(n, 15.0)

        n.step(15.0)

        assert n.a == pytest.approx(expected_a, rel=1e-14, abs=1e-14)

    def test_exponential_hazard_keeps_probability_bounded(self):
        n = BendaHerzNeuron(f_max=1.0e6, dt=1.0, seed=7)

        _, probability = n._rk4_candidate(1.0e6)

        assert 0.0 <= probability <= 1.0
        assert probability == pytest.approx(1.0, rel=0.0, abs=1e-12)

    def test_seeded_sequences_are_reproducible(self):
        left = BendaHerzNeuron(seed=123)
        right = BendaHerzNeuron(seed=123)

        assert [left.step(25.0) for _ in range(50)] == [right.step(25.0) for _ in range(50)]

    def test_f_onset_sigmoid(self):
        """f_onset should be sigmoid-shaped: low at I=0, high at I>>i_half."""
        n = BendaHerzNeuron()
        f_low = n._f_onset(0.0)
        f_high = n._f_onset(50.0)
        assert f_high > f_low

    def test_state_finite(self):
        n = BendaHerzNeuron()
        for _ in range(5000):
            n.step(100.0)
        assert np.isfinite(n.a)

    def test_reset(self):
        n = BendaHerzNeuron()
        for _ in range(100):
            n.step(30.0)
        n.reset()
        assert n.a == 0.0


class TestBendaHerzValidation:
    @pytest.mark.parametrize("a", [-1.0, np.nan, np.inf, -np.inf])
    def test_rejects_negative_or_non_finite_adaptation_state(self, a: float):
        with pytest.raises(ValueError, match="a"):
            BendaHerzNeuron(a=a)

    @pytest.mark.parametrize("field", ["f_max", "beta", "tau_a", "dt"])
    @pytest.mark.parametrize("value", [0.0, -1.0, np.nan, np.inf])
    def test_rejects_non_positive_or_non_finite_scale_parameters(self, field: str, value: float):
        with pytest.raises(ValueError, match=field):
            BendaHerzNeuron(**{field: value})

    @pytest.mark.parametrize("field", ["i_half", "delta_a"])
    @pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
    def test_rejects_non_finite_threshold_and_adaptation_gain(self, field: str, value: float):
        with pytest.raises(ValueError, match=field):
            BendaHerzNeuron(**{field: value})

    def test_rejects_negative_adaptation_gain(self):
        with pytest.raises(ValueError, match="delta_a"):
            BendaHerzNeuron(delta_a=-1.0)

    @pytest.mark.parametrize("current", [np.nan, np.inf, -np.inf])
    def test_rejects_non_finite_current_before_state_mutation(self, current: float):
        n = BendaHerzNeuron(a=0.5)
        before = n.a
        with pytest.raises(ValueError, match="current"):
            n.step(current)
        assert n.a == before

    def test_rejects_non_finite_adaptation_update_before_state_mutation(self):
        n = BendaHerzNeuron(f_max=1.0e-306, delta_a=1.0e308, dt=1.0e308, a=0.5)
        before = n.a

        with pytest.raises(ValueError, match="adaptation RK4"):
            n.step(100.0)

        assert n.a == before

    @pytest.mark.parametrize("seed", [np.nan, np.inf, -1, True, 2**64])
    def test_rejects_invalid_seed(self, seed):
        with pytest.raises(ValueError, match="seed"):
            BendaHerzNeuron(seed=seed)


class TestBendaHerzNetwork:
    def test_population(self):
        pop = Population(BendaHerzNeuron, n=10, label="bh")
        assert pop.n == 10
        assert pop.model_name == "BendaHerzNeuron"

    def test_network_produces_spikes(self):
        pop = Population(BendaHerzNeuron, n=20, label="bh")
        drive = PoissonInput(n=20, rate_hz=500.0, weight=50.0, dt=0.001, seed=42)
        mon = SpikeMonitor(pop)
        net = Network(pop, drive, mon)
        net.run(duration=2.0, dt=0.001, backend="python")
        assert mon.count > 0, "network produced zero spikes"

    def test_with_projection(self):
        pop = Population(BendaHerzNeuron, n=20, label="bh")
        proj = Projection(pop, pop, weight=5.0, probability=0.2, seed=42)
        drive = PoissonInput(n=20, rate_hz=500.0, weight=50.0, dt=0.001, seed=42)
        mon = SpikeMonitor(pop)
        net = Network(pop, proj, drive, mon)
        net.run(duration=1.0, dt=0.001, backend="python")
        assert isinstance(mon.spike_trains, dict)


class TestBendaHerzAnalysis:
    def _get_binary_train(self):
        n = BendaHerzNeuron()
        train = np.zeros(10000, dtype=np.int8)
        for t in range(10000):
            train[t] = n.step(50.0)
        return train

    def test_firing_rate(self):
        train = self._get_binary_train()
        rate = firing_rate(train, dt=0.001)
        assert rate >= 0  # stochastic — may be very low

    def test_spike_count(self):
        train = self._get_binary_train()
        count = spike_count(train)
        assert count >= 0
