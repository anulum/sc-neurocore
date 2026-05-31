# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Module-specific test: ButeraRespiratoryNeuron

"""Module-specific behavioural tests for ButeraRespiratoryNeuron (Butera, Rinzel & Smith 1999).

Pre-Bötzinger respiratory neuron with persistent Na⁺ current and
slow h_nap inactivation. Bursting at high current."""

from __future__ import annotations

import numpy as np
import pytest

from sc_neurocore.neurons.models.butera_respiratory import ButeraRespiratoryNeuron
from sc_neurocore.network.population import Population
from sc_neurocore.network.projection import Projection
from sc_neurocore.network.network import Network
from sc_neurocore.network.monitor import SpikeMonitor
from sc_neurocore.network.stimulus import PoissonInput
from sc_neurocore.analysis.spike_stats.basic import firing_rate, spike_count, isi


class TestButeraIsolation:
    def test_construction(self):
        n = ButeraRespiratoryNeuron()
        assert n.v == -50.0
        assert n.h_nap == 0.5

    def test_step_returns_binary(self):
        n = ButeraRespiratoryNeuron()
        assert n.step(0.0) in (0, 1)

    def test_subthreshold(self):
        n = ButeraRespiratoryNeuron()
        spikes = sum(n.step(10.0) for _ in range(10_000))
        assert spikes == 0

    def test_spikes_at_high_current(self):
        n = ButeraRespiratoryNeuron()
        spikes = sum(n.step(100.0) for _ in range(100_000))
        assert spikes > 100, f"too few spikes at I=100: {spikes}"

    def test_persistent_na_inactivation(self):
        """h_nap should change from initial value under sustained drive."""
        n = ButeraRespiratoryNeuron()
        h_init = n.h_nap
        for _ in range(100_000):
            n.step(100.0)
        assert n.h_nap != h_init

    def test_numerical_stability(self):
        for I in [0, 10, 50, 100]:
            n = ButeraRespiratoryNeuron()
            for _ in range(50_000):
                n.step(float(I))
            assert np.isfinite(n.v), f"v NaN at I={I}"
            assert np.isfinite(n.n), f"n NaN at I={I}"
            assert np.isfinite(n.h_nap), f"h_nap NaN at I={I}"

    def test_gating_bounded(self):
        n = ButeraRespiratoryNeuron()
        for _ in range(50_000):
            n.step(100.0)
        assert 0 <= n.n <= 1
        assert 0 <= n.h_nap <= 1

    def test_reset(self):
        n = ButeraRespiratoryNeuron()
        for _ in range(1000):
            n.step(100.0)
        n.reset()
        assert n.v == -50.0
        assert n.n == 0.01
        assert n.h_nap == 0.5


class TestButeraNetwork:
    def test_population(self):
        pop = Population(ButeraRespiratoryNeuron, n=5, label="butera")
        assert pop.n == 5

    def test_network_spikes(self):
        pop = Population(ButeraRespiratoryNeuron, n=10, label="butera")
        drive = PoissonInput(n=10, rate_hz=500.0, weight=100.0, dt=0.001, seed=42)
        mon = SpikeMonitor(pop)
        net = Network(pop, drive, mon)
        net.run(duration=1.0, dt=0.001, backend="python")
        assert mon.count > 0

    def test_with_projection(self):
        pop = Population(ButeraRespiratoryNeuron, n=10, label="butera")
        proj = Projection(pop, pop, weight=5.0, probability=0.3, seed=42)
        drive = PoissonInput(n=10, rate_hz=500.0, weight=100.0, dt=0.001, seed=42)
        mon = SpikeMonitor(pop)
        net = Network(pop, proj, drive, mon)
        net.run(duration=0.5, dt=0.001, backend="python")
        assert isinstance(mon.spike_trains, dict)


class TestButeraAnalysis:
    def _get_train(self):
        n = ButeraRespiratoryNeuron()
        train = np.zeros(100_000, dtype=np.int8)
        for t in range(100_000):
            train[t] = n.step(100.0)
        return train

    def test_firing_rate(self):
        train = self._get_train()
        rate = firing_rate(train, dt=0.0001)
        assert rate > 0

    def test_spike_count(self):
        train = self._get_train()
        assert spike_count(train) > 100

    def test_isi(self):
        train = self._get_train()
        intervals = isi(train, dt=0.0001)
        if intervals.size > 0:
            assert np.all(np.isfinite(intervals))


def _butera_rates(v: float) -> tuple[float, float, float, float, float, float]:
    m_na_inf = 1.0 / (1.0 + np.exp(-(v + 34.0) / 5.0))
    m_nap_inf = 1.0 / (1.0 + np.exp(-(v + 40.0) / 6.0))
    h_nap_inf = 1.0 / (1.0 + np.exp((v + 48.0) / 6.0))
    n_inf = 1.0 / (1.0 + np.exp(-(v + 29.0) / 4.0))
    tau_n = max(10.0 / max(np.cosh((v + 29.0) / 8.0), 1e-12), 0.01)
    tau_h = max(10000.0 / max(np.cosh((v + 48.0) / 12.0), 1e-12), 0.1)
    return m_na_inf, m_nap_inf, h_nap_inf, n_inf, tau_n, tau_h


def _butera_derivatives(state: tuple[float, float, float], current: float, params: dict[str, float]) -> tuple[float, float, float]:
    v, n, h_nap = state
    m_na_inf, m_nap_inf, h_nap_inf, n_inf, tau_n, tau_h = _butera_rates(v)
    i_na = params["g_na"] * m_na_inf**3 * (1.0 - n) * (v - params["e_na"])
    i_nap = params["g_nap"] * m_nap_inf * h_nap * (v - params["e_na"])
    i_k = params["g_k"] * n**4 * (v - params["e_k"])
    i_l = params["g_l"] * (v - params["e_l"])
    return (
        -i_na - i_nap - i_k - i_l + current,
        (n_inf - n) / tau_n,
        (h_nap_inf - h_nap) / tau_h,
    )


def _butera_reference_rk4(neuron: ButeraRespiratoryNeuron, current: float) -> tuple[float, float, float]:
    state = (neuron.v, neuron.n, neuron.h_nap)
    params = {
        "g_na": neuron.g_na,
        "g_nap": neuron.g_nap,
        "g_k": neuron.g_k,
        "g_l": neuron.g_l,
        "e_na": neuron.e_na,
        "e_k": neuron.e_k,
        "e_l": neuron.e_l,
    }
    dt = neuron.dt
    k1 = _butera_derivatives(state, current, params)
    k2 = _butera_derivatives(tuple(s + 0.5 * dt * k for s, k in zip(state, k1)), current, params)
    k3 = _butera_derivatives(tuple(s + 0.5 * dt * k for s, k in zip(state, k2)), current, params)
    k4 = _butera_derivatives(tuple(s + dt * k for s, k in zip(state, k3)), current, params)
    return tuple(s + dt * (a + 2.0 * b + 2.0 * c + d) / 6.0 for s, a, b, c, d in zip(state, k1, k2, k3, k4))


def test_butera_matches_independent_rk4_contract() -> None:
    """Butera respiratory step follows the module RK4 integration contract."""
    neuron = ButeraRespiratoryNeuron(v=-48.0, n=0.08, h_nap=0.62, dt=0.025)
    expected = _butera_reference_rk4(neuron, current=18.0)

    spike = neuron.step(18.0)

    assert spike in (0, 1)
    assert (neuron.v, neuron.n, neuron.h_nap) == pytest.approx(expected, rel=1e-10, abs=1e-10)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("v", float("nan")),
        ("n", -0.01),
        ("h_nap", 1.01),
        ("g_na", -1.0),
        ("g_nap", -1.0),
        ("g_k", -1.0),
        ("g_l", -1.0),
        ("tau_h", 0.0),
        ("dt", 0.0),
    ],
)
def test_butera_rejects_invalid_physical_parameters(field: str, value: float) -> None:
    """Invalid Butera state or physical parameters are rejected at construction."""
    with pytest.raises((TypeError, ValueError)):
        ButeraRespiratoryNeuron(**{field: value})


def test_butera_rejects_non_finite_current_without_mutation() -> None:
    """Invalid respiratory drive preserves voltage and gate state."""
    neuron = ButeraRespiratoryNeuron(v=-49.0, n=0.04, h_nap=0.55)
    before = (neuron.v, neuron.n, neuron.h_nap)

    with pytest.raises((TypeError, ValueError, FloatingPointError)):
        neuron.step(float("nan"))

    assert (neuron.v, neuron.n, neuron.h_nap) == before


def test_butera_rejects_corrupted_runtime_state_without_mutation() -> None:
    """Runtime gate corruption cannot produce a partially committed candidate."""
    neuron = ButeraRespiratoryNeuron()
    neuron.n = float("inf")
    before = (neuron.v, neuron.n, neuron.h_nap)

    with pytest.raises((TypeError, ValueError, FloatingPointError)):
        neuron.step(20.0)

    assert neuron.v == before[0]
    assert np.isinf(neuron.n)
    assert neuron.h_nap == before[2]
