# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Independent Fardet-Levina eLIF equation checks

"""Bind EnergyLIFNeuron to the authors' Brian-reference recurrence."""

from __future__ import annotations

import math

import pytest

from sc_neurocore.neurons.models.energy_lif import EnergyLIFNeuron


def _rhs(n: EnergyLIFNeuron, v: float, epsilon: float, current: float) -> tuple[float, float]:
    leak = n.e_0 + (n.e_u - n.e_0) * (1.0 - epsilon / n.epsilon_0)
    dv = (n.g_leak * (leak - v) + current) / n.capacitance
    de = ((1.0 - epsilon / (n.alpha * n.epsilon_0)) ** 3 - (v - n.e_f) / (n.e_d - n.e_f)) / n.tau_e
    return dv, de


def _oracle(n: EnergyLIFNeuron, current: float) -> tuple[float, float, int]:
    dt = n.dt
    k1 = _rhs(n, n.v, n.epsilon, current)
    k2 = _rhs(n, n.v + dt * k1[0] / 2.0, n.epsilon + dt * k1[1] / 2.0, current)
    k3 = _rhs(n, n.v + dt * k2[0] / 2.0, n.epsilon + dt * k2[1] / 2.0, current)
    k4 = _rhs(n, n.v + dt * k3[0], n.epsilon + dt * k3[1], current)
    v = n.v + dt * (k1[0] + 2.0 * k2[0] + 2.0 * k3[0] + k4[0]) / 6.0
    epsilon = n.epsilon + dt * (k1[1] + 2.0 * k2[1] + 2.0 * k3[1] + k4[1]) / 6.0
    event = int(v > n.v_threshold and epsilon > n.epsilon_c)
    return (n.v_reset if event else v, epsilon - n.delta if event else epsilon, event)


def test_author_brian_defaults_are_pinned() -> None:
    n = EnergyLIFNeuron()
    assert (n.v, n.epsilon, n.dt) == (-61.0, 0.32, 0.1)
    assert (n.capacitance, n.g_leak, n.tau_e) == (100.0, 9.0, 200.0)
    assert (n.alpha, n.epsilon_0, n.epsilon_c, n.delta) == (1.0, 0.5, 0.18, 0.01)


@pytest.mark.parametrize("current", [-60.0, 0.0, 30.0, 80.0, 120.0])
def test_one_step_matches_independent_coupled_rk4(current: float) -> None:
    n = EnergyLIFNeuron()
    expected_v, expected_epsilon, expected_event = _oracle(n, current)
    assert n.step(current) == expected_event
    assert n.v == pytest.approx(expected_v, abs=1.0e-15)
    assert n.epsilon == pytest.approx(expected_epsilon, abs=1.0e-15)


def test_energy_changes_the_leak_and_spike_gate() -> None:
    high = EnergyLIFNeuron(v=-58.8, epsilon=0.32)
    low = EnergyLIFNeuron(v=-58.8, epsilon=0.17)
    assert high.step(0.0) == 1
    assert low.step(0.0) == 0
    assert high.v == high.v_reset
    assert low.v > low.v_threshold


def test_failure_is_atomic() -> None:
    n = EnergyLIFNeuron()
    before = (n.v, n.epsilon)
    with pytest.raises(ValueError, match="current"):
        n.step(math.nan)
    assert (n.v, n.epsilon) == before
