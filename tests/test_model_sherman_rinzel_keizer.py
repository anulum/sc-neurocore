# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Sherman-Rinzel-Keizer neuron behavioural tests

from __future__ import annotations

import math

import pytest

from sc_neurocore.analysis.spike_stats.basic import spike_count
from sc_neurocore.network.population import Population
from sc_neurocore.neurons.models.sherman_rinzel_keizer import ShermanRinzelKeizerNeuron


def _sigmoid(arg: float) -> float:
    arg = max(-80.0, min(80.0, arg))
    return 1.0 / (1.0 + math.exp(-arg))


def _rhs(
    model: ShermanRinzelKeizerNeuron,
    v: float,
    n_gate: float,
    s_gate: float,
    current: float,
) -> tuple[float, float, float]:
    m_inf = _sigmoid((v + 20.0) / 12.0)
    n_inf = _sigmoid((v + 16.0) / 5.0)
    s_inf = _sigmoid((v + 35.0) / 10.0)
    i_ca = model.g_ca * m_inf * (v - model.e_ca)
    i_k = model.g_k * n_gate * (v - model.e_k)
    i_s = model.g_s * s_gate * (v - model.e_k)
    return -i_ca - i_k - i_s + current, (n_inf - n_gate) / 9.09, (s_inf - s_gate) / model.tau_s


def _rk4_reference(model: ShermanRinzelKeizerNeuron, current: float) -> tuple[float, float, float]:
    half_dt = 0.5 * model.dt
    k1 = _rhs(model, model.v, model.n, model.s, current)
    k2 = _rhs(
        model,
        model.v + half_dt * k1[0],
        model.n + half_dt * k1[1],
        model.s + half_dt * k1[2],
        current,
    )
    k3 = _rhs(
        model,
        model.v + half_dt * k2[0],
        model.n + half_dt * k2[1],
        model.s + half_dt * k2[2],
        current,
    )
    k4 = _rhs(
        model,
        model.v + model.dt * k3[0],
        model.n + model.dt * k3[1],
        model.s + model.dt * k3[2],
        current,
    )
    return (
        model.v + model.dt * (k1[0] + 2.0 * k2[0] + 2.0 * k3[0] + k4[0]) / 6.0,
        model.n + model.dt * (k1[1] + 2.0 * k2[1] + 2.0 * k3[1] + k4[1]) / 6.0,
        model.s + model.dt * (k1[2] + 2.0 * k2[2] + 2.0 * k3[2] + k4[2]) / 6.0,
    )


def test_step_matches_candidate_first_rk4_reference() -> None:
    model = ShermanRinzelKeizerNeuron(v=-50.0, n=0.1, s=0.1)
    expected = _rk4_reference(model, 5.0)

    spike = model.step(5.0)

    assert spike == 0
    assert model.v == pytest.approx(expected[0], abs=1e-12)
    assert model.n == pytest.approx(expected[1], abs=1e-12)
    assert model.s == pytest.approx(expected[2], abs=1e-12)


def test_rk4_separates_from_former_euler_increment() -> None:
    model = ShermanRinzelKeizerNeuron(v=-50.0, n=0.1, s=0.1)
    euler_dv, euler_dn, euler_ds = _rhs(model, model.v, model.n, model.s, 5.0)
    euler_candidate = (
        model.v + model.dt * euler_dv,
        model.n + model.dt * euler_dn,
        model.s + model.dt * euler_ds,
    )

    model.step(5.0)

    assert abs(model.v - euler_candidate[0]) > 0.1
    assert abs(model.n - euler_candidate[1]) > 1e-4
    assert abs(model.s - euler_candidate[2]) < 1e-3


def test_gate_variables_remain_bounded_under_sustained_drive() -> None:
    model = ShermanRinzelKeizerNeuron(dt=0.05)
    for _ in range(2000):
        model.step(8.0)
        assert 0.0 <= model.n <= 1.0
        assert 0.0 <= model.s <= 1.0
        assert math.isfinite(model.v)


def test_slow_gate_separates_from_fast_gate() -> None:
    model = ShermanRinzelKeizerNeuron(dt=0.05)
    n0 = model.n
    s0 = model.s
    for _ in range(80):
        model.step(4.0)

    assert abs(model.n - n0) > 100.0 * abs(model.s - s0)


def test_current_balance_signs_match_beta_cell_contract() -> None:
    model = ShermanRinzelKeizerNeuron()
    m_inf = _sigmoid((model.v + 20.0) / 12.0)
    i_ca = model.g_ca * m_inf * (model.v - model.e_ca)
    i_k = model.g_k * model.n * (model.v - model.e_k)
    i_s = model.g_s * model.s * (model.v - model.e_k)

    assert i_ca < 0.0
    assert i_k > 0.0
    assert i_s > 0.0


def test_invalid_physical_parameters_rejected_before_mutation() -> None:
    invalid_cases = (
        ("g_ca", 0.0),
        ("g_k", -1.0),
        ("g_s", -1.0),
        ("tau_s", 0.0),
        ("dt", 0.0),
    )
    for attr, value in invalid_cases:
        model = ShermanRinzelKeizerNeuron()
        setattr(model, attr, value)
        previous = (model.v, model.n, model.s)

        with pytest.raises(ValueError):
            model.step(1.0)

        assert (model.v, model.n, model.s) == previous


def test_non_finite_current_rejected_before_mutation() -> None:
    model = ShermanRinzelKeizerNeuron()
    previous = (model.v, model.n, model.s)

    with pytest.raises(ValueError):
        model.step(math.inf)

    assert (model.v, model.n, model.s) == previous


def test_corrupted_gate_state_rejected_before_mutation() -> None:
    model = ShermanRinzelKeizerNeuron(n=1.2)
    previous = (model.v, model.n, model.s)

    with pytest.raises(ValueError):
        model.step(1.0)

    assert (model.v, model.n, model.s) == previous


def test_invalid_candidate_rejected_before_mutation() -> None:
    model = ShermanRinzelKeizerNeuron(dt=50.0)
    previous = (model.v, model.n, model.s)

    with pytest.raises(ValueError):
        model.step(1000.0)

    assert (model.v, model.n, model.s) == previous


def test_intermediate_stage_escape_rejected_before_mutation() -> None:
    model = ShermanRinzelKeizerNeuron(dt=1.0e308)
    previous = (model.v, model.n, model.s)

    with pytest.raises(ValueError):
        model.step(1.0)

    assert (model.v, model.n, model.s) == previous


def test_nonfinite_derivative_rejected_before_mutation() -> None:
    model = ShermanRinzelKeizerNeuron(g_ca=1.0e308)
    previous = (model.v, model.n, model.s)

    with pytest.raises(ValueError):
        model.step(1.0)

    assert (model.v, model.n, model.s) == previous


def test_reset_restores_dynamic_state_only() -> None:
    model = ShermanRinzelKeizerNeuron(g_ca=4.2, dt=0.05)
    for _ in range(10):
        model.step(2.0)

    model.reset()

    assert (model.v, model.n, model.s) == (-50.0, 0.1, 0.1)
    assert model.g_ca == 4.2
    assert model.dt == 0.05


def test_population_network_and_analysis_wiring() -> None:
    population = Population(ShermanRinzelKeizerNeuron, n=3)

    spikes = [neuron.step(5.0) for neuron in population.neurons]

    assert len(population.neurons) == 3
    assert spike_count(spikes) == sum(spikes)
