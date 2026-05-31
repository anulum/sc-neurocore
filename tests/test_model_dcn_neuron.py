# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - DCNNeuron behavioural tests

from __future__ import annotations

import math

import pytest

from sc_neurocore.neurons.models.dcn_neuron import DCNNeuron


def _safe_rate(a: float, vhalf: float, v: float, k: float, fallback: float) -> float:
    d = v + vhalf
    if abs(d) < 1e-7:
        return fallback
    return a * d / (1.0 - math.exp(-d / k))


def _exact_hh_gate(value: float, alpha: float, beta: float, phi: float, dt: float) -> float:
    rate = phi * (alpha + beta)
    target = alpha / (alpha + beta)
    return target + (value - target) * math.exp(-rate * dt)


def _exact_relax(value: float, target: float, tau: float, dt: float) -> float:
    return target + (value - target) * math.exp(-dt / tau)


def _snapshot(neuron: DCNNeuron) -> tuple[float, ...]:
    return (neuron.v, neuron.h, neuron.n, neuron.p, neuron.s, neuron.r, neuron.ca)


def test_default_step_preserves_physical_state_bounds() -> None:
    neuron = DCNNeuron()

    spike = neuron.step(0.0)

    assert spike in (0, 1)
    assert -100.0 <= neuron.v <= 60.0
    assert 0.0 <= neuron.h <= 1.0
    assert 0.0 <= neuron.n <= 1.0
    assert 0.0 <= neuron.p <= 1.0
    assert 0.0 <= neuron.s <= 1.0
    assert 0.0 <= neuron.r <= 1.0
    assert neuron.ca >= 0.0


def test_seven_current_conductance_surface_is_present() -> None:
    neuron = DCNNeuron()

    assert neuron.g_na > 0.0
    assert neuron.g_nap > 0.0
    assert neuron.g_k > 0.0
    assert neuron.g_t > 0.0
    assert neuron.g_ahp > 0.0
    assert neuron.g_h > 0.0
    assert neuron.g_l > 0.0


def test_t_type_deinactivation_increases_during_hyperpolarisation() -> None:
    neuron = DCNNeuron()
    before = neuron.s

    for _ in range(500):
        neuron.step(-5.0)

    assert neuron.s >= before


def test_ih_depolarises_from_hyperpolarised_state() -> None:
    with_ih = DCNNeuron(v=-80.0)
    without_ih = DCNNeuron(v=-80.0, g_h=0.0)

    for _ in range(400):
        with_ih.step(0.0)
        without_ih.step(0.0)

    assert with_ih.v > without_ih.v


def test_gate_and_calcium_kinetics_use_closed_form_relaxation() -> None:
    neuron = DCNNeuron(g_na=0.0, g_nap=0.0, g_k=0.0, g_t=0.0, g_ahp=0.0, g_h=0.0, g_l=0.0, gain=0.0)
    before = _snapshot(neuron)
    v0 = neuron.v

    alpha_h = 0.07 * math.exp(-(v0 + 58.0) / 20.0)
    beta_h = 1.0 / (1.0 + math.exp(-(v0 + 28.0) / 10.0))
    alpha_n = _safe_rate(0.01, 34.0, v0, 10.0, 0.1)
    beta_n = 0.125 * math.exp(-(v0 + 44.0) / 80.0)
    p_inf = 1.0 / (1.0 + math.exp(-(v0 + 48.0) / 5.0))
    tau_p = 5.0 + 15.0 / max(0.01, 1.0 + ((v0 + 48.0) / 10.0) ** 2)
    s_inf = 1.0 / (1.0 + math.exp((v0 + 60.0) / 6.5))
    tau_s = 20.0 + 50.0 / (1.0 + math.exp((v0 + 65.0) / 10.0))
    r_inf = 1.0 / (1.0 + math.exp((v0 + 80.0) / 10.0))
    tau_r = 100.0 + 200.0 / (1.0 + math.exp((v0 + 70.0) / 10.0))

    expected = (
        v0,
        _exact_hh_gate(before[1], alpha_h, beta_h, neuron.phi, neuron.dt),
        _exact_hh_gate(before[2], alpha_n, beta_n, neuron.phi, neuron.dt),
        _exact_relax(before[3], p_inf, tau_p, neuron.dt),
        _exact_relax(before[4], s_inf, tau_s, neuron.dt),
        _exact_relax(before[5], r_inf, tau_r, neuron.dt),
        _exact_relax(before[6], 0.0, neuron.tau_ca, neuron.dt),
    )

    neuron.step(0.0)

    for observed, expected_value in zip(_snapshot(neuron), expected, strict=True):
        assert observed == pytest.approx(expected_value, rel=1e-12, abs=1e-12)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"v": 60.1},
        {"h": -0.1},
        {"n": 1.1},
        {"ca": -1e-6},
        {"g_na": -1.0},
        {"c_m": 0.0},
        {"phi": 0.0},
        {"tau_ca": 0.0},
        {"kd_ahp": 0.0},
        {"dt": 0.0},
        {"gain": -1.0},
        {"_sub_steps": 0},
        {"v": math.inf},
    ],
)
def test_invalid_physical_configuration_is_rejected(kwargs: dict[str, float]) -> None:
    with pytest.raises(ValueError):
        DCNNeuron(**kwargs)


def test_non_finite_current_does_not_mutate_state() -> None:
    neuron = DCNNeuron()
    before = _snapshot(neuron)

    with pytest.raises(ValueError):
        neuron.step(math.nan)

    assert _snapshot(neuron) == before


def test_unstable_candidate_does_not_mutate_state() -> None:
    neuron = DCNNeuron()
    neuron.g_na = math.inf
    before = _snapshot(neuron)

    with pytest.raises(ValueError):
        neuron.step(5.0)

    assert _snapshot(neuron) == before
