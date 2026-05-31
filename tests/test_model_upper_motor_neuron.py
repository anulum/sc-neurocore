# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Upper motor neuron model tests

from __future__ import annotations

import math

import pytest

from sc_neurocore.neurons.models.upper_motor_neuron import UpperMotorNeuron


def _rate_exp(value: float) -> float:
    return math.exp(max(-60.0, min(60.0, value)))


def _gate(previous: float, alpha: float, beta: float, dt: float) -> float:
    total = alpha + beta
    steady = alpha / total
    return min(1.0, max(0.0, steady + (previous - steady) * _rate_exp(-total * dt)))


def _gate_inf(previous: float, steady: float, tau: float, dt: float) -> float:
    return min(1.0, max(0.0, steady + (previous - steady) * _rate_exp(-dt / tau)))


def _reference_upper_motor_step(
    cell: UpperMotorNeuron, current: float
) -> tuple[float, float, float, float, float, float]:
    v = cell.v
    m = cell.m
    h = cell.h
    n = cell.n
    p = cell.p
    s = cell.s
    vt = -56.2
    for _ in range(4):
        dv = v - vt
        x_m = dv - 13.0
        alpha_m = 0.32 * 4.0 if abs(x_m) < 1e-6 else -0.32 * x_m / (_rate_exp(-x_m / 4.0) - 1.0)
        x_h = dv - 17.0
        beta_m = 0.28 * 5.0 if abs(x_h) < 1e-6 else 0.28 * x_h / (_rate_exp(x_h / 5.0) - 1.0)
        alpha_h = 0.128 * _rate_exp(-(dv - 17.0) / 18.0)
        beta_h = 4.0 / (1.0 + _rate_exp(-(dv - 40.0) / 5.0))
        x_n = dv - 15.0
        alpha_n = 0.032 * 5.0 if abs(x_n) < 1e-6 else -0.032 * x_n / (_rate_exp(-x_n / 5.0) - 1.0)
        beta_n = 0.5 * _rate_exp(-(dv - 10.0) / 40.0)

        m = _gate(m, alpha_m, beta_m, cell.dt)
        h = _gate(h, alpha_h, beta_h, cell.dt)
        n = _gate(n, alpha_n, beta_n, cell.dt)

        p_inf = 1.0 / (1.0 + _rate_exp(-(v + 35.0) / 10.0))
        tau_p = 400.0 / (3.3 * _rate_exp((v + 35.0) / 20.0) + _rate_exp(-(v + 35.0) / 20.0))
        p = _gate_inf(p, p_inf, tau_p, cell.dt)

        s_inf = 1.0 / (1.0 + _rate_exp(-(v + 20.0) / 5.0))
        s = _gate_inf(s, s_inf, 10.0, cell.dt)

        g_na = cell.g_na * m**3 * h
        g_k = cell.g_k * n**4
        g_m = cell.g_m * p
        g_ca = cell.g_ca * s**2
        g_total = g_na + g_k + g_m + g_ca + cell.g_l
        steady_v = (
            current
            + g_na * cell.e_na
            + g_k * cell.e_k
            + g_m * cell.e_k
            + g_ca * cell.e_ca
            + cell.g_l * cell.e_l
        ) / g_total
        v = steady_v + (v - steady_v) * _rate_exp(-(g_total / cell.c_m) * cell.dt)
    return v, m, h, n, p, s


def test_upper_motor_neuron_uses_exact_gate_and_conductance_membrane_step() -> None:
    cell = UpperMotorNeuron()
    expected = _reference_upper_motor_step(cell, 5.0)

    spike = cell.step(5.0)

    assert spike == 0
    for observed, target in zip(
        (cell.v, cell.m, cell.h, cell.n, cell.p, cell.s), expected, strict=True
    ):
        assert math.isclose(observed, target, rel_tol=0.0, abs_tol=1e-12)


def test_upper_motor_neuron_rejects_invalid_configuration() -> None:
    with pytest.raises(ValueError, match="c_m"):
        UpperMotorNeuron(c_m=0.0)
    with pytest.raises(ValueError, match="g_na"):
        UpperMotorNeuron(g_na=-1.0)
    with pytest.raises(ValueError, match="m"):
        UpperMotorNeuron(m=1.5)


def test_upper_motor_neuron_preserves_state_on_invalid_current() -> None:
    cell = UpperMotorNeuron(v=-64.0, m=0.1, h=0.7, n=0.2, p=0.1, s=0.2)

    with pytest.raises(ValueError, match="current"):
        cell.step(math.nan)

    assert (cell.v, cell.m, cell.h, cell.n, cell.p, cell.s) == (-64.0, 0.1, 0.7, 0.2, 0.1, 0.2)
