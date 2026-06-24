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
        x_bm = dv - 40.0
        beta_m = 0.28 * 5.0 if abs(x_bm) < 1e-6 else 0.28 * x_bm / (_rate_exp(x_bm / 5.0) - 1.0)
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


def _count_spikes(current: float, steps: int) -> int:
    cell = UpperMotorNeuron()
    return sum(cell.step(current) for _ in range(steps))


class TestUpperMotorBetaMRegression:
    """Guards the corrected β_m offset against the depolarisation-block bug.

    With the earlier ``V - V_T - 17`` numerator the cell fired exactly three spikes
    then settled at a fixed point near threshold for any stimulus; the published
    ``-40`` offset restores a monotone frequency-current relation.
    """

    def test_quiescent_without_drive(self) -> None:
        assert _count_spikes(0.0, 20000) == 0

    def test_firing_rate_increases_with_current(self) -> None:
        low = _count_spikes(2.0, 30000)
        mid = _count_spikes(5.0, 30000)
        high = _count_spikes(10.0, 30000)
        assert low < mid < high

    def test_no_depolarisation_block_at_strong_drive(self) -> None:
        assert _count_spikes(5.0, 40000) > 100

    def test_membrane_recovers_below_threshold_after_drive(self) -> None:
        cell = UpperMotorNeuron()
        for _ in range(40000):
            cell.step(5.0)
        assert cell.v < cell.v_threshold

    def test_cross_backend_reference_spike_count(self) -> None:
        # Pins the Python reference the Rust/Julia/Go kernels reproduce exactly.
        assert _count_spikes(5.0, 40000) == 377
