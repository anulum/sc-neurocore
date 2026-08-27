# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Support for former test_model_connor_stevens.py

from __future__ import annotations

"""Module-specific tests for the Connor-Walter-McKown 1977 parameterization.

HH-type model with A-type potassium current (I_A), Type-I excitability.
6 state variables: v, m (Na act), h (Na inact), n (K), a (A-type act),
b (A-type inact). 4 ionic currents: I_Na(g=120, m³h), I_K(g=20, n⁴),
I_A(g=47.7, a³b), I_L(g=0.3).

100 sub-steps per step() call (dt=0.01, 1/dt=100). Type-I: continuous
f-I curve from zero frequency (saddle-node on invariant circle bifurcation).
A-current delays spike onset → long latency at rheobase.
~536 steps/s (100 sub-steps × HH complexity)."""
import time
import numpy as np
import pytest
from sc_neurocore.neurons.models.connor_stevens import ConnorStevensNeuron
from sc_neurocore.network.population import Population
from sc_neurocore.network.projection import Projection
from sc_neurocore.network.network import Network
from sc_neurocore.network.monitor import SpikeMonitor
from sc_neurocore.network.stimulus import PoissonInput
from sc_neurocore.analysis.spike_stats.basic import spike_count, firing_rate, isi


def _run(neuron: ConnorStevensNeuron, current: float, steps: int) -> list[int]:
    return [t for t in range(steps) if neuron.step(current) == 1]


def _connor_reference_rate(
    scale: float, shift: float, v: float, denom: float, limit: float
) -> float:
    delta = v + shift
    x = delta / denom
    if abs(x) < 1e-9:
        return scale * denom
    return scale * delta / (1.0 - np.exp(-x))


def _connor_reference_derivatives(
    state: tuple[float, float, float, float, float, float], current: float, params: dict[str, float]
) -> tuple[float, float, float, float, float, float]:
    v, m, h, n, a, b = state
    alpha_m = _connor_reference_rate(0.38, 29.7, v, 10.0, 3.8)
    beta_m = 15.2 * np.exp(-(v + 54.7) / 18.0)
    alpha_h = 0.266 * np.exp(-(v + 48.0) / 20.0)
    beta_h = 3.8 / (1.0 + np.exp(-(v + 18.0) / 10.0))
    alpha_n = _connor_reference_rate(0.02, 45.7, v, 10.0, 0.2)
    beta_n = 0.25 * np.exp(-(v + 55.7) / 80.0)
    a_inf = (0.0761 * np.exp((v + 94.22) / 31.84) / (1.0 + np.exp((v + 1.17) / 28.93))) ** (
        1.0 / 3.0
    )
    tau_a = 0.3632 + 1.158 / (1.0 + np.exp((v + 55.96) / 20.12))
    b_inf = (1.0 / (1.0 + np.exp((v + 53.3) / 14.54))) ** 4
    tau_b = 1.24 + 2.678 / (1.0 + np.exp((v + 50.0) / 16.027))

    i_na = params["g_na"] * m**3 * h * (v - params["e_na"])
    i_k = params["g_k"] * n**4 * (v - params["e_k"])
    i_a = params["g_a"] * a**3 * b * (v - params["e_a"])
    i_l = params["g_l"] * (v - params["e_l"])
    dv = (-i_na - i_k - i_a - i_l + current) / params["c_m"]
    return (
        dv,
        alpha_m * (1.0 - m) - beta_m * m,
        alpha_h * (1.0 - h) - beta_h * h,
        alpha_n * (1.0 - n) - beta_n * n,
        (a_inf - a) / tau_a,
        (b_inf - b) / tau_b,
    )


def _connor_reference_rk4(
    neuron: ConnorStevensNeuron, current: float
) -> tuple[float, float, float, float, float, float]:
    params = {
        "g_na": neuron.g_na,
        "g_k": neuron.g_k,
        "g_a": neuron.g_a,
        "g_l": neuron.g_l,
        "e_na": neuron.e_na,
        "e_k": neuron.e_k,
        "e_a": neuron.e_a,
        "e_l": neuron.e_l,
        "c_m": neuron.c_m,
    }
    state = (neuron.v, neuron.m, neuron.h, neuron.n, neuron.a, neuron.b)
    dt = neuron.dt
    for _ in range(int(1.0 / max(dt, 0.001))):
        k1 = _connor_reference_derivatives(state, current, params)
        k2 = _connor_reference_derivatives(
            tuple(s + 0.5 * dt * k for s, k in zip(state, k1)), current, params
        )
        k3 = _connor_reference_derivatives(
            tuple(s + 0.5 * dt * k for s, k in zip(state, k2)), current, params
        )
        k4 = _connor_reference_derivatives(
            tuple(s + dt * k for s, k in zip(state, k3)), current, params
        )
        state = tuple(
            s + dt * (a + 2.0 * b + 2.0 * c + d) / 6.0
            for s, a, b, c, d in zip(state, k1, k2, k3, k4)
        )
    return state


__all__ = [
    "time",
    "np",
    "pytest",
    "ConnorStevensNeuron",
    "Population",
    "Projection",
    "Network",
    "SpikeMonitor",
    "PoissonInput",
    "spike_count",
    "firing_rate",
    "isi",
    "_run",
    "_connor_reference_rate",
    "_connor_reference_derivatives",
    "_connor_reference_rk4",
]
