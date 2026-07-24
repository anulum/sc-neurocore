# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Support for former test_model_yamada.py

from __future__ import annotations

"""Full pipeline test for YamadaNeuron (Yamada, Kashimori & Kambara 1989).

Subcritical Hopf burster: 3 ODEs (V, n, q). q is ultra-slow (tau_q=300ms)
controlling burst envelope. Square-wave bursting via slow modulation."""
import numpy as np
import pytest
from sc_neurocore.neurons.models.yamada import YamadaNeuron
from sc_neurocore.network.population import Population
from sc_neurocore.network.projection import Projection
from sc_neurocore.network.network import Network
from sc_neurocore.network.monitor import SpikeMonitor
from sc_neurocore.network.stimulus import PoissonInput
from sc_neurocore.analysis.spike_stats.basic import spike_count, firing_rate


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + np.exp(-x))


def _tau_n(v: float) -> float:
    x = (v + 40.0) / 12.0
    if x > 709.0:
        return 1.0
    return 1.0 + 7.5 / (1.0 + np.exp(x))


def _rhs(
    neuron: YamadaNeuron, v: float, n_gate: float, q_gate: float, current: float
) -> tuple[float, float, float]:
    m_inf = _sigmoid((v + 30.0) / 9.5)
    n_inf = _sigmoid((v + 30.0) / 10.0)
    q_inf = _sigmoid((v + 50.0) / 10.0)
    i_na = neuron.g_na * m_inf**3 * (1.0 - n_gate) * (v - neuron.e_na)
    i_k = neuron.g_k * n_gate**4 * (v - neuron.e_k)
    i_q = neuron.g_q * q_gate * (v - neuron.e_q)
    i_l = neuron.g_l * (v - neuron.e_l)
    return (
        -i_na - i_k - i_q - i_l + current,
        (n_inf - n_gate) / _tau_n(v),
        (q_inf - q_gate) / neuron.tau_q,
    )


def _rk4_reference(neuron: YamadaNeuron, current: float) -> tuple[float, float, float]:
    v0, n0, q0 = neuron.v, neuron.n, neuron.q
    dt = neuron.dt
    k1 = _rhs(neuron, v0, n0, q0, current)
    k2 = _rhs(neuron, v0 + 0.5 * dt * k1[0], n0 + 0.5 * dt * k1[1], q0 + 0.5 * dt * k1[2], current)
    k3 = _rhs(neuron, v0 + 0.5 * dt * k2[0], n0 + 0.5 * dt * k2[1], q0 + 0.5 * dt * k2[2], current)
    k4 = _rhs(neuron, v0 + dt * k3[0], n0 + dt * k3[1], q0 + dt * k3[2], current)
    return (
        v0 + dt * (k1[0] + 2.0 * k2[0] + 2.0 * k3[0] + k4[0]) / 6.0,
        n0 + dt * (k1[1] + 2.0 * k2[1] + 2.0 * k3[1] + k4[1]) / 6.0,
        q0 + dt * (k1[2] + 2.0 * k2[2] + 2.0 * k3[2] + k4[2]) / 6.0,
    )


def _run(neuron: YamadaNeuron, current: float, steps: int) -> list[int]:
    return [t for t in range(steps) if neuron.step(current) == 1]


__all__ = [
    "np",
    "pytest",
    "YamadaNeuron",
    "Population",
    "Projection",
    "Network",
    "SpikeMonitor",
    "PoissonInput",
    "spike_count",
    "firing_rate",
    "_sigmoid",
    "_tau_n",
    "_rhs",
    "_rk4_reference",
    "_run",
]
