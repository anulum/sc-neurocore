# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Support for former test_model_gutkin_ermentrout.py

from __future__ import annotations

"""Full pipeline test for GutkinErmentroutNeuron (Gutkin & Ermentrout 1998).

Minimal 2D conductance model: persistent Na + delayed-rectifier K.
I_Na: g=20, m_inf (instantaneous Boltzmann v_half=-20, k=15)
I_K: g=10, n (tau=1ms, Boltzmann v_half=-25, k=5)
I_L: g=8, ohmic leak

Candidate-first RK4 step (dt=0.05). m_Na instantaneous.
Simple enough for full analytical verification.
FULL PIPELINE WIRED + PERFORMANCE."""
import time
import math
import numpy as np
import pytest
from tests.performance_guard import assert_throughput_guard
from sc_neurocore.neurons.models.gutkin_ermentrout import GutkinErmentroutNeuron
from sc_neurocore.network.population import Population
from sc_neurocore.network.projection import Projection
from sc_neurocore.network.network import Network
from sc_neurocore.network.monitor import SpikeMonitor
from sc_neurocore.network.stimulus import PoissonInput
from sc_neurocore.analysis.spike_stats.basic import spike_count, firing_rate, isi


def _run(neuron: GutkinErmentroutNeuron, current: float, steps: int) -> list[int]:
    return [t for t in range(steps) if neuron.step(current) == 1]


def _m_inf(v: float) -> float:
    return 1.0 / (1.0 + math.exp(-(v + 20.0) / 15.0))


def _n_inf(v: float) -> float:
    return 1.0 / (1.0 + math.exp(-(v + 25.0) / 5.0))


def _rhs(
    neuron: GutkinErmentroutNeuron, v: float, n_gate: float, current: float
) -> tuple[float, float]:
    m_inf = _m_inf(v)
    n_inf = _n_inf(v)
    i_na = neuron.g_na * m_inf * (v - neuron.e_na)
    i_k = neuron.g_k * n_gate * (v - neuron.e_k)
    i_l = neuron.g_l * (v - neuron.e_l)
    return -i_na - i_k - i_l + current, n_inf - n_gate


def _rk4_reference(neuron: GutkinErmentroutNeuron, current: float) -> tuple[float, float]:
    v0, n0 = neuron.v, neuron.n
    k1_v, k1_n = _rhs(neuron, v0, n0, current)
    k2_v, k2_n = _rhs(neuron, v0 + 0.5 * neuron.dt * k1_v, n0 + 0.5 * neuron.dt * k1_n, current)
    k3_v, k3_n = _rhs(neuron, v0 + 0.5 * neuron.dt * k2_v, n0 + 0.5 * neuron.dt * k2_n, current)
    k4_v, k4_n = _rhs(neuron, v0 + neuron.dt * k3_v, n0 + neuron.dt * k3_n, current)
    next_v = v0 + neuron.dt * (k1_v + 2.0 * k2_v + 2.0 * k3_v + k4_v) / 6.0
    next_n = n0 + neuron.dt * (k1_n + 2.0 * k2_n + 2.0 * k3_n + k4_n) / 6.0
    return next_v, next_n


__all__ = [
    "time",
    "math",
    "np",
    "pytest",
    "assert_throughput_guard",
    "GutkinErmentroutNeuron",
    "Population",
    "Projection",
    "Network",
    "SpikeMonitor",
    "PoissonInput",
    "spike_count",
    "firing_rate",
    "isi",
    "_run",
    "_m_inf",
    "_n_inf",
    "_rhs",
    "_rk4_reference",
]
