# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Support for former test_model_pernarowski.py

from __future__ import annotations

"""Module-specific behavioural tests for PernarowskiNeuron.

The tests verify the three-state ODE, RK4 integration, slow-variable
modulation, finite-domain rejection, and the module's public network and
analysis integration contracts without bucket-style coverage assertions."""
import math
import numpy as np
import pytest
from sc_neurocore.neurons.models.pernarowski import PernarowskiNeuron
from sc_neurocore.network.population import Population
from sc_neurocore.network.network import Network
from sc_neurocore.network.monitor import SpikeMonitor
from sc_neurocore.network.stimulus import PoissonInput
from sc_neurocore.analysis.spike_stats.basic import spike_count


def _run_and_collect(
    neuron: PernarowskiNeuron, current: float, steps: int
) -> tuple[list[int], list[float]]:
    """Return (spike_times, voltage_trace)."""
    spike_times: list[int] = []
    voltages: list[float] = []
    for t in range(steps):
        s = neuron.step(current)
        if s == 1:
            spike_times.append(t)
        voltages.append(neuron.v)
    return spike_times, voltages


def _rhs(
    neuron: PernarowskiNeuron, v: float, w: float, z: float, current: float
) -> tuple[float, float, float]:
    return (
        v - v * v * v / 3.0 - w - z + current,
        neuron.eps1 * (v - neuron.gamma * w + neuron.alpha),
        neuron.eps2 * (neuron.beta * (v + 0.7) - z),
    )


def _rk4_reference(
    neuron: PernarowskiNeuron, v: float, w: float, z: float, current: float
) -> tuple[float, float, float]:
    dt = neuron.dt
    k1 = _rhs(neuron, v, w, z, current)
    k2 = _rhs(
        neuron,
        v + 0.5 * dt * k1[0],
        w + 0.5 * dt * k1[1],
        z + 0.5 * dt * k1[2],
        current,
    )
    k3 = _rhs(
        neuron,
        v + 0.5 * dt * k2[0],
        w + 0.5 * dt * k2[1],
        z + 0.5 * dt * k2[2],
        current,
    )
    k4 = _rhs(neuron, v + dt * k3[0], w + dt * k3[1], z + dt * k3[2], current)
    return (
        v + dt * (k1[0] + 2.0 * k2[0] + 2.0 * k3[0] + k4[0]) / 6.0,
        w + dt * (k1[1] + 2.0 * k2[1] + 2.0 * k3[1] + k4[1]) / 6.0,
        z + dt * (k1[2] + 2.0 * k2[2] + 2.0 * k3[2] + k4[2]) / 6.0,
    )


__all__ = [
    "math",
    "np",
    "pytest",
    "PernarowskiNeuron",
    "Population",
    "Network",
    "SpikeMonitor",
    "PoissonInput",
    "spike_count",
    "_run_and_collect",
    "_rhs",
    "_rk4_reference",
]
