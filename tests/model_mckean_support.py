# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Support for former test_model_mckean.py

from __future__ import annotations

"""Compatibility contract for the retained SC triangular recurrence.

The test surface is module-specific by default. Cross-module checks exercise the
real public workflow contract for using the SC recurrence inside Population,
Projection, Network, SpikeMonitor, and spike-stat analysis APIs; they are not
coverage bucket tests.

Model equations:
dv/dt = f(v) - w + I
dw/dt = epsilon * (v - gamma*w)

f(v) = -v             if v < a/2
     = v - a          if a/2 <= v < (1+a)/2
     = 1 - v          if v >= (1+a)/2

The production integrator is candidate-first RK4 over the coupled (v, w) state.
"""
import time
import numpy as np
import pytest
from sc_neurocore.analysis.spike_stats.basic import firing_rate, isi, spike_count
from sc_neurocore.network.monitor import SpikeMonitor
from sc_neurocore.network.network import Network
from sc_neurocore.network.population import Population
from sc_neurocore.network.projection import Projection
from sc_neurocore.network.stimulus import PoissonInput
from sc_neurocore.neurons.models.sc_triangular_mckean import (
    SCTriangularMcKeanNeuron as McKeanNeuron,
)


def _rhs(neuron: McKeanNeuron, v: float, w: float, current: float) -> tuple[float, float]:
    return neuron._f(v) - w + current, neuron.epsilon * (v - neuron.gamma * w)


def _rk4_reference(neuron: McKeanNeuron, current: float) -> tuple[float, float]:
    v0, w0 = neuron.v, neuron.w
    dt = neuron.dt
    k1 = _rhs(neuron, v0, w0, current)
    k2 = _rhs(neuron, v0 + 0.5 * dt * k1[0], w0 + 0.5 * dt * k1[1], current)
    k3 = _rhs(neuron, v0 + 0.5 * dt * k2[0], w0 + 0.5 * dt * k2[1], current)
    k4 = _rhs(neuron, v0 + dt * k3[0], w0 + dt * k3[1], current)
    return (
        v0 + dt * (k1[0] + 2.0 * k2[0] + 2.0 * k3[0] + k4[0]) / 6.0,
        w0 + dt * (k1[1] + 2.0 * k2[1] + 2.0 * k3[1] + k4[1]) / 6.0,
    )


def _run(neuron: McKeanNeuron, current: float, steps: int) -> list[int]:
    return [t for t in range(steps) if neuron.step(current) == 1]


__all__ = [
    "time",
    "np",
    "pytest",
    "firing_rate",
    "isi",
    "spike_count",
    "SpikeMonitor",
    "Network",
    "Population",
    "Projection",
    "PoissonInput",
    "McKeanNeuron",
    "_rhs",
    "_rk4_reference",
    "_run",
]
