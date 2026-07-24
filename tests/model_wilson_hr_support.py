# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Support for former test_model_wilson_hr.py

from __future__ import annotations

"""Behavioural contract for the Wilson 1999 polynomial cortical neuron.

The module-specific tests validate the coupled Wilson-HR ODE, candidate-first
RK4 integration, finite-state error boundaries, reset semantics, and the named
public workflow contract inside the Python simulator.
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
from sc_neurocore.neurons.models.wilson_hr import WilsonHRNeuron


def _rhs(neuron: WilsonHRNeuron, v: float, r: float, current: float) -> tuple[float, float]:
    poly = -(17.81 + 47.71 * v + 32.63 * v * v) * (v - 0.55)
    syn = -26.0 * r * (v + 0.92)
    return poly + syn + current, (-r + 1.35 * v + 1.03) / neuron.tau_r


def _rk4_reference(neuron: WilsonHRNeuron, current: float) -> tuple[float, float]:
    v0, r0 = neuron.v, neuron.r
    dt = neuron.dt
    k1 = _rhs(neuron, v0, r0, current)
    k2 = _rhs(neuron, v0 + 0.5 * dt * k1[0], r0 + 0.5 * dt * k1[1], current)
    k3 = _rhs(neuron, v0 + 0.5 * dt * k2[0], r0 + 0.5 * dt * k2[1], current)
    k4 = _rhs(neuron, v0 + dt * k3[0], r0 + dt * k3[1], current)
    return (
        v0 + dt * (k1[0] + 2.0 * k2[0] + 2.0 * k3[0] + k4[0]) / 6.0,
        r0 + dt * (k1[1] + 2.0 * k2[1] + 2.0 * k3[1] + k4[1]) / 6.0,
    )


def _run(neuron: WilsonHRNeuron, current: float, steps: int) -> list[int]:
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
    "WilsonHRNeuron",
    "_rhs",
    "_rk4_reference",
    "_run",
]
