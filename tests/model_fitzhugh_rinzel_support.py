# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Support for former test_model_fitzhugh_rinzel.py

from __future__ import annotations

"""Module-specific tests for FitzHughRinzelNeuron.

The model is the FitzHugh-Nagumo fast subsystem plus ultra-slow Rinzel
modulation. Tests validate RK4 integration, three-timescale dynamics,
fail-closed numerical boundaries, pipeline wiring, and measured throughput.
"""
import math
import time
import numpy as np
import pytest
from sc_neurocore.analysis.spike_stats.basic import firing_rate, spike_count
from sc_neurocore.network.monitor import SpikeMonitor
from sc_neurocore.network.network import Network
from sc_neurocore.network.population import Population
from sc_neurocore.network.projection import Projection
from sc_neurocore.network.stimulus import PoissonInput
from sc_neurocore.neurons.models.fitzhugh_rinzel import FitzHughRinzelNeuron
def _run(neuron: FitzHughRinzelNeuron, current: float, steps: int) -> list[int]:
    return [t for t in range(steps) if neuron.step(current) == 1]
def _rhs(
    v: float,
    w: float,
    y: float,
    current: float,
    *,
    a=0.7,
    b=0.8,
    c=-0.775,
    d=1.0,
    delta=0.08,
    mu=0.0001,
):
    return (
        v - v**3 / 3.0 - w + y + current,
        delta * (a + v - b * w),
        mu * (c - v - d * y),
    )
def _rk4_reference(v: float, w: float, y: float, current: float, dt: float):
    k1 = _rhs(v, w, y, current)
    k2 = _rhs(v + 0.5 * dt * k1[0], w + 0.5 * dt * k1[1], y + 0.5 * dt * k1[2], current)
    k3 = _rhs(v + 0.5 * dt * k2[0], w + 0.5 * dt * k2[1], y + 0.5 * dt * k2[2], current)
    k4 = _rhs(v + dt * k3[0], w + dt * k3[1], y + dt * k3[2], current)
    return (
        v + dt * (k1[0] + 2.0 * k2[0] + 2.0 * k3[0] + k4[0]) / 6.0,
        w + dt * (k1[1] + 2.0 * k2[1] + 2.0 * k3[1] + k4[1]) / 6.0,
        y + dt * (k1[2] + 2.0 * k2[2] + 2.0 * k3[2] + k4[2]) / 6.0,
    )

__all__ = ['math', 'time', 'np', 'pytest', 'firing_rate', 'spike_count', 'SpikeMonitor', 'Network', 'Population', 'Projection', 'PoissonInput', 'FitzHughRinzelNeuron', '_run', '_rhs', '_rk4_reference']
