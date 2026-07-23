# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Support for former test_model_fitzhugh_nagumo.py

from __future__ import annotations

"""Module-specific test for FitzHughNagumoNeuron (FitzHugh 1961).

2D qualitative model: dv/dt = v - v³/3 - w + I, dw/dt = ε(v+a-bw).
Oscillatory band I∈[0.5, 1.0]. Hopf bifurcation on both sides.
Nullcline analysis: V-nullcline w = v-v³/3+I, w-nullcline w = (v+a)/b.
The production default is RK4 over the published two-state ODE."""
import os
import math
import time
import numpy as np
import pytest
from tests.performance_guard import assert_throughput_guard
from sc_neurocore.neurons.models.fitzhugh_nagumo import FitzHughNagumoNeuron
from sc_neurocore.network.population import Population
from sc_neurocore.network.projection import Projection
from sc_neurocore.network.network import Network
from sc_neurocore.network.monitor import SpikeMonitor
from sc_neurocore.network.stimulus import PoissonInput
from sc_neurocore.analysis.spike_stats.basic import spike_count, isi, firing_rate
def _run(neuron: FitzHughNagumoNeuron, current: float, steps: int) -> list[int]:
    return [t for t in range(steps) if neuron.step(current) == 1]
def _rhs(
    v: float, w: float, current: float, *, a: float = 0.7, b: float = 0.8, epsilon: float = 0.08
):
    return v - v**3 / 3.0 - w + current, epsilon * (v + a - b * w)
def _rk4_reference(
    v: float,
    w: float,
    current: float,
    dt: float,
    *,
    a: float = 0.7,
    b: float = 0.8,
    epsilon: float = 0.08,
):
    k1v, k1w = _rhs(v, w, current, a=a, b=b, epsilon=epsilon)
    k2v, k2w = _rhs(v + 0.5 * dt * k1v, w + 0.5 * dt * k1w, current, a=a, b=b, epsilon=epsilon)
    k3v, k3w = _rhs(v + 0.5 * dt * k2v, w + 0.5 * dt * k2w, current, a=a, b=b, epsilon=epsilon)
    k4v, k4w = _rhs(v + dt * k3v, w + dt * k3w, current, a=a, b=b, epsilon=epsilon)
    return (
        v + dt * (k1v + 2.0 * k2v + 2.0 * k3v + k4v) / 6.0,
        w + dt * (k1w + 2.0 * k2w + 2.0 * k3w + k4w) / 6.0,
    )

__all__ = ['os', 'math', 'time', 'np', 'pytest', 'assert_throughput_guard', 'FitzHughNagumoNeuron', 'Population', 'Projection', 'Network', 'SpikeMonitor', 'PoissonInput', 'spike_count', 'isi', 'firing_rate', '_run', '_rhs', '_rk4_reference']
